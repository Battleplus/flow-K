"""
批量筛选 SP500 股票 — 按多维信号+回测排名，输出推荐标的

用法:
    python src/screener.py                 # 默认 top 100 股票，中线
    python src/screener.py --top 50        # top 50
    python src/screener.py --hold short    # 短线筛选
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import argparse
import warnings
warnings.filterwarnings("ignore")

from src.data_loader import fetch_data
from src.factors import add_all_factors
from src.strategies import (
    STRATEGIES, backtest_all, rank_strategies, get_latest_consensus, HOLD_PROFILES
)
from src.patterns import add_indicators, detect_all, signal_summary


# ── 统一分析一只股票 ──────────────────────────────────────────
def analyze_ticker(
    ticker: str,
    period: str = "6mo",
    hold_profile: str = "medium",
    worst_case: bool = True,
):
    """对单只股票运行完整管线，返回摘要 dict"""
    try:
        df = fetch_data(ticker, period=period)
        if len(df) < 50:
            return None
    except Exception:
        return None

    df = add_all_factors(df)
    df = add_indicators(df)
    patterns = detect_all(df)

    # ── 信号评分 ──
    s_summary = signal_summary(patterns)
    signal_bull = sum(1 for p in patterns.get("recent", []) if p.get("direction") == "bullish")
    signal_bear = sum(1 for p in patterns.get("recent", []) if p.get("direction") == "bearish")
    signal_total = signal_bull - signal_bear

    # ── 策略共识 ──
    consensus = get_latest_consensus(df)
    consensus_text = consensus.get("consensus", "中性")
    aggregate_score = consensus.get("aggregate_score", 0)

    # 转换为统一方向标签
    if "强烈偏多" in consensus_text:
        direction = "strong_bullish"
    elif "偏多" in consensus_text:
        direction = "bullish"
    elif "强烈偏空" in consensus_text:
        direction = "strong_bearish"
    elif "偏空" in consensus_text:
        direction = "bearish"
    else:
        direction = "neutral"

    # ── 回测 ──
    bt = backtest_all(df, min_trades=3, worst_case=worst_case, hold_profile=hold_profile)
    ranked = rank_strategies(bt, "sharpe_ratio")

    # Top5 策略平均夏普
    top5_sharpes = [bt[sid].sharpe_ratio for sid, _, _ in ranked[:5] if sid in bt]
    avg_sharpe = np.mean(top5_sharpes) if top5_sharpes else 0

    # 等权Top5 权益曲线收益
    from src.strategy_portfolio import backtest_portfolio
    top5_ids = [sid for sid, _, _ in ranked[:5] if sid in bt]
    if top5_ids:
        hp = HOLD_PROFILES.get(hold_profile, HOLD_PROFILES["medium"])
        pf = backtest_portfolio(df, top5_ids, worst_case=worst_case,
                                stop_loss_pct=hp["sl"], take_profit_pct=hp["tp"],
                                hold_days_min=hp["hold_min"], hold_days_max=hp["hold_max"])
        pf_return = pf.total_return
    else:
        pf_return = 0

    # ── 市场状态 ──
    from src.strategy_portfolio import detect_market_regime
    regime = detect_market_regime(df)

    # 最强策略名称
    best_sid = ranked[0][0] if ranked else None
    best_name = STRATEGIES[best_sid].name if best_sid and best_sid in STRATEGIES else "--"
    best_sharpe = round(ranked[0][2], 2) if ranked else 0

    return {
        "ticker": ticker,
        "n_days": len(df),
        "close": round(df["Close"].iloc[-1], 2),
        "consensus_text": consensus_text,
        "consensus_direction": direction,
        "aggregate_score": round(aggregate_score, 1),
        "signal_bull": signal_bull,
        "signal_bear": signal_bear,
        "signal_net": signal_total,
        "market_regime": regime.name,
        "regime_score": round(regime.score, 0),
        "avg_sharpe_top5": round(avg_sharpe, 2),
        "pf_return": round(pf_return * 100, 2),
        "best_strategy": best_name,
        "best_sharpe": best_sharpe,
    }


# ── 获取候选股票池 ────────────────────────────────────────────
def get_candidate_stocks(n: int = 100):
    """读取 SP500 数据，选有足够数据的股票（按最近收盘价高低取大市值）"""
    csv_path = ROOT / "SP500_Historical_Data.csv"
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    cutoff = df["Date"].max() - pd.Timedelta(days=180)
    recent = df[df["Date"] >= cutoff]
    tickers = recent["Ticker"].unique().tolist()
    # 优先取最近收盘价高的（大市值 proxy）
    last_close = recent.groupby("Ticker")["Close"].last().sort_values(ascending=False)
    return last_close.index[:n].tolist()


# ── 排名 & 输出 ───────────────────────────────────────────────
def score_row(r):
    """综合打分：共识方向 + 回测表现 + 信号强度"""
    s = 0
    if r["consensus_direction"] == "strong_bullish":
        s += 30
    elif r["consensus_direction"] == "bullish":
        s += 15
    elif r["consensus_direction"] == "strong_bearish":
        s -= 30
    elif r["consensus_direction"] == "bearish":
        s -= 15
    s += r["aggregate_score"] * 2
    s += r["avg_sharpe_top5"] * 10
    s += r["pf_return"] * 0.5
    s += r["signal_net"] * 2
    s += r["regime_score"] * 0.5
    return round(s, 1)


def screen(n: int = 100, period: str = "6mo", hold: str = "medium"):
    tickers = get_candidate_stocks(n)
    print(f"🔍 正在扫描 {len(tickers)} 只 SP500 股票 (持仓: {HOLD_PROFILES[hold]['label']})...\n")
    results = []
    for i, t in enumerate(tickers):
        sys.stdout.write(f"\r  [{i+1}/{len(tickers)}] {t:6s} ...")
        sys.stdout.flush()
        r = analyze_ticker(t, period=period, hold_profile=hold, worst_case=True)
        if r:
            results.append(r)
    print("\n")

    if not results:
        print("❌ 无有效结果")
        return

    df = pd.DataFrame(results)
    df["score"] = df.apply(score_row, axis=1)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    # ── 输出表格 ──
    cols = ["ticker", "close", "consensus_text", "aggregate_score",
            "signal_net", "market_regime", "avg_sharpe_top5", "pf_return",
            "best_strategy", "best_sharpe", "score"]
    display = df[cols].copy()
    display.columns = ["代码", "价格", "策略共识", "共识分", "信号净",
                       "市场状态", "Top5夏普", "组合收益%", "最强策略", "最佳夏普", "综合分"]

    print("=" * 120)
    print(f"  🏆 SP500 股票多头推荐 (持仓: {HOLD_PROFILES[hold]['label']}, 数据: {period})")
    print("=" * 120)
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_colwidth", 16)
    print(display.head(30).to_string(index=True))
    print("=" * 120)

    # ── 分类汇总 ──
    bullish = df[df["consensus_direction"].isin(["bullish", "strong_bullish"])]
    bearish = df[df["consensus_direction"].isin(["bearish", "strong_bearish"])]
    neutral = df[df["consensus_direction"] == "neutral"]
    print(f"\n📊 统计: 看多 {len(bullish)}只 | 看空 {len(bearish)}只 | 中性 {len(neutral)}只")
    if bullish.shape[0] > 0:
        print(f"📊 多头股: {', '.join(bullish['ticker'].head(10))}")
    print(f"📊 平均 Top5 夏普: {df['avg_sharpe_top5'].mean():.2f}")
    print(f"📊 平均 组合收益: {df['pf_return'].mean():.1f}%")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=100, help="扫描前 N 只股票")
    parser.add_argument("--period", type=str, default="6mo", help="数据周期")
    parser.add_argument("--hold", type=str, default="medium", choices=["short", "medium", "long"])
    args = parser.parse_args()
    screen(n=args.top, period=args.period, hold=args.hold)
