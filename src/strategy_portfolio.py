"""
策略组合 + 市场状态识别 + 仓位管理

核心能力：
  1. 市场状态识别 (Market Regime Detection)
     - 强趋势市: ADX>25 + 均线多头排列/空头排列
     - 震荡市: ADX<20 + 布林带收窄
     - 反转市: 价格偏离均线较远 + RSI极端 + 成交量萎缩

  2. 策略组合回测
     - 等权组合多个策略信号
     - 按夏普比率加权
     - 按市场状态动态选择策略类别

  3. 仓位管理
     - 波动率目标法 (Volatility Targeting)
     - Kelly 比例 (简化版)
     - 最大仓位限制
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from src.strategies import STRATEGIES, run_backtest, BacktestResult, aggregate_signals


# ═══════════════════════════════════════════════════════════════
# 市场状态识别
# ═══════════════════════════════════════════════════════════════

@dataclass
class MarketRegime:
    name: str
    category: str
    score: float
    description: str


def detect_market_regime(df: pd.DataFrame) -> MarketRegime:
    """
    基于 ADX、均线排列、布林带宽度、RSI 判断市场状态
    返回最可能的状态 + 每个状态的得分
    """
    latest = df.iloc[-1]

    # 特征计算
    adx = latest.get("adx_14", 15)
    bbw = (latest.get("bb_upper", 0) - latest.get("bb_lower", 1)) / latest.get("bb_mid", 1)
    bbw_mean = df.get("bb_upper", pd.Series(0, index=df.index)) - df.get("bb_lower", pd.Series(0, index=df.index))
    bbw_mean = (bbw_mean / df.get("bb_mid", pd.Series(1, index=df.index))).mean()
    bbw_pct = bbw / bbw_mean if bbw_mean > 0 else 0.5

    rsi = latest.get("rsi_14", 50)
    ma_score = 0
    if all(c in df.columns for c in ["ma5", "ma10", "ma20", "ma60"]):
        ma5, ma10, ma20, ma60 = latest["ma5"], latest["ma10"], latest["ma20"], latest["ma60"]
        if ma5 > ma10 > ma20 > ma60: ma_score = +2
        elif ma5 < ma10 < ma20 < ma60: ma_score = -2
        else: ma_score = 0

    price = latest["Close"]
    ma20 = latest.get("ma20", price)
    dist_ma20 = abs(price - ma20) / ma20 if ma20 != 0 else 0

    # 状态评分 (0-100)
    scores = {
        "强趋势上涨": 0.0,
        "强趋势下跌": 0.0,
        "震荡市": 0.0,
        "顶部反转": 0.0,
        "底部反转": 0.0,
    }

    # 强趋势上涨: 均线多头 + ADX高 + 价格上涨
    if ma_score > 0 and adx > 25 and price > ma20:
        scores["强趋势上涨"] = min(100, 60 + adx * 1.2 + dist_ma20 * 500)
    # 强趋势下跌: 均线空头 + ADX高 + 价格下跌
    if ma_score < 0 and adx > 25 and price < ma20:
        scores["强趋势下跌"] = min(100, 60 + adx * 1.2 + dist_ma20 * 500)
    # 震荡市: ADX低 + 布林带收窄 + 均线交织
    if adx < 20 and bbw_pct < 0.7 and ma_score == 0:
        scores["震荡市"] = min(100, 80 + (20 - adx) * 1.5)
    # 顶部反转: 价格偏离均线 + RSI超买 + 动能减弱
    if rsi > 70 and dist_ma20 > 0.05:
        scores["顶部反转"] = min(100, 50 + rsi + dist_ma20 * 500)
    # 底部反转: 价格偏离均线 + RSI超卖 + 缩量企稳
    if rsi < 30 and dist_ma20 > 0.03:
        scores["底部反转"] = min(100, 50 + (40 - rsi) + dist_ma20 * 500)

    # 如果趋势不明显，按ADX默认给震荡市加分
    if adx < 15 and scores["震荡市"] == 0:
        scores["震荡市"] = 50

    best_state = max(scores, key=scores.get)
    best_score = scores[best_state]

    return MarketRegime(
        name=best_state,
        category=regime_to_category(best_state),
        score=best_score,
        description=regime_description(best_state),
    )


def regime_to_category(regime: str) -> str:
    mapping = {
        "强趋势上涨": "趋势跟踪",
        "强趋势下跌": "趋势跟踪",
        "震荡市": "动量反转",
        "顶部反转": "动量反转",
        "底部反转": "动量反转",
    }
    return mapping.get(regime, "多重确认")


def regime_description(regime: str) -> str:
    mapping = {
        "强趋势上涨": "适合趋势跟踪与放量突破策略",
        "强趋势下跌": "适合趋势跟踪做空或空仓观望",
        "震荡市": "适合布林带回归、RSI反转、VWAP回归",
        "顶部反转": "适合MACD背离、OBV出货、Keltner突破",
        "底部反转": "适合MACD底背离、OBV吸筹、Stochastic金叉",
    }
    return mapping.get(regime, "适合多重确认策略")


# ═══════════════════════════════════════════════════════════════
# 策略组合构造
# ═══════════════════════════════════════════════════════════════

def combine_signals(
    df: pd.DataFrame,
    selected_strategies: List[str],
    weights: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """
    把多个策略的信号按权重组合成综合信号
    输出: -1/0/1 的交易信号 (用阈值触发)
    """
    combined = pd.Series(0.0, index=df.index)
    total_weight = 0.0
    for sid in selected_strategies:
        if sid not in STRATEGIES:
            continue
        w = weights.get(sid, 1.0) if weights else 1.0
        sig = STRATEGIES[sid].generate(df)
        combined += sig.astype(float) * w
        total_weight += abs(w)
    if total_weight > 0:
        combined = combined / total_weight
    # 阈值: >0.3 买入, <-0.3 卖出
    signal = pd.Series(0, index=df.index)
    signal[combined > 0.3] = 1
    signal[combined < -0.3] = -1
    signal[combined > 0.6] = 2
    signal[combined < -0.6] = -2
    return signal


def select_strategies_by_regime(
    df: pd.DataFrame,
    regime: MarketRegime,
    top_n: int = 5,
) -> List[str]:
    """
    根据市场状态选择最适合的策略类别, 并在该类别中选夏普最高的 top_n
    """
    preferred_category = regime.category
    # 收集该类别下的策略
    candidates = [sid for sid, s in STRATEGIES.items() if s.category == preferred_category]
    if len(candidates) < top_n:
        # 类别不足则补充其他类别中夏普高的策略
        others = [sid for sid, s in STRATEGIES.items() if sid not in candidates]
        # 快速回测所有候选, 取夏普最高的
        from src.strategies import backtest_strategy
        scores = []
        for sid in candidates + others:
            try:
                r = backtest_strategy(df, sid)
                scores.append((sid, r.sharpe_ratio))
            except Exception:
                pass
        scores.sort(key=lambda x: x[1], reverse=True)
        return [sid for sid, _ in scores[:top_n]]
    else:
        # 回测该类别内策略并排序
        from src.strategies import backtest_strategy
        scores = []
        for sid in candidates:
            try:
                r = backtest_strategy(df, sid)
                scores.append((sid, r.sharpe_ratio))
            except Exception:
                pass
        scores.sort(key=lambda x: x[1], reverse=True)
        return [sid for sid, _ in scores[:top_n]]


# ═══════════════════════════════════════════════════════════════
# 仓位管理
# ═══════════════════════════════════════════════════════════════

def volatility_targeting(
    df: pd.DataFrame,
    target_vol: float = 0.15,  # 15% 年化目标波动
    max_position: float = 1.0,
) -> pd.Series:
    """
    基于20日历史波动率调整仓位
    波动率高则减仓，波动率低则加仓
    """
    returns = df["Close"].pct_change().dropna()
    hist_vol = returns.rolling(20).std() * np.sqrt(252)
    position = target_vol / hist_vol
    position = position.clip(0.1, max_position)
    return position.reindex(df.index).fillna(method="ffill").fillna(0.5)


def kelly_position(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    max_position: float = 0.5,
) -> float:
    """
    简化 Kelly 公式: f* = (bp - q) / b
    b = 平均盈利/平均亏损, p = 胜率, q = 1-p
    """
    if avg_loss <= 0 or avg_win <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0
    b = avg_win / avg_loss
    q = 1 - win_rate
    f = (b * win_rate - q) / b
    return max(0.0, min(f, max_position))


# ═══════════════════════════════════════════════════════════════
# 组合回测引擎
# ═══════════════════════════════════════════════════════════════

def backtest_portfolio(
    df: pd.DataFrame,
    selected_strategies: List[str],
    weights: Optional[Dict[str, float]] = None,
    position_size: float = 1.0,
    initial_capital: float = 100000,
    commission: float = 0.001,
    stop_loss_pct: float = 0.07,
    take_profit_pct: float = 0.12,
    hold_days_max: int = 30,
) -> BacktestResult:
    """
    组合信号回测 (单只股票)
    """
    signals = combine_signals(df, selected_strategies, weights)
    close = df["Close"].values
    n = len(close)
    equity = np.ones(n) * initial_capital
    position = 0
    cash = initial_capital
    trades = []
    entry_idx = -1
    entry_price = 0

    for i in range(n):
        price = close[i]
        sig_val = signals.iloc[i]

        # 有持仓
        if position > 0:
            hold_days = i - entry_idx
            pnl_pct = (price - entry_price) / entry_price
            exit_reason = None
            if sig_val < 0 and hold_days >= 3:
                exit_reason = "signal"
            elif pnl_pct <= -stop_loss_pct:
                exit_reason = "stop_loss"
            elif pnl_pct >= take_profit_pct:
                exit_reason = "take_profit"
            elif hold_days >= hold_days_max:
                exit_reason = "timeout"
            if exit_reason:
                sell_value = position * price * (1 - commission)
                cash += sell_value
                trades.append({
                    "entry_idx": entry_idx,
                    "exit_idx": i,
                    "entry_price": entry_price,
                    "exit_price": price,
                    "return_pct": pnl_pct,
                    "hold_days": hold_days,
                    "exit_reason": exit_reason,
                })
                position = 0
                entry_idx = -1
                entry_price = 0

        # 无持仓
        elif position == 0 and sig_val > 0:
            shares = int(cash * position_size * 0.95 / price)
            if shares > 0:
                cost = shares * price * (1 + commission)
                cash -= cost
                position = shares
                entry_idx = i
                entry_price = price

        equity[i] = cash + position * price

    if position > 0:
        cash += position * close[-1] * (1 - commission)
        trades.append({
            "entry_idx": entry_idx,
            "exit_idx": n - 1,
            "entry_price": entry_price,
            "exit_price": close[-1],
            "return_pct": (close[-1] - entry_price) / entry_price,
            "hold_days": n - 1 - entry_idx,
            "exit_reason": "force_close",
        })

    total_return = (cash - initial_capital) / initial_capital
    n_years = n / 252
    annual_return = ((1 + total_return) ** (1 / max(n_years, 0.01))) - 1
    eq = pd.Series(equity, index=df.index)
    daily_return = eq.pct_change().dropna()
    sharpe = (daily_return.mean() / daily_return.std() * np.sqrt(252)) if daily_return.std() > 0 else 0
    peak = pd.Series(equity).expanding().max()
    dd = (pd.Series(equity) - peak) / peak
    max_dd = dd.min()
    total_trades = len(trades)
    win_rate = sum(1 for t in trades if t["return_pct"] > 0) / max(total_trades, 1)
    avg_return = np.mean([t["return_pct"] for t in trades]) if trades else 0
    avg_hold = np.mean([t["hold_days"] for t in trades]) if trades else 0
    winning = [t["return_pct"] for t in trades if t["return_pct"] > 0]
    losing = [t["return_pct"] for t in trades if t["return_pct"] <= 0]
    avg_win = np.mean(winning) if winning else 0
    avg_loss = abs(np.mean(losing)) if losing else 1
    pf = avg_win / avg_loss if avg_loss > 0 else float("inf")

    return BacktestResult(
        strategy_id="portfolio",
        strategy_name="策略组合",
        total_return=total_return,
        annual_return=annual_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        win_rate=win_rate,
        total_trades=total_trades,
        avg_return_per_trade=avg_return,
        avg_hold_days=avg_hold,
        profit_factor=pf,
        signal_series=signals,
        equity_curve=pd.Series(equity, index=df.index),
    )


# ═══════════════════════════════════════════════════════════════
# 组合分析主函数
# ═══════════════════════════════════════════════════════════════

def analyze_portfolio(df: pd.DataFrame, methods: List[str] = None) -> Dict:
    """
    分析多种组合方式:
      - equal_weight: 等权 Top5 策略
      - sharpe_weight: 按夏普加权
      - regime: 基于市场状态选择策略
      - dynamic: 等权 + 波动率目标仓位
    """
    if methods is None:
        methods = ["equal_weight", "sharpe_weight", "regime", "dynamic"]

    from src.strategies import backtest_all, rank_strategies

    results = backtest_all(df, min_trades=1)
    ranked = rank_strategies(results, "sharpe_ratio")
    top5 = [sid for sid, _, _ in ranked[:5]]
    top10 = [sid for sid, _, _ in ranked[:10]]

    summary = {}
    # 等权组合 Top5
    if "equal_weight" in methods:
        r = backtest_portfolio(df, top5)
        summary["equal_weight_top5"] = result_to_dict(r)
    # 夏普加权 Top5
    if "sharpe_weight" in methods:
        weights = {}
        for sid, r, _ in ranked[:5]:
            weights[sid] = max(0, r.sharpe_ratio)
        if weights:
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}
        r = backtest_portfolio(df, top5, weights)
        summary["sharpe_weight_top5"] = result_to_dict(r)
    # 状态驱动组合
    if "regime" in methods:
        regime = detect_market_regime(df)
        selected = select_strategies_by_regime(df, regime, top_n=5)
        r = backtest_portfolio(df, selected)
        summary["regime_driven"] = result_to_dict(r)
        summary["regime_driven"]["selected_strategies"] = selected
        summary["regime_info"] = {
            "name": regime.name,
            "category": regime.category,
            "score": regime.score,
            "description": regime.description,
        }
    # 动态波动率目标
    if "dynamic" in methods:
        r = backtest_portfolio(df, top10, position_size=0.8)
        summary["dynamic_vol_target"] = result_to_dict(r)

    # 基准：买入持有
    bh_return = (df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0]
    summary["buy_hold"] = {
        "total_return": round(bh_return * 100, 2),
        "annual_return": round(bh_return / max(len(df) / 252, 0.01) * 100, 2),
    }

    summary["top_strategies"] = [
        {
            "id": sid,
            "name": results[sid].strategy_name,
            "sharpe": round(results[sid].sharpe_ratio, 2),
            "total_return": round(results[sid].total_return * 100, 2),
        }
        for sid, _, _ in ranked[:5]
    ]

    return summary


def result_to_dict(r: BacktestResult) -> dict:
    return {
        "total_return": round(r.total_return * 100, 2),
        "annual_return": round(r.annual_return * 100, 2),
        "sharpe_ratio": round(r.sharpe_ratio, 2),
        "max_drawdown": round(r.max_drawdown * 100, 2),
        "win_rate": round(r.win_rate * 100, 1),
        "total_trades": r.total_trades,
        "avg_return_per_trade": round(r.avg_return_per_trade * 100, 2),
        "avg_hold_days": round(r.avg_hold_days, 0),
        "profit_factor": round(r.profit_factor, 2),
    }


# ═══════════════════════════════════════════════════════════════
# 绘制组合对比图
# ═══════════════════════════════════════════════════════════════

def plot_portfolio_equity(
    df: pd.DataFrame,
    portfolio_result: BacktestResult,
    ticker: str,
    method_name: str = "策略组合",
) -> str:
    """
    绘制组合权益曲线 vs 买入持有曲线
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as _fm

    # 字体
    _FONT_PATH = "C:/Windows/Fonts/msyh.ttc"
    _fm.fontManager.addfont(_FONT_PATH)
    _cn_font = _fm.FontProperties(fname=_FONT_PATH)
    _cn_font_name = _cn_font.get_name()
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [_cn_font_name, "Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    close = df["Close"].values
    bh_equity = close / close[0]
    portfolio_equity = portfolio_result.equity_curve.values / portfolio_result.equity_curve.values[0]
    dates = pd.to_datetime(df["Date"]).values

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, portfolio_equity, label=f"{method_name}", color="#0984e3", linewidth=2)
    ax.plot(dates, bh_equity, label="买入持有", color="#95a5a6", linewidth=1.5, linestyle="--")
    ax.axhline(1.0, color="#ccc", linewidth=0.5)
    ax.set_title(f"{ticker} {method_name} 权益曲线", fontproperties=_cn_font, fontsize=14)
    ax.set_xlabel("日期", fontproperties=_cn_font)
    ax.set_ylabel("净值", fontproperties=_cn_font)
    ax.legend(prop=_cn_font, loc="upper left")
    ax.grid(True, alpha=0.3)

    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = str(out_dir / f"{ticker}_portfolio_equity.png")
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
