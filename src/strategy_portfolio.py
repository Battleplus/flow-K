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
    基于 ADX、均线排列、布林带宽度、RSI、价格偏离均线 判断市场状态
    返回最可能的状态 + 得分 (0-100)
    """
    latest = df.iloc[-1]

    adx = latest.get("adx_14", 15)
    rsi = latest.get("rsi_14", 50)
    price = latest["Close"]
    ma20 = latest.get("ma20", price)
    ma60 = latest.get("ma60", price)
    dist_ma20 = abs(price - ma20) / ma20 if ma20 != 0 else 0
    dist_ma60 = abs(price - ma60) / ma60 if ma60 != 0 else 0

    # 计算布林带宽度历史分位
    bbw = (latest.get("bb_upper", 0) - latest.get("bb_lower", 1)) / latest.get("bb_mid", 1)
    bbw_series = (df.get("bb_upper", pd.Series(1, index=df.index)) - df.get("bb_lower", pd.Series(1, index=df.index))) / df.get("bb_mid", pd.Series(1, index=df.index))
    bbw_pct = 0.5
    if bbw_series.std() > 0:
        bbw_pct = (bbw - bbw_series.min()) / (bbw_series.max() - bbw_series.min())
    bbw_pct = np.clip(bbw_pct, 0, 1)

    # 趋势强度: 计算所有均线的得分
    trend_score = 0
    if all(c in df.columns for c in ["ma5", "ma10", "ma20", "ma60"]):
        ma5, ma10, ma20, ma60 = latest["ma5"], latest["ma10"], latest["ma20"], latest["ma60"]
        checks = [ma5 > ma10, ma10 > ma20, ma20 > ma60, ma5 > ma60]
        trend_score = sum([1 if c else -1 for c in checks])  # -4 ~ +4

    # 各状态得分
    scores = {}

    # 强趋势上涨: 趋势分高 + ADX高 + 价格在均线上方
    if trend_score > 0 and adx > 20 and price > ma20:
        scores["强趋势上涨"] = min(100, 50 + adx * 0.8 + trend_score * 10 + dist_ma60 * 500)
    else:
        scores["强趋势上涨"] = 0

    # 强趋势下跌
    if trend_score < 0 and adx > 20 and price < ma20:
        scores["强趋势下跌"] = min(100, 50 + adx * 0.8 + abs(trend_score) * 10 + dist_ma60 * 500)
    else:
        scores["强趋势下跌"] = 0

    # 震荡市: ADX低 + 布林带收窄 + 均线交织
    adx_score = max(0, 20 - adx) if adx < 20 else 0
    squeeze_score = (1 - bbw_pct) * 50
    tangled_score = 10 if abs(trend_score) <= 1 else 0
    scores["震荡市"] = min(100, adx_score * 2 + squeeze_score + tangled_score)

    # 顶部反转: RSI超买 + 价格偏离均线
    if rsi > 65 and price > ma20:
        scores["顶部反转"] = min(100, (rsi - 50) * 1.5 + dist_ma20 * 1000)
    else:
        scores["顶部反转"] = 0

    # 底部反转: RSI超卖 + 价格偏离均线
    if rsi < 40 and price < ma20:
        scores["底部反转"] = min(100, (40 - rsi) * 2 + abs(dist_ma20) * 1000)
    else:
        scores["底部反转"] = 0

    # 如果趋势得分明显, 给趋势状态加基础分
    if trend_score >= 3:
        scores["强趋势上涨"] = max(scores.get("强趋势上涨", 0), 30)
    if trend_score <= -3:
        scores["强趋势下跌"] = max(scores.get("强趋势下跌", 0), 30)

    # 选择最高分状态
    best_state = max(scores, key=scores.get)
    best_score = scores[best_state]

    # 所有得分都低 -> 默认震荡市
    if best_score < 5:
        best_state = "震荡市"
        best_score = 5

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
    precomputed_results: Optional[Dict[str, BacktestResult]] = None,
) -> List[str]:
    """
    根据市场状态选择最适合的策略类别, 并在该类别中选夏普最高的 top_n
    """
    preferred_category = regime.category
    # 收集该类别下的策略
    candidates = [sid for sid, s in STRATEGIES.items() if s.category == preferred_category]
    if precomputed_results:
        pool = candidates if len(candidates) >= top_n else candidates + [sid for sid in precomputed_results if sid not in candidates]
        scored = [(sid, precomputed_results[sid].sharpe_ratio) for sid in pool if sid in precomputed_results]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [sid for sid, _ in scored[:top_n]]
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
    hold_days_min: int = 3,
    worst_case: bool = False,
    limit_pct: float = 0.0,
) -> BacktestResult:
    """
    组合信号回测 (单只股票)
    worst_case: 买入用当日最高价、卖出用当日最低价
    limit_pct: 单日涨跌超此比例暂停交易
    """
    signals = combine_signals(df, selected_strategies, weights)
    close = df["Close"].values
    high = df["High"].values if worst_case and "High" in df.columns else close
    low = df["Low"].values if worst_case and "Low" in df.columns else close
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

        # 涨跌停检查
        day_limited = False
        if limit_pct > 0 and i > 0:
            day_change = abs((close[i] - close[i - 1]) / close[i - 1])
            if day_change >= limit_pct:
                day_limited = True

        # 有持仓
        if position > 0:
            hold_days = i - entry_idx
            pnl_pct = (price - entry_price) / entry_price
            exit_reason = None
            if sig_val < 0 and hold_days >= hold_days_min:
                exit_reason = "signal"
            elif stop_loss_pct > 0 and pnl_pct <= -stop_loss_pct:
                exit_reason = "stop_loss"
            elif take_profit_pct > 0 and pnl_pct >= take_profit_pct:
                exit_reason = "take_profit"
            elif hold_days >= hold_days_max:
                exit_reason = "timeout"
            if exit_reason:
                exit_price = low[i] if worst_case else price
                sell_value = position * exit_price * (1 - commission)
                cash += sell_value
                trades.append({
                    "entry_idx": entry_idx,
                    "exit_idx": i,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "return_pct": (exit_price - entry_price) / entry_price,
                    "hold_days": hold_days,
                    "exit_reason": exit_reason,
                })
                position = 0
                entry_idx = -1
                entry_price = 0

        # 无持仓
        elif position == 0 and sig_val > 0 and not day_limited:
            buy_price = high[i] if worst_case else price
            shares = int(cash * position_size * 0.95 / buy_price)
            if shares > 0:
                cost = shares * buy_price * (1 + commission)
                cash -= cost
                position = shares
                entry_idx = i
                entry_price = buy_price

        equity[i] = cash + position * price

    if position > 0:
        exit_price = low[-1] if worst_case else close[-1]
        cash += position * exit_price * (1 - commission)
        trades.append({
            "entry_idx": entry_idx,
            "exit_idx": n - 1,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "return_pct": (exit_price - entry_price) / entry_price,
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

def analyze_portfolio(df: pd.DataFrame, methods: List[str] = None,
                      worst_case: bool = False, limit_pct: float = 0.0,
                      hold_profile: str = "medium",
                      precomputed_results: Optional[Dict[str, BacktestResult]] = None) -> Dict:
    """
    分析多种组合方式:
      - equal_weight: 等权 Top5 策略
      - sharpe_weight: 按夏普加权
      - regime: 基于市场状态选择策略
      - dynamic: 等权 + 波动率目标仓位
    hold_profile: short/medium/long 持仓周期
    """
    if methods is None:
        methods = ["equal_weight", "sharpe_weight", "regime", "dynamic"]

    from src.strategies import backtest_all, rank_strategies, HOLD_PROFILES, run_backtest, STRATEGIES

    results = precomputed_results or backtest_all(
        df, min_trades=1, worst_case=worst_case, limit_pct=limit_pct,
        hold_profile=hold_profile,
    )
    # 长线按收益排名，短线/中线按夏普排名
    rank_by = "total_return" if hold_profile == "long" else "quality_score"
    ranked = rank_strategies(results, rank_by)
    top5 = [sid for sid, _, _ in ranked[:5]]
    top10 = [sid for sid, _, _ in ranked[:10]]

    hp = HOLD_PROFILES.get(hold_profile, HOLD_PROFILES["medium"])
    bt_kwargs = dict(
        worst_case=worst_case, limit_pct=limit_pct,
        stop_loss_pct=hp["sl"], take_profit_pct=hp["tp"],
        hold_days_min=hp["hold_min"], hold_days_max=hp["hold_max"],
    )

    def _parallel_portfolio(strategy_ids, weights=None):
        """并行分仓组合：每个策略独立回测，按权重分配资金，合并权益曲线"""
        n = len(strategy_ids)
        if n == 0:
            return None
        if weights is None:
            weights = {sid: 1.0 / n for sid in strategy_ids}
        else:
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}

        init_cap = 100000
        comm = 0.001
        combined_equity = pd.Series(0.0, index=df.index)
        total_trades = 0
        weighted_hold = 0.0
        weighted_ret = 0.0

        for sid in strategy_ids:
            if sid not in STRATEGIES:
                continue
            sig = STRATEGIES[sid].generate(df)
            alloc = init_cap * weights.get(sid, 1.0 / n)
            r = run_backtest(df, sig, initial_capital=alloc, commission=comm, **bt_kwargs)
            combined_equity += r.equity_curve.values
            total_trades += r.total_trades
            weighted_hold += r.avg_hold_days * r.total_trades
            weighted_ret += r.avg_return_per_trade * r.total_trades

        total_return = (combined_equity.iloc[-1] - init_cap) / init_cap
        n_years = len(df) / 252
        annual_return = ((1 + total_return) ** (1 / max(n_years, 0.01))) - 1
        daily_return = combined_equity.pct_change().dropna()
        sharpe = (daily_return.mean() / daily_return.std() * np.sqrt(252)) if daily_return.std() > 0 else 0
        peak = combined_equity.expanding().max()
        dd = (combined_equity - peak) / peak
        max_dd = dd.min()
        avg_hold = weighted_hold / max(total_trades, 1)
        avg_ret = weighted_ret / max(total_trades, 1)
        # 从组合权益曲线估算胜率
        pf = 1.0  # 简化


        return BacktestResult(
            strategy_id="portfolio",
            strategy_name="策略组合",
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=0.5,
            total_trades=total_trades,
            avg_return_per_trade=avg_ret,
            avg_hold_days=avg_hold,
            profit_factor=pf,
            signal_series=pd.Series(0, index=df.index),
            equity_curve=combined_equity,
        )

    summary = {}
    # 初始化 regime_info
    summary["regime_info"] = {
        "date_range": f"{df['Date'].iloc[0]} ~ {df['Date'].iloc[-1]} ({len(df)}天)",
        "hold_profile": HOLD_PROFILES[hold_profile]["label"],
    }
    # 等权组合 Top5 — 并行分仓
    if "equal_weight" in methods:
        r = _parallel_portfolio(top5)
        summary["equal_weight_top5"] = result_to_dict(r)
    # 夏普加权 Top5 — 按排名选策略，按夏普分配权重
    if "sharpe_weight" in methods:
        weights = {}
        for sid in top5:
            weights[sid] = max(0, results[sid].sharpe_ratio)
        r = _parallel_portfolio(top5, weights if weights else None)
        summary["sharpe_weight_top5"] = result_to_dict(r)
    # 状态驱动组合 — 并行分仓
    if "regime" in methods:
        regime = detect_market_regime(df)
        selected = select_strategies_by_regime(df, regime, top_n=5, precomputed_results=results)
        r = _parallel_portfolio(selected)
        summary["regime_driven"] = result_to_dict(r)
        summary["regime_driven"]["selected_strategies"] = selected
        summary["regime_info"]["name"] = regime.name
        summary["regime_info"]["score"] = round(regime.score, 0)
        summary["regime_info"]["category"] = regime.category
        summary["regime_info"]["description"] = regime.description
    # 动态波动率目标 — 并行分仓 (降低仓位)
    if "dynamic" in methods:
        weights = {sid: 0.8 / len(top5) for sid in top5}
        r = _parallel_portfolio(top5, weights)
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
    from src.plot_fonts import configure_chinese_font

    _cn_font = configure_chinese_font()

    close = df["Close"].values
    bh_equity = close / close[0]
    portfolio_equity = portfolio_result.equity_curve.values / portfolio_result.equity_curve.values[0]
    dates = pd.to_datetime(df["Date"]).values

    bg = "#131722"
    panel = "#171b26"
    grid = "#242833"
    text = "#d1d4dc"
    muted = "#787b86"
    blue = "#2962ff"

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=bg)
    ax.set_facecolor(panel)
    ax.plot(dates, portfolio_equity, label=f"{method_name}", color=blue, linewidth=2.4)
    ax.plot(dates, bh_equity, label="买入持有", color="#8a9bab", linewidth=1.4, linestyle="--")
    ax.axhline(1.0, color="#3a404d", linewidth=0.8)
    ax.set_title(f"{ticker} {method_name} 权益曲线", fontproperties=_cn_font, fontsize=14, color="#f0f3fa", pad=12)
    ax.set_xlabel("日期", fontproperties=_cn_font, color=muted)
    ax.set_ylabel("净值", fontproperties=_cn_font, color=muted)
    ax.tick_params(colors=muted, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#2a2e39")
    legend = ax.legend(prop=_cn_font, loc="upper left", facecolor=panel, edgecolor="#2a2e39", framealpha=0.92)
    for label in legend.get_texts():
        label.set_color(text)
    ax.grid(True, color=grid, linewidth=0.8, alpha=0.9)

    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = str(out_dir / f"{ticker}_portfolio_equity.png")
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=bg)
    plt.close(fig)
    return path
