"""
策略引擎 —— 多因子组合 → 交易信号 → 回测 → 绩效评估

策略分类：
  A. 趋势跟踪 (Trend Following)
  B. 动量反转 (Momentum/Reversal)
  C. 成交量确认 (Volume Confirmation)
  D. 波动率突破 (Volatility Breakout)
  E. 多重确认 (Multi-Confirmation)

每个策略 = 规则集合 → 产生信号(BUY/SELL/HOLD) → 可回测
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


# ═══════════════════════════════════════════════════════════════
# 信号与策略数据结构
# ═══════════════════════════════════════════════════════════════

class Signal(Enum):
    STRONG_SELL = -2
    SELL = -1
    HOLD = 0
    BUY = 1
    STRONG_BUY = 2


@dataclass
class Strategy:
    id: str
    name: str
    category: str
    description: str
    factors_used: List[str]
    # 核心逻辑: (df) -> pd.Series of Signal values
    generate: Callable
    # 回测参数
    stop_loss_pct: float = 0.08      # 止损 8%
    take_profit_pct: float = 0.15    # 止盈 15%
    hold_days_min: int = 3           # 最少持仓天数
    hold_days_max: int = 30          # 最长持仓天数
    position_size: float = 1.0       # 仓位比例


@dataclass
class BacktestResult:
    strategy_id: str
    strategy_name: str
    total_return: float           # 总收益率
    annual_return: float          # 年化收益率
    sharpe_ratio: float           # 夏普比率
    max_drawdown: float           # 最大回撤
    win_rate: float               # 胜率
    total_trades: int             # 总交易次数
    avg_return_per_trade: float   # 平均每笔收益
    avg_hold_days: float          # 平均持仓天数
    profit_factor: float          # 盈亏比
    signal_series: pd.Series      # 信号序列
    equity_curve: pd.Series       # 权益曲线


# ═══════════════════════════════════════════════════════════════
# 信号生成辅助函数
# ═══════════════════════════════════════════════════════════════

def _cross_above(a: pd.Series, b: pd.Series) -> pd.Series:
    """a 上穿 b (金叉)"""
    return (a > b) & (a.shift(1) <= b.shift(1))

def _cross_below(a: pd.Series, b: pd.Series) -> pd.Series:
    """a 下穿 b (死叉)"""
    return (a < b) & (a.shift(1) >= b.shift(1))

def _rising(series: pd.Series, n: int = 3) -> pd.Series:
    """连续 n 天上升"""
    return series.diff().rolling(n).apply(lambda x: (x > 0).all(), raw=True).fillna(0).astype(bool)

def _falling(series: pd.Series, n: int = 3) -> pd.Series:
    """连续 n 天下降"""
    return series.diff().rolling(n).apply(lambda x: (x < 0).all(), raw=True).fillna(0).astype(bool)

def _above_ma(df: pd.Series, ma_col: str) -> pd.Series:
    """价格在均线上方"""
    return df["Close"] > df[ma_col]

def _below_ma(df: pd.Series, ma_col: str) -> pd.Series:
    """价格在均线下方"""
    return df["Close"] < df[ma_col]


# ═══════════════════════════════════════════════════════════════
# 回测引擎
# ═══════════════════════════════════════════════════════════════

def run_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    stop_loss_pct: float = 0.08,
    take_profit_pct: float = 0.15,
    hold_days_min: int = 3,
    hold_days_max: int = 30,
    initial_capital: float = 100000,
    commission: float = 0.001,
    worst_case: bool = False,
    limit_pct: float = 0.0,
) -> BacktestResult:
    """
    简易事件驱动回测引擎
    - 买入: 信号 > 0
    - 卖出: 信号 < 0 或 止损/止盈/超时
    - 不做空，只做多

    worst_case: 开启后，买入用当日最高价、卖出用当日最低价（最差执行）
    limit_pct: >0 时模拟涨跌停，单日涨跌幅超过此值暂停交易
    """
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
    hold_days = 0
    skip_count = 0

    for i in range(n):
        price = close[i]
        sig_val = signals.iloc[i] if i < len(signals) else 0

        # 涨跌停检查: 当日涨跌幅超过阈值则暂停交易
        day_limited = False
        if limit_pct > 0 and i > 0:
            day_change = abs((close[i] - close[i - 1]) / close[i - 1])
            if day_change >= limit_pct:
                day_limited = True
                skip_count += 1

        # 有持仓: 检查出场条件
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
                # 最差执行: 卖出用当日最低价
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

        # 无持仓: 检查入场条件（涨跌停日不交易）
        elif position == 0 and sig_val > 0 and not day_limited:
            # 最差执行: 买入用当日最高价
            buy_price = high[i] if worst_case else price
            position = int(cash * 0.95 / buy_price)
            if position > 0:
                cost = position * buy_price * (1 + commission)
                cash -= cost
                entry_idx = i
                entry_price = buy_price

        # 更新权益（按收盘价估值）
        equity[i] = cash + position * price

    # 强制平仓
    if position > 0:
        # 最差执行: 最后一天用最低价卖出
        exit_price = low[-1] if worst_case else close[-1]
        final_value = position * exit_price * (1 - commission)
        cash += final_value
        trades.append({
            "entry_idx": entry_idx,
            "exit_idx": n - 1,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "return_pct": (exit_price - entry_price) / entry_price,
            "hold_days": n - 1 - entry_idx,
            "exit_reason": "force_close",
        })

    # 计算绩效指标
    total_return = (cash - initial_capital) / initial_capital
    n_years = n / 252
    annual_return = ((1 + total_return) ** (1 / max(n_years, 0.01))) - 1

    # 夏普比率
    eq = pd.Series(equity, index=df.index)
    daily_return = eq.pct_change().dropna()
    sharpe = (daily_return.mean() / daily_return.std() * np.sqrt(252)) if daily_return.std() > 0 else 0

    # 最大回撤
    peak = pd.Series(equity).expanding().max()
    dd = (pd.Series(equity) - peak) / peak
    max_dd = dd.min()

    # 交易统计
    total_trades = len(trades)
    win_rate = sum(1 for t in trades if t["return_pct"] > 0) / max(total_trades, 1)
    avg_return = np.mean([t["return_pct"] for t in trades]) if trades else 0
    avg_hold = np.mean([t["hold_days"] for t in trades]) if trades else 0

    # 盈亏比
    winning = [t["return_pct"] for t in trades if t["return_pct"] > 0]
    losing = [t["return_pct"] for t in trades if t["return_pct"] <= 0]
    avg_win = np.mean(winning) if winning else 0
    avg_loss = abs(np.mean(losing)) if losing else 1
    profit_factor = avg_win / avg_loss if avg_loss > 0 else float("inf")

    return BacktestResult(
        strategy_id="",
        strategy_name="",
        total_return=total_return,
        annual_return=annual_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        win_rate=win_rate,
        total_trades=total_trades,
        avg_return_per_trade=avg_return,
        avg_hold_days=avg_hold,
        profit_factor=profit_factor,
        signal_series=signals,
        equity_curve=pd.Series(equity, index=df.index),
    )


# ═══════════════════════════════════════════════════════════════
# A. 趋势跟踪策略
# ═══════════════════════════════════════════════════════════════

def _strategy_ma_ribbon(df: pd.DataFrame) -> pd.Series:
    """
    均线多头排列策略
    BUY:  MA5>MA10>MA20>MA60 且价格在MA20上方且RSI 50-70
    STRONG_BUY: 上述条件 + MA20斜率上升 + 放量
    SELL: 均线空头排列 或 价格跌破MA60
    """
    sig = pd.Series(0, index=df.index)
    bull_align = (df["ma5"] > df["ma10"]) & (df["ma10"] > df["ma20"]) & (df["ma20"] > df["ma60"])
    price_above = df["Close"] > df["ma20"]
    rsi_ok = (df.get("rsi_14", pd.Series(50, index=df.index)) > 50) & \
             (df.get("rsi_14", pd.Series(50, index=df.index)) < 75)
    vol_up = df.get("volume_ratio", pd.Series(1, index=df.index)) > 1.0
    slope_up = df.get("trend_slope_ma20", pd.Series(0, index=df.index)) > 0

    bear_align = (df["ma5"] < df["ma10"]) & (df["ma10"] < df["ma20"]) & (df["ma20"] < df["ma60"])
    price_below_60 = df["Close"] < df["ma60"]

    sig[bull_align & price_above & rsi_ok & slope_up & vol_up] = Signal.STRONG_BUY.value
    sig[bull_align & price_above & rsi_ok] = Signal.BUY.value
    sig[bear_align | price_below_60] = Signal.SELL.value
    return sig


def _strategy_ema_cross(df: pd.DataFrame) -> pd.Series:
    """
    EMA交叉策略
    BUY:  EMA12 上穿 EMA26 + 价格在EMA50上方
    SELL: EMA12 下穿 EMA26 + 价格在EMA50下方
    """
    sig = pd.Series(0, index=df.index)
    golden = _cross_above(df["ema12"], df["ema26"])
    dead = _cross_below(df["ema12"], df["ema26"])
    sig[golden & (df["Close"] > df["ema50"])] = Signal.BUY.value
    sig[dead | ((df["Close"] < df["ema50"]) & (df["ema12"] < df["ema26"]))] = Signal.SELL.value
    # 同时发生死叉+EMA空头排列 → STRONG_SELL
    sig[dead & (df["Close"] < df["ema50"]) & (df["ema12"] < df["ema50"])] = Signal.STRONG_SELL.value
    return sig


def _strategy_trend_breakout(df: pd.DataFrame) -> pd.Series:
    """
    趋势突破策略
    BUY:  价格突破布林上轨 + ADX > 25 + 成交量 > 1.5倍
    SELL: 价格跌破布林中轨 + ADX > 20
    """
    sig = pd.Series(0, index=df.index)
    has_bb = all(c in df.columns for c in ["bb_upper", "bb_mid", "bb_lower"])
    has_adx = "adx_14" in df.columns
    has_vol = "volume_ratio" in df.columns
    if not (has_bb and has_adx):
        return sig

    price = df["Close"]
    breakout = price > df["bb_upper"]
    adx_strong = df["adx_14"] > 25
    vol_confirm = df.get("volume_ratio", pd.Series(1, index=df.index)) > 1.5 if has_vol else pd.Series(True, index=df.index)
    sig[breakout & adx_strong & vol_confirm] = Signal.BUY.value
    sig[(price < df["bb_mid"]) & (df["adx_14"] > 20)] = Signal.SELL.value
    # 突破 + 强力确认
    sig[breakout & adx_strong & vol_confirm & (df["adx_14"] > 35)] = Signal.STRONG_BUY.value
    return sig


def _strategy_ma_pullback(df: pd.DataFrame) -> pd.Series:
    """
    均线回踩买入策略
    BUY:  多头排列中价格回调至MA10/MA20附近(<3%) + RSI 40-55
    SELL: 价格跌破MA60 或 空头排列
    """
    sig = pd.Series(0, index=df.index)
    bull_align = (df["ma5"] > df["ma10"]) & (df["ma10"] > df["ma20"]) & (df["ma20"] > df["ma60"])
    dist_ma20 = abs(df["Close"] - df["ma20"]) / df["ma20"]
    pullback = dist_ma20 < 0.03  # 价格在MA20 3%范围内
    rsi_mid = (df.get("rsi_14", pd.Series(50, index=df.index)) > 40) & \
              (df.get("rsi_14", pd.Series(50, index=df.index)) < 55)
    sig[bull_align & pullback & rsi_mid] = Signal.BUY.value
    sig[(df["Close"] < df["ma60"]) | (df["ma5"] < df["ma60"])] = Signal.SELL.value
    return sig


def _strategy_trend_acceleration(df: pd.DataFrame) -> pd.Series:
    """
    趋势加速度策略
    检测MA20斜率加速上升（二阶导 > 0）
    BUY:  斜率加速 + 前一周斜率 < 当前斜率
    SELL: 斜率减速变负
    """
    sig = pd.Series(0, index=df.index)
    if "trend_slope_ma20" not in df.columns:
        return sig
    slope = df["trend_slope_ma20"]
    slope_ma5 = slope.rolling(5).mean()
    slope_ma5_prev = slope_ma5.shift(5)
    # 加速: 当前斜率 > 前5天斜率 且 > 0
    accel_up = (slope_ma5 > slope_ma5_prev) & (slope_ma5 > 0) & (slope > 0)
    sig[accel_up] = Signal.BUY.value
    # 减速/转负
    decel = (slope_ma5 < slope_ma5_prev) | (slope < -0.5)
    sig[decel] = Signal.SELL.value
    return sig


# ═══════════════════════════════════════════════════════════════
# B. 动量反转策略
# ═══════════════════════════════════════════════════════════════

def _strategy_rsi_extreme(df: pd.DataFrame) -> pd.Series:
    """
    RSI超买超卖策略
    BUY:  RSI < 30 后回升 + 成交量放大
    SELL: RSI > 75 后回落
    """
    sig = pd.Series(0, index=df.index)
    if "rsi_14" not in df.columns:
        return sig
    rsi = df["rsi_14"]
    oversold = rsi < 30
    overbought = rsi > 75
    rsi_rising = rsi > rsi.shift(1)
    rsi_falling = rsi < rsi.shift(1)
    sig[oversold & rsi_rising] = Signal.BUY.value
    sig[overbought & rsi_falling] = Signal.SELL.value
    # 极端值
    sig[rsi < 20] = Signal.STRONG_BUY.value
    sig[rsi > 85] = Signal.STRONG_SELL.value
    return sig


def _strategy_bb_mean_reversion(df: pd.DataFrame) -> pd.Series:
    """
    布林带回归策略
    BUY:  价格跌破布林下轨 + RSI < 35 + 次日回升
    SELL: 价格突破布林上轨 + RSI > 70
    然后等待回归中轨
    """
    sig = pd.Series(0, index=df.index)
    if not all(c in df.columns for c in ["bb_upper", "bb_mid", "bb_lower"]):
        return sig
    price = df["Close"]
    # 下轨超卖
    below_lower = price < df["bb_lower"]
    rsi_low = df.get("rsi_14", pd.Series(50, index=df.index)) < 35
    sig[below_lower & rsi_low] = Signal.BUY.value
    # 上轨超买
    above_upper = price > df["bb_upper"]
    rsi_high = df.get("rsi_14", pd.Series(50, index=df.index)) > 70
    sig[above_upper & rsi_high] = Signal.SELL.value
    # 回归中轨后平仓
    sig[(price < df["bb_mid"]) & (price.shift(1) > df["bb_mid"].shift(1))] = Signal.SELL.value
    return sig


def _strategy_stochastic_cross(df: pd.DataFrame) -> pd.Series:
    """
    Stochastic 双线交叉策略
    BUY:  %K 上穿 %D 且两者都在超卖区 (<25)
    SELL: %K 下穿 %D 且两者都在超买区 (>75)
    """
    sig = pd.Series(0, index=df.index)
    if not all(c in df.columns for c in ["stochastic_k", "stochastic_d"]):
        return sig
    k, d = df["stochastic_k"], df["stochastic_d"]
    golden = _cross_above(k, d)
    dead = _cross_below(k, d)
    sig[golden & (k < 25)] = Signal.BUY.value
    sig[dead & (k > 75)] = Signal.SELL.value
    # 强信号
    sig[golden & (k < 15)] = Signal.STRONG_BUY.value
    sig[dead & (k > 85)] = Signal.STRONG_SELL.value
    return sig


def _strategy_macd_divergence(df: pd.DataFrame) -> pd.Series:
    """
    MACD背离策略
    底背离: 价格新低但MACD柱不新低 → BUY
    顶背离: 价格新高但MACD柱不新高 → SELL
    """
    sig = pd.Series(0, index=df.index)
    if "macd_hist" not in df.columns:
        return sig
    price = df["Close"]
    hist = df["macd_hist"]
    # 底背离检测 (用20天窗口)
    n = 20
    price_20_low = price.rolling(n).apply(lambda x: x.argmin(), raw=True).fillna(0).astype(int)
    hist_20_low = hist.rolling(n).apply(lambda x: x.argmin(), raw=True).fillna(0).astype(int)
    div_bull = (price_20_low == 0) & (hist_20_low != 0) & (hist_20_low > 3)
    # 顶背离
    price_20_high = price.rolling(n).apply(lambda x: x.argmax(), raw=True).fillna(0).astype(int)
    hist_20_high = hist.rolling(n).apply(lambda x: x.argmax(), raw=True).fillna(0).astype(int)
    div_bear = (price_20_high == 0) & (hist_20_high != 0) & (hist_20_high > 3)
    sig[div_bull] = Signal.BUY.value
    sig[div_bear] = Signal.SELL.value
    return sig


def _strategy_williams_r_reversal(df: pd.DataFrame) -> pd.Series:
    """
    Williams %R 反转策略
    BUY:  %R < -80 后回升到 -50 以上
    SELL: %R > -20 后回落到 -50 以下
    """
    sig = pd.Series(0, index=df.index)
    if "williams_r" not in df.columns:
        return sig
    wr = df["williams_r"]
    sig[(wr < -80) & (wr > wr.shift(1))] = Signal.BUY.value
    sig[(wr > -20) & (wr < wr.shift(1))] = Signal.SELL.value
    return sig


# ═══════════════════════════════════════════════════════════════
# C. 成交量确认策略
# ═══════════════════════════════════════════════════════════════

def _strategy_volume_breakout(df: pd.DataFrame) -> pd.Series:
    """
    放量突破策略
    BUY:  价格突破20日高 + 成交量 > 2倍均量 + MA多头
    SELL: 价格跌破20日低 + 缩量
    """
    sig = pd.Series(0, index=df.index)
    high_20 = df["Close"].rolling(20).max().shift(1)
    low_20 = df["Close"].rolling(20).min().shift(1)
    vol_ratio = df.get("volume_ratio", pd.Series(1, index=df.index))
    bull_align = (df["ma5"] > df["ma10"]) & (df["ma10"] > df["ma20"])
    sig[(df["Close"] > high_20) & (vol_ratio > 2) & bull_align] = Signal.STRONG_BUY.value
    sig[(df["Close"] > high_20) & (vol_ratio > 1.5)] = Signal.BUY.value
    sig[(df["Close"] < low_20) & (vol_ratio < 0.7)] = Signal.SELL.value
    return sig


def _strategy_obv_divergence(df: pd.DataFrame) -> pd.Series:
    """
    OBV背离策略
    价格与OBV同向 = 趋势确认
    价格与OBV背离 = 反转信号
    BUY:  价格走平/下跌但OBV上升 (吸筹)
    SELL: 价格走平/上升但OBV下降 (出货)
    """
    sig = pd.Series(0, index=df.index)
    if "obv" not in df.columns:
        return sig
    n = 10
    price_change = df["Close"].pct_change(n)
    obv_change = df["obv"].pct_change(n)
    # 筹码收集: 价格跌/横 但 OBV 涨
    accu = (price_change <= 0.01) & (obv_change > 0.02)
    sig[accu] = Signal.BUY.value
    # 出货: 价格涨/横 但 OBV 跌
    distr = (price_change >= -0.01) & (obv_change < -0.02)
    sig[distr] = Signal.SELL.value
    return sig


def _strategy_vwap_mean_reversion(df: pd.DataFrame) -> pd.Series:
    """
    VWAP回归策略
    BUY:  价格远低于VWAP (> 5%) + 缩量 → 预计回归
    SELL: 价格远高于VWAP (> 5%) → 预计回归
    """
    sig = pd.Series(0, index=df.index)
    if "vwap" not in df.columns:
        return sig
    dist = (df["Close"] - df["vwap"]) / df["vwap"]
    vol_ratio = df.get("volume_ratio", pd.Series(1, index=df.index))
    sig[(dist < -0.05) & (vol_ratio < 0.8)] = Signal.BUY.value
    sig[dist > 0.05] = Signal.SELL.value
    sig[dist < -0.08] = Signal.STRONG_BUY.value
    sig[dist > 0.08] = Signal.STRONG_SELL.value
    return sig


# ═══════════════════════════════════════════════════════════════
# D. 波动率策略
# ═══════════════════════════════════════════════════════════════

def _strategy_bb_squeeze(df: pd.DataFrame) -> pd.Series:
    """
    布林带收缩突破策略 (经典的 Bollinger Squeeze)
    布林带宽度缩到极小 → 变盘前兆 → 放量突破方向
    BUY:  BB带宽 < 5%分位历史值 + 价格突破上轨 + 放量
    SELL: BB带宽 < 5%分位历史值 + 价格跌破下轨
    """
    sig = pd.Series(0, index=df.index)
    if not all(c in df.columns for c in ["bb_upper", "bb_lower", "bb_mid"]):
        return sig
    bbw = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    threshold = bbw.quantile(0.1)  # 历史10%分位
    squeeze = bbw < threshold
    vol_ratio = df.get("volume_ratio", pd.Series(1, index=df.index))
    sig[squeeze & (df["Close"] > df["bb_upper"]) & (vol_ratio > 1.5)] = Signal.BUY.value
    sig[squeeze & (df["Close"] < df["bb_lower"])] = Signal.SELL.value
    return sig


def _strategy_keltner_breakout(df: pd.DataFrame) -> pd.Series:
    """
    Keltner通道突破策略
    BUY:  价格突破Keltner上轨 + ADX > 20
    SELL: 价格跌破Keltner下轨
    """
    sig = pd.Series(0, index=df.index)
    if not all(c in df.columns for c in ["keltner_upper", "keltner_lower"]):
        return sig
    adx_ok = df.get("adx_14", pd.Series(20, index=df.index)) > 20
    sig[(df["Close"] > df["keltner_upper"]) & adx_ok] = Signal.BUY.value
    sig[df["Close"] < df["keltner_lower"]] = Signal.SELL.value
    return sig


def _strategy_atr_band(df: pd.DataFrame) -> pd.Series:
    """
    ATR轨道触及策略
    BUY:  价格触及ATR下轨 + RSI超卖
    SELL: 价格触及ATR上轨 + RSI超买
    (ATR轨道用于极端波动捕捉)
    """
    sig = pd.Series(0, index=df.index)
    if not all(c in df.columns for c in ["atr_upper_20", "atr_lower_20"]):
        return sig
    rsi = df.get("rsi_14", pd.Series(50, index=df.index))
    sig[(df["Close"] < df["atr_lower_20"]) & (rsi < 35)] = Signal.BUY.value
    sig[(df["Close"] > df["atr_upper_20"]) & (rsi > 70)] = Signal.SELL.value
    return sig


# ═══════════════════════════════════════════════════════════════
# E. 多重确认策略 (高胜率)
# ═══════════════════════════════════════════════════════════════

def _strategy_triple_confirm_long(df: pd.DataFrame) -> pd.Series:
    """
    三重确认做多
    BUY:  MA多头 + RSI > 50 + ADX > 20 三者同时满足
    SELL: 任一条件失效
    """
    sig = pd.Series(0, index=df.index)
    bull_align = (df["ma5"] > df["ma10"]) & (df["ma10"] > df["ma20"]) & (df["ma20"] > df["ma60"])
    rsi_ok = df.get("rsi_14", pd.Series(50, index=df.index)) > 50
    adx_ok = df.get("adx_14", pd.Series(15, index=df.index)) > 20
    sig[bull_align & rsi_ok & adx_ok] = Signal.BUY.value
    sig[~(bull_align | rsi_ok)] = Signal.SELL.value
    # 超强信号
    di_ok = df.get("di_plus", pd.Series(20, index=df.index)) > df.get("di_minus", pd.Series(20, index=df.index))
    sig[bull_align & rsi_ok & adx_ok & di_ok & (df["adx_14"] > 30)] = Signal.STRONG_BUY.value
    return sig


def _strategy_triple_confirm_short(df: pd.DataFrame) -> pd.Series:
    """
    三重确认做空
    SELL: MA空头 + RSI < 50 + ADX > 20 三者同时满足
    (此策略主要用于识别做空机会，信号只产生SELL)
    """
    sig = pd.Series(0, index=df.index)
    bear_align = (df["ma5"] < df["ma10"]) & (df["ma10"] < df["ma20"]) & (df["ma20"] < df["ma60"])
    rsi_ok = df.get("rsi_14", pd.Series(50, index=df.index)) < 50
    adx_ok = df.get("adx_14", pd.Series(15, index=df.index)) > 20
    di_ok = df.get("di_minus", pd.Series(20, index=df.index)) > df.get("di_plus", pd.Series(20, index=df.index))
    sig[bear_align & rsi_ok & adx_ok] = Signal.SELL.value
    sig[bear_align & rsi_ok & adx_ok & di_ok & (df["adx_14"] > 30)] = Signal.STRONG_SELL.value
    return sig


def _strategy_ichimoku_cloud(df: pd.DataFrame) -> pd.Series:
    """
    一目均衡表策略
    BUY:  转换线 上穿 基准线 + 价格在云上方
    SELL: 转换线 下穿 基准线 + 价格在云下方
    """
    sig = pd.Series(0, index=df.index)
    if not all(c in df.columns for c in ["ichimoku_conv", "ichimoku_base"]):
        return sig
    conv, base = df["ichimoku_conv"], df["ichimoku_base"]
    golden = _cross_above(conv, base)
    dead = _cross_below(conv, base)
    # 云的位置用 span_a 和 span_b
    above_cloud = True
    if all(c in df.columns for c in ["ichimoku_span_a", "ichimoku_span_b"]):
        cloud_top = df[["ichimoku_span_a", "ichimoku_span_b"]].max(axis=1)
        cloud_bot = df[["ichimoku_span_a", "ichimoku_span_b"]].min(axis=1)
        above_cloud = df["Close"] > cloud_top
        below_cloud = df["Close"] < cloud_bot
        sig[golden & above_cloud] = Signal.STRONG_BUY.value
        sig[dead & below_cloud] = Signal.STRONG_SELL.value
    sig[golden & ~above_cloud] = Signal.BUY.value
    sig[dead] = Signal.SELL.value
    return sig


def _strategy_full_monty(df: pd.DataFrame) -> pd.Series:
    """
    全维度确认策略 (最强信号)
    BUY:  均线多头 + RSI金叉(快线>慢线) + MACD柱转正 + 放量 + ADX > 25
         5个条件同时满足 → STRONG_BUY
    SELL: 均线空头 + RSI死叉 + MACD柱转负
    """
    sig = pd.Series(0, index=df.index)
    # 均线
    bull_align = (df["ma5"] > df["ma10"]) & (df["ma10"] > df["ma20"]) & (df["ma20"] > df["ma60"])
    # RSI
    rsi_golden = df.get("rsi_14", pd.Series(50, index=df.index)) > df.get("rsi_28", pd.Series(50, index=df.index))
    # MACD
    macd_positive = df.get("macd_hist", pd.Series(0, index=df.index)) > 0
    # 成交量
    vol_up = df.get("volume_ratio", pd.Series(1, index=df.index)) > 1.2
    # ADX
    adx_ok = df.get("adx_14", pd.Series(15, index=df.index)) > 25

    all_bull = bull_align & rsi_golden & macd_positive & vol_up & adx_ok
    sig[all_bull] = Signal.STRONG_BUY.value
    sig[bull_align & rsi_golden & macd_positive] = Signal.BUY.value

    # 空头
    bear_align = (df["ma5"] < df["ma10"]) & (df["ma10"] < df["ma20"]) & (df["ma20"] < df["ma60"])
    macd_negative = df.get("macd_hist", pd.Series(0, index=df.index)) < 0
    sig[bear_align & ~rsi_golden & macd_negative] = Signal.SELL.value
    sig[bear_align & ~rsi_golden & macd_negative & (df.get("adx_14", pd.Series(15, index=df.index)) > 25)] = Signal.STRONG_SELL.value
    return sig


# ═══════════════════════════════════════════════════════════════
# 策略注册表
# ═══════════════════════════════════════════════════════════════

STRATEGIES: Dict[str, Strategy] = {
    # A. 趋势跟踪
    "ma_ribbon": Strategy(
        id="ma_ribbon", name="均线多头排列", category="趋势跟踪",
        description="MA5>MA10>MA20>MA60多头排列 + RSI确认 + 放量",
        factors_used=["ma5","ma10","ma20","ma60","rsi_14","volume_ratio","trend_slope_ma20"],
        generate=_strategy_ma_ribbon, stop_loss_pct=0.06, take_profit_pct=0.12,
    ),
    "ema_cross": Strategy(
        id="ema_cross", name="EMA交叉", category="趋势跟踪",
        description="EMA12上穿EMA26 + EMA50确认",
        factors_used=["ema12","ema26","ema50"],
        generate=_strategy_ema_cross, stop_loss_pct=0.06, take_profit_pct=0.10,
    ),
    "trend_breakout": Strategy(
        id="trend_breakout", name="趋势突破", category="趋势跟踪",
        description="价格突破布林上轨 + ADX确认 + 放量",
        factors_used=["bb_upper","bb_mid","bb_lower","adx_14","volume_ratio"],
        generate=_strategy_trend_breakout, stop_loss_pct=0.07, take_profit_pct=0.15,
    ),
    "ma_pullback": Strategy(
        id="ma_pullback", name="均线回踩", category="趋势跟踪",
        description="多头排列中价格回调至MA20附近(3%) + RSI回踩",
        factors_used=["ma5","ma10","ma20","ma60","rsi_14"],
        generate=_strategy_ma_pullback, stop_loss_pct=0.05, take_profit_pct=0.10,
    ),
    "trend_acceleration": Strategy(
        id="trend_acceleration", name="趋势加速度", category="趋势跟踪",
        description="MA20斜率加速上升检测",
        factors_used=["trend_slope_ma20"],
        generate=_strategy_trend_acceleration, stop_loss_pct=0.06, take_profit_pct=0.12,
    ),

    # B. 动量反转
    "rsi_extreme": Strategy(
        id="rsi_extreme", name="RSI超买超卖", category="动量反转",
        description="RSI<30超卖反弹 + RSI>75超买回落",
        factors_used=["rsi_14"],
        generate=_strategy_rsi_extreme, stop_loss_pct=0.06, take_profit_pct=0.08,
    ),
    "bb_mean_reversion": Strategy(
        id="bb_mean_reversion", name="布林带回归", category="动量反转",
        description="价格跌破布林下轨后回归 + RSI确认",
        factors_used=["bb_upper","bb_lower","bb_mid","rsi_14"],
        generate=_strategy_bb_mean_reversion, stop_loss_pct=0.05, take_profit_pct=0.08,
    ),
    "stochastic_cross": Strategy(
        id="stochastic_cross", name="Stochastic双线交叉", category="动量反转",
        description="%K上穿%D在超卖区 / 下穿超买区",
        factors_used=["stochastic_k","stochastic_d"],
        generate=_strategy_stochastic_cross, stop_loss_pct=0.05, take_profit_pct=0.08,
    ),
    "macd_divergence": Strategy(
        id="macd_divergence", name="MACD背离", category="动量反转",
        description="价格与MACD柱的顶底背离检测",
        factors_used=["macd_hist"],
        generate=_strategy_macd_divergence, stop_loss_pct=0.06, take_profit_pct=0.12,
    ),
    "williams_r_reversal": Strategy(
        id="williams_r_reversal", name="Williams %R反转", category="动量反转",
        description="%R极端值后反转",
        factors_used=["williams_r"],
        generate=_strategy_williams_r_reversal, stop_loss_pct=0.05, take_profit_pct=0.08,
    ),

    # C. 成交量确认
    "volume_breakout": Strategy(
        id="volume_breakout", name="放量突破", category="成交量确认",
        description="价格突破20日高 + 成交量>2倍 + MA多头",
        factors_used=["ma5","ma10","ma20","ma60","volume_ratio"],
        generate=_strategy_volume_breakout, stop_loss_pct=0.07, take_profit_pct=0.15,
    ),
    "obv_divergence": Strategy(
        id="obv_divergence", name="OBV背离", category="成交量确认",
        description="OBV与价格的背离检测(吸筹/出货)",
        factors_used=["obv"],
        generate=_strategy_obv_divergence, stop_loss_pct=0.05, take_profit_pct=0.10,
    ),
    "vwap_mean_reversion": Strategy(
        id="vwap_mean_reversion", name="VWAP回归", category="成交量确认",
        description="价格偏离VWAP >5%后回归",
        factors_used=["vwap","volume_ratio"],
        generate=_strategy_vwap_mean_reversion, stop_loss_pct=0.05, take_profit_pct=0.08,
    ),

    # D. 波动率
    "bb_squeeze": Strategy(
        id="bb_squeeze", name="布林带收缩突破", category="波动率突破",
        description="BB带宽缩到历史10%分位 → 方向性突破",
        factors_used=["bb_upper","bb_lower","bb_mid","volume_ratio"],
        generate=_strategy_bb_squeeze, stop_loss_pct=0.08, take_profit_pct=0.18,
    ),
    "keltner_breakout": Strategy(
        id="keltner_breakout", name="Keltner突破", category="波动率突破",
        description="价格突破Keltner通道",
        factors_used=["keltner_upper","keltner_lower","adx_14"],
        generate=_strategy_keltner_breakout, stop_loss_pct=0.07, take_profit_pct=0.15,
    ),
    "atr_band": Strategy(
        id="atr_band", name="ATR轨道触及", category="波动率突破",
        description="极端波动时触及ATR轨道",
        factors_used=["atr_upper_20","atr_lower_20","rsi_14"],
        generate=_strategy_atr_band, stop_loss_pct=0.06, take_profit_pct=0.10,
    ),

    # E. 多重确认
    "triple_confirm_long": Strategy(
        id="triple_confirm_long", name="三重确认做多", category="多重确认",
        description="MA多头 + RSI>50 + ADX>20 三重确认",
        factors_used=["ma5","ma10","ma20","ma60","rsi_14","adx_14","di_plus","di_minus"],
        generate=_strategy_triple_confirm_long, stop_loss_pct=0.06, take_profit_pct=0.15,
    ),
    "triple_confirm_short": Strategy(
        id="triple_confirm_short", name="三重确认做空", category="多重确认",
        description="MA空头 + RSI<50 + ADX>20 做空信号",
        factors_used=["ma5","ma10","ma20","ma60","rsi_14","adx_14","di_plus","di_minus"],
        generate=_strategy_triple_confirm_short, stop_loss_pct=0.06, take_profit_pct=0.12,
    ),
    "ichimoku_cloud": Strategy(
        id="ichimoku_cloud", name="一目均衡云", category="多重确认",
        description="转换线上穿基准线 + 云层突破",
        factors_used=["ichimoku_conv","ichimoku_base","ichimoku_span_a","ichimoku_span_b"],
        generate=_strategy_ichimoku_cloud, stop_loss_pct=0.07, take_profit_pct=0.15,
    ),
    "full_monty": Strategy(
        id="full_monty", name="全维度确认", category="多重确认",
        description="均线+RSI+MACD+成交量+ADX 五维共振",
        factors_used=["ma5","ma10","ma20","ma60","rsi_14","rsi_28","macd_hist","volume_ratio","adx_14"],
        generate=_strategy_full_monty, stop_loss_pct=0.07, take_profit_pct=0.15,
    ),
}


# ═══════════════════════════════════════════════════════════════
# 策略执行与批量回测
# ═══════════════════════════════════════════════════════════════

def generate_strategy_signals(df: pd.DataFrame, strategy_id: str) -> pd.Series:
    """对单个策略生成信号序列"""
    strat = STRATEGIES.get(strategy_id)
    if not strat:
        raise ValueError(f"未知策略: {strategy_id}")
    return strat.generate(df)


def backtest_strategy(
    df: pd.DataFrame, strategy_id: str,
    initial_capital: float = 100000,
    worst_case: bool = False,
    limit_pct: float = 0.0,
    hold_override: Optional[Dict[str, int]] = None,
) -> BacktestResult:
    """
    回测单个策略
    hold_override: 可选，覆盖持仓参数 {"hold_min": 3, "hold_max": 30, "sl": 0.06, "tp": 0.12}
    """
    strat = STRATEGIES[strategy_id]
    signals = strat.generate(df)
    params = {
        "stop_loss_pct": strat.stop_loss_pct,
        "take_profit_pct": strat.take_profit_pct,
        "hold_days_min": strat.hold_days_min,
        "hold_days_max": strat.hold_days_max,
    }
    if hold_override:
        if "hold_min" in hold_override:
            params["hold_days_min"] = hold_override["hold_min"]
        if "hold_max" in hold_override:
            params["hold_days_max"] = hold_override["hold_max"]
        if "sl" in hold_override:
            params["stop_loss_pct"] = hold_override["sl"]
        if "tp" in hold_override:
            params["take_profit_pct"] = hold_override["tp"]
    result = run_backtest(
        df, signals,
        initial_capital=initial_capital,
        worst_case=worst_case,
        limit_pct=limit_pct,
        **params,
    )
    result.strategy_id = strategy_id
    result.strategy_name = strat.name
    return result


# 持仓周期预设
HOLD_PROFILES = {
    "short":  {"hold_min": 1,   "hold_max": 15,  "sl": 0.04, "tp": 0.08,  "label": "短线 (1-15天)"},
    "medium": {"hold_min": 30,  "hold_max": 180, "sl": 0.0,  "tp": 0.0,   "label": "中线 (30-180天,无止损)"},
    "long":   {"hold_min": 180, "hold_max": 360, "sl": 0.0,  "tp": 0.0,   "label": "长线 (180-360天,无止损)"},
}


def backtest_all(
    df: pd.DataFrame,
    initial_capital: float = 100000,
    min_trades: int = 1,
    worst_case: bool = False,
    limit_pct: float = 0.0,
    hold_profile: str = "medium",
) -> Dict[str, BacktestResult]:
    """对所有注册策略进行回测"""
    hold_override = HOLD_PROFILES.get(hold_profile)
    results = {}
    for sid in STRATEGIES:
        try:
            r = backtest_strategy(df, sid, initial_capital=initial_capital,
                                  worst_case=worst_case, limit_pct=limit_pct,
                                  hold_override=hold_override)
            if r.total_trades >= min_trades:
                results[sid] = r
        except Exception as e:
            print(f"  [WARN] 策略 {sid} 回测失败: {e}")
    return results


def rank_strategies(results: Dict[str, BacktestResult], metric: str = "sharpe_ratio") -> List[Tuple[str, BacktestResult, float]]:
    """按指定指标排序策略"""
    ranked = [(sid, r, getattr(r, metric, 0)) for sid, r in results.items()]
    ranked.sort(key=lambda x: x[2], reverse=True)
    return ranked


def get_strategy_summary(results: Dict[str, BacktestResult]) -> pd.DataFrame:
    """生成策略评估汇总表"""
    rows = []
    for sid, r in results.items():
        strat = STRATEGIES[sid]
        rows.append({
            "策略ID": sid,
            "策略名称": r.strategy_name,
            "分类": strat.category,
            "总收益率": f"{r.total_return * 100:.2f}%",
            "年化收益": f"{r.annual_return * 100:.2f}%",
            "夏普比率": f"{r.sharpe_ratio:.2f}",
            "最大回撤": f"{r.max_drawdown * 100:.2f}%",
            "胜率": f"{r.win_rate * 100:.1f}%",
            "交易次数": r.total_trades,
            "平均收益": f"{r.avg_return_per_trade * 100:.2f}%",
            "平均持仓": f"{r.avg_hold_days:.0f}天",
            "盈亏比": f"{r.profit_factor:.2f}",
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
# 策略信号汇总 (多策略综合打分)
# ═══════════════════════════════════════════════════════════════

def aggregate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    对每个策略生成信号并汇总
    返回 DataFrame，列 = 策略ID，值 = 信号(-2 ~ +2)
    另有 aggregate_score 列 = 所有策略信号的总和
    """
    agg = pd.DataFrame(index=df.index)
    for sid in STRATEGIES:
        try:
            agg[sid] = STRATEGIES[sid].generate(df)
        except Exception:
            agg[sid] = 0
    agg["aggregate_score"] = agg.sum(axis=1)
    agg["bullish_ratio"] = (agg > 0).sum(axis=1) / max(len(STRATEGIES), 1)
    agg["consensus"] = pd.cut(
        agg["aggregate_score"],
        bins=[-float("inf"), -3, 0, 3, float("inf")],
        labels=["强烈偏空", "偏空", "偏多", "强烈偏多"],
    )
    return agg


def get_latest_consensus(df: pd.DataFrame) -> dict:
    """获取最新的多策略共识"""
    agg = aggregate_signals(df)
    latest = agg.iloc[-1]
    signals_detail = {}
    for col in agg.columns:
        if col not in ("aggregate_score", "bullish_ratio", "consensus"):
            sig_val = int(latest[col])
            if sig_val != 0:
                signals_detail[col] = {
                    "name": STRATEGIES[col].name,
                    "category": STRATEGIES[col].category,
                    "signal": sig_val,
                }
    return {
        "date": str(df["Date"].iloc[-1]),
        "aggregate_score": float(latest["aggregate_score"]),
        "bullish_ratio": float(latest["bullish_ratio"]),
        "consensus": str(latest["consensus"]),
        "active_strategies": signals_detail,
    }
