"""K 线信号系统 — 趋势线 + 斜率 + 曲线 + 经典形态 四维分析

信号分类:
  A. 趋势线信号 (斜线):  趋势线突破、通道突破、趋势线回踩
  B. 斜率信号:           MA斜率加速/减速、斜率背离
  C. 曲线信号:           金叉死叉、均线收敛发散、布林带
  D. 经典K线形态:        12种形态 + 原有4个增强版

设计原则:
  1. 每个信号都有确认条件，不单靠一个因素
  2. 趋势线用 pivot 拟合，斜率用多周期对比
  3. 信号返回 (int, float) — 是否触发 + 强度评分
"""

import pandas as pd
import numpy as np
from scipy import stats as sp_stats


# ═══════════════════════════════════════════════════════════════
# 信号注册表
# ═══════════════════════════════════════════════════════════════
# 格式: key -> (中文名, 人话描述, 方向: bullish/bearish/neutral)
SIGNALS = {
    # ── A. 趋势线信号 ──
    "trendline_break_up": (
        "趋势线向上突破",
        "价格突破下降趋势线，且放量确认，趋势可能反转向上",
        "bullish",
    ),
    "trendline_break_down": (
        "趋势线向下突破",
        "价格跌破上升趋势线，且放量确认，趋势可能反转向下",
        "bearish",
    ),
    "trendline_bounce_up": (
        "趋势线支撑反弹",
        "价格回踩上升趋势线后反弹，下影线+放量确认",
        "bullish",
    ),
    "trendline_bounce_down": (
        "趋势线压力回落",
        "价格触及下降趋势线后回落，上影线+放量确认",
        "bearish",
    ),
    "channel_top_touch": (
        "通道上轨触及",
        "价格触及上升通道上轨，可能回调",
        "bearish",
    ),
    "channel_bottom_touch": (
        "通道下轨触及",
        "价格触及下降通道下轨，可能反弹",
        "bullish",
    ),

    # ── B. 斜率信号 ──
    "slope_accelerating_up": (
        "斜率加速上升",
        "MA20斜率在加速变大，涨势加强，趋势延续",
        "bullish",
    ),
    "slope_decelerating_up": (
        "上升减速 (可能见顶)",
        "MA20上升斜率在减小，涨势衰竭，警惕回调",
        "bearish",
    ),
    "slope_accelerating_down": (
        "斜率加速下降",
        "MA20斜率在加速变负，跌势加剧",
        "bearish",
    ),
    "slope_decelerating_down": (
        "下跌减速 (可能见底)",
        "MA20下降斜率在收窄，跌势减弱，可能反转",
        "bullish",
    ),
    "slope_divergence_bullish": (
        "底背离",
        "价格新低但MA20斜率未新低，下跌动能衰竭",
        "bullish",
    ),
    "slope_divergence_bearish": (
        "顶背离",
        "价格新高但MA20斜率未新高，上涨动能衰竭",
        "bearish",
    ),

    # ── C. 曲线信号 ──
    "ma_golden_cross": (
        "MA金叉",
        "MA5上穿MA20，短期趋势转多",
        "bullish",
    ),
    "ma_death_cross": (
        "MA死叉",
        "MA5下穿MA20，短期趋势转空",
        "bearish",
    ),
    "ma_convergence": (
        "均线收敛",
        "多周期均线间距收窄，即将变盘",
        "neutral",
    ),
    "ma_divergence_bullish": (
        "均线多头发散",
        "多周期均线间距扩大，多头趋势加强",
        "bullish",
    ),
    "ma_divergence_bearish": (
        "均线空头发散",
        "多周期均线间距扩大，空头趋势加强",
        "bearish",
    ),
    "ma_ribbon_bullish": (
        "均线多头排列",
        "MA5>MA10>MA20>MA60，经典多头结构",
        "bullish",
    ),
    "ma_ribbon_bearish": (
        "均线空头排列",
        "MA5<MA10<MA20<MA60，经典空头结构",
        "bearish",
    ),
    "bb_squeeze_long": (
        "布林带收窄做多",
        "布林带宽度收窄至低位后价格向上突破中轨",
        "bullish",
    ),
    "bb_squeeze_short": (
        "布林带收窄做空",
        "布林带宽度收窄至低位后价格向下突破中轨",
        "bearish",
    ),

    # ── D. 经典K线形态 (增强版) ──
    "dark_cloud_cover": (
        "乌云盖顶",
        "上涨后跳空高开收阴线，收盘低于前阳线中点，看跌反转",
        "bearish",
    ),
    "piercing_line": (
        "刺透形态",
        "下跌后跳空低开收阳线，收盘高于前阴线中点，看涨反转",
        "bullish",
    ),
    "morning_star": (
        "晨星",
        "三K线：长阴+小K线+长阳，底部反转信号",
        "bullish",
    ),
    "evening_star": (
        "黄昏之星",
        "三K线：长阳+小K线+长阴，顶部反转信号",
        "bearish",
    ),
    "hammer": (
        "锤头线",
        "下跌后出现长下影线小实体，下影线>=实体2倍，看涨反转",
        "bullish",
    ),
    "shooting_star": (
        "射击之星",
        "上涨后出现长上影线小实体，上影线>=实体2倍，看跌反转",
        "bearish",
    ),
    "bullish_engulfing": (
        "看涨吞没",
        "阴线后紧跟更大阳线完全包裹前阴线，强烈看涨",
        "bullish",
    ),
    "bearish_engulfing": (
        "看跌吞没",
        "阳线后紧跟更大阴线完全包裹前阳线，强烈看跌",
        "bearish",
    ),
    "three_white_soldiers": (
        "白三兵",
        "连续三根递增阳线，每根收盘接近最高，强势上涨",
        "bullish",
    ),
    "three_black_crows": (
        "三只乌鸦",
        "连续三根递减阴线，每根收盘接近最低，强势下跌",
        "bearish",
    ),

    # ── E. 原有信号增强版 ──
    "big_bull_breakout": (
        "大阳线突破",
        "涨幅>3%+突破20日高点+放量+MA5斜率>0",
        "bullish",
    ),
    "support_bounce": (
        "支撑位反弹",
        "低点接近20日低点+收盘>开盘+放量+下影线确认",
        "bullish",
    ),
    "reversal_after_decline": (
        "回调后转强",
        "前3日下跌+今日收阳突破昨日高点+放量+MA5转向上",
        "bullish",
    ),
}


# ═══════════════════════════════════════════════════════════════
# 基础指标计算
# ═══════════════════════════════════════════════════════════════

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算所有基础指标"""
    df = df.copy()

    # 价格列
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    df["_price"] = df[price_col]

    # 均线 (多周期)
    for p in [5, 10, 20, 60]:
        df[f"ma{p}"] = df["_price"].rolling(p, min_periods=1).mean()

    # 成交量均线
    df["vol_ma20"] = df["Volume"].rolling(20, min_periods=1).mean()
    df["vol_ratio"] = df["Volume"] / df["vol_ma20"]

    # 20 日高低点
    df["high_20"] = df["High"].rolling(20, min_periods=1).max()
    df["low_20"] = df["Low"].rolling(20, min_periods=1).min()
    df["high_60"] = df["High"].rolling(60, min_periods=1).max()
    df["low_60"] = df["Low"].rolling(60, min_periods=1).min()

    # 日收益率
    df["daily_ret"] = df["_price"].pct_change()

    # K 线实体与影线
    df["body"] = (df["Close"] - df["Open"]).abs()
    df["body_dir"] = np.where(df["Close"] >= df["Open"], 1, -1)  # 1=阳线, -1=阴线
    df["lower_shadow"] = df[["Open", "Close"]].min(axis=1) - df["Low"]
    df["upper_shadow"] = df["High"] - df[["Open", "Close"]].max(axis=1)
    df["total_range"] = df["High"] - df["Low"]

    # ATR (14日平均真实波幅)
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["_price"].shift(1)).abs()
    low_close = (df["Low"] - df["_price"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()

    # 布林带 (20日, 2倍标准差)
    df["bb_mid"] = df["ma20"]
    df["bb_std"] = df["_price"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]  # 带宽百分比

    # 均线间距 (用于收敛/发散判断)
    df["ma_spread"] = (df["ma5"] - df["ma60"]).abs() / df["ma60"]

    return df


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def _calc_slope(series: pd.Series, lookback: int = 10) -> float:
    """计算序列最近 lookback 日的归一化斜率"""
    s = series.dropna().tail(lookback)
    if len(s) < max(3, lookback // 2):
        return 0.0
    x = np.arange(len(s))
    slope, _, _, _, _ = sp_stats.linregress(x, s.values)
    mean_val = s.mean()
    if mean_val == 0:
        return 0.0
    return slope / mean_val


def _find_pivots(series: pd.Series, window: int = 5, mode: str = "high") -> np.ndarray:
    """找局部极值点 (pivot highs 或 pivot lows)"""
    arr = series.values
    n = len(arr)
    pivots = np.zeros(n, dtype=bool)
    for i in range(window, n - window):
        if mode == "high":
            if arr[i] == np.max(arr[i - window : i + window + 1]):
                pivots[i] = True
        else:
            if arr[i] == np.min(arr[i - window : i + window + 1]):
                pivots[i] = True
    return pivots


def _fit_trendline(x: np.ndarray, y: np.ndarray) -> tuple:
    """线性回归拟合趋势线，返回 (斜率, 截距, R²)"""
    if len(x) < 3:
        return 0, 0, 0
    slope, intercept, r_value, _, _ = sp_stats.linregress(x, y)
    return slope, intercept, r_value ** 2


def _trendline_value(slope: float, intercept: float, x: float) -> float:
    """计算趋势线在 x 位置的值"""
    return slope * x + intercept


def _is_volume_confirmed(df: pd.DataFrame, idx: int) -> bool:
    """检查当日成交量是否大于20日均量"""
    if idx < 0 or idx >= len(df):
        return False
    return bool(df["Volume"].iloc[idx] > df["vol_ma20"].iloc[idx] * 0.8)


# ═══════════════════════════════════════════════════════════════
# A. 趋势线信号检测
# ═══════════════════════════════════════════════════════════════

def _detect_trendlines(df: pd.DataFrame, lookback: int = 60) -> dict:
    """
    检测主要趋势线 (最多检测最近 lookback 日内)
    返回 dict 包含上升趋势线和下降趋势线信息
    """
    n = len(df)
    start = max(0, n - lookback)
    segment = df.iloc[start:].copy()
    seg_n = len(segment)

    result = {
        "uptrend_line": None,    # 上升趋势线 (连接低点)
        "downtrend_line": None,  # 下降趋势线 (连接高点)
        "channel_upper": None,
        "channel_lower": None,
    }

    # 找 pivot lows → 上升趋势线
    low_pivots = _find_pivots(df["Low"], window=3, mode="low")
    low_indices = np.where(low_pivots[start:])[0]
    if len(low_indices) >= 3:
        # 取最近的 pivot lows
        recent_lows = low_indices[-min(5, len(low_indices)):]
        y_vals = df["Low"].iloc[start:].iloc[recent_lows].values
        x_vals = recent_lows.astype(float)
        slope, intercept, r2 = _fit_trendline(x_vals, y_vals)
        if r2 > 0.5:  # 拟合度阈值
            result["uptrend_line"] = {
                "slope": slope,
                "intercept": intercept,
                "r2": r2,
                "start_idx": int(start + recent_lows[0]),
                "end_idx": int(start + recent_lows[-1]),
                "values": [_trendline_value(slope, intercept, xi) for xi in x_vals],
            }

    # 找 pivot highs → 下降趋势线
    high_pivots = _find_pivots(df["High"], window=3, mode="high")
    high_indices = np.where(high_pivots[start:])[0]
    if len(high_indices) >= 3:
        recent_highs = high_indices[-min(5, len(high_indices)):]
        y_vals = df["High"].iloc[start:].iloc[recent_highs].values
        x_vals = recent_highs.astype(float)
        slope, intercept, r2 = _fit_trendline(x_vals, y_vals)
        if r2 > 0.5:
            result["downtrend_line"] = {
                "slope": slope,
                "intercept": intercept,
                "r2": r2,
                "start_idx": int(start + recent_highs[0]),
                "end_idx": int(start + recent_highs[-1]),
                "values": [_trendline_value(slope, intercept, xi) for xi in x_vals],
            }

    # 通道检测: 如果有上升趋势线，找平行上轨
    if result["uptrend_line"]:
        ul = result["uptrend_line"]
        # 上轨 = 上升趋势线上方 ATR*2
        channel_width = df["atr14"].iloc[-1] * 3 if not pd.isna(df["atr14"].iloc[-1]) else df["_price"].iloc[-1] * 0.05
        result["channel_upper"] = {
            "slope": ul["slope"],
            "intercept": ul["intercept"] + channel_width,
        }
        result["channel_lower"] = {
            "slope": ul["slope"],
            "intercept": ul["intercept"],
        }

    return result


def _signal_trendline_break_up(df: pd.DataFrame) -> pd.Series:
    """价格向上突破下降趋势线"""
    n = len(df)
    signal = pd.Series(0, index=df.index)

    if n < 40:
        return signal

    # 滚动检测：每20天重新计算趋势线
    for i in range(60, n):
        sub = df.iloc[max(0, i - 60) : i + 1]
        tls = _detect_trendlines(sub, lookback=60)
        dt = tls.get("downtrend_line")
        if dt is None:
            continue

        # 当前价格
        curr_price = df["_price"].iloc[i]
        prev_price = df["_price"].iloc[i - 1]
        # 趋势线在当前点的值
        local_idx = len(sub) - 1
        tl_val = _trendline_value(dt["slope"], dt["intercept"], local_idx)

        # 突破条件: 前一日在线上方附近，今日收盘在线上方超过ATR*0.3
        atr = df["atr14"].iloc[i]
        if pd.isna(atr) or atr == 0:
            atr = curr_price * 0.01

        if (curr_price > tl_val + atr * 0.3
            and curr_price > prev_price
            and _is_volume_confirmed(df, i)
            and df["daily_ret"].iloc[i] > 0.01):
            signal.iloc[i] = 2  # 强度2

    return signal


def _signal_trendline_break_down(df: pd.DataFrame) -> pd.Series:
    """价格向下跌破上升趋势线"""
    n = len(df)
    signal = pd.Series(0, index=df.index)

    if n < 40:
        return signal

    for i in range(60, n):
        sub = df.iloc[max(0, i - 60) : i + 1]
        tls = _detect_trendlines(sub, lookback=60)
        ut = tls.get("uptrend_line")
        if ut is None:
            continue

        curr_price = df["_price"].iloc[i]
        prev_price = df["_price"].iloc[i - 1]
        local_idx = len(sub) - 1
        tl_val = _trendline_value(ut["slope"], ut["intercept"], local_idx)

        atr = df["atr14"].iloc[i]
        if pd.isna(atr) or atr == 0:
            atr = curr_price * 0.01

        if (curr_price < tl_val - atr * 0.3
            and curr_price < prev_price
            and _is_volume_confirmed(df, i)
            and df["daily_ret"].iloc[i] < -0.01):
            signal.iloc[i] = 2

    return signal


def _signal_trendline_bounce_up(df: pd.DataFrame) -> pd.Series:
    """价格回踩上升趋势线后反弹"""
    n = len(df)
    signal = pd.Series(0, index=df.index)

    if n < 40:
        return signal

    for i in range(60, n):
        sub = df.iloc[max(0, i - 60) : i + 1]
        tls = _detect_trendlines(sub, lookback=60)
        ut = tls.get("uptrend_line")
        if ut is None:
            continue

        curr_price = df["_price"].iloc[i]
        low_price = df["Low"].iloc[i]
        local_idx = len(sub) - 1
        tl_val = _trendline_value(ut["slope"], ut["intercept"], local_idx)

        atr = df["atr14"].iloc[i]
        if pd.isna(atr) or atr == 0:
            atr = curr_price * 0.01

        # 回踩条件: 最低价接近趋势线 + 收盘价远离趋势线(反弹) + 阳线 + 下影线
        near_tl = abs(low_price - tl_val) < atr * 0.5
        bounced = curr_price > tl_val + atr * 0.5
        is_bullish = df["body_dir"].iloc[i] == 1
        has_shadow = df["lower_shadow"].iloc[i] > df["body"].iloc[i] * 0.5

        if near_tl and bounced and is_bullish and has_shadow and _is_volume_confirmed(df, i):
            signal.iloc[i] = 1

    return signal


def _signal_trendline_bounce_down(df: pd.DataFrame) -> pd.Series:
    """价格触及下降趋势线后回落"""
    n = len(df)
    signal = pd.Series(0, index=df.index)

    if n < 40:
        return signal

    for i in range(60, n):
        sub = df.iloc[max(0, i - 60) : i + 1]
        tls = _detect_trendlines(sub, lookback=60)
        dt = tls.get("downtrend_line")
        if dt is None:
            continue

        curr_price = df["_price"].iloc[i]
        high_price = df["High"].iloc[i]
        local_idx = len(sub) - 1
        tl_val = _trendline_value(dt["slope"], dt["intercept"], local_idx)

        atr = df["atr14"].iloc[i]
        if pd.isna(atr) or atr == 0:
            atr = curr_price * 0.01

        near_tl = abs(high_price - tl_val) < atr * 0.5
        rejected = curr_price < tl_val - atr * 0.3
        is_bearish = df["body_dir"].iloc[i] == -1
        has_shadow = df["upper_shadow"].iloc[i] > df["body"].iloc[i] * 0.5

        if near_tl and rejected and is_bearish and has_shadow and _is_volume_confirmed(df, i):
            signal.iloc[i] = 1

    return signal


def _signal_channel_touch(df: pd.DataFrame) -> tuple:
    """通道上轨/下轨触及信号"""
    n = len(df)
    sig_top = pd.Series(0, index=df.index)
    sig_bottom = pd.Series(0, index=df.index)

    if n < 60:
        return sig_top, sig_bottom

    for i in range(60, n):
        sub = df.iloc[max(0, i - 60) : i + 1]
        tls = _detect_trendlines(sub, lookback=60)
        cu = tls.get("channel_upper")
        cl = tls.get("channel_lower")
        if cu is None or cl is None:
            continue

        curr = df["_price"].iloc[i]
        high = df["High"].iloc[i]
        low = df["Low"].iloc[i]
        local_idx = len(sub) - 1
        upper_val = _trendline_value(cu["slope"], cu["intercept"], local_idx)
        lower_val = _trendline_value(cl["slope"], cl["intercept"], local_idx)

        atr = df["atr14"].iloc[i]
        if pd.isna(atr) or atr == 0:
            atr = curr * 0.01

        # 上轨触及
        if high >= upper_val - atr * 0.3 and df["body_dir"].iloc[i] == -1:
            sig_top.iloc[i] = 1
        # 下轨触及
        if low <= lower_val + atr * 0.3 and df["body_dir"].iloc[i] == 1:
            sig_bottom.iloc[i] = 1

    return sig_top, sig_bottom


# ═══════════════════════════════════════════════════════════════
# B. 斜率信号检测
# ═══════════════════════════════════════════════════════════════

def _signal_slope_analysis(df: pd.DataFrame) -> dict:
    """完整的斜率分析，返回多个信号 Series"""
    n = len(df)
    signals = {
        "slope_accelerating_up": pd.Series(0, index=df.index),
        "slope_decelerating_up": pd.Series(0, index=df.index),
        "slope_accelerating_down": pd.Series(0, index=df.index),
        "slope_decelerating_down": pd.Series(0, index=df.index),
        "slope_divergence_bullish": pd.Series(0, index=df.index),
        "slope_divergence_bearish": pd.Series(0, index=df.index),
    }

    if n < 30:
        return signals

    # 计算 MA20 的滚动斜率序列 (每个点用前10天计算)
    slope_series = np.full(n, np.nan)
    for i in range(20, n):
        slope_series[i] = _calc_slope(df["ma20"].iloc[max(0, i - 10) : i + 1], lookback=10)

    # 斜率变化检测
    for i in range(30, n):
        s_curr = slope_series[i]
        s_prev = slope_series[i - 5]  # 5天前的斜率
        s_prev2 = slope_series[i - 10]  # 10天前的斜率

        if np.isnan(s_curr) or np.isnan(s_prev):
            continue

        # 斜率加速上升: s_curr > s_prev > s_prev2 > 0
        if s_curr > 0 and s_prev > 0 and s_curr > s_prev * 1.3:
            signals["slope_accelerating_up"].iloc[i] = 1

        # 上升减速: s_curr < s_prev (但都>0)
        if s_curr > 0 and s_prev > 0 and s_curr < s_prev * 0.6:
            signals["slope_decelerating_up"].iloc[i] = 1

        # 斜率加速下降: s_curr < s_prev < s_prev2 < 0
        if s_curr < 0 and s_prev < 0 and s_curr < s_prev * 1.3:
            signals["slope_accelerating_down"].iloc[i] = 1

        # 下降减速: s_curr > s_prev (但都<0)
        if s_curr < 0 and s_prev < 0 and s_curr > s_prev * 0.6:
            signals["slope_decelerating_down"].iloc[i] = 1

    # 背离检测
    for i in range(40, n):
        s_curr = slope_series[i]
        if np.isnan(s_curr):
            continue

        # 底背离: 价格比20天前低，但斜率比20天前高
        price_now = df["_price"].iloc[i]
        price_20d = df["_price"].iloc[i - 20]
        slope_20d = slope_series[i - 20]

        if not np.isnan(slope_20d) and price_now < price_20d and s_curr > slope_20d and s_curr < 0 and slope_20d < 0:
            signals["slope_divergence_bullish"].iloc[i] = 1

        # 顶背离: 价格比20天前高，但斜率比20天前低
        if not np.isnan(slope_20d) and price_now > price_20d and s_curr < slope_20d and s_curr > 0 and slope_20d > 0:
            signals["slope_divergence_bearish"].iloc[i] = 1

    return signals


# ═══════════════════════════════════════════════════════════════
# C. 曲线信号检测
# ═══════════════════════════════════════════════════════════════

def _signal_ma_cross(df: pd.DataFrame) -> tuple:
    """金叉/死叉检测"""
    n = len(df)
    gold_cross = pd.Series(0, index=df.index)
    death_cross = pd.Series(0, index=df.index)

    if n < 21:
        return gold_cross, death_cross

    diff = df["ma5"] - df["ma20"]
    for i in range(21, n):
        prev_diff = diff.iloc[i - 1]
        curr_diff = diff.iloc[i]
        if prev_diff < 0 and curr_diff > 0:
            gold_cross.iloc[i] = 1
        elif prev_diff > 0 and curr_diff < 0:
            death_cross.iloc[i] = 1

    return gold_cross, death_cross


def _signal_ma_convergence_divergence(df: pd.DataFrame) -> tuple:
    """均线收敛/发散检测"""
    n = len(df)
    convergence = pd.Series(0, index=df.index)
    divergence_bull = pd.Series(0, index=df.index)
    divergence_bear = pd.Series(0, index=df.index)

    if n < 30:
        return convergence, divergence_bull, divergence_bear

    for i in range(30, n):
        spread_now = df["ma_spread"].iloc[i]
        spread_10d = df["ma_spread"].iloc[i - 10]

        if pd.isna(spread_now) or pd.isna(spread_10d) or spread_10d == 0:
            continue

        ratio = spread_now / spread_10d

        # 收敛: 间距缩小到原来的60%以下
        if ratio < 0.6:
            convergence.iloc[i] = 1

        # 多头发散: 间距扩大 + 价格在MA60上方
        if ratio > 1.5 and df["_price"].iloc[i] > df["ma60"].iloc[i]:
            divergence_bull.iloc[i] = 1

        # 空头发散: 间距扩大 + 价格在MA60下方
        if ratio > 1.5 and df["_price"].iloc[i] < df["ma60"].iloc[i]:
            divergence_bear.iloc[i] = 1

    return convergence, divergence_bull, divergence_bear


def _signal_ma_ribbon(df: pd.DataFrame) -> tuple:
    """均线排列检测"""
    n = len(df)
    bull_ribbon = pd.Series(0, index=df.index)
    bear_ribbon = pd.Series(0, index=df.index)

    for i in range(60, n):
        ma5 = df["ma5"].iloc[i]
        ma10 = df["ma10"].iloc[i]
        ma20 = df["ma20"].iloc[i]
        ma60 = df["ma60"].iloc[i]

        if ma5 > ma10 > ma20 > ma60:
            bull_ribbon.iloc[i] = 1
        elif ma5 < ma10 < ma20 < ma60:
            bear_ribbon.iloc[i] = 1

    return bull_ribbon, bear_ribbon


def _signal_bb_squeeze(df: pd.DataFrame) -> tuple:
    """布林带收窄信号"""
    n = len(df)
    squeeze_long = pd.Series(0, index=df.index)
    squeeze_short = pd.Series(0, index=df.index)

    if n < 30:
        return squeeze_long, squeeze_short

    # 计算布林带宽度的历史分位数
    for i in range(30, n):
        bw = df["bb_width"].iloc[i]
        bw_history = df["bb_width"].iloc[max(0, i - 60) : i + 1]
        percentile = (bw_history < bw).mean()

        # 带宽处于低位 (最低20%)
        if percentile > 0.8:
            continue

        # 价格突破中轨
        if df["_price"].iloc[i] > df["bb_mid"].iloc[i] and df["_price"].iloc[i - 1] <= df["bb_mid"].iloc[i - 1]:
            if _is_volume_confirmed(df, i):
                squeeze_long.iloc[i] = 1
        elif df["_price"].iloc[i] < df["bb_mid"].iloc[i] and df["_price"].iloc[i - 1] >= df["bb_mid"].iloc[i - 1]:
            if _is_volume_confirmed(df, i):
                squeeze_short.iloc[i] = 1

    return squeeze_long, squeeze_short


# ═══════════════════════════════════════════════════════════════
# D. 经典K线形态检测 (增强版)
# ═══════════════════════════════════════════════════════════════

def _signal_dark_cloud_cover(df: pd.DataFrame) -> pd.Series:
    """乌云盖顶: 上涨趋势 + 阳线 + 次日跳空高开收阴 + 收盘低于前阳线中点"""
    n = len(df)
    signal = pd.Series(0, index=df.index)

    for i in range(5, n):
        # 确认上涨趋势：近5日均线上行
        ma10_slope = _calc_slope(df["ma10"].iloc[i - 5 : i + 1], lookback=5)
        if ma10_slope <= 0.001:
            continue

        # 前一日是实体较大的阳线
        prev_body = df["body"].iloc[i - 1]
        prev_open = df["Open"].iloc[i - 1]
        prev_close = df["Close"].iloc[i - 1]
        if prev_body < df["atr14"].iloc[i - 1] * 0.5 or df["body_dir"].iloc[i - 1] != 1:
            continue

        # 今日跳空高开
        if df["Open"].iloc[i] <= prev_close:
            continue

        # 今日收阴
        if df["body_dir"].iloc[i] != -1:
            continue

        # 收盘价低于前阳线实体中点
        mid_point = (prev_open + prev_close) / 2
        if df["Close"].iloc[i] >= mid_point:
            continue

        signal.iloc[i] = 1

    return signal


def _signal_piercing_line(df: pd.DataFrame) -> pd.Series:
    """刺透形态 (乌云盖顶的反向): 下跌后跳空低开收阳 > 前阴线中点"""
    n = len(df)
    signal = pd.Series(0, index=df.index)

    for i in range(5, n):
        ma10_slope = _calc_slope(df["ma10"].iloc[i - 5 : i + 1], lookback=5)
        if ma10_slope >= -0.001:
            continue

        prev_body = df["body"].iloc[i - 1]
        prev_open = df["Open"].iloc[i - 1]
        prev_close = df["Close"].iloc[i - 1]
        if prev_body < df["atr14"].iloc[i - 1] * 0.5 or df["body_dir"].iloc[i - 1] != -1:
            continue

        if df["Open"].iloc[i] >= prev_close:
            continue

        if df["body_dir"].iloc[i] != 1:
            continue

        mid_point = (prev_open + prev_close) / 2
        if df["Close"].iloc[i] <= mid_point:
            continue

        signal.iloc[i] = 1

    return signal


def _signal_morning_star(df: pd.DataFrame) -> pd.Series:
    """晨星: 长阴 + 小K线 + 长阳，底部反转"""
    n = len(df)
    signal = pd.Series(0, index=df.index)

    for i in range(3, n):
        # 第一根: 阴线，实体较大
        k1_body = df["body"].iloc[i - 2]
        if k1_body < df["atr14"].iloc[i - 2] * 0.5 or df["body_dir"].iloc[i - 2] != -1:
            continue

        # 第二根: 小实体 (无论阴阳)
        k2_body = df["body"].iloc[i - 1]
        if k2_body > k1_body * 0.5:
            continue

        # 第三根: 阳线，实体较大，收盘收复第一根大部分跌幅
        k3_body = df["body"].iloc[i]
        if k3_body < df["atr14"].iloc[i] * 0.5 or df["body_dir"].iloc[i] != 1:
            continue

        recovery = df["Close"].iloc[i] > (df["Open"].iloc[i - 2] + df["Close"].iloc[i - 2]) / 2
        if recovery and _is_volume_confirmed(df, i):
            signal.iloc[i] = 2  # 强度2

    return signal


def _signal_evening_star(df: pd.DataFrame) -> pd.Series:
    """黄昏之星: 长阳 + 小K线 + 长阴，顶部反转"""
    n = len(df)
    signal = pd.Series(0, index=df.index)

    for i in range(3, n):
        k1_body = df["body"].iloc[i - 2]
        if k1_body < df["atr14"].iloc[i - 2] * 0.5 or df["body_dir"].iloc[i - 2] != 1:
            continue

        k2_body = df["body"].iloc[i - 1]
        if k2_body > k1_body * 0.5:
            continue

        k3_body = df["body"].iloc[i]
        if k3_body < df["atr14"].iloc[i] * 0.5 or df["body_dir"].iloc[i] != -1:
            continue

        drop = df["Close"].iloc[i] < (df["Open"].iloc[i - 2] + df["Close"].iloc[i - 2]) / 2
        if drop and _is_volume_confirmed(df, i):
            signal.iloc[i] = 2

    return signal


def _signal_hammer(df: pd.DataFrame) -> pd.Series:
    """锤头线: 下跌后，长下影>=实体2倍，小上影，阳线"""
    n = len(df)
    signal = pd.Series(0, index=df.index)

    for i in range(5, n):
        # 确认下跌趋势：前5日价格走低
        if df["_price"].iloc[i] > df["_price"].iloc[i - 5]:
            continue

        body = df["body"].iloc[i]
        lower = df["lower_shadow"].iloc[i]
        upper = df["upper_shadow"].iloc[i]
        atr = df["atr14"].iloc[i]

        if body == 0 or pd.isna(atr):
            continue

        # 下影线 >= 实体 2 倍
        if lower < body * 2:
            continue
        # 上影线很短
        if upper > body * 0.5:
            continue
        # 收盘 > 开盘 (阳线)
        if df["body_dir"].iloc[i] != 1:
            continue

        if _is_volume_confirmed(df, i):
            signal.iloc[i] = 1

    return signal


def _signal_shooting_star(df: pd.DataFrame) -> pd.Series:
    """射击之星: 上涨后，长上影>=实体2倍，小下影，阴线"""
    n = len(df)
    signal = pd.Series(0, index=df.index)

    for i in range(5, n):
        if df["_price"].iloc[i] < df["_price"].iloc[i - 5]:
            continue

        body = df["body"].iloc[i]
        lower = df["lower_shadow"].iloc[i]
        upper = df["upper_shadow"].iloc[i]
        atr = df["atr14"].iloc[i]

        if body == 0 or pd.isna(atr):
            continue

        if upper < body * 2:
            continue
        if lower > body * 0.5:
            continue
        if df["body_dir"].iloc[i] != -1:
            continue

        if _is_volume_confirmed(df, i):
            signal.iloc[i] = 1

    return signal


def _signal_bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """看涨吞没: 阴线后紧跟更大阳线，完全包裹前阴线 + 放量"""
    n = len(df)
    signal = pd.Series(0, index=df.index)

    for i in range(1, n):
        if df["body_dir"].iloc[i - 1] != -1:
            continue
        if df["body_dir"].iloc[i] != 1:
            continue

        # 阳线完全包裹阴线
        if df["Open"].iloc[i] >= df["Close"].iloc[i - 1]:
            continue
        if df["Close"].iloc[i] <= df["Open"].iloc[i - 1]:
            continue

        # 阳线实体 > 阴线实体
        if df["body"].iloc[i] <= df["body"].iloc[i - 1]:
            continue

        # 在下跌趋势中更有意义
        if df["_price"].iloc[i] < df["ma20"].iloc[i]:
            if _is_volume_confirmed(df, i):
                signal.iloc[i] = 2
            else:
                signal.iloc[i] = 1

    return signal


def _signal_bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    """看跌吞没: 阳线后紧跟更大阴线，完全包裹前阳线 + 放量"""
    n = len(df)
    signal = pd.Series(0, index=df.index)

    for i in range(1, n):
        if df["body_dir"].iloc[i - 1] != 1:
            continue
        if df["body_dir"].iloc[i] != -1:
            continue

        if df["Open"].iloc[i] <= df["Close"].iloc[i - 1]:
            continue
        if df["Close"].iloc[i] >= df["Open"].iloc[i - 1]:
            continue

        if df["body"].iloc[i] <= df["body"].iloc[i - 1]:
            continue

        if df["_price"].iloc[i] > df["ma20"].iloc[i]:
            if _is_volume_confirmed(df, i):
                signal.iloc[i] = 2
            else:
                signal.iloc[i] = 1

    return signal


def _signal_three_white_soldiers(df: pd.DataFrame) -> pd.Series:
    """白三兵: 连续三根递增阳线，每根收盘接近最高价"""
    n = len(df)
    signal = pd.Series(0, index=df.index)

    for i in range(2, n):
        d0 = df.iloc[i - 2]
        d1 = df.iloc[i - 1]
        d2 = df.iloc[i]

        # 三根都是阳线
        if not (d0["Close"] > d0["Open"] and d1["Close"] > d1["Open"] and d2["Close"] > d2["Open"]):
            continue

        # 每根收盘价递增
        if not (d0["Close"] < d1["Close"] < d2["Close"]):
            continue

        # 每根开盘价在前一根实体范围内
        if not (d0["Open"] < d1["Open"] < d2["Open"]):
            continue

        # 每根收盘接近最高价 (上影线短)
        if (d0["upper_shadow"] > d0["body"] * 0.3 or
            d1["upper_shadow"] > d1["body"] * 0.3 or
            d2["upper_shadow"] > d2["body"] * 0.3):
            continue

        signal.iloc[i] = 1

    return signal


def _signal_three_black_crows(df: pd.DataFrame) -> pd.Series:
    """三只乌鸦: 连续三根递减阴线，每根收盘接近最低价"""
    n = len(df)
    signal = pd.Series(0, index=df.index)

    for i in range(2, n):
        d0 = df.iloc[i - 2]
        d1 = df.iloc[i - 1]
        d2 = df.iloc[i]

        if not (d0["Close"] < d0["Open"] and d1["Close"] < d1["Open"] and d2["Close"] < d2["Open"]):
            continue

        if not (d0["Close"] > d1["Close"] > d2["Close"]):
            continue

        if not (d0["Open"] > d1["Open"] > d2["Open"]):
            continue

        if (d0["lower_shadow"] > d0["body"] * 0.3 or
            d1["lower_shadow"] > d1["body"] * 0.3 or
            d2["lower_shadow"] > d2["body"] * 0.3):
            continue

        signal.iloc[i] = 1

    return signal


# ═══════════════════════════════════════════════════════════════
# E. 原有信号 (增强版)
# ═══════════════════════════════════════════════════════════════

def _signal_big_bull_breakout(df: pd.DataFrame) -> pd.Series:
    """大阳线突破: 涨幅>3% + 突破20日高点 + 放量 + MA5斜率向上"""
    ret_ok = df["daily_ret"] > 0.03
    breakout = df["Close"] > df["high_20"].shift(1)
    vol_ok = df["Volume"] > df["vol_ma20"] * 0.8
    # 增强: MA5 斜率向上确认
    ma5_ok = pd.Series(False, index=df.index)
    for i in range(5, len(df)):
        ma5_ok.iloc[i] = _calc_slope(df["ma5"].iloc[i - 4 : i + 1], lookback=5) > 0.001
    return (ret_ok & breakout & vol_ok & ma5_ok).astype(int) * 2  # 强度2


def _signal_support_bounce(df: pd.DataFrame) -> pd.Series:
    """支撑位反弹: 低点接近20日低点 + 阳线 + 放量 + 长下影线确认"""
    near_support = df["Low"] <= df["low_20"].shift(1) * 1.02
    bullish = df["Close"] > df["Open"]
    vol_ok = df["Volume"] > df["vol_ma20"] * 0.8
    shadow_ok = df["lower_shadow"] > df["body"] * 0.8  # 增强: 要求有明显下影线
    return (near_support & bullish & vol_ok & shadow_ok).astype(int)


def _signal_reversal_after_decline(df: pd.DataFrame) -> pd.Series:
    """回调后转强: 前3日连续下跌 + 今日收阳突破昨日高点 + 放量 + MA5转向"""
    decline = (
        (df["Close"].shift(1) < df["Close"].shift(2))
        & (df["Close"].shift(2) < df["Close"].shift(3))
        & (df["Close"].shift(3) < df["Close"].shift(4))
    )
    reversal = df["Close"] > df["High"].shift(1)
    vol_ok = df["Volume"] > df["vol_ma20"] * 0.8

    # 增强: MA5 斜率从负转正
    ma5_turning = pd.Series(False, index=df.index)
    for i in range(10, len(df)):
        s_prev = _calc_slope(df["ma5"].iloc[i - 9 : i], lookback=5)
        s_curr = _calc_slope(df["ma5"].iloc[i - 4 : i + 1], lookback=5)
        ma5_turning.iloc[i] = (s_prev < 0 and s_curr > -0.0005)

    return (decline & reversal & vol_ok & ma5_turning).astype(int) * 2


# ═══════════════════════════════════════════════════════════════
# 统一入口
# ═══════════════════════════════════════════════════════════════

def detect_all(df: pd.DataFrame) -> pd.DataFrame:
    """检测所有信号，在 df 中添加 signal_xxx 列"""
    df = df.copy()

    # ── 趋势线信号 ──
    df["signal_trendline_break_up"] = _signal_trendline_break_up(df)
    df["signal_trendline_break_down"] = _signal_trendline_break_down(df)
    df["signal_trendline_bounce_up"] = _signal_trendline_bounce_up(df)
    df["signal_trendline_bounce_down"] = _signal_trendline_bounce_down(df)
    sig_ch_top, sig_ch_bot = _signal_channel_touch(df)
    df["signal_channel_top_touch"] = sig_ch_top
    df["signal_channel_bottom_touch"] = sig_ch_bot

    # ── 斜率信号 ──
    slope_signals = _signal_slope_analysis(df)
    for key, series in slope_signals.items():
        df[f"signal_{key}"] = series

    # ── 曲线信号 ──
    gold, death = _signal_ma_cross(df)
    df["signal_ma_golden_cross"] = gold
    df["signal_ma_death_cross"] = death

    conv, div_bull, div_bear = _signal_ma_convergence_divergence(df)
    df["signal_ma_convergence"] = conv
    df["signal_ma_divergence_bullish"] = div_bull
    df["signal_ma_divergence_bearish"] = div_bear

    bull_r, bear_r = _signal_ma_ribbon(df)
    df["signal_ma_ribbon_bullish"] = bull_r
    df["signal_ma_ribbon_bearish"] = bear_r

    sq_long, sq_short = _signal_bb_squeeze(df)
    df["signal_bb_squeeze_long"] = sq_long
    df["signal_bb_squeeze_short"] = sq_short

    # ── 经典形态 ──
    df["signal_dark_cloud_cover"] = _signal_dark_cloud_cover(df)
    df["signal_piercing_line"] = _signal_piercing_line(df)
    df["signal_morning_star"] = _signal_morning_star(df)
    df["signal_evening_star"] = _signal_evening_star(df)
    df["signal_hammer"] = _signal_hammer(df)
    df["signal_shooting_star"] = _signal_shooting_star(df)
    df["signal_bullish_engulfing"] = _signal_bullish_engulfing(df)
    df["signal_bearish_engulfing"] = _signal_bearish_engulfing(df)
    df["signal_three_white_soldiers"] = _signal_three_white_soldiers(df)
    df["signal_three_black_crows"] = _signal_three_black_crows(df)

    # ── 原有增强信号 ──
    df["signal_big_bull_breakout"] = _signal_big_bull_breakout(df)
    df["signal_support_bounce"] = _signal_support_bounce(df)
    df["signal_reversal_after_decline"] = _signal_reversal_after_decline(df)

    return df


def get_active_signals(df: pd.DataFrame, recent_days: int = 5) -> list[dict]:
    """获取最近 N 天内出现的所有信号"""
    df = df.copy()
    if not any(col.startswith("signal_") for col in df.columns):
        return []

    n = len(df)
    start = max(0, n - recent_days)
    recent = df.iloc[start:]

    active = []
    for col in sorted(df.columns):
        if not col.startswith("signal_"):
            continue
        key = col[7:]  # 去掉 "signal_" 前缀
        if key not in SIGNALS:
            continue

        # 找最近触发的信号
        triggered = recent[recent[col] > 0]
        if len(triggered) > 0:
            info = SIGNALS[key]
            latest_idx = triggered.index[-1]
            latest_date = str(df.loc[latest_idx, "Date"]) if "Date" in df.columns else ""
            active.append({
                "key": key,
                "name": info[0],
                "description": info[1],
                "direction": info[2],
                "strength": int(triggered[col].iloc[-1]),
                "last_date": latest_date,
                "count_recent": len(triggered),
            })

    return active


def signal_summary(df: pd.DataFrame) -> dict:
    """生成信号摘要：统计各类信号的触发情况"""
    active = get_active_signals(df, recent_days=5)

    bullish_count = sum(1 for s in active if s["direction"] == "bullish")
    bearish_count = sum(1 for s in active if s["direction"] == "bearish")
    neutral_count = sum(1 for s in active if s["direction"] == "neutral")

    # 加权评分
    score = 0
    for s in active:
        w = s["strength"]
        if s["direction"] == "bullish":
            score += w
        elif s["direction"] == "bearish":
            score -= w

    # 判定
    if score >= 3:
        verdict = "强偏多"
    elif score >= 1:
        verdict = "偏多"
    elif score <= -3:
        verdict = "强偏空"
    elif score <= -1:
        verdict = "偏空"
    else:
        verdict = "中性/震荡"

    return {
        "score": score,
        "verdict": verdict,
        "bullish_count": bullish_count,
        "bearish_count": bearish_count,
        "neutral_count": neutral_count,
        "active_signals": active,
    }
