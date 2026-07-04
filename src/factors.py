"""
因子计算模块 — 所有"线"背后的计算逻辑
每条线 = 一个因子，按类别组织:

  A. 趋势因子:  MA/EMA/MA Ribbon/趋势线斜率
  B. 波动率因子: BB/Keltner/ATR轨道/历史波动率
  C. 动量因子:  RSI/Stochastic/MACD/ROC/Williams %R
  D. 成交量因子: VWAP/OBV/成交量MA/量价背离
  E. 结构因子:  Ichimoku/Donchian/Pivot Points
  F. 自适应因子: ADX/DMI/市场状态检测
"""

import pandas as pd
import numpy as np
from scipy import stats as sp_stats


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=1).mean()


def _std(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=1).std()


# ═══════════════════════════════════════════════════════════════
# A. 趋势因子
# ═══════════════════════════════════════════════════════════════

def add_trend_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加趋势类因子到 df，返回 df（inplace 风格）
    新增列:
      ema12, ema26, ema50, ema100
      ma_ribbon_* (MA5~MA60 的密集程度)
      trend_slope_ma20 (MA20的滚动斜率)
      ma_distance (价格离MA20的百分比距离)
    """
    df = df.copy()
    price = df["Close"]

    # EMA 系列
    for p, name in [(12, "ema12"), (26, "ema26"), (50, "ema50"), (100, "ema100")]:
        df[name] = _ema(price, p)

    # MA Ribbon 宽度 (各MA之间的最大间距，越小=收敛=变盘前兆)
    ma_cols = ["ma5", "ma10", "ma20", "ma60"]
    if all(c in df.columns for c in ma_cols):
        ribbon_vals = df[ma_cols].values
        df["ma_ribbon_width"] = np.max(ribbon_vals, axis=1) - np.min(ribbon_vals, axis=1)
        df["ma_ribbon_width_pct"] = df["ma_ribbon_width"] / df["ma20"]

    # MA20 滚动斜率 (短期趋势强度)
    df["trend_slope_ma20"] = (
        df["ma20"].rolling(10, min_periods=3).apply(
            lambda x: sp_stats.linregress(np.arange(len(x)), x)[0] / x.mean() * 100
            if x.mean() != 0 else 0, raw=False
        )
    )

    # 价格离 MA20 的距离 (%)
    df["ma_distance"] = (price - df["ma20"]) / df["ma20"] * 100

    # 均线排列得分 (-4 ~ +4)
    # 多头排列: +4, 空头排列: -4, 混乱: 0
    def _ribbon_score(row):
        ma5, ma10, ma20, ma60 = row["ma5"], row["ma10"], row["ma20"], row["ma60"]
        score = 0
        if ma5 > ma10: score += 1
        else: score -= 1
        if ma10 > ma20: score += 1
        else: score -= 1
        if ma20 > ma60: score += 1
        else: score -= 1
        if ma5 > ma60: score += 1
        else: score -= 1
        return score
    df["ma_ribbon_score"] = df.apply(_ribbon_score, axis=1)

    return df


# ═══════════════════════════════════════════════════════════════
# B. 波动率因子
# ═══════════════════════════════════════════════════════════════

def add_volatility_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    波动率类因子:
      bb_upper/mid/lower (已在 patterns.py，这里补充带宽)
      keltner_upper/mid/lower (基于ATR的通道)
      atr_upper/atr_lower (ATR轨道，多条)
      hist_vol_20 (20日历史波动率)
      vol_rank (当前波动率在60日中的分位数)
    """
    df = df.copy()
    price = df["Close"]
    high = df["High"]
    low = df["Low"]

    # ── Keltner 通道 (基于EMA的中轨 + ATR通道) ──
    ema20 = _ema(price, 20)
    atr = df.get("atr14", _calc_atr(high, low, price, 14))
    df["keltner_mid"] = ema20
    df["keltner_upper"] = ema20 + 2 * atr
    df["keltner_lower"] = ema20 - 2 * atr

    # ── ATR 轨道 (多倍ATR，类似支撑压力轨道) ──
    for mult in [1.0, 1.5, 2.0, 2.5]:
        df[f"atr_upper_{int(mult*10)}"] = price + mult * atr
        df[f"atr_lower_{int(mult*10)}"] = price - mult * atr

    # ── 历史波动率 (20日年化) ──
    ret = price.pct_change()
    df["hist_vol_20"] = ret.rolling(20).std() * np.sqrt(252) * 100  # 年化%

    # ── 波动率分位数 (60日) ──
    df["vol_rank"] = df["hist_vol_20"].rolling(60, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    # ── 布林带宽度 (补充) ──
    if "bb_upper" in df.columns and "bb_lower" in df.columns:
        df["bb_width_pct"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"] * 100
        # 带宽分位数 (低=收敛, 高=发散)
        df["bb_squeeze_rank"] = df["bb_width_pct"].rolling(60, min_periods=20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

    return df


def _calc_atr(high, low, close, period=14):
    """计算ATR序列"""
    hl = high - low
    hc = (high - close.shift(1)).abs()
    lc = (low - close.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


# ═══════════════════════════════════════════════════════════════
# C. 动量因子
# ═══════════════════════════════════════════════════════════════

def add_momentum_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    动量类因子:
      rsi_14, rsi_28
      stochastic_k, stochastic_d (KDJ的K/D)
      macd_line, macd_signal, macd_hist
      roc_10 (Rate of Change)
      williams_r (Williams %R)
      cci (Commodity Channel Index)
    """
    df = df.copy()
    price = df["Close"]
    high = df["High"]
    low = df["Low"]

    # ── RSI ──
    for p in [14, 28]:
        delta = price.diff()
        gain = delta.clip(lower=0).rolling(p, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(p, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        df[f"rsi_{p}"] = 100 - (100 / (1 + rs))

    # ── Stochastic (KD) ──
    low_14 = low.rolling(14, min_periods=1).min()
    high_14 = high.rolling(14, min_periods=1).max()
    stoch_k = 100 * (price - low_14) / (high_14 - low_14).replace(0, np.nan)
    df["stochastic_k"] = stoch_k
    df["stochastic_d"] = stoch_k.rolling(3, min_periods=1).mean()

    # ── MACD ──
    ema12 = _ema(price, 12)
    ema26 = _ema(price, 26)
    df["macd_line"] = ema12 - ema26
    df["macd_signal"] = _ema(df["macd_line"], 9)
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]

    # ── ROC (Rate of Change) ──
    df["roc_10"] = price.pct_change(10) * 100
    df["roc_20"] = price.pct_change(20) * 100

    # ── Williams %R ──
    df["williams_r"] = -100 * (high_14 - price) / (high_14 - low_14).replace(0, np.nan)

    # ── CCI (Commodity Channel Index) ──
    tp = (high + low + price) / 3  # Typical Price
    tp_ma = tp.rolling(20, min_periods=1).mean()
    tp_std = tp.rolling(20, min_periods=1).std()
    df["cci_20"] = (tp - tp_ma) / (0.015 * tp_std)

    return df


# ═══════════════════════════════════════════════════════════════
# D. 成交量因子
# ═══════════════════════════════════════════════════════════════

def add_volume_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    成交量类因子:
      vwap (VWAP)
      obv (On-Balance Volume)
      volume_ma_ratio (成交量相对均线倍数)
      volume_price_trend (量价趋势)
      accumulation_distribution (AD线)
    """
    df = df.copy()
    price = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    # ── VWAP (成交量加权均价, 重置版: 每个周期从头算) ──
    # 累计VWAP
    cum_vol = vol.cumsum()
    cum_vol_price = (vol * price).cumsum()
    df["vwap"] = cum_vol_price / cum_vol.replace(0, 1)

    # ── OBV ──
    obv = [0]
    for i in range(1, len(price)):
        if price.iloc[i] > price.iloc[i - 1]:
            obv.append(obv[-1] + vol.iloc[i])
        elif price.iloc[i] < price.iloc[i - 1]:
            obv.append(obv[-1] - vol.iloc[i])
        else:
            obv.append(obv[-1])
    df["obv"] = obv

    # ── 成交量相对均线 ──
    vol_ma = vol.rolling(20, min_periods=1).mean()
    df["volume_ratio"] = vol / vol_ma
    df["volume_ma20"] = vol_ma

    # ── 量价背离检测 (简化: 价格新高但成交量未新高) ──
    price_20d_high = price.rolling(20, min_periods=1).max()
    vol_20d_high = vol.rolling(20, min_periods=1).max()
    df["price_new_high"] = price == price_20d_high
    df["vol_not_new_high"] = vol < vol_20d_high.shift(1)
    # 量价背离: 价格新高但量未新高
    df["volume_divergence_bear"] = df["price_new_high"] & df["vol_not_new_high"]

    # ── Accumulation/Distribution Line ──
    clv = ((price - low) - (high - price)) / (high - low).replace(0, np.nan)  # Close Location Value
    ad = (clv * vol).cumsum()
    df["ad_line"] = ad

    return df


# ═══════════════════════════════════════════════════════════════
# E. 结构因子 (Ichimoku / Donchian / Pivot)
# ═══════════════════════════════════════════════════════════════

def add_structure_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    结构类因子 (Ichimoku 云图为主要新增):
      ichimoku_conv (转换线, 9日)
      ichimoku_base (基准线, 26日)
      ichimoku_span_a (先行span A)
      ichimoku_span_b (先行span B)
      ichimoku_lag (滞后span)
      donchian_high/donchian_low (已在chart.py，这里写入df)
    """
    df = df.copy()
    high = df["High"]
    low = df["Low"]
    price = df["Close"]

    # ── Ichimoku 云图 ──
    # 转换线 (Tenkan-sen): (9日高 + 9日低) / 2
    df["ichimoku_conv"] = (high.rolling(9, min_periods=1).max() + low.rolling(9, min_periods=1).min()) / 2

    # 基准线 (Kijun-sen): (26日高 + 26日低) / 2
    df["ichimoku_base"] = (high.rolling(26, min_periods=1).max() + low.rolling(26, min_periods=1).min()) / 2

    # 先行span A (Senkou Span A): (转换线 + 基准线) / 2，向前平移26日
    span_a = ((df["ichimoku_conv"] + df["ichimoku_base"]) / 2)
    df["ichimoku_span_a"] = span_a.shift(26)

    # 先行span B (Senkou Span B): (52日高 + 52日低) / 2，向前平移26日
    span_b = (high.rolling(52, min_periods=1).max() + low.rolling(52, min_periods=1).min()) / 2
    df["ichimoku_span_b"] = span_b.shift(26)

    # 滞后span (Chikou Span): 收盘价向后平移26日
    df["ichimoku_lag"] = price.shift(-26)

    # ── Donchian 通道 ──
    df["donchian_high"] = high.rolling(20, min_periods=1).max()
    df["donchian_low"] = low.rolling(20, min_periods=1).min()
    df["donchian_mid"] = (df["donchian_high"] + df["donchian_low"]) / 2

    # ── 经典 Pivot Points (每日) ──
    # 用前一日数据计算今日pivot
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = price.shift(1)
    pp = (prev_high + prev_low + prev_close) / 3
    df["pivot_pp"] = pp
    df["pivot_r1"] = 2 * pp - prev_low
    df["pivot_s1"] = 2 * pp - prev_high
    df["pivot_r2"] = pp + (prev_high - prev_low)
    df["pivot_s2"] = pp - (prev_high - prev_low)

    return df


# ═══════════════════════════════════════════════════════════════
# F. 自适应因子 (市场状态 / ADX / DMI)
# ═══════════════════════════════════════════════════════════════

def add_adaptive_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    自适应因子:
      adx_14 (平均趋向指数)
      di_plus, di_minus (动向指标)
      market_regime (市场状态: trending/ranging/volatile)
      trend_strength (趋势强度评分 0-3)
    """
    df = df.copy()
    high = df["High"]
    low = df["Low"]
    price = df["Close"]

    # ── DMI (+DI, -DI, ADX) ──
    # True Range
    tr = pd.concat([
        high - low,
        (high - price.shift(1)).abs(),
        (low - price.shift(1)).abs(),
    ], axis=1).max(axis=1)

    # +DM, -DM
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    # 约束: +DM > -DM 时 -DM=0; 反之亦然
    mask = plus_dm > minus_dm
    minus_dm = minus_dm.where(mask, 0)
    plus_dm = plus_dm.where(~mask, 0)

    tr_smooth = tr.rolling(14, min_periods=1).sum()
    plus_di = 100 * plus_dm.rolling(14, min_periods=1).sum() / tr_smooth.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(14, min_periods=1).sum() / tr_smooth.replace(0, np.nan)
    df["di_plus"] = plus_di
    df["di_minus"] = minus_di

    # ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx_14"] = dx.rolling(14, min_periods=1).mean()

    # ── 趋势强度 (综合评分) ──
    # ADX > 25 = 有趋势; > 40 = 强趋势
    # +DI > -DI = 多头; 反之空头
    def _trend_strength(row):
        adx = row["adx_14"]
        dp = row["di_plus"]
        dm = row["di_minus"]
        if pd.isna(adx) or pd.isna(dp) or pd.isna(dm):
            return 0
        if adx > 40:
            return 3 if dp > dm else -3
        elif adx > 25:
            return 2 if dp > dm else -2
        elif adx > 15:
            return 1 if dp > dm else -1
        else:
            return 0  # 无明显趋势
    df["trend_strength"] = df.apply(_trend_strength, axis=1)

    # ── 市场状态标注 ──
    # 用 bb_squeeze_rank + adx 判断
    if "bb_squeeze_rank" in df.columns:
        def _regime(row):
            adx = row.get("adx_14", 20)
            sq = row.get("bb_squeeze_rank", 0.5)
            vol = row.get("vol_rank", 0.5)
            if sq < 0.2 and adx < 20:
                return "ranging"         # 收敛横盘
            elif adx > 30:
                return "trending"        # 强趋势
            elif vol > 0.7:
                return "volatile"        # 高波动
            else:
                return "ranging"
        df["market_regime"] = df.apply(_regime, axis=1)
    else:
        df["market_regime"] = "unknown"

    return df


# ═══════════════════════════════════════════════════════════════
# 统一入口
# ═══════════════════════════════════════════════════════════════

def add_all_factors(df: pd.DataFrame) -> pd.DataFrame:
    """计算所有因子，返回增强后的 df。会自动先计算基础指标。"""
    df = df.copy()
    # 确保基础指标已计算
    if "ma20" not in df.columns:
        try:
            from src.patterns import add_indicators as _add_ind
            df = _add_ind(df)
        except Exception:
            # 如果导入失败，手动计算基础MA
            price = df["Close"]
            for p in [5, 10, 20, 60]:
                df[f"ma{p}"] = price.rolling(p, min_periods=1).mean()
            df["atr14"] = _calc_atr(df["High"], df["Low"], price, 14)
            df["bb_mid"] = df["ma20"]
            bb_std = price.rolling(20).std()
            df["bb_upper"] = df["bb_mid"] + 2 * bb_std
            df["bb_lower"] = df["bb_mid"] - 2 * bb_std
            df["Volume"] = df.get("Volume", pd.Series(1, index=df.index))
            df["vol_ma20"] = df["Volume"].rolling(20, min_periods=1).mean()

    df = add_trend_factors(df)
    df = add_volatility_factors(df)
    df = add_momentum_factors(df)
    df = add_volume_factors(df)
    df = add_structure_factors(df)
    df = add_adaptive_factors(df)
    return df


# ═══════════════════════════════════════════════════════════════
# 因子元数据 (用于图表分组和说明)
# ═══════════════════════════════════════════════════════════════

FACTOR_META = {
    # 趋势因子 → 主图
    "ma5":            ("趋势", "MA5 短期均线", "#ec2c2c", 0.8),
    "ma10":           ("趋势", "MA10 短期均线", "#2196f3", 0.8),
    "ma20":           ("趋势", "MA20 中期均线", "#9c27b0", 1.2),
    "ma60":           ("趋势", "MA60 长期均线", "#4caf50", 1.5),
    "ema12":          ("趋势", "EMA12 指数均线", "#00bcd4", 0.7),
    "ema26":          ("趋势", "EMA26 指数均线", "#795548", 0.7),
    "ema50":          ("趋势", "EMA50 指数均线", "#ff9800", 0.8),
    "ema100":         ("趋势", "EMA100 指数均线", "#9c27b0", 0.9),
    "trend_slope_ma20": ("趋势", "MA20斜率(滚动)", "#e91e63", 0.0),  # 不在主图
    "ma_ribbon_score":  ("趋势", "均线排列得分", "#607d8b", 0.0),

    # 波动率因子 → 主图 or 副图
    "bb_upper":       ("波动率", "布林上轨", "#e53935", 0.6),
    "bb_mid":         ("波动率", "布林中轨", "#78909c", 0.5),
    "bb_lower":       ("波动率", "布林下轨", "#43a047", 0.6),
    "keltner_upper":  ("波动率", "Keltner上轨(EMA+ATR)", "#ff6f00", 0.5),
    "keltner_mid":    ("波动率", "Keltner中轨(EMA20)", "#ff6f00", 0.7),
    "keltner_lower":  ("波动率", "Keltner下轨(EMA-ATR)", "#ff6f00", 0.5),
    "atr_upper_20":   ("波动率", "ATR轨道 +2.0×ATR", "#ef5350", 0.4),
    "atr_lower_20":   ("波动率", "ATR轨道 -2.0×ATR", "#66bb6a", 0.4),
    "hist_vol_20":    ("波动率", "20日历史波动率(%)", "#9c27b0", 0.0),

    # 动量因子 → 副图 (独立面板)
    "rsi_14":         ("动量", "RSI(14)", "#e91e63", 0.0),
    "rsi_28":         ("动量", "RSI(28)", "#9c27b0", 0.0),
    "stochastic_k":   ("动量", "Stochastic %K", "#ff9800", 0.0),
    "stochastic_d":   ("动量", "Stochastic %D", "#2196f3", 0.0),
    "macd_line":      ("动量", "MACD线", "#ff5722", 0.0),
    "macd_signal":    ("动量", "MACD信号线", "#448aff", 0.0),
    "macd_hist":      ("动量", "MACD柱状图", "#9e9e9e", 0.0),
    "roc_10":         ("动量", "ROC(10) 变化率(%)", "#00bcd4", 0.0),
    "williams_r":     ("动量", "Williams %R", "#8bc34a", 0.0),

    # 成交量因子 → 副图
    "vwap":           ("成交量", "VWAP 成交量加权均价", "#e91e63", 1.0),
    "obv":            ("成交量", "OBV 能量潮", "#9c27b0", 0.0),
    "volume_ratio":   ("成交量", "成交量相对倍数", "#ff9800", 0.0),
    "ad_line":        ("成交量", "A/D 积累分配线", "#4caf50", 0.0),

    # 结构因子 → 主图
    "ichimoku_conv":  ("结构", "Ichimoku 转换线(9)", "#e91e63", 0.7),
    "ichimoku_base":  ("结构", "Ichimoku 基准线(26)", "#00bcd4", 0.7),
    "ichimoku_span_a": ("结构", "Ichimoku 先行Span A", "#4caf50", 0.4),
    "ichimoku_span_b": ("结构", "Ichimoku 先行Span B", "#ef5350", 0.4),
    "donchian_high":  ("结构", "Donchian 上轨(20)", "#ff6f00", 0.5),
    "donchian_low":   ("结构", "Donchian 下轨(20)", "#ff6f00", 0.5),
    "pivot_pp":       ("结构", "Pivot Point", "#9c27b0", 0.5),

    # 自适应因子
    "adx_14":         ("自适应", "ADX 趋向指数", "#ff5722", 0.0),
    "di_plus":        ("自适应", "+DI 正向动向", "#4caf50", 0.0),
    "di_minus":       ("自适应", "-DI 负向动向", "#ef5350", 0.0),
    "trend_strength": ("自适应", "趋势强度(-3~+3)", "#607d8b", 0.0),
}

# 因子分组 → 图表面板映射
#   "main":   主图 (K线叠加)
#   "sub_momentum": 副图1 (动量)
#   "sub_volatility": 副图2 (波动率)
#   "sub_volume": 副图3 (成交量)
FACTOR_PANEL = {
    "main": [
        "ma5", "ma10", "ma20", "ma60",
        "ema12", "ema26", "ema50",
        "bb_upper", "bb_mid", "bb_lower",
        "keltner_upper", "keltner_mid", "keltner_lower",
        "atr_upper_20", "atr_lower_20",
        "vwap",
        "ichimoku_conv", "ichimoku_base",
        "ichimoku_span_a", "ichimoku_span_b",
        "donchian_high", "donchian_low",
        "pivot_pp", "pivot_r1", "pivot_s1",
    ],
    "sub_momentum": [
        "rsi_14", "rsi_28",
        "stochastic_k", "stochastic_d",
        "macd_hist",
        "roc_10", "williams_r",
    ],
    "sub_volatility": [
        "hist_vol_20", "vol_rank",
        "bb_width_pct", "bb_squeeze_rank",
    ],
    "sub_volume": [
        "volume_ratio", "obv", "ad_line",
    ],
    "sub_directional": [
        "adx_14", "di_plus", "di_minus", "trend_strength",
    ],
}
