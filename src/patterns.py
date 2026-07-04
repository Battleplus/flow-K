"""K 线信号定义模块 — 技术指标计算 + 4 个经典 K 线信号检测

信号列表:
  1. big_bull_breakout      大阳线突破: 涨幅>3%，突破20日高点，放量确认
  2. support_bounce         支撑位反弹: 低点接近20日低点，收盘>开盘，放量
  3. long_lower_shadow      长下影线:   下影线>实体2倍，收盘>开盘
  4. reversal_after_decline 回调后转强: 前3日下跌，今日收盘>昨日高点，放量
"""

import pandas as pd
import numpy as np


# ── 信号注册表 ──────────────────────────────────────────────
# 格式: key -> (中文名, 人话描述)
SIGNALS = {
    "big_bull_breakout": (
        "大阳线突破",
        "今日涨幅 > 3%，收盘价突破过去 20 日高点，成交量放大",
    ),
    "support_bounce": (
        "支撑位反弹",
        "最低价接近过去 20 日低点，收盘价高于开盘价，成交量放大",
    ),
    "long_lower_shadow": (
        "长下影线",
        "下影线长度 > 实体 2 倍，收盘价高于开盘价",
    ),
    "reversal_after_decline": (
        "回调后转强",
        "前 3 日连续下跌，今日收盘价高于昨日高点，成交量放大",
    ),
}


# ── 技术指标 ────────────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算 MA、成交量均线、20 日高低点等辅助指标"""
    df = df.copy()

    # 均线
    df["ma5"] = df["Adj Close"].rolling(5, min_periods=1).mean()
    df["ma20"] = df["Adj Close"].rolling(20, min_periods=1).mean()
    df["ma60"] = df["Adj Close"].rolling(60, min_periods=1).mean()

    # 成交量均线
    df["vol_ma20"] = df["Volume"].rolling(20, min_periods=1).mean()

    # 20 日高低点（用于支撑/压力判断）
    df["high_20"] = df["High"].rolling(20, min_periods=1).max()
    df["low_20"] = df["Low"].rolling(20, min_periods=1).min()

    # 日收益率
    df["daily_ret"] = df["Adj Close"].pct_change()

    # K 线实体与影线
    body = (df["Close"] - df["Open"]).abs()
    lower_shadow = df[["Open", "Close"]].min(axis=1) - df["Low"]
    upper_shadow = df["High"] - df[["Open", "Close"]].max(axis=1)

    df["body"] = body
    df["lower_shadow"] = lower_shadow
    df["upper_shadow"] = upper_shadow

    return df


# ── 信号检测函数 ────────────────────────────────────────────
def _signal_big_bull_breakout(df: pd.DataFrame) -> pd.Series:
    """大阳线突破: 涨幅>3% + 突破20日高点 + 放量"""
    ret_ok = df["daily_ret"] > 0.03
    breakout = df["Close"] > df["high_20"].shift(1)
    vol_ok = df["Volume"] > df["vol_ma20"] * 0.8
    return (ret_ok & breakout & vol_ok).astype(int)


def _signal_support_bounce(df: pd.DataFrame) -> pd.Series:
    """支撑位反弹: 低点接近20日低点 + 收盘>开盘 + 放量"""
    near_support = df["Low"] <= df["low_20"].shift(1) * 1.02
    bullish = df["Close"] > df["Open"]
    vol_ok = df["Volume"] > df["vol_ma20"] * 0.8
    return (near_support & bullish & vol_ok).astype(int)


def _signal_long_lower_shadow(df: pd.DataFrame) -> pd.Series:
    """长下影线: 下影线>实体2倍 + 下影线>上影线 + 收盘>开盘"""
    body_ok = df["body"] > 0  # 阳线
    shadow_ok = df["lower_shadow"] > df["body"] * 2
    upper_ok = df["lower_shadow"] > df["upper_shadow"]
    return (body_ok & shadow_ok & upper_ok).astype(int)


def _signal_reversal_after_decline(df: pd.DataFrame) -> pd.Series:
    """回调后转强: 前3日下跌 + 今日收盘>昨日高点 + 放量"""
    decline = (
        (df["Close"].shift(1) < df["Close"].shift(2))
        & (df["Close"].shift(2) < df["Close"].shift(3))
        & (df["Close"].shift(3) < df["Close"].shift(4))
    )
    reversal = df["Close"] > df["High"].shift(1)
    vol_ok = df["Volume"] > df["vol_ma20"] * 0.8
    return (decline & reversal & vol_ok).astype(int)


# ── 统一入口 ───────────────────────────────────────────────
def detect_all(df: pd.DataFrame) -> pd.DataFrame:
    """检测所有信号，在 df 中添加 signal_xxx 列"""
    df = df.copy()
    df["signal_big_bull_breakout"] = _signal_big_bull_breakout(df)
    df["signal_support_bounce"] = _signal_support_bounce(df)
    df["signal_long_lower_shadow"] = _signal_long_lower_shadow(df)
    df["signal_reversal_after_decline"] = _signal_reversal_after_decline(df)
    return df
