"""K 线形态识别模块 - 4 个核心信号"""

import pandas as pd
import numpy as np


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """添加基础技术指标"""
    df = df.copy()
    df["ma5"] = df["Adj Close"].rolling(5).mean()
    df["ma20"] = df["Adj Close"].rolling(20).mean()
    df["ma60"] = df["Adj Close"].rolling(60).mean()
    df["vol_ma20"] = df["Volume"].rolling(20).mean()
    df["high_20"] = df["High"].rolling(20).max()
    df["low_20"] = df["Low"].rolling(20).min()
    df["ret_1d"] = df["Adj Close"].pct_change()
    df["ret_3d"] = df["Adj Close"].pct_change(3)
    df["body"] = abs(df["Close"] - df["Open"])
    df["upper_shadow"] = df["High"] - df[["Open", "Close"]].max(axis=1)
    df["lower_shadow"] = df[["Open", "Close"]].min(axis=1) - df["Low"]
    return df


# -------- 4 个信号函数 --------

def signal_big_bull_breakout(df: pd.DataFrame, pct_threshold: float = 0.03) -> pd.Series:
    """大阳线突破：涨幅 > pct_threshold，收盘突破 20 日高点"""
    cond1 = df["ret_1d"] > pct_threshold
    cond2 = df["Close"] > df["high_20"].shift(1)
    cond3 = df["Volume"] > df["vol_ma20"]
    return (cond1 & cond2 & cond3).astype(int)


def signal_support_bounce(df: pd.DataFrame) -> pd.Series:
    """支撑位反弹：最低价接近 20 日低点，收盘价 > 开盘价"""
    close_to_support = (df["Low"] - df["low_20"].shift(1)).abs() / df["low_20"].shift(1) < 0.02
    cond1 = close_to_support
    cond2 = df["Close"] > df["Open"]
    cond3 = df["Volume"] > df["vol_ma20"] * 0.8
    return (cond1 & cond2 & cond3).astype(int)


def signal_long_lower_shadow(df: pd.DataFrame) -> pd.Series:
    """长下影线：下影线长度 > 实体 2 倍，收盘价 > 开盘价"""
    cond1 = df["lower_shadow"] > df["body"] * 2
    cond2 = df["lower_shadow"] > df["upper_shadow"]
    cond3 = df["Close"] > df["Open"]
    cond4 = df["Volume"] > df["vol_ma20"] * 0.6
    return (cond1 & cond2 & cond3 & cond4).astype(int)


def signal_reversal_after_decline(df: pd.DataFrame) -> pd.Series:
    """连续回调后转强：前 3 日下跌，今日收盘 > 昨日高点"""
    cond1 = (df["ret_1d"].shift(1) < 0) & (df["ret_1d"].shift(2) < 0) & (df["ret_1d"].shift(3) < 0)
    cond2 = df["Close"] > df["High"].shift(1)
    cond3 = df["Volume"] > df["vol_ma20"]
    return (cond1 & cond2 & cond3).astype(int)


# -------- 聚合 --------

SIGNALS = {
    "big_bull_breakout": ("大阳线突破", signal_big_bull_breakout),
    "support_bounce": ("支撑位反弹", signal_support_bounce),
    "long_lower_shadow": ("长下影线", signal_long_lower_shadow),
    "reversal_after_decline": ("回调后转强", signal_reversal_after_decline),
}


def detect_all(df: pd.DataFrame) -> pd.DataFrame:
    """在带指标 df 上计算所有信号，返回含信号列的 df"""
    df = df.copy()
    for key, (name, func) in SIGNALS.items():
        df[f"signal_{key}"] = func(df)
    return df
