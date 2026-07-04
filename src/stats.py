"""统计模块 - 扫描历史信号，计算未来 N 日收益/胜率/回撤"""

import pandas as pd
import numpy as np

FORWARD_PERIODS = [1, 3, 5, 10, 20]


def compute_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    """计算未来 N 日收益率"""
    df = df.copy()
    for n in FORWARD_PERIODS:
        df[f"fwd_ret_{n}d"] = df["Adj Close"].shift(-n) / df["Adj Close"] - 1
    return df


def signal_stats(df: pd.DataFrame, signal_col: str) -> dict:
    """
    统计某个信号出现后的未来收益。

    返回：
    {
        "total_occurrences": int,
        "forward_1d": {"avg_ret": float, "win_rate": float, "max_dd": float, "best": float, "worst": float},
        ...
    }
    """
    df = df.dropna(subset=[signal_col] + [f"fwd_ret_{n}d" for n in FORWARD_PERIODS])
    mask = df[signal_col] == 1
    total = mask.sum()

    if total < 3:
        return {
            "total_occurrences": int(total),
            "forward": {},
            "note": "样本不足，无法统计"
        }

    result = {"total_occurrences": int(total), "forward": {}}
    for n in FORWARD_PERIODS:
        col = f"fwd_ret_{n}d"
        vals = df.loc[mask, col].dropna()
        if len(vals) < 2:
            continue
        # 计算滚动最大回撤
        cum = (1 + vals).cumprod()
        rolling_max = cum.expanding().max()
        dd = (cum / rolling_max - 1).min()

        result["forward"][f"{n}d"] = {
            "avg_ret": round(float(vals.mean()) * 100, 2),
            "win_rate": round(float((vals > 0).mean()) * 100, 1),
            "max_dd": round(float(dd) * 100, 2),
            "best": round(float(vals.max()) * 100, 2),
            "worst": round(float(vals.min()) * 100, 2),
            "median": round(float(vals.median()) * 100, 2),
        }
    return result


def all_signal_stats(df: pd.DataFrame) -> dict:
    """对 df 中所有 signal_xxx 列计算统计"""
    df = compute_forward_returns(df)
    results = {}
    for col in df.columns:
        if col.startswith("signal_"):
            name = col.replace("signal_", "")
            results[name] = signal_stats(df, col)
    return results
