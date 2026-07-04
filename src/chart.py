"""K 线绘图模块"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from pathlib import Path

# 中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
CHART_DIR = ROOT / "outputs" / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

# 信号名称 -> 中文标签映射
SIGNAL_LABELS = {
    "big_bull_breakout": "大阳线突破",
    "support_bounce": "支撑位反弹",
    "long_lower_shadow": "长下影线",
    "reversal_after_decline": "回调后转强",
}

# 信号 -> 标记样式
SIGNAL_MARKERS = {
    "signal_big_bull_breakout": ("^", "#26a69a", "大阳线突破"),
    "signal_support_bounce": ("s", "#42a5f5", "支撑位反弹"),
    "signal_long_lower_shadow": ("v", "#ff9800", "长下影线"),
    "signal_reversal_after_decline": ("D", "#ab47bc", "回调后转强"),
}


def plot_kline_with_signals(
    df: pd.DataFrame,
    ticker: str,
    signal_cols: list[str] | None = None,
    lookback: int = 120,
    save: bool = True,
) -> str:
    """绘制最近 lookback 天的 K 线 + MA + 信号标记。返回保存路径。"""
    df = df.tail(lookback).copy()
    df["idx"] = range(len(df))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    # K 线
    colors = np.where(df["Close"] >= df["Open"], "#26a69a", "#ef5350")
    ax1.bar(df["idx"], df["High"] - df["Low"], bottom=df["Low"],
            color=colors, width=0.6, linewidth=0.5)
    ax1.bar(df["idx"], abs(df["Close"] - df["Open"]),
            bottom=df[["Open", "Close"]].min(axis=1),
            color=colors, width=0.8, linewidth=0.5)

    # 均线
    for col, ls, lw in [("ma5", "-", 1), ("ma20", "--", 1.2), ("ma60", ":", 1)]:
        if col in df.columns:
            ax1.plot(df["idx"], df[col], label=col.upper(), linewidth=lw, linestyle=ls, alpha=0.8)

    # 信号标记（中文标签）
    if signal_cols:
        for sc in signal_cols:
            if sc in df.columns and sc in SIGNAL_MARKERS:
                m, c, label = SIGNAL_MARKERS[sc]
                pts = df[df[sc] == 1]
                if not pts.empty:
                    ax1.scatter(pts["idx"], pts["Low"] * 0.995, marker=m,
                                color=c, s=80, zorder=5, label=label, edgecolors="white")

    ax1.set_title(f"{ticker} K线图与信号标记", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8, ncol=2)
    ax1.set_ylabel("价格")
    ax1.grid(True, alpha=0.3)

    # 成交量
    vol_colors = np.where(df["Close"] >= df["Open"], "#26a69a", "#ef5350")
    ax2.bar(df["idx"], df["Volume"], color=vol_colors, width=0.8, alpha=0.7)
    if "vol_ma20" in df.columns:
        ax2.plot(df["idx"], df["vol_ma20"], color="black", linewidth=0.8, linestyle="--", label="VOL MA20")
    ax2.set_ylabel("成交量")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # x 轴标签
    tick_step = max(1, len(df) // 8)
    tick_idx = df["idx"].iloc[::tick_step]
    tick_labels = df["Date"].iloc[::tick_step].dt.strftime("%Y-%m-%d")
    ax2.set_xticks(tick_idx)
    ax2.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    fpath = CHART_DIR / f"{ticker}_kline_signals.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(fpath)


def plot_forward_distribution(
    df: pd.DataFrame,
    signal_col: str,
    ticker: str,
    forward_n: int = 5,
    save: bool = True,
) -> str:
    """绘制信号后未来 N 日收益分布直方图"""
    col = f"fwd_ret_{forward_n}d"
    if col not in df.columns or signal_col not in df.columns:
        return ""
    vals = df.loc[df[signal_col] == 1, col].dropna() * 100
    if len(vals) < 3:
        return ""

    signal_name = signal_col.replace("signal_", "")
    chinese_name = SIGNAL_LABELS.get(signal_name, signal_name)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(vals, bins=30, color="#4a90d9", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.2)
    ax.axvline(vals.mean(), color="green", linestyle="-", linewidth=1.5, label=f"均值: {vals.mean():.2f}%")
    ax.set_title(f"{ticker} - {chinese_name} 未来{forward_n}日收益分布 (n={len(vals)})", fontsize=12, fontweight="bold")
    ax.set_xlabel("收益率 %")
    ax.set_ylabel("次数")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fpath = CHART_DIR / f"{ticker}_{signal_col}_fwd{forward_n}d_dist.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(fpath)
