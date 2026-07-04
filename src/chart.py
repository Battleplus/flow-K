"""图表模块 - mplfinance K线 + 多色均线 + 趋势标注"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
CHART_DIR = ROOT / "outputs" / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

# 均线颜色方案
MA_COLORS = {
    "ma5": "#ff9800",   # 橙色
    "ma10": "#2196f3",  # 蓝色
    "ma20": "#9c27b0",  # 紫色
    "ma60": "#4caf50",  # 绿色
}
MA_LINEWIDTH = {"ma5": 0.8, "ma10": 0.8, "ma20": 1.2, "ma60": 1.5}


def plot_kline_trend(
    df: pd.DataFrame,
    ticker: str,
    analysis: dict,
    save: bool = True,
) -> str:
    """绘制专业 K 线图 + 多色均线 + 趋势标注"""

    # 准备 mplfinance 格式数据
    df_plot = df.copy()
    df_plot["Date"] = pd.to_datetime(df_plot["Date"])
    df_plot = df_plot.set_index("Date")

    # 构建均线 addplot
    apds = []
    for col in ["ma5", "ma10", "ma20", "ma60"]:
        if col in df_plot.columns:
            apds.append(
                mpf.make_addplot(
                    df_plot[col],
                    color=MA_COLORS[col],
                    width=MA_LINEWIDTH[col],
                    label=col.upper(),
                )
            )

    # 自定义风格
    mc = mpf.make_marketcolors(
        up="#26a69a", down="#ef5350",
        edge="inherit", wick="inherit",
        volume="inherit",
    )
    s = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle=":", gridcolor="#e0e0e0",
        facecolor="#fafbfc",
        figcolor="#ffffff",
    )

    fig, axes = mpf.plot(
        df_plot,
        type="candle",
        style=s,
        addplot=apds,
        volume=True,
        figsize=(16, 9),
        title=f"{ticker}  趋势: {analysis['overall_trend']} | {analysis['ma_alignment']['alignment']}",
        returnfig=True,
        warn_too_much_data=len(df_plot) + 1,
    )

    ax_main = axes[0]
    ax_vol = axes[2]

    # 标注支撑/压力位
    sr = analysis.get("support_resistance", {})
    support = sr.get("support")
    resistance = sr.get("resistance")
    if support:
        ax_main.axhline(y=support, color="#4caf50", linestyle="--", linewidth=1, alpha=0.7)
        ax_main.text(len(df_plot) - 1, support, f" 支撑 {support}", fontsize=9, color="#4caf50", va="bottom")
    if resistance:
        ax_main.axhline(y=resistance, color="#ef5350", linestyle="--", linewidth=1, alpha=0.7)
        ax_main.text(len(df_plot) - 1, resistance, f" 压力 {resistance}", fontsize=9, color="#ef5350", va="top")

    # 标注均线交叉点
    crosses = analysis.get("crosses", [])
    for c in crosses[-5:]:  # 最近 5 个交叉
        try:
            cross_date = pd.to_datetime(c["date"])
            if cross_date in df_plot.index:
                idx = df_plot.index.get_loc(cross_date)
                label = "▲金叉" if c["type"] == "金叉" else "▼死叉"
                color = "#26a69a" if c["type"] == "金叉" else "#ef5350"
                price = df_plot.iloc[idx]["Low"] * 0.98
                ax_main.annotate(
                    label, (idx, price),
                    fontsize=8, color=color, fontweight="bold",
                    ha="center", va="top",
                )
        except Exception:
            pass

    # 图例
    legend_lines = []
    legend_labels = []
    for col in ["ma5", "ma10", "ma20", "ma60"]:
        if col in df_plot.columns:
            from matplotlib.lines import Line2D
            legend_lines.append(Line2D([0], [0], color=MA_COLORS[col], linewidth=1.5))
            legend_labels.append(col.upper())
    if legend_lines:
        ax_main.legend(legend_lines, legend_labels, loc="upper left", fontsize=9, ncol=4)

    # 添加趋势信息文本框
    slopes = analysis.get("slopes", {})
    slope_text = "均线斜率:\n"
    for name, info in slopes.items():
        direction_symbol = "↑" if info["direction"] == "上升" else "↓" if info["direction"] == "下降" else "→"
        slope_text += f"  {name}: {info['slope']}% {direction_symbol}  {info['direction']}\n"

    ax_main.text(
        0.01, 0.97, slope_text.strip(),
        transform=ax_main.transAxes,
        fontsize=8, fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#ccc"),
    )

    fpath = CHART_DIR / f"{ticker}_kline_trend.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(fpath)
