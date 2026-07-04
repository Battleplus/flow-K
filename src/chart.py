"""图表模块 - mplfinance K线 + 趋势线 + 斜率标注 + 信号标记"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.font_manager as _fm
import warnings

# 抑制字体警告
warnings.filterwarnings("ignore", message="Glyph.*missing from font")

# 强制注册微软雅黑字体
_FONT_PATH = "C:/Windows/Fonts/msyh.ttc"
_fm.fontManager.addfont(_FONT_PATH)
_cn_font = _fm.FontProperties(fname=_FONT_PATH)
_cn_font_name = _cn_font.get_name()

# 全局字体设置
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [_cn_font_name, "Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# 清除字体缓存以确保新字体生效
try:
    _fm.fontManager._load_fonts(try_read_cache=False)
except AttributeError:
    pass  # 新版 matplotlib 不需要手动重建缓存

ROOT = Path(__file__).resolve().parent.parent
CHART_DIR = ROOT / "outputs" / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

# 均线颜色方案
MA_COLORS = {
    "ma5": "#ff9800",
    "ma10": "#2196f3",
    "ma20": "#9c27b0",
    "ma60": "#4caf50",
}
MA_LINEWIDTH = {"ma5": 0.8, "ma10": 0.8, "ma20": 1.2, "ma60": 1.5}

# 方向颜色
BULL_COLOR = "#ec2c2c"   # 红色 (A股习惯: 涨红)
BEAR_COLOR = "#19a15f"   # 绿色 (A股习惯: 跌绿)
TL_UP_COLOR = "#ec2c2c"
TL_DN_COLOR = "#19a15f"


def _fit_trendline_pivots(df_plot, window=5):
    """从 pivot 点拟合趋势线，返回用于绘制的数据"""
    n = len(df_plot)
    if n < window * 2 + 1:
        return {}, {}

    # 找 pivot lows (上升趋势线)
    lows = df_plot["Low"].values
    low_pivots = []
    for i in range(window, n - window):
        if lows[i] == np.min(lows[i - window : i + window + 1]):
            low_pivots.append(i)

    # 找 pivot highs (下降趋势线)
    highs = df_plot["High"].values
    high_pivots = []
    for i in range(window, n - window):
        if highs[i] == np.max(highs[i - window : i + window + 1]):
            high_pivots.append(i)

    result = {}

    # 上升趋势线: 用最近几个 pivot lows 拟合
    if len(low_pivots) >= 3:
        recent = low_pivots[-min(5, len(low_pivots)):]
        x_vals = np.array(recent, dtype=float)
        y_vals = lows[recent]
        from scipy import stats as sp_stats
        slope, intercept, r_value, _, _ = sp_stats.linregress(x_vals, y_vals)
        if r_value ** 2 > 0.4:
            x_line = np.array([recent[0], recent[-1] + int(n * 0.1)])
            y_line = slope * x_line + intercept
            result["uptrend"] = {
                "x": x_line, "y": y_line,
                "slope": slope, "r2": r_value ** 2,
            }

    # 下降趋势线
    if len(high_pivots) >= 3:
        recent = high_pivots[-min(5, len(high_pivots)):]
        x_vals = np.array(recent, dtype=float)
        y_vals = highs[recent]
        from scipy import stats as sp_stats
        slope, intercept, r_value, _, _ = sp_stats.linregress(x_vals, y_vals)
        if r_value ** 2 > 0.4:
            x_line = np.array([recent[0], recent[-1] + int(n * 0.1)])
            y_line = slope * x_line + intercept
            result["downtrend"] = {
                "x": x_line, "y": y_line,
                "slope": slope, "r2": r_value ** 2,
            }

    # 通道: 基于上升趋势线做平行上轨
    if "uptrend" in result:
        ul = result["uptrend"]
        # 通道宽度 = 价格到趋势线最大距离的 1.2 倍
        max_dist = 0
        for i in range(low_pivots[-1], n):
            tl_y = ul["slope"] * i + (ul["y"][0] - ul["slope"] * ul["x"][0])
            dist = df_plot["High"].values[i] - tl_y
            max_dist = max(max_dist, dist)
        if max_dist > 0:
            upper_slope = ul["slope"]
            upper_intercept = ul["y"][0] - ul["slope"] * ul["x"][0] + max_dist * 1.1
            x_ch = np.array([ul["x"][0], min(ul["x"][-1], n - 1)])
            y_ch_upper = upper_slope * x_ch + upper_intercept
            y_ch_lower = ul["slope"] * x_ch + (ul["y"][0] - ul["slope"] * ul["x"][0])
            result["channel_upper"] = {"x": x_ch, "y": y_ch_upper}
            result["channel_lower"] = {"x": x_ch, "y": y_ch_lower}

    return result


def plot_kline_trend(
    df: pd.DataFrame,
    ticker: str,
    analysis: dict,
    signals: dict = None,
    save: bool = True,
) -> str:
    """绘制专业 K 线图 + 趋势线 + 均线 + 信号标注"""

    df_plot = df.copy()
    df_plot["Date"] = pd.to_datetime(df_plot["Date"])
    df_plot = df_plot.set_index("Date")

    # 拟合趋势线
    trend_lines = _fit_trendline_pivots(df_plot, window=5)

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

    # 自定义风格 (A股习惯: 红涨绿跌)
    mc = mpf.make_marketcolors(
        up="#ec2c2c", down="#19a15f",
        edge="inherit", wick="inherit",
        volume="inherit",
    )
    s = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle=":", gridcolor="#e0e0e0",
        facecolor="#fafbfc",
        figcolor="#ffffff",
    )

    # 标题
    title_parts = [ticker]
    if analysis:
        title_parts.append(f"趋势: {analysis.get('overall_trend', '')}")
    if signals:
        title_parts.append(f"信号: {signals.get('verdict', '')} (评分{signals.get('score', 0)})")
    title = " | ".join(title_parts)

    fig, axes = mpf.plot(
        df_plot,
        type="candle",
        style=s,
        addplot=apds,
        volume=True,
        figsize=(18, 10),
        title="",  # 手动设置标题以使用中文字体
        returnfig=True,
        warn_too_much_data=len(df_plot) + 1,
    )

    ax_main = axes[0]
    ax_vol = axes[2]
    n = len(df_plot)

    # 手动设置主标题 (使用中文字体)
    ax_main.set_title(
        title,
        fontsize=14, fontweight="bold", fontproperties=_cn_font, pad=10
    )

    # ── 绘制趋势线 ──
    # 上升趋势线 (红色虚线)
    if "uptrend" in trend_lines:
        tl = trend_lines["uptrend"]
        ax_main.plot(tl["x"], tl["y"], color=TL_UP_COLOR, linestyle="--", linewidth=1.5, alpha=0.8)
        mid_idx = int((tl["x"][0] + tl["x"][-1]) / 2)
        mid_y = tl["slope"] * mid_idx + (tl["y"][0] - tl["slope"] * tl["x"][0])
        ax_main.annotate(
            f"上升趋势线 R²={tl['r2']:.2f}",
            (mid_idx, mid_y),
            fontsize=7, color=TL_UP_COLOR, alpha=0.8,
            fontproperties=_cn_font,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
        )

    # 下降趋势线 (绿色虚线)
    if "downtrend" in trend_lines:
        tl = trend_lines["downtrend"]
        ax_main.plot(tl["x"], tl["y"], color=TL_DN_COLOR, linestyle="--", linewidth=1.5, alpha=0.8)
        mid_idx = int((tl["x"][0] + tl["x"][-1]) / 2)
        mid_y = tl["slope"] * mid_idx + (tl["y"][0] - tl["slope"] * tl["x"][0])
        ax_main.annotate(
            f"下降趋势线 R²={tl['r2']:.2f}",
            (mid_idx, mid_y),
            fontsize=7, color=TL_DN_COLOR, alpha=0.8,
            fontproperties=_cn_font,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
        )

    # 通道线
    if "channel_upper" in trend_lines and "channel_lower" in trend_lines:
        cu = trend_lines["channel_upper"]
        cl = trend_lines["channel_lower"]
        ax_main.plot(cu["x"], cu["y"], color="#ff9800", linestyle=":", linewidth=1, alpha=0.6)
        ax_main.plot(cl["x"], cl["y"], color="#ff9800", linestyle=":", linewidth=1, alpha=0.6)
        ax_main.fill_between(
            cu["x"], cl["y"], cu["y"],
            alpha=0.05, color="#ff9800",
        )

    # ── 标注支撑/压力位 ──
    sr = analysis.get("support_resistance", {}) if analysis else {}
    support = sr.get("support")
    resistance = sr.get("resistance")
    if support:
        ax_main.axhline(y=support, color="#4caf50", linestyle="--", linewidth=1, alpha=0.7)
        ax_main.text(n - 1, support, f" 支撑 {support}", fontsize=9, color="#4caf50", va="bottom", fontproperties=_cn_font)
    if resistance:
        ax_main.axhline(y=resistance, color="#ef5350", linestyle="--", linewidth=1, alpha=0.7)
        ax_main.text(n - 1, resistance, f" 压力 {resistance}", fontsize=9, color="#ef5350", va="top", fontproperties=_cn_font)

    # ── 标注金叉死叉 ──
    crosses = analysis.get("crosses", []) if analysis else []
    for c in crosses[-5:]:
        try:
            cross_date = pd.to_datetime(c["date"])
            if cross_date in df_plot.index:
                idx = df_plot.index.get_loc(cross_date)
                label = "▲金叉" if c["type"] == "金叉" else "▼死叉"
                color = BULL_COLOR if c["type"] == "金叉" else BEAR_COLOR
                price = df_plot.iloc[idx]["Low"] * 0.98
                ax_main.annotate(
                    label, (idx, price),
                    fontsize=8, color=color, fontweight="bold",
                    ha="center", va="top", fontproperties=_cn_font,
                )
        except Exception:
            pass

    # ── 标注主动信号 ──
    if signals and signals.get("active"):
        for sig in signals["active"][-8:]:  # 最多显示8个
            try:
                sig_date = pd.to_datetime(sig.get("last_date", ""))
                if sig_date in df_plot.index:
                    idx = df_plot.index.get_loc(sig_date)
                    direction = sig.get("direction", "neutral")
                    if direction == "bullish":
                        marker = "▲"
                        color = BULL_COLOR
                        y_offset = -0.06
                    elif direction == "bearish":
                        marker = "▼"
                        color = BEAR_COLOR
                        y_offset = 0.06
                    else:
                        marker = "◆"
                        color = "#ff9800"
                        y_offset = 0

                    price = df_plot.iloc[idx]["Close"] * (1 + y_offset)
                    ax_main.annotate(
                        marker,
                        (idx, price),
                        fontsize=10, color=color, fontweight="bold",
                        ha="center", va="center",
                    )
            except Exception:
                pass

    # ── 图例 ──
    legend_lines = []
    legend_labels = []
    for col in ["ma5", "ma10", "ma20", "ma60"]:
        if col in df_plot.columns:
            from matplotlib.lines import Line2D
            legend_lines.append(Line2D([0], [0], color=MA_COLORS[col], linewidth=1.5))
            legend_labels.append(col.upper())
    if legend_lines:
        ax_main.legend(legend_lines, legend_labels, loc="upper left", fontsize=9, ncol=4, prop=_cn_font)

    # ── 斜率信息文本框 ──
    slopes = analysis.get("slopes", {}) if analysis else {}
    slope_text = "均线斜率:\n"
    for name, info in slopes.items():
        direction_symbol = "↑" if info["direction"] == "上升" else "↓" if info["direction"] == "下降" else "→"
        slope_text += f"  {name}: {info['slope']}% {direction_symbol}  {info['direction']}\n"

    ax_main.text(
        0.01, 0.97, slope_text.strip(),
        transform=ax_main.transAxes,
        fontsize=8, fontproperties=_cn_font,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#ccc"),
    )

    # ── 信号评分文本框 ──
    if signals:
        sig_text = f"信号评分: {signals.get('score', 0)}\n"
        sig_text += f"判定: {signals.get('verdict', 'N/A')}\n"
        sig_text += f"偏多:{signals.get('bullish_count', 0)} 偏空:{signals.get('bearish_count', 0)}"
        ax_main.text(
            0.99, 0.97, sig_text.strip(),
            transform=ax_main.transAxes,
            fontsize=9, fontproperties=_cn_font,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="#1a1a2e", alpha=0.85,
                edgecolor="#0984e3", linewidth=1,
            ),
            color="#ffffff",
        )

    fpath = CHART_DIR / f"{ticker}_kline_trend.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(fpath)
