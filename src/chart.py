"""多面板图表模块 — 因子按类别分面板绘制
支持两种模式:
  - "single": 单面板 (原风格，K线+叠加线)
  - "multi":  多面板 (K线 / 动量 / 方向 / 成交量 四面板)

新增因子全部接入对应面板。
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
import matplotlib.patches as mpatches
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import mplfinance as mpf

warnings.filterwarnings("ignore", message="Glyph.*missing from font")

# 字体
_FONT_PATH = "C:/Windows/Fonts/msyh.ttc"
_fm.fontManager.addfont(_FONT_PATH)
_cn_font = _fm.FontProperties(fname=_FONT_PATH)
_cn_font_name = _cn_font.get_name()
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [_cn_font_name, "Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
CHART_DIR = ROOT / "outputs" / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

BULL_COLOR = "#ec2c2c"
BEAR_COLOR = "#19a15f"


# ═══════════════════════════════════════════════════════════════
# 趋势线拟合 (复用)
# ═══════════════════════════════════════════════════════════════

def _fit_trendlines(df_plot, windows=(3, 5, 8, 13)):
    n = len(df_plot)
    lows = df_plot["Low"].values
    highs = df_plot["High"].values
    results = {"uptrends": [], "downtrends": []}
    from scipy import stats as sp_stats

    for w in windows:
        if n < w * 4:
            continue
        low_pivots = [i for i in range(w, n - w)
                      if lows[i] == np.min(lows[i - w : i + w + 1])]
        if len(low_pivots) >= 3:
            for num in [min(3, len(low_pivots)), min(4, len(low_pivots)), min(5, len(low_pivots))]:
                if num < 3:
                    continue
                recent = low_pivots[-num:]
                xv = np.array(recent, dtype=float)
                yv = lows[recent]
                slope, intercept, r_value, _, _ = sp_stats.linregress(xv, yv)
                r2 = r_value ** 2
                if r2 > 0.35:
                    x_line = np.array([recent[0], n - 1])
                    y_line = slope * x_line + intercept
                    results["uptrends"].append({"x": x_line, "y": y_line,
                                                "slope": slope, "r2": r2, "pivots": recent})
                    break

        high_pivots = [i for i in range(w, n - w)
                       if highs[i] == np.max(highs[i - w : i + w + 1])]
        if len(high_pivots) >= 3:
            for num in [min(3, len(high_pivots)), min(4, len(high_pivots)), min(5, len(high_pivots))]:
                if num < 3:
                    continue
                recent = high_pivots[-num:]
                xv = np.array(recent, dtype=float)
                yv = highs[recent]
                slope, intercept, r_value, _, _ = sp_stats.linregress(xv, yv)
                r2 = r_value ** 2
                if r2 > 0.35:
                    x_line = np.array([recent[0], n - 1])
                    y_line = slope * x_line + intercept
                    results["downtrends"].append({"x": x_line, "y": y_line,
                                                 "slope": slope, "r2": r2, "pivots": recent})
                    break

    def _deduplicate(lines, slope_thresh=0.00005):
        kept = []
        for line in sorted(lines, key=lambda x: x["r2"], reverse=True):
            if all(abs(line["slope"] - k["slope"]) > slope_thresh for k in kept):
                kept.append(line)
            if len(kept) >= 4:
                break
        return kept

    results["uptrends"] = _deduplicate(results["uptrends"])
    results["downtrends"] = _deduplicate(results["downtrends"])
    return results


# ═══════════════════════════════════════════════════════════════
# 单面板模式 (原风格升级版)
# ═══════════════════════════════════════════════════════════════

def plot_kline_single(df, ticker, analysis=None, signals=None, save=True):
    """单面板: K线 + 所有主图因子叠加"""
    df_plot = df.copy()
    df_plot["Date"] = pd.to_datetime(df_plot["Date"])
    df_plot = df_plot.set_index("Date")
    n = len(df_plot)
    close = df_plot["Close"].values
    high = df_plot["High"].values
    low = df_plot["Low"].values

    # 确保有关键因子
    if "ema12" not in df_plot.columns:
        try:
            from src.factors import add_all_factors
            df_plot = df_plot.reset_index()
            df_plot = add_all_factors(df_plot)
            df_plot["Date"] = pd.to_datetime(df_plot["Date"])
            df_plot = df_plot.set_index("Date")
        except Exception:
            pass

    # ── addplot 构建 ──
    apds = []

    # 均线 (MA + EMA)
    ma_specs = [
        ("ma5", "#ec2c2c", 0.8), ("ma10", "#2196f3", 0.8),
        ("ma20", "#9c27b0", 1.2), ("ma60", "#4caf50", 1.5),
        ("ema12", "#00bcd4", 0.6), ("ema26", "#795548", 0.6),
    ]
    for col, color, width in ma_specs:
        if col in df_plot.columns:
            apds.append(mpf.make_addplot(df_plot[col], color=color, width=width, type="line"))

    # 布林带
    for band, color, style in [("bb_upper","#e53935","--"),("bb_lower","#43a047","--"),("bb_mid","#78909c",":")]:
        if band in df_plot.columns:
            apds.append(mpf.make_addplot(df_plot[band], color=color, width=0.6, type="line", linestyle=style))

    # Keltner 通道
    for band, color in [("keltner_upper","#ff6f00"),("keltner_lower","#ff6f00")]:
        if band in df_plot.columns:
            apds.append(mpf.make_addplot(df_plot[band], color=color, width=0.5, type="line", linestyle=":", alpha=0.6))

    # VWAP
    if "vwap" in df_plot.columns:
        apds.append(mpf.make_addplot(df_plot["vwap"], color="#e91e63", width=1.0, type="line"))

    # Donchian
    if "donchian_high" in df_plot.columns:
        apds.append(mpf.make_addplot(df_plot["donchian_high"], color="#ff6f00", width=0.5, type="line", linestyle=":", alpha=0.5))
        apds.append(mpf.make_addplot(df_plot["donchian_low"], color="#ff6f00", width=0.5, type="line", linestyle=":", alpha=0.5))

    # Ichimoku 云 (填充)
    if "ichimoku_span_a" in df_plot.columns and "ichimoku_span_b" in df_plot.columns:
        # 用 addplot 画填充区
        span_a = df_plot["ichimoku_span_a"].values
        span_b = df_plot["ichimoku_span_b"].values
        # 云填充通过 matplotlib 后续处理

    # 信号散点
    if signals and signals.get("active"):
        bull_idx, bull_px, bear_idx, bear_px = [], [], [], []
        for sig in signals["active"][-12:]:
            try:
                sd = pd.to_datetime(sig.get("last_date", ""))
                if sd in df_plot.index:
                    idx = df_plot.index.get_loc(sd)
                    d = sig.get("direction", "neutral")
                    if d == "bullish":
                        bull_idx.append(idx)
                        bull_px.append(df_plot.iloc[idx]["Low"] * 0.97)
                    elif d == "bearish":
                        bear_idx.append(idx)
                        bear_px.append(df_plot.iloc[idx]["High"] * 1.03)
            except Exception:
                pass
        if bull_idx:
            s = pd.Series(np.nan, index=df_plot.index)
            for d, p in zip(bull_idx, bull_px):
                s.iloc[d] = p
            apds.append(mpf.make_addplot(s, type="scatter", marker="^", markersize=90, color=BULL_COLOR, alpha=0.9))
        if bear_idx:
            s = pd.Series(np.nan, index=df_plot.index)
            for d, p in zip(bear_idx, bear_px):
                s.iloc[d] = p
            apds.append(mpf.make_addplot(s, type="scatter", marker="v", markersize=90, color=BEAR_COLOR, alpha=0.9))

    # ── 绘图 ──
    mc = mpf.make_marketcolors(up="#ec2c2c", down="#19a15f", edge="inherit", wick="inherit", volume="inherit")
    style = mpf.make_mpf_style(marketcolors=mc, gridstyle=":", gridcolor="#e8e8e8", facecolor="#fafbfc", figcolor="#ffffff")

    title = ticker
    if analysis:
        title += f" | {analysis.get('overall_trend','')}"
    if signals:
        title += f" | {signals.get('verdict','')}({signals.get('score',0)})"

    fig, axes = mpf.plot(
        df_plot, type="candle", style=style, addplot=apds,
        volume=True, figsize=(22, 12), title="", returnfig=True,
        warn_too_much_data=len(df_plot) + 1,
    )
    ax_main = axes[0]

    # ── 手动叠加: 趋势线 / 斐波那契 / 支撑压力 / Ichimoku云 ──
    trend_lines = _fit_trendlines(df_plot)
    TL_COLORS = ["#e53935","#ff7043","#ff9800","#43a047","#1e88e5","#5e35b1"]

    for i, tl in enumerate(trend_lines.get("uptrends", [])):
        c = TL_COLORS[i % len(TL_COLORS)]
        ax_main.plot(tl["x"], tl["y"], color=c, linestyle="--", linewidth=1.4, alpha=0.8)
        mid = int((tl["x"][0]+tl["x"][-1])/2)
        intercept = tl["y"][0] - tl["slope"]*tl["x"][0]
        ax_main.annotate(f"上升{i+1}", (mid, tl["slope"]*mid+intercept),
                         fontsize=7, color=c, fontproperties=_cn_font,
                         bbox=dict(facecolor="white",alpha=0.5,edgecolor="none",pad=1))

    for i, tl in enumerate(trend_lines.get("downtrends", [])):
        c = TL_COLORS[(i+4)%len(TL_COLORS)]
        ax_main.plot(tl["x"], tl["y"], color=c, linestyle="--", linewidth=1.4, alpha=0.8)
        mid = int((tl["x"][0]+tl["x"][-1])/2)
        intercept = tl["y"][0] - tl["slope"]*tl["x"][0]
        ax_main.annotate(f"下降{i+1}", (mid, tl["slope"]*mid+intercept),
                         fontsize=7, color=c, fontproperties=_cn_font,
                         bbox=dict(facecolor="white",alpha=0.5,edgecolor="none",pad=1))

    # Ichimoku 云填充
    if "ichimoku_span_a" in df_plot.columns and "ichimoku_span_b" in df_plot.columns:
        span_a = df_plot["ichimoku_span_a"].values
        span_b = df_plot["ichimoku_span_b"].values
        x = np.arange(n)
        ax_main.fill_between(x, span_a, span_b, where=span_a>=span_b, alpha=0.1, color="#4caf50")
        ax_main.fill_between(x, span_a, span_b, where=span_a<span_b, alpha=0.1, color="#ef5350")
        ax_main.plot(x, span_a, color="#4caf50", linewidth=0.8, alpha=0.6)
        ax_main.plot(x, span_b, color="#ef5350", linewidth=0.8, alpha=0.6)

    # 斐波那契
    lo_all = low[-min(n,120):].min()
    hi_all = high[-min(n,120):].max()
    curr_p = close[-1]
    for ratio in [0.236, 0.382, 0.5, 0.618, 0.786]:
        level = lo_all + (hi_all-lo_all)*ratio
        if abs(level-curr_p)/curr_p < 0.20:
            ax_main.axhline(y=level, color="#ab47bc", linestyle=":", linewidth=0.7, alpha=0.5)
            ax_main.text(n-1, level, f" Fib{int(ratio*1000)}", fontsize=6.5, color="#ab47bc",
                         va="center", fontproperties=_cn_font, alpha=0.6)

    # 支撑/压力
    def _find_sr(df_plot):
        n = len(df_plot)
        lows = df_plot["Low"].values
        highs = df_plot["High"].values
        atr = float(np.mean(df_plot["High"].iloc[-20:].values - df_plot["Low"].iloc[-20:].values)) if n>=20 else curr_p*0.02
        def cluster(prices):
            if not prices: return []
            s = sorted(prices)
            out = [[s[0]]]
            for p in s[1:]:
                if p - out[-1][-1] < atr*0.4: out[-1].append(p)
                else: out.append([p])
            return [float(np.mean(c)) for c in out]
        sup = [s for s in cluster([lows[i] for i in range(5,n-5) if lows[i]==np.min(lows[max(0,i-5):min(n,i+6)])]) if s < curr_p][-3:]
        res = [r for r in cluster([highs[i] for i in range(5,n-5) if highs[i]==np.max(highs[max(0,i-5):min(n,i+6)])]) if r > curr_p][:3]
        return sup, res

    sup_levels, res_levels = _find_sr(df_plot)
    for j, s in enumerate(sup_levels):
        ax_main.axhline(y=s, color="#4caf50", linestyle="-.", linewidth=1, alpha=0.6)
        ax_main.text(n-1, s, f" 支撑{j+1}", fontsize=7.5, color="#2e7d32", va="bottom", fontproperties=_cn_font, alpha=0.8)
    for j, r in enumerate(res_levels):
        ax_main.axhline(y=r, color="#ef5350", linestyle="-.", linewidth=1, alpha=0.6)
        ax_main.text(n-1, r, f" 压力{j+1}", fontsize=7.5, color="#c62828", va="top", fontproperties=_cn_font, alpha=0.8)

    # ── 信息框 ──
    if signals:
        sc = signals.get("score", 0)
        vd = signals.get("verdict", "N/A")
        bg = "#1a1a2e" if abs(sc)>=2 else "#2d2d2d"
        ax_main.text(0.99, 0.97, f"评分:{sc}\n{vd}", transform=ax_main.transAxes,
                     fontsize=9, fontproperties=_cn_font, ha="right", va="top",
                     bbox=dict(boxstyle="round,pad=0.4", facecolor=bg, alpha=0.85, edgecolor="#0984e3", linewidth=1),
                     color="#ffffff")

    curr = df_plot["Close"].iloc[-1]
    prev = df_plot["Close"].iloc[-2] if n>=2 else curr
    chg = (curr-prev)/prev*100 if prev else 0
    ax_main.text(0.99, 0.01, f"{curr:.2f}\n{chg:+.2f}%", transform=ax_main.transAxes,
                 fontsize=8, fontproperties=_cn_font, ha="right", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white",alpha=0.8,edgecolor="#ddd"))

    ax_main.set_title(title, fontsize=14, fontweight="bold", fontproperties=_cn_font, pad=12)

    # 图例
    from matplotlib.lines import Line2D
    items = [Line2D([0],[0],color=c,linewidth=1.5) for c in ["#ec2c2c","#2196f3","#9c27b0","#4caf50","#00bcd4","#e91e63"]]
    labels = ["MA5","MA10","MA20","MA60","EMA12","VWAP"]
    ax_main.legend(items, labels, loc="upper left", fontsize=7.5, ncol=3, prop=_cn_font, framealpha=0.9)

    fpath = CHART_DIR / f"{ticker}_single.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(fpath)


# ═══════════════════════════════════════════════════════════════
# 多面板模式 (4面板)
# ═══════════════════════════════════════════════════════════════

def plot_kline_multi(df, ticker, analysis=None, signals=None, save=True):
    """
    多面板图表:
      Panel 0: K线 + 趋势/波动率因子 (主图)
      Panel 1: 动量因子 (RSI / Stochastic / MACD)
      Panel 2: 方向因子 (ADX / +DI / -DI / 趋势强度)
      Panel 3: 成交量 + OBV
    """
    df_plot = df.copy()
    df_plot["Date"] = pd.to_datetime(df_plot["Date"])
    df_plot = df_plot.set_index("Date")
    n = len(df_plot)

    # 确保因子已计算
    if "rsi_14" not in df_plot.columns:
        try:
            from src.factors import add_all_factors
            df_plot = df_plot.reset_index()
            df_plot = add_all_factors(df_plot)
            df_plot["Date"] = pd.to_datetime(df_plot["Date"])
            df_plot = df_plot.set_index("Date")
        except Exception:
            pass

    # ── 构建 matplotlib 多面板 ──
    fig = plt.figure(figsize=(22, 16), facecolor="#fafbfc")
    # 4 行: 主图(6) / 动量(2) / 方向(2) / 成交量(2)  [高度比例]
    gs = fig.add_gridspec(4, 1, height_ratios=[6, 2, 2, 2], hspace=0.08)
    ax_main = fig.add_subplot(gs[0])
    ax_momentum = fig.add_subplot(gs[1], sharex=ax_main)
    ax_dir = fig.add_subplot(gs[2], sharex=ax_main)
    ax_vol = fig.add_subplot(gs[3], sharex=ax_main)

    # ── Panel 0: K线 + 主图因子 ──
    _plot_main_panel(ax_main, df_plot, ticker, analysis, signals)

    # ── Panel 1: 动量因子 ──
    _plot_momentum_panel(ax_momentum, df_plot)

    # ── Panel 2: 方向因子 ──
    _plot_directional_panel(ax_dir, df_plot)

    # ── Panel 3: 成交量 ──
    _plot_volume_panel(ax_vol, df_plot)

    # 隐藏多余x轴标签
    for ax in [ax_main, ax_momentum, ax_dir]:
        plt.setp(ax.get_xticklabels(), visible=False)

    fpath = CHART_DIR / f"{ticker}_multi.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(fpath)


def _plot_main_panel(ax, df_plot, ticker, analysis, signals):
    """主面板: K线 + 均线 + 布林带 + 趋势线 + 支撑压力"""
    n = len(df_plot)
    x = np.arange(n)
    close = df_plot["Close"].values

    # K线 (手动画, 不用mplfinance)
    for i in range(n):
        o, c = df_plot["Open"].iloc[i], df_plot["Close"].iloc[i]
        h, l = df_plot["High"].iloc[i], df_plot["Low"].iloc[i]
        color = BULL_COLOR if c >= o else BEAR_COLOR
        # 影线
        ax.plot([x[i], x[i]], [l, h], color=color, linewidth=0.6, alpha=0.8)
        # 实体
        body_h = abs(c - o)
        body_b = min(o, c)
        ax.add_patch(plt.Rectangle((x[i]-0.3, body_b), 0.6, body_h,
                                    color=color, alpha=0.9, ec=color, linewidth=0.5))

    # 均线
    for col, color, width in [
        ("ma5","#ec2c2c",0.8),("ma10","#2196f3",0.8),
        ("ma20","#9c27b0",1.2),("ma60","#4caf50",1.5),
        ("ema12","#00bcd4",0.6),("ema26","#795548",0.6),
    ]:
        if col in df_plot.columns:
            ax.plot(x, df_plot[col].values, color=color, linewidth=width, alpha=0.85)

    # 布林带填充
    if "bb_upper" in df_plot.columns and "bb_lower" in df_plot.columns:
        ax.fill_between(x, df_plot["bb_lower"].values, df_plot["bb_upper"].values,
                        alpha=0.04, color="#9c27b0")
        ax.plot(x, df_plot["bb_upper"].values, color="#e53935", linewidth=0.5, alpha=0.5)
        ax.plot(x, df_plot["bb_lower"].values, color="#43a047", linewidth=0.5, alpha=0.5)

    # Keltner 通道
    if "keltner_upper" in df_plot.columns:
        ax.plot(x, df_plot["keltner_upper"].values, color="#ff6f00", linewidth=0.5, alpha=0.4, linestyle=":")
        ax.plot(x, df_plot["keltner_lower"].values, color="#ff6f00", linewidth=0.5, alpha=0.4, linestyle=":")

    # VWAP
    if "vwap" in df_plot.columns:
        ax.plot(x, df_plot["vwap"].values, color="#e91e63", linewidth=1.0, alpha=0.8)

    # Ichimoku 云
    if "ichimoku_span_a" in df_plot.columns and "ichimoku_span_b" in df_plot.columns:
        sa = df_plot["ichimoku_span_a"].values
        sb = df_plot["ichimoku_span_b"].values
        ax.fill_between(x, sa, sb, where=sa>=sb, alpha=0.08, color="#4caf50")
        ax.fill_between(x, sa, sb, where=sa<sb, alpha=0.08, color="#ef5350")
        ax.plot(x, sa, color="#4caf50", linewidth=0.7, alpha=0.5)
        ax.plot(x, sb, color="#ef5350", linewidth=0.7, alpha=0.5)

    # 趋势线
    trend_lines = _fit_trendlines(df_plot)
    TL_COLORS = ["#e53935","#ff7043","#ff9800","#43a047"]
    for i, tl in enumerate(trend_lines.get("uptrends",[])[:3]):
        c = TL_COLORS[i % len(TL_COLORS)]
        ax.plot(tl["x"], tl["y"], color=c, linestyle="--", linewidth=1.3, alpha=0.8)
    for i, tl in enumerate(trend_lines.get("downtrends",[])[:3]):
        c = TL_COLORS[(i+2)%len(TL_COLORS)]
        ax.plot(tl["x"], tl["y"], color=c, linestyle="--", linewidth=1.3, alpha=0.8)

    # 斐波那契
    lo = df_plot["Low"].values[-min(n,120):].min()
    hi = df_plot["High"].values[-min(n,120):].max()
    curr_p = close[-1]
    for ratio, label in [(0.236,"Fib23.6"),(0.382,"Fib38.2"),(0.5,"Fib50"),(0.618,"Fib61.8"),(0.786,"Fib78.6")]:
        level = lo + (hi-lo)*ratio
        if abs(level-curr_p)/curr_p < 0.18:
            ax.axhline(y=level, color="#ab47bc", linestyle=":", linewidth=0.6, alpha=0.4)
            ax.text(n-1, level, f" {label}", fontsize=6, color="#ab47bc", va="center", fontproperties=_cn_font, alpha=0.5)

    # 支撑/压力
    atr = float(np.mean(df_plot["High"].iloc[-20:].values - df_plot["Low"].iloc[-20:].values)) if n>=20 else curr_p*0.02
    def cluster(prices):
        if not prices: return []
        s = sorted(prices)
        out = [[s[0]]]
        for p in s[1:]:
            if p - out[-1][-1] < atr*0.4: out[-1].append(p)
            else: out.append([p])
        return [float(np.mean(c)) for c in out]
    sup = []
    res = []
    w = 5
    lows = df_plot["Low"].values
    highs = df_plot["High"].values
    for i in range(w, n-w):
        if lows[i] == np.min(lows[i-w:i+w+1]): sup.append(lows[i])
        if highs[i] == np.max(highs[i-w:i+w+1]): res.append(highs[i])
    sup_levels = sorted([s for s in cluster(sup) if s < curr_p])[-3:]
    res_levels = sorted([r for r in cluster(res) if r > curr_p])[:3]
    for s in sup_levels:
        ax.axhline(y=s, color="#4caf50", linestyle="-.", linewidth=0.8, alpha=0.5)
    for r in res_levels:
        ax.axhline(y=r, color="#ef5350", linestyle="-.", linewidth=0.8, alpha=0.5)

    # 信号标记
    if signals and signals.get("active"):
        for sig in signals["active"][-10:]:
            try:
                sd = pd.to_datetime(sig.get("last_date",""))
                if sd in df_plot.index:
                    idx = df_plot.index.get_loc(sd)
                    d = sig.get("direction","neutral")
                    if d == "bullish":
                        ax.annotate("▲", (idx, df_plot.iloc[idx]["Low"]*0.97),
                                    fontsize=10, color=BULL_COLOR, fontweight="bold",
                                    ha="center", va="center")
                    elif d == "bearish":
                        ax.annotate("▼", (idx, df_plot.iloc[idx]["High"]*1.03),
                                    fontsize=10, color=BEAR_COLOR, fontweight="bold",
                                    ha="center", va="center")
            except Exception:
                pass

    # 标题 & 信息
    title = ticker
    if analysis:
        title += f"  |  {analysis.get('overall_trend','')}"
    if signals:
        title += f"  |  {signals.get('verdict','')} (评分:{signals.get('score',0)})"
    ax.set_title(title, fontsize=13, fontweight="bold", fontproperties=_cn_font, pad=10, loc="left")

    # 评分框
    if signals:
        sc = signals.get("score",0)
        bg = "#1a1a2e" if abs(sc)>=2 else "#2d2d2d"
        ax.text(0.99, 0.97, f"{sc}", transform=ax.transAxes,
                fontsize=14, fontproperties=_cn_font, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3",facecolor=bg,alpha=0.9,edgecolor="#0984e3"),
                color="#ffffff")

    ax.set_ylabel("价格", fontproperties=_cn_font, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.3, color="#ddd")


def _plot_momentum_panel(ax, df_plot):
    """动量面板: RSI + Stochastic + MACD直方图"""
    n = len(df_plot)
    x = np.arange(n)

    # RSI
    if "rsi_14" in df_plot.columns:
        ax.plot(x, df_plot["rsi_14"].values, color="#e91e63", linewidth=1.0, label="RSI14")
    if "rsi_28" in df_plot.columns:
        ax.plot(x, df_plot["rsi_28"].values, color="#9c27b0", linewidth=0.8, alpha=0.6, label="RSI28")

    # RSI 超买/超卖线
    ax.axhline(y=70, color="#ef5350", linewidth=0.5, alpha=0.5, linestyle="--")
    ax.axhline(y=30, color="#4caf50", linewidth=0.5, alpha=0.5, linestyle="--")
    ax.axhline(y=50, color="#9e9e9e", linewidth=0.5, alpha=0.3, linestyle="-.")

    # Stochastic
    if "stochastic_k" in df_plot.columns:
        ax.plot(x, df_plot["stochastic_k"].values, color="#ff9800", linewidth=0.7, alpha=0.7, label="K")
    if "stochastic_d" in df_plot.columns:
        ax.plot(x, df_plot["stochastic_d"].values, color="#2196f3", linewidth=0.7, alpha=0.7, label="D")

    ax.set_ylabel("动量", fontproperties=_cn_font, fontsize=8)
    ax.set_ylim(-10, 110)
    ax.grid(True, linestyle=":", alpha=0.3)
    from matplotlib.lines import Line2D
    ax.legend([Line2D([0],[0],color="#e91e63",lw=1), Line2D([0],[0],color="#ff9800",lw=0.7)],
              ["RSI14","K"], loc="upper left", fontsize=7, prop=_cn_font, framealpha=0.8, ncol=3)


def _plot_directional_panel(ax, df_plot):
    """方向面板: ADX + +DI/-DI + 趋势强度"""
    n = len(df_plot)
    x = np.arange(n)

    # +DI / -DI
    if "di_plus" in df_plot.columns:
        ax.plot(x, df_plot["di_plus"].values, color="#4caf50", linewidth=1.0, label="+DI")
    if "di_minus" in df_plot.columns:
        ax.plot(x, df_plot["di_minus"].values, color="#ef5350", linewidth=1.0, label="-DI")

    # ADX (右轴)
    ax2 = ax.twinx()
    if "adx_14" in df_plot.columns:
        ax2.plot(x, df_plot["adx_14"].values, color="#ff5722", linewidth=1.0, label="ADX")
    ax2.axhline(y=25, color="#ff5722", linewidth=0.5, alpha=0.5, linestyle="--")
    ax2.set_ylim(0, 80)
    ax2.set_ylabel("ADX", fontsize=8)

    ax.set_ylabel("DI", fontproperties=_cn_font, fontsize=8)
    ax.set_ylim(-5, 60)
    ax.grid(True, linestyle=":", alpha=0.3)
    from matplotlib.lines import Line2D
    ax.legend([Line2D([0],[0],color="#4caf50",lw=1), Line2D([0],[0],color="#ef5350",lw=1)],
              ["+DI","-DI"], loc="upper left", fontsize=7, prop=_cn_font, framealpha=0.8, ncol=2)


def _plot_volume_panel(ax, df_plot):
    """成交量面板: 成交量柱状图 + OBV"""
    n = len(df_plot)
    x = np.arange(n)

    # 成交量柱状图
    vol = df_plot["Volume"].values
    close = df_plot["Close"].values
    open_ = df_plot["Open"].values
    colors = [BULL_COLOR if c >= o else BEAR_COLOR for c, o in zip(close, open_)]
    ax.bar(x, vol, color=colors, alpha=0.5, width=0.7)

    # 成交量MA
    if "volume_ma20" in df_plot.columns:
        ax.plot(x, df_plot["volume_ma20"].values, color="#ff9800", linewidth=1.0, alpha=0.8, label="VOL MA20")

    # OBV (右轴)
    if "obv" in df_plot.columns:
        ax2 = ax.twinx()
        obv = df_plot["obv"].values
        obv_norm = (obv - obv.min()) / (obv.max() - obv.min()) * vol.max() * 0.8 if obv.max()!=obv.min() else obv
        ax2.plot(x, obv_norm, color="#9c27b0", linewidth=0.8, alpha=0.7, label="OBV")
        ax2.set_ylabel("OBV(归一)", fontsize=8)
        ax2.set_ylim(0, vol.max()*1.2)

    ax.set_ylabel("成交量", fontproperties=_cn_font, fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.set_xlim(-1, n)


# ═══════════════════════════════════════════════════════════════
# 统一入口
# ═══════════════════════════════════════════════════════════════

def plot_kline_trend(df, ticker, analysis=None, signals=None, save=True, mode="multi"):
    """
    统一绘图入口
    mode:
      "single" -> 单面板 (原风格, 多线叠加)
      "multi"  -> 多面板 (4面板, 因子分类显示)
    """
    if mode == "multi":
        return plot_kline_multi(df, ticker, analysis, signals, save)
    else:
        return plot_kline_single(df, ticker, analysis, signals, save)
