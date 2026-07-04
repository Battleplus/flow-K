"""图表模块 - mplfinance K线 + 全量趋势线/曲线/信号标注
新增: VWAP, Donchian通道, 线性回归通道, EMA, 更多趋势线
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import mplfinance as mpf

warnings.filterwarnings("ignore", message="Glyph.*missing from font")

# 强制注册微软雅黑字体
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

MA_COLORS = {"ma5": "#ff9800", "ma10": "#2196f3", "ma20": "#9c27b0", "ma60": "#4caf50"}
MA_LINEWIDTH = {"ma5": 0.8, "ma10": 0.8, "ma20": 1.2, "ma60": 1.5}
BULL_COLOR = "#ec2c2c"
BEAR_COLOR = "#19a15f"
TL_COLORS = ["#e53935", "#ff7043", "#ff9800", "#fdd835", "#43a047", "#1e88e5", "#5e35b1", "#8e24aa"]


# ═══════════════════════════════════════════════════════════════
# 趋势线拟合 (多窗口, 低阈值)
# ═══════════════════════════════════════════════════════════════

def _fit_trendline_pivots_multi(df_plot, windows=(3, 5, 8, 13)):
    """多窗口拟合多条趋势线, R² 阈值降低到 0.35"""
    n = len(df_plot)
    lows = df_plot["Low"].values
    highs = df_plot["High"].values
    results = {"uptrends": [], "downtrends": []}
    from scipy import stats as sp_stats

    for w in windows:
        if n < w * 4:
            continue
        # Pivot Lows
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
                    results["uptrends"].append({
                        "x": x_line, "y": y_line,
                        "slope": slope, "r2": r2,
                        "pivots": recent, "window": w,
                    })
                    break

        # Pivot Highs
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
                    results["downtrends"].append({
                        "x": x_line, "y": y_line,
                        "slope": slope, "r2": r2,
                        "pivots": recent, "window": w,
                    })
                    break

    results["uptrends"] = _deduplicate(results["uptrends"])
    results["downtrends"] = _deduplicate(results["downtrends"])
    return results


def _deduplicate(lines: list, slope_thresh=0.00005) -> list:
    if not lines:
        return []
    kept = []
    for line in sorted(lines, key=lambda x: x["r2"], reverse=True):
        if all(abs(line["slope"] - k["slope"]) > slope_thresh for k in kept):
            kept.append(line)
        if len(kept) >= 4:
            break
    return kept


# ═══════════════════════════════════════════════════════════════
# 支撑压力位 (多级别)
# ═══════════════════════════════════════════════════════════════

def _find_support_resistance(df_plot):
    """返回 (support_levels, resistance_levels) 各3个"""
    n = len(df_plot)
    lows = df_plot["Low"].values
    highs = df_plot["High"].values
    curr = df_plot["Close"].iloc[-1]
    atr = float(np.mean(df_plot["High"].iloc[-20:].values - df_plot["Low"].iloc[-20:].values)) if n >= 20 else curr * 0.02

    def cluster(prices):
        if not prices:
            return []
        s = sorted(prices)
        out = [[s[0]]]
        for p in s[1:]:
            if p - out[-1][-1] < atr * 0.4:
                out[-1].append(p)
            else:
                out.append([p])
        return [float(np.mean(c)) for c in out]

    # 局部低点聚类 → 支撑
    sup_candidates = []
    w = 5
    for i in range(w, n - w):
        if lows[i] == np.min(lows[i - w : i + w + 1]):
            sup_candidates.append(lows[i])
    sup_levels = [s for s in cluster(sup_candidates) if s < curr][-3:]

    # 局部高点聚类 → 压力
    res_candidates = []
    for i in range(w, n - w):
        if highs[i] == np.max(highs[i - w : i + w + 1]):
            res_candidates.append(highs[i])
    res_levels = [r for r in cluster(res_candidates) if r > curr][:3]

    return sup_levels, res_levels


# ═══════════════════════════════════════════════════════════════
# 主绘图函数
# ═══════════════════════════════════════════════════════════════

def plot_kline_trend(df, ticker, analysis=None, signals=None, save=True):
    """绘制 K 线 + 全量线条 + 信号"""

    df_plot = df.copy()
    df_plot["Date"] = pd.to_datetime(df_plot["Date"])
    df_plot = df_plot.set_index("Date")
    n = len(df_plot)

    # ── 计算额外指标 (如果还没有) ──
    close = df_plot["Close"].values
    high = df_plot["High"].values
    low = df_plot["Low"].values
    volume = df_plot["Volume"].values

    # VWAP
    if "vwap" not in df_plot.columns:
        cum_vol = np.cumsum(volume)
        cum_vol_price = np.cumsum(volume * close)
        df_plot["vwap"] = cum_vol_price / np.where(cum_vol == 0, 1, cum_vol)

    # EMA12 / EMA26
    if "ema12" not in df_plot.columns:
        df_plot["ema12"] = pd.Series(close).ewm(span=12, adjust=False).mean().values
    if "ema26" not in df_plot.columns:
        df_plot["ema26"] = pd.Series(close).ewm(span=26, adjust=False).mean().values

    # Donchian 通道 (20日)
    df_plot["donchian_high"] = pd.Series(high).rolling(20, min_periods=1).max().values
    df_plot["donchian_low"] = pd.Series(low).rolling(20, min_periods=1).min().values

    # 线性回归通道
    x_arr = np.arange(n)
    slope, intercept = np.polyfit(x_arr, close, 1)
    reg_line = slope * x_arr + intercept
    residuals = close - reg_line
    std_res = float(np.std(residuals))
    df_plot["lr_mid"] = reg_line
    df_plot["lr_upper1"] = reg_line + std_res
    df_plot["lr_lower1"] = reg_line - std_res
    df_plot["lr_upper2"] = reg_line + 2 * std_res
    df_plot["lr_lower2"] = reg_line - 2 * std_res

    # ── 趋势线 ──
    trend_lines = _fit_trendline_pivots_multi(df_plot, windows=(3, 5, 8, 13))

    # ── addplot 构建 ──
    apds = []

    # 均线
    for col in ["ma5", "ma10", "ma20", "ma60"]:
        if col in df_plot.columns:
            apds.append(mpf.make_addplot(df_plot[col], color=MA_COLORS[col],
                         width=MA_LINEWIDTH[col], type="line"))

    # EMA
    if "ema12" in df_plot.columns:
        apds.append(mpf.make_addplot(df_plot["ema12"], color="#00bcd4",
                     width=0.7, type="line", linestyle="-.", alpha=0.8))
    if "ema26" in df_plot.columns:
        apds.append(mpf.make_addplot(df_plot["ema26"], color="#795548",
                     width=0.7, type="line", linestyle="-.", alpha=0.8))

    # 布林带
    for band, color, width, style in [
        ("bb_upper", "#e53935", 0.6, "--"),
        ("bb_lower", "#43a047", 0.6, "--"),
        ("bb_mid", "#78909c", 0.5, ":"),
    ]:
        if band in df_plot.columns:
            apds.append(mpf.make_addplot(df_plot[band], color=color,
                         width=width, type="line", linestyle=style))

    # VWAP
    apds.append(mpf.make_addplot(df_plot["vwap"], color="#e91e63",
                 width=1.0, type="line", linestyle="-"))

    # Donchian 通道
    apds.append(mpf.make_addplot(df_plot["donchian_high"], color="#ff6f00",
                 width=0.5, type="line", linestyle=":", alpha=0.6))
    apds.append(mpf.make_addplot(df_plot["donchian_low"], color="#ff6f00",
                 width=0.5, type="line", linestyle=":", alpha=0.6))

    # 线性回归通道
    for band, color, alpha in [
        ("lr_mid", "#9c27b0", 0.7),
        ("lr_upper1", "#ce93d8", 0.4),
        ("lr_lower1", "#ce93d8", 0.4),
        ("lr_upper2", "#e1bee7", 0.25),
        ("lr_lower2", "#e1bee7", 0.25),
    ]:
        style = "-" if band == "lr_mid" else "--"
        apds.append(mpf.make_addplot(df_plot[band], color=color,
                     width=0.7 if band == "lr_mid" else 0.5,
                     type="line", linestyle=style, alpha=alpha))

    # 信号散点标记
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
            apds.append(mpf.make_addplot(s, type="scatter",
                         marker="^", markersize=90, color=BULL_COLOR, alpha=0.9))
        if bear_idx:
            s = pd.Series(np.nan, index=df_plot.index)
            for d, p in zip(bear_idx, bear_px):
                s.iloc[d] = p
            apds.append(mpf.make_addplot(s, type="scatter",
                         marker="v", markersize=90, color=BEAR_COLOR, alpha=0.9))

    # ── mplfinance 绘图 ──
    mc = mpf.make_marketcolors(up="#ec2c2c", down="#19a15f",
                               edge="inherit", wick="inherit", volume="inherit")
    style = mpf.make_mpf_style(marketcolors=mc, gridstyle=":",
                               gridcolor="#e8e8e8", facecolor="#fafbfc", figcolor="#ffffff")

    title = ticker
    if analysis:
        title += f" | 趋势:{analysis.get('overall_trend','')}"
    if signals:
        title += f" | {signals.get('verdict','')}({signals.get('score',0)})"

    fig, axes = mpf.plot(
        df_plot, type="candle", style=style, addplot=apds,
        volume=True, figsize=(22, 12), title="", returnfig=True,
        warn_too_much_data=len(df_plot) + 1,
    )
    ax_main = axes[0]
    ax_main.set_title(title, fontsize=14, fontweight="bold",
                      fontproperties=_cn_font, pad=12)

    # ── 在 matplotlib 轴上画额外线条 ──

    # 多条趋势线
    for i, tl in enumerate(trend_lines.get("uptrends", [])):
        c = TL_COLORS[i % len(TL_COLORS)]
        ax_main.plot(tl["x"], tl["y"], color=c, linestyle="--",
                     linewidth=1.4, alpha=0.8)
        mid = int((tl["x"][0] + tl["x"][-1]) / 2)
        intercept = tl["y"][0] - tl["slope"] * tl["x"][0]
        ax_main.annotate(f"上升{i+1} R²={tl['r2']:.2f}",
                         (mid, tl["slope"] * mid + intercept),
                         fontsize=7, color=c, fontproperties=_cn_font,
                         bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=1))

    for i, tl in enumerate(trend_lines.get("downtrends", [])):
        c = TL_COLORS[(i + 4) % len(TL_COLORS)]
        ax_main.plot(tl["x"], tl["y"], color=c, linestyle="--",
                     linewidth=1.4, alpha=0.8)
        mid = int((tl["x"][0] + tl["x"][-1]) / 2)
        intercept = tl["y"][0] - tl["slope"] * tl["x"][0]
        ax_main.annotate(f"下降{i+1} R²={tl['r2']:.2f}",
                         (mid, tl["slope"] * mid + intercept),
                         fontsize=7, color=c, fontproperties=_cn_font,
                         bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=1))

    # 斐波那契回撤 (全趋势都画)
    lo_all = low[-min(n, 120):].min()
    hi_all = high[-min(n, 120):].max()
    diff = hi_all - lo_all
    curr_p = close[-1]
    for ratio in [0.236, 0.382, 0.5, 0.618, 0.786]:
        level = lo_all + diff * ratio
        if abs(level - curr_p) / curr_p < 0.20:
            ax_main.axhline(y=level, color="#ab47bc", linestyle=":",
                            linewidth=0.7, alpha=0.5)
            ax_main.text(n - 1, level, f" Fib{int(ratio*1000)}",
                         fontsize=6.5, color="#ab47bc", va="center",
                         fontproperties=_cn_font, alpha=0.6)

    # 多级别支撑/压力
    sup_levels, res_levels = _find_support_resistance(df_plot)
    for j, s in enumerate(sup_levels):
        ax_main.axhline(y=s, color="#4caf50", linestyle="-.", linewidth=1, alpha=0.6)
        ax_main.text(n - 1, s, f" 支撑{j+1}", fontsize=7.5, color="#2e7d32",
                     va="bottom", fontproperties=_cn_font, alpha=0.8)
    for j, r in enumerate(res_levels):
        ax_main.axhline(y=r, color="#ef5350", linestyle="-.", linewidth=1, alpha=0.6)
        ax_main.text(n - 1, r, f" 压力{j+1}", fontsize=7.5, color="#c62828",
                     va="top", fontproperties=_cn_font, alpha=0.8)

    # 来自 analysis 的关键支撑压力
    if analysis:
        sr = analysis.get("support_resistance", {})
        if sr.get("support"):
            ax_main.axhline(y=sr["support"], color="#1b5e20", linestyle="--",
                            linewidth=1.2, alpha=0.7)
            ax_main.text(n - 1, sr["support"], f" 关键支撑", fontsize=8,
                         color="#1b5e20", va="bottom", fontproperties=_cn_font)
        if sr.get("resistance"):
            ax_main.axhline(y=sr["resistance"], color="#b71c1c", linestyle="--",
                            linewidth=1.2, alpha=0.7)
            ax_main.text(n - 1, sr["resistance"], f" 关键压力", fontsize=8,
                         color="#b71c1c", va="top", fontproperties=_cn_font)

    # 金叉死叉标注
    if analysis:
        for c in (analysis.get("crosses", []) or [])[-8:]:
            try:
                cd = pd.to_datetime(c["date"])
                if cd in df_plot.index:
                    idx = df_plot.index.get_loc(cd)
                    is_gold = c["type"] == "金叉"
                    lbl = "▲金叉" if is_gold else "▼死叉"
                    col = BULL_COLOR if is_gold else BEAR_COLOR
                    yp = df_plot.iloc[idx]["Low"] * 0.955 if is_gold else df_plot.iloc[idx]["High"] * 1.045
                    ax_main.annotate(lbl, (idx, yp), fontsize=8, color=col,
                                     fontweight="bold", ha="center", va="center",
                                     fontproperties=_cn_font,
                                     bbox=dict(facecolor="white", alpha=0.7,
                                               edgecolor=col, pad=2))
            except Exception:
                pass

    # ── 图例 ──
    legend_items = []
    legend_labels = []
    for col, lbl in [("ma5","MA5"),("ma10","MA10"),("ma20","MA20"),("ma60","MA60")]:
        if col in df_plot.columns:
            from matplotlib.lines import Line2D
            legend_items.append(Line2D([0],[0], color=MA_COLORS[col], linewidth=1.5))
            legend_labels.append(lbl)
    for lbl, color, style in [("EMA12","#00bcd4","-"),("EMA26","#795548","-"),
                               ("VWAP","#e91e63","-"),("BB","#e53935","--"),
                               ("Donchian","#ff6f00",":"),("LR通道","#9c27b0","-")]:
        legend_items.append(Line2D([0],[0], color=color, linewidth=1.2, linestyle=style))
        legend_labels.append(lbl)

    if legend_items:
        ax_main.legend(legend_items, legend_labels, loc="upper left",
                       fontsize=7.5, ncol=3, prop=_cn_font, framealpha=0.9)

    # ── 信息框 ──
    # 左上: 斜率
    if analysis and analysis.get("slopes"):
        slopes = analysis["slopes"]
        txt = "均线斜率:\n"
        for name, info in slopes.items():
            sym = "↑" if info["direction"] == "上升" else "↓" if info["direction"] == "下降" else "→"
            txt += f"  {name}: {info['slope']:.3f}% {sym}\n"
        ax_main.text(0.01, 0.97, txt.strip(), transform=ax_main.transAxes,
                     fontsize=7.5, fontproperties=_cn_font,
                     verticalalignment="top",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                               alpha=0.85, edgecolor="#ccc"))

    # 右上: 信号评分
    if signals:
        sc = signals.get("score", 0)
        vd = signals.get("verdict", "N/A")
        bg = "#1a1a2e" if abs(sc) >= 2 else "#2d2d2d"
        txt = f"评分: {sc}\n{vd}\n↑{signals.get('bullish_count',0)} ↓{signals.get('bearish_count',0)}"
        ax_main.text(0.99, 0.97, txt.strip(), transform=ax_main.transAxes,
                     fontsize=9, fontproperties=_cn_font,
                     verticalalignment="top", horizontalalignment="right",
                     bbox=dict(boxstyle="round,pad=0.4", facecolor=bg, alpha=0.85,
                               edgecolor="#0984e3", linewidth=1),
                     color="#ffffff")

    # 右下: 价格信息
    curr = df_plot["Close"].iloc[-1]
    prev = df_plot["Close"].iloc[-2] if n >= 2 else curr
    chg = (curr - prev) / prev * 100 if prev else 0
    vol_m = df_plot["Volume"].iloc[-1] / 1e6
    txt = f"收盘: {curr:.2f}\n涨跌: {chg:+.2f}%\n成交量: {vol_m:.1f}M"
    ax_main.text(0.99, 0.01, txt.strip(), transform=ax_main.transAxes,
                 fontsize=8, fontproperties=_cn_font,
                 verticalalignment="bottom", horizontalalignment="right",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                           alpha=0.8, edgecolor="#ddd"))

    # ── 保存 ──
    fpath = CHART_DIR / f"{ticker}_kline_trend.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(fpath)
