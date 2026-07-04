"""技术指标模块 - 均线、斜率、趋势、交叉、支撑压力"""

import pandas as pd
import numpy as np
from scipy import stats as sp_stats


def add_ma(df: pd.DataFrame, periods: list[int] = None) -> pd.DataFrame:
    """添加多周期均线"""
    if periods is None:
        periods = [5, 10, 20, 60]
    df = df.copy()
    for p in periods:
        df[f"ma{p}"] = df["Close"].rolling(p).mean()
    return df


def calc_slope(series: pd.Series, lookback: int = 5) -> float:
    """计算最近 lookback 日的线性回归斜率（角度制）"""
    s = series.dropna().tail(lookback)
    if len(s) < lookback:
        return 0.0
    x = np.arange(len(s))
    slope, _, _, _, _ = sp_stats.linregress(x, s.values)
    # 归一化：斜率 / 均值 → 变化率
    mean_val = s.mean()
    if mean_val == 0:
        return 0.0
    return slope / mean_val


def trend_direction(slope: float, threshold: float = 0.001) -> str:
    """根据斜率判断方向"""
    if slope > threshold:
        return "上升"
    elif slope < -threshold:
        return "下降"
    return "走平"


def slope_summary(df: pd.DataFrame, mas: list[int] = None) -> dict:
    """计算所有均线的当前斜率及方向"""
    if mas is None:
        mas = [5, 10, 20, 60]
    result = {}
    for p in mas:
        col = f"ma{p}"
        if col in df.columns:
            s = calc_slope(df[col], lookback=5)
            result[f"MA{p}"] = {
                "slope": round(s * 100, 4),  # 百分比
                "direction": trend_direction(s),
                "value": round(df[col].iloc[-1], 2),
            }
    return result


def ma_alignment(df: pd.DataFrame) -> dict:
    """判断均线排列：多头/空头/交织"""
    import re
    latest = df.iloc[-1]
    mas = {}
    for col in df.columns:
        if col.startswith("ma"):
            m = re.match(r"^ma(\d+)$", col)
            if m:
                p = int(m.group(1))
                mas[p] = latest[col]

    if not mas:
        return {"alignment": "无数据", "detail": ""}

    sorted_mas = sorted(mas.items())
    values = [v for _, v in sorted_mas]
    periods = [p for p, _ in sorted_mas]

    # 多头排列：短均 > 长均
    bullish = all(values[i] > values[i + 1] for i in range(len(values) - 1))
    # 空头排列：短均 < 长均
    bearish = all(values[i] < values[i + 1] for i in range(len(values) - 1))

    if bullish:
        return {"alignment": "多头排列", "detail": " > ".join(f"MA{p}" for p in periods)}
    elif bearish:
        return {"alignment": "空头排列", "detail": " < ".join(f"MA{p}" for p in periods)}
    else:
        return {"alignment": "交织震荡", "detail": "均线无序排列"}


def find_cross(df: pd.DataFrame, ma1: int = 5, ma2: int = 20, recent: int = 20) -> list[dict]:
    """找最近 N 天内的均线交叉点"""
    col1 = f"ma{ma1}"
    col2 = f"ma{ma2}"
    if col1 not in df.columns or col2 not in df.columns:
        return []

    diff = df[col1] - df[col2]
    crosses = []
    for i in range(1, min(len(df), recent + 5)):
        idx = len(df) - i
        if idx < 1:
            break
        prev = diff.iloc[idx - 1]
        curr = diff.iloc[idx]
        if (prev < 0 and curr > 0) or (prev > 0 and curr < 0):
            crosses.append({
                "date": str(df["Date"].iloc[idx]),
                "type": "金叉" if curr > 0 else "死叉",
                "ma1": f"MA{ma1}",
                "ma2": f"MA{ma2}",
            })
    return crosses


def find_support_resistance(df: pd.DataFrame, window: int = 20) -> dict:
    """找近期支撑位和压力位"""
    recent = df.tail(window)
    support = recent["Low"].min()
    resistance = recent["High"].max()
    current = recent["Close"].iloc[-1]
    return {
        "support": round(support, 2),
        "resistance": round(resistance, 2),
        "current": round(current, 2),
        "pct_to_support": round((current - support) / support * 100, 2),
        "pct_to_resistance": round((resistance - current) / current * 100, 2),
    }


def trend_analysis(df: pd.DataFrame) -> dict:
    """综合趋势分析"""
    df = add_ma(df)
    slopes = slope_summary(df)
    alignment = ma_alignment(df)
    crosses = find_cross(df, 5, 20, recent=30)
    sr = find_support_resistance(df)

    # 整体趋势判断
    ma20_slope = slopes.get("MA20", {}).get("slope", 0)
    if ma20_slope > 0.5:
        overall = "上升趋势"
    elif ma20_slope < -0.5:
        overall = "下降趋势"
    else:
        overall = "横盘震荡"

    return {
        "overall_trend": overall,
        "ma_alignment": alignment,
        "slopes": slopes,
        "crosses": crosses,
        "support_resistance": sr,
        "latest_close": round(df["Close"].iloc[-1], 2),
        "latest_date": str(df["Date"].iloc[-1]),
    }
