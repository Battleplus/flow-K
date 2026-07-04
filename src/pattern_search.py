"""
形态相似搜索模块

功能:
  1. 当前走势 vs 该股历史走势 (找历史相似区间)
  2. 当前走势 vs 其他股票近期走势 (找相似股票)

方法:
  - 价格序列归一化 (z-score)
  - 相关系数 (Pearson Correlation)
  - DTW 距离 (可选, 处理时间轴拉伸)
  - 返回最相似的 Top-N 结果
"""

import pandas as pd
import numpy as np
from scipy import stats as sp_stats
from scipy.spatial.distance import euclidean
try:
    from fastdtw import fastdtw
    HAS_DTW = True
except ImportError:
    HAS_DTW = False


def _normalize_sequence(arr: np.ndarray) -> np.ndarray:
    """z-score 归一化, 消除量纲影响"""
    if len(arr) == 0:
        return arr
    mu = np.mean(arr)
    sigma = np.std(arr)
    if sigma == 0:
        return np.zeros_like(arr)
    return (arr - mu) / sigma


def _extract_window(df: pd.DataFrame, end_idx: int, window: int) -> np.ndarray:
    """提取 end_idx 前 window 天的收盘价序列, 归一化"""
    start = max(0, end_idx - window + 1)
    prices = df["Close"].iloc[start:end_idx + 1].values
    if len(prices) < window:
        # 不足 window 天, 左边补 NaN 标记
        padded = np.full(window, np.nan)
        padded[-len(prices):] = prices
        prices = padded[~np.isnan(padded)]
    return _normalize_sequence(prices)


def search_historical_similar(
    df: pd.DataFrame,
    window: int = 30,
    min_gap: int = 60,
    top_n: int = 5,
    use_dtw: bool = False,
) -> list[dict]:
    """
    在当前股票的历史数据中搜索与最近 window 天走势最相似的区间

    参数:
      df:       含有 Date, Close 的 DataFrame
      window:   匹配窗口长度 (天数)
      min_gap:  与当前区间的最小间隔 (避免返回相邻区间)
      top_n:    返回最相似的 N 个结果
      use_dtw:  是否使用 DTW (慢但更准确)

    返回:
      [{"start_date", "end_date", "correlation", "distance", "score"}, ...]
    """
    n = len(df)
    if n < window * 2 + min_gap:
        return []

    # 当前窗口 (最后 window 天)
    curr_prices = df["Close"].iloc[-window:].values
    curr_norm = _normalize_sequence(curr_prices)

    results = []

    # 在历史数据中滑动窗口
    for start in range(0, n - window - min_gap):
        end = start + window - 1
        hist_prices = df["Close"].iloc[start:end + 1].values
        hist_norm = _normalize_sequence(hist_prices)

        if len(curr_norm) != len(hist_norm):
            continue

        # 相关系数 (越高越相似)
        if np.std(curr_norm) == 0 or np.std(hist_norm) == 0:
            corr = 0
        else:
            corr = np.corrcoef(curr_norm, hist_norm)[0, 1]

        # 欧氏距离 (越小越相似)
        dist = euclidean(curr_norm, hist_norm)

        # DTW 距离 (可选)
        dtw_dist = None
        if use_dtw and HAS_DTW:
            dtw_dist, _ = fastdtw(curr_norm, hist_norm)

        # 综合评分: 相关系数主导
        score = corr - dist / 10  # 简单加权

        results.append({
            "start_date": str(df["Date"].iloc[start]),
            "end_date": str(df["Date"].iloc[end]),
            "start_idx": int(start),
            "end_idx": int(end),
            "correlation": round(float(corr), 4),
            "distance": round(float(dist), 4),
            "dtw_distance": round(float(dtw_dist), 4) if dtw_dist is not None else None,
            "score": round(float(score), 4),
        })

    # 按评分排序 (降序, 越高越好)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]


def search_cross_stock_similar(
    target_df: pd.DataFrame,
    candidates: list[pd.DataFrame],
    candidate_names: list[str],
    window: int = 30,
    top_n: int = 5,
) -> list[dict]:
    """
    跨股票搜索相似走势

    参数:
      target_df:       目标股票的 DataFrame
      candidates:      候选股票 DataFrame 列表
      candidate_names: 候选股票名称列表
      window:          匹配窗口长度
      top_n:           返回最相似的 N 个

    返回:
      [{"ticker", "end_date", "correlation", "score"}, ...]
    """
    if len(target_df) < window:
        return []

    curr_prices = target_df["Close"].iloc[-window:].values
    curr_norm = _normalize_sequence(curr_prices)
    results = []

    for cand_df, name in zip(candidates, candidate_names):
        if len(cand_df) < window:
            continue

        # 取候选股票的最后 window 天
        cand_prices = cand_df["Close"].iloc[-window:].values
        cand_norm = _normalize_sequence(cand_prices)

        if len(curr_norm) != len(cand_norm):
            continue

        if np.std(curr_norm) == 0 or np.std(cand_norm) == 0:
            corr = 0
        else:
            corr = np.corrcoef(curr_norm, cand_norm)[0, 1]

        dist = euclidean(curr_norm, cand_norm)
        score = corr - dist / 10

        results.append({
            "ticker": name,
            "end_date": str(cand_df["Date"].iloc[-1]),
            "correlation": round(float(corr), 4),
            "distance": round(float(dist), 4),
            "score": round(float(score), 4),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]


def get_pattern_features(df: pd.DataFrame, window: int = 30) -> dict:
    """
    提取当前走势的形态特征, 用于展示
    """
    if len(df) < window:
        return {}

    prices = df["Close"].iloc[-window:].values
    returns = np.diff(prices) / prices[:-1]

    features = {
        "trend_direction": "up" if prices[-1] > prices[0] else "down",
        "total_return": round((prices[-1] / prices[0] - 1) * 100, 2),
        "volatility": round(float(np.std(returns) * np.sqrt(252) * 100), 2),
        "max_drawdown": round(float(_calc_max_drawdown(prices)), 4),
        "sharpe_approx": round(float(np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0, 2),
        "slope": round(float(sp_stats.linregress(np.arange(len(prices)), prices)[0]), 4),
        "pattern_type": _classify_pattern(prices),
    }
    return features


def _calc_max_drawdown(prices: np.ndarray) -> float:
    """计算最大回撤"""
    peak = np.maximum.accumulate(prices)
    drawdown = (prices - peak) / peak
    return float(np.min(drawdown))


def _classify_pattern(prices: np.ndarray) -> str:
    """简单形态分类"""
    if len(prices) < 10:
        return "unknown"

    slope, _, r_value, _, _ = sp_stats.linregress(np.arange(len(prices)), prices)
    r2 = r_value ** 2
    ret = (prices[-1] - prices[0]) / prices[0]

    if r2 > 0.7:
        if slope > 0:
            return "strong_uptrend"
        else:
            return "strong_downtrend"
    elif r2 > 0.3:
        if slope > 0:
            return "uptrend"
        else:
            return "downtrend"
    else:
        if abs(ret) < 0.05:
            return "sideways"
        elif ret > 0:
            return "choppy_up"
        else:
            return "choppy_down"


# ═══════════════════════════════════════════════════════════════
# 可视化: 把相似历史区间画在图上
# ═══════════════════════════════════════════════════════════════

def highlight_similar_periods(df: pd.DataFrame, similar_list: list[dict],
                              ax, color="#ff9800", alpha=0.15):
    """
    在主图上用背景色高亮相似历史区间
    similar_list: search_historical_similar 的返回结果
    """
    n = len(df)
    for item in similar_list:
        start_idx = item.get("start_idx", 0)
        end_idx = item.get("end_idx", n - 1)
        corr = item.get("correlation", 0)
        # 用矩形高亮
        ax.axvspan(start_idx, end_idx, alpha=alpha, color=color,
                   label=f"相似区间 corr={corr:.2f}" if corr > 0.7 else None)
