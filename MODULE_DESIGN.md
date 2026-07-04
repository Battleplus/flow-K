# FLOW AI Trader · 新模块设计与集成方案

> 聚焦：消息面分析 + 板块ETF轮动 怎么设计、怎么和现有代码对接

---

## 0. 设计起点：现有代码有什么

写新代码之前，先摸清楚已有的能复用什么，避免重造轮子。

### 0.1 现有函数清单（新模块会直接调用）

| 模块 | 函数 | 输入 | 输出 | 新模块用途 |
|------|------|------|------|-----------|
| `indicators.py` | `add_ma(df)` | DataFrame | 加 ma5/10/20/60 列 | ETF加均线 |
| `indicators.py` | `calc_slope(series, lookback=5)` | pd.Series | float (归一化斜率) | ETF趋势方向 |
| `indicators.py` | `ma_alignment(df)` | DataFrame | `{"alignment": "多头排列"/"空头排列"/"交织震荡"}` | ETF均线排列判断 |
| `indicators.py` | `slope_summary(df)` | DataFrame | dict (各MA斜率+方向) | ETF趋势细节 |
| `patterns.py` | `add_indicators(df)` | DataFrame | 加 MACD/RSI/BB/Stochastic/vol_ratio 等列 | ETF加全部指标 |
| `patterns.py` | `detect_all(df)` | DataFrame | 加 signal_xxx 列 | ETF跑35个信号 |
| `patterns.py` | `signal_summary(df)` | DataFrame | `{"bullish": n, "bearish": m, "score": x}` | ETF信号汇总 |
| `strategies.py` | `aggregate_signals(df)` | DataFrame | DataFrame (20策略信号+aggregate_score) | ETF多策略共识 |
| `strategies.py` | `get_latest_consensus(df)` | DataFrame | `{"aggregate_score": f, "consensus": "偏多", ...}` | ETF最终共识 |
| `data_loader.py` | `fetch_data(ticker, period)` | str | DataFrame (OHLCV) | 加载ETF行情 |
| `factors.py` | `add_all_factors(df)` | DataFrame | 加85因子列 | 个股分析(已有) |
| `screener.py` | `analyze_ticker(ticker, ...)` | str | dict (完整分析结果) | 个股扫描(已有) |

### 0.2 现有数据清单

| 文件 | 内容 | 新模块怎么用 |
|------|------|-------------|
| `data/company_info.csv` | 503家公司，含 Symbol/Company/Sector/BusinessSummary | 消息面TF-IDF + 板块→个股映射 |
| `data/sector_etf_prices.csv` | 12只ETF(SPY+QQQ+10板块)，2000-2026.2 | 板块ETF行情分析 |
| `data/vix_data.csv` | VIX日线，2000-2026.2 | 风控/仓位调整 |
| `data/event_sector_mapping.json` | 17组关键词→板块映射，含direction/weight | 消息面关键词匹配 |
| `SP500_Historical_Data.csv` | 472只个股日线 | 个股K线分析(已有) |

**关键发现**：现有 `event_sector_mapping.json` 已经包含了关键词→板块映射和评分规则，但从来没有人调用过它。消息面分析模块就是把这个数据激活。

---

## 1. 消息面分析模块设计 (`src/news_analysis.py`)

### 1.1 核心问题

输入一段新闻文本，输出11个板块各自的消息面评分(-2~+2)。

难点在于：关键词匹配能覆盖已知事件，但覆盖不了新事件；TF-IDF能覆盖语义但不够精确。两者互补。

### 1.2 数据流

```
用户输入: "NVIDIA发布B300芯片，AI推理性能提升60%"
    │
    ├──→ 路径A: 关键词匹配 (精确, 权重0.4)
    │    │
    │    │  加载 event_sector_mapping.json
    │    │  遍历17组关键词
    │    │  "AI" 命中 → Technology, direction=positive, weight=1.5
    │    │  "芯片" 命中 → Technology, direction=positive, weight=1.5
    │    │  "NVIDIA" 命中 → Technology, direction=positive, weight=1.5
    │    │  → Technology 板块 keyword_score = 3次命中 × 1.5权重 = 4.5
    │    │
    │    └──→ 归一化到 [-2, +2]: min(keyword_score * direction, 2)
    │
    ├──→ 路径B: TF-IDF语义匹配 (模糊, 权重0.6)
    │    │
    │    │  加载预训练 TF-IDF vectorizer (470股 × 2000维)
    │    │  将新闻文本向量化
    │    │  与470只股票的 BusinessSummary 向量计算余弦相似度
    │    │  → top匹配: NVDA(0.95), AMD(0.87), AVGO(0.81)...
    │    │
    │    │  按板块聚合: 取每个板块内 top5 匹配的均值
    │    │  Technology 板块 top5 均值 = 0.82
    │    │  → 归一化到 [-2, +2]: similarity * 2 (正向) 或 similarity * -2 (负向)
    │    │
    │    └──→ 负向判断: 如果新闻含负面词(禁令/制裁/暴跌/亏损)，
    │         则取负值
    │
    └──→ 融合:
         news_score[Technology] = keyword_score * 0.4 + tfidf_score * 0.6
                                 = 2.0 * 0.4 + 1.64 * 0.6 = 1.78 → 钳制到 +2
```

### 1.3 函数设计

```python
# src/news_analysis.py

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent

# ── 板块ETF映射（从json加载，不硬编码）──
_SECTOR_ETF_MAP = None  # lazy load
_TFIDF_VECTORIZER = None
_TFIDF_MATRIX = None
_STOCK_SECTOR_MAP = None  # ticker → sector

def _load_mapping():
    """加载 event_sector_mapping.json"""
    global _SECTOR_ETF_MAP
    if _SECTOR_ETF_MAP is None:
        with open(ROOT / "data" / "event_sector_mapping.json", "r", encoding="utf-8") as f:
            _SECTOR_ETF_MAP = json.load(f)
    return _SECTOR_ETF_MAP


def _load_tfidf():
    """加载/构建 TF-IDF 向量库（懒加载，首次调用时构建）"""
    global _TFIDF_VECTORIZER, _TFIDF_MATRIX, _STOCK_SECTOR_MAP
    if _TFIDF_VECTORIZER is not None:
        return _TFIDF_VECTORIZER, _TFIDF_MATRIX, _STOCK_SECTOR_MAP

    ci = pd.read_csv(ROOT / "data" / "company_info.csv")
    # 过滤掉没有 BusinessSummary 的
    ci = ci[ci["BusinessSummary"].notna() & (ci["BusinessSummary"].str.len() > 10)]

    summaries = ci["BusinessSummary"].tolist()
    tickers = ci["Symbol"].tolist()
    sectors = ci["Sector"].tolist()

    vec = TfidfVectorizer(max_features=2000, stop_words="english")
    matrix = vec.fit_transform(summaries)

    _TFIDF_VECTORIZER = vec
    _TFIDF_MATRIX = matrix
    _STOCK_SECTOR_MAP = dict(zip(tickers, sectors))
    return vec, matrix, _STOCK_SECTOR_MAP


# ── 负面词表（用于方向判定）──
_NEGATIVE_WORDS = {
    "禁令", "ban", "禁止", "制裁", "sanction", "暴跌", "crash", "崩盘",
    "亏损", "loss", "破产", "bankrupt", "退市", "delist", "调查", "investigation",
    "罚款", "fine", "诉讼", "lawsuit", "召回", "recall", "警告", "warning",
    "不及预期", "miss", "下调", "downgrade", "裁员", "layoff", "违约", "default",
}


def _keyword_match(text: str) -> dict:
    """
    路径A: 关键词匹配
    返回: {sector: {"score": float, "hits": [keyword, ...], "direction": str}}
    """
    mapping = _load_mapping()
    text_lower = text.lower()
    results = {}

    for group in mapping["event_keywords"]:
        sector = group["sector"]
        direction = group["direction"]  # positive / negative / mixed / neutral
        weight = group["weight"]
        hits = []

        for kw in group["keywords"]:
            if kw.lower() in text_lower:
                hits.append(kw)

        if hits and sector:
            # 方向判断
            if direction == "negative":
                dir_score = -1
            elif direction == "positive":
                dir_score = 1
            elif direction == "mixed":
                # 检查文本里有没有负面词
                has_negative = any(w in text_lower for w in _NEGATIVE_WORDS)
                dir_score = -1 if has_negative else 1
            else:
                dir_score = 0

            raw_score = len(hits) * weight * dir_score
            # 钳制到 [-2, 2]
            clamped = max(-2.0, min(2.0, raw_score))

            if sector not in results or abs(clamped) > abs(results[sector]["score"]):
                results[sector] = {
                    "score": clamped,
                    "hits": hits,
                    "direction": direction,
                    "weight": weight,
                }

    return results


def _tfidf_match(text: str) -> dict:
    """
    路径B: TF-IDF语义匹配
    返回: {sector: {"score": float, "top_stocks": [(ticker, sim), ...]}}
    """
    vec, matrix, stock_sector = _load_tfidf()

    # 将新闻文本向量化
    text_vec = vec.transform([text])
    sims = cosine_similarity(text_vec, matrix).flatten()

    # 按板块聚合
    sector_scores = {}
    for i, sim in enumerate(sims):
        ticker = list(stock_sector.keys())[i]
        sector = stock_sector[ticker]
        if sector not in sector_scores:
            sector_scores[sector] = []
        sector_scores[sector].append((ticker, sim))

    # 每个板块取 top5 的均值
    results = {}
    has_negative = any(w in text.lower() for w in _NEGATIVE_WORDS)
    direction = -1 if has_negative else 1

    for sector, stock_sims in sector_scores.items():
        stock_sims.sort(key=lambda x: x[1], reverse=True)
        top5 = stock_sims[:5]
        avg_sim = np.mean([s for _, s in top5])
        # 归一化: similarity 0~1 → score 0~2
        score = avg_sim * 2 * direction
        results[sector] = {
            "score": max(-2.0, min(2.0, score)),
            "top_stocks": top5,
        }

    return results


def analyze_news(text: str) -> dict:
    """
    主函数: 输入新闻文本 → 返回11板块消息面评分

    返回:
    {
        "text": "NVIDIA发布B300芯片...",
        "sectors": {
            "Technology": {
                "score": 2.0,
                "keyword_score": 2.0,
                "tfidf_score": 1.64,
                "matched_keywords": ["AI", "芯片", "NVIDIA"],
                "top_tfidf_stocks": [("NVDA", 0.95), ("AMD", 0.87), ...],
                "reason": "命中AI/芯片/NVIDIA关键词 + TF-IDF相似度0.82"
            },
            ...
        },
        "matched_keywords": ["AI", "芯片", "NVIDIA"],
        "sentiment": "positive"
    }
    """
    if not text or not text.strip():
        return _empty_result()

    kw_results = _keyword_match(text)
    tfidf_results = _tfidf_match(text)

    # 获取所有板块名
    mapping = _load_mapping()
    all_sectors = list(mapping["sector_etf_map"].keys())

    # 融合两条路径
    sectors = {}
    all_keywords = []

    for sector in all_sectors:
        kw = kw_results.get(sector, {"score": 0, "hits": []})
        tf = tfidf_results.get(sector, {"score": 0, "top_stocks": []})

        # 融合: 关键词权重0.4 + TF-IDF权重0.6
        # 但如果关键词没命中，纯靠TF-IDF要打折（因为TF-IDF有噪声）
        if kw["score"] != 0:
            fused = kw["score"] * 0.4 + tf["score"] * 0.6
        else:
            # 关键词没命中，TF-IDF单独打分，阈值过滤
            if abs(tf["score"]) < 0.5:
                fused = 0  # 太低，当噪声
            else:
                fused = tf["score"] * 0.5  # 打折

        fused = max(-2.0, min(2.0, fused))

        # 生成原因说明
        reasons = []
        if kw["hits"]:
            reasons.append(f"命中关键词: {', '.join(kw['hits'])}")
        if tf["top_stocks"] and tf["top_stocks"][0][1] > 0.3:
            top_stock = tf["top_stocks"][0]
            reasons.append(f"TF-IDF最相关: {top_stock[0]}(相似度{top_stock[1]:.2f})")

        sectors[sector] = {
            "score": round(fused, 2),
            "keyword_score": round(kw["score"], 2),
            "tfidf_score": round(tf["score"], 2),
            "matched_keywords": kw["hits"],
            "top_tfidf_stocks": [
                {"ticker": t, "similarity": round(s, 3)}
                for t, s in tf["top_stocks"][:5]
            ],
            "reason": "; ".join(reasons) if reasons else "无相关信号",
        }
        all_keywords.extend(kw["hits"])

    # 整体情感
    all_scores = [s["score"] for s in sectors.values()]
    avg_score = np.mean(all_scores)
    if avg_score > 0.5:
        sentiment = "positive"
    elif avg_score < -0.5:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return {
        "text": text,
        "sectors": sectors,
        "matched_keywords": list(set(all_keywords)),
        "sentiment": sentiment,
    }
```

### 1.4 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| TF-IDF权重 > 关键词权重 | 0.6 vs 0.4 | TF-IDF能覆盖关键词表里没有的新事件，更通用 |
| 关键词没命中时TF-IDF打折 | × 0.5 + 阈值0.5 | 纯语义匹配噪声大，需要关键词确认才给满分 |
| 负面词表硬编码 | 30个词 | 简单有效，不需要NLP模型 |
| 方向判定用 `direction` 字段 | 从json读 | `event_sector_mapping.json`已经有positive/negative/mixed标记 |
| `mixed`方向看负面词 | 有负面词→负，否则→正 | "降息"对金融是mixed，要看上下文 |
| TF-IDF向量库懒加载 | 首次调用时构建 | 避免启动慢，构建一次约2秒 |
| 按板块取top5均值 | 不是全板块均值 | top5代表最相关的，全板块均值会被无关股票稀释 |

---

## 2. 板块ETF轮动模块设计 (`src/sector_rotation.py`)

### 2.1 核心问题

用11只板块ETF的行情数据，判断资金在往哪个板块流。

核心洞察：**资金总量守恒**——不需要看Level2资金流数据，板块ETF相对SPY的超额收益就是资金流向的"水表"。

### 2.2 三个子评分如何复用现有代码

```
对每只板块ETF (XLK, XLV, XLF, ...):
    │
    ├── ① ETF趋势评分 (复用 indicators + patterns + strategies)
    │   │
    │   │  df = load_etf("XLK")                    # 新函数，读sector_etf_prices.csv
    │   │  df = add_indicators(df)                 # 现有: 加MA/MACD/RSI/BB/vol_ratio
    │   │  df = detect_all(df)                     # 现有: 35个信号
    │   │  summary = signal_summary(df)            # 现有: 信号汇总
    │   │  consensus = get_latest_consensus(df)    # 现有: 20策略共识
    │   │  alignment = ma_alignment(df)            # 现有: 多头/空头排列
    │   │
    │   │  综合判断:
    │   │    alignment == "多头排列" + consensus偏多 → etf_trend = +1
    │   │    alignment == "空头排列" + consensus偏空 → etf_trend = -1
    │   │    否则 → 0
    │   │
    │   └── 额外: vol_ratio > 2 (放量) 时额外 ±0.3
    │
    ├── ② 相对强度评分 (新逻辑，但数据来自现有CSV)
    │   │
    │   │  etf_return_20d = XLK近20日收益率
    │   │  spy_return_20d = SPY近20日收益率
    │   │  excess_20d = etf_return_20d - spy_return_20d
    │   │  (同理算60日)
    │   │
    │   │  rs_raw = 0.6 * excess_20d + 0.4 * excess_60d
    │   │
    │   │  11只ETF按 rs_raw 排名:
    │   │    排名1-3 → rs_score = +1 (资金流入区)
    │   │    排名4-8 → rs_score = 0  (中性区)
    │   │    排名9-11→ rs_score = -1 (资金流出区)
    │   │
    │   └── 这就是"资金轮动水表"
    │
    └── ③ 资金确认矩阵 (新逻辑，交叉判断①和②)
       │
       │  趋势 +1 & RS +1 → "inflow" (资金确认流入)     最强
       │  趋势 +1 & RS -1 → "distribution" (高位派发)   警惕
       │  趋势 -1 & RS +1 → "rotation" (轮动早期)       重点关注
       │  趋势 -1 & RS -1 → "outflow" (资金确认流出)    最弱
       │  趋势  0 & 任意  → "neutral" (无方向)
       │
       └── 这个矩阵是整个模块的核心输出
```

### 2.3 函数设计

```python
# src/sector_rotation.py

import pandas as pd
import numpy as np
from pathlib import Path

from src.patterns import add_indicators, detect_all, signal_summary
from src.strategies import get_latest_consensus
from src.indicators import ma_alignment, calc_slope

ROOT = Path(__file__).resolve().parent.parent
ETF_CSV = ROOT / "data" / "sector_etf_prices.csv"

# 11板块ETF → 中文名映射
ETF_NAMES = {
    "XLK": "Technology", "XLV": "Healthcare", "XLF": "Financials",
    "XLE": "Energy", "XLI": "Industrials", "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples", "XLU": "Utilities", "XLB": "Materials",
    "XLRE": "Real Estate", "XLC": "Communication Services",
}
ETF_CN = {
    "XLK": "科技", "XLV": "医疗", "XLF": "金融", "XLE": "能源",
    "XLI": "工业", "XLY": "可选消费", "XLP": "必需消费", "XLU": "公用",
    "XLB": "材料", "XLRE": "房地产", "XLC": "通信",
}


def load_etf(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    加载ETF历史数据（从 sector_etf_prices.csv）
    复用 data_loader 的列格式，确保和现有函数兼容
    """
    df = pd.read_csv(ETF_CSV, parse_dates=["Date"])
    df = df[df["Ticker"] == ticker].copy()
    df = df.sort_values("Date").reset_index(drop=True)

    # 截取period
    days = {"3mo": 90, "6mo": 180, "1y": 365, "2y": 730}.get(period, 365)
    cutoff = df["Date"].max() - pd.Timedelta(days=days)
    df = df[df["Date"] >= cutoff].reset_index(drop=True)
    df["Date"] = df["Date"].dt.date
    return df


def _calc_etf_trend(df: pd.DataFrame) -> dict:
    """
    子评分①: ETF趋势评分
    全部复用现有函数
    返回: {"trend_score": -1/0/+1, "alignment": str, "consensus": str, "vol_ratio": float}
    """
    df = add_indicators(df)     # 加MA/MACD/RSI/BB/...
    df = detect_all(df)         # 35个信号
    summary = signal_summary(df)
    consensus = get_latest_consensus(df)
    alignment = ma_alignment(df)

    # 最新成交量比率
    vol_ratio = df["vol_ratio"].iloc[-1] if "vol_ratio" in df.columns else 1.0

    # 综合判断趋势
    agg_score = consensus.get("aggregate_score", 0)
    consensus_text = consensus.get("consensus", "中性")

    # 均线排列 + 策略共识 双确认
    if "多头" in alignment["alignment"] and agg_score > 0:
        trend = 1
    elif "空头" in alignment["alignment"] and agg_score < 0:
        trend = -1
    else:
        trend = 0

    # 放量加成
    if vol_ratio > 2.0:
        trend = trend + (0.3 if trend > 0 else -0.3 if trend < 0 else 0)
        trend = max(-1.0, min(1.0, trend))

    return {
        "trend_score": trend,
        "alignment": alignment["alignment"],
        "consensus": consensus_text,
        "aggregate_score": agg_score,
        "vol_ratio": round(vol_ratio, 2),
    }


def _calc_relative_strength(etf_ticker: str, spy_df: pd.DataFrame,
                            etf_df: pd.DataFrame) -> dict:
    """
    子评分②: 相对强度评分
    计算 ETF 相对于 SPY 的超额收益
    返回: {"rs_score": -1/0/+1, "excess_20d": float, "excess_60d": float, "rank": int}
    """
    def _return(df, days):
        if len(df) < days + 1:
            return 0.0
        return (df["Close"].iloc[-1] / df["Close"].iloc[-days-1] - 1) * 100

    ret_20 = _return(etf_df, 20)
    ret_60 = _return(etf_df, 60)
    spy_20 = _return(spy_df, 20)
    spy_60 = _return(spy_df, 60)

    excess_20 = ret_20 - spy_20
    excess_60 = ret_60 - spy_60
    rs_raw = 0.6 * excess_20 + 0.4 * excess_60

    return {
        "excess_20d": round(excess_20, 2),
        "excess_60d": round(excess_60, 2),
        "rs_raw": round(rs_raw, 2),
    }


def _fund_flow_matrix(trend: float, rs_score: int) -> dict:
    """
    子评分③: 资金确认矩阵
    输入: 趋势评分(-1~+1) + 相对强度排名分(-1/0/+1)
    输出: {"flow": str, "desc": str, "action": str}
    """
    trend_dir = 1 if trend > 0.3 else (-1 if trend < -0.3 else 0)

    if trend_dir == 1 and rs_score == 1:
        return {"flow": "inflow", "desc": "资金确认流入", "action": "重点选股"}
    elif trend_dir == 1 and rs_score == -1:
        return {"flow": "distribution", "desc": "高位派发", "action": "警惕见顶"}
    elif trend_dir == -1 and rs_score == 1:
        return {"flow": "rotation", "desc": "板块轮动早期", "action": "重点关注"}
    elif trend_dir == -1 and rs_score == -1:
        return {"flow": "outflow", "desc": "资金确认流出", "action": "坚决回避"}
    else:
        return {"flow": "neutral", "desc": "无明确方向", "action": "观望"}


def analyze_sectors(news_scores: dict = None) -> dict:
    """
    主函数: 分析11板块ETF → 返回板块综合排名

    参数:
        news_scores: 消息面评分, {"Technology": 2.0, ...} (可选, 来自news_analysis)
                     如果不传，sector_score 只含 ETF趋势 + RS 两个维度

    返回:
    {
        "benchmark": {"spy_price": 612, "spy_20d": "+2.1%", ...},
        "sectors": [
            {
                "ticker": "XLK", "name": "Technology", "name_cn": "科技",
                "price": 235.8, "return_20d": 5.2, "return_60d": 12.1,
                "trend_score": 1, "alignment": "多头排列",
                "rs_rank": 1, "rs_score": 1,
                "excess_20d": 3.1, "excess_60d": 6.8,
                "fund_flow": "inflow", "fund_flow_desc": "资金确认流入",
                "news_score": 2.0,  # 如果传了news_scores
                "sector_score": 1.0,  # 综合评分
                "zone": "strong"  # strong/watch/neutral/avoid
            },
            ...按sector_score降序
        ]
    }
    """
    spy_df = load_etf("SPY", "1y")

    # 计算所有ETF的原始数据
    all_results = []
    for etf_ticker, sector_name in ETF_NAMES.items():
        try:
            etf_df = load_etf(etf_ticker, "1y")
            if len(etf_df) < 60:
                continue
        except Exception:
            continue

        trend_info = _calc_etf_trend(etf_df)
        rs_info = _calc_relative_strength(etf_ticker, spy_df, etf_df)

        all_results.append({
            "ticker": etf_ticker,
            "name": sector_name,
            "name_cn": ETF_CN[etf_ticker],
            "price": round(etf_df["Close"].iloc[-1], 2),
            "trend_score": trend_info["trend_score"],
            "alignment": trend_info["alignment"],
            "consensus": trend_info["consensus"],
            "vol_ratio": trend_info["vol_ratio"],
            "rs_raw": rs_info["rs_raw"],
            "excess_20d": rs_info["excess_20d"],
            "excess_60d": rs_info["excess_60d"],
        })

    # 按rs_raw排名 → rs_score
    all_results.sort(key=lambda x: x["rs_raw"], reverse=True)
    for i, r in enumerate(all_results):
        r["rs_rank"] = i + 1
        if i < 3:
            r["rs_score"] = 1
        elif i >= len(all_results) - 3:
            r["rs_score"] = -1
        else:
            r["rs_score"] = 0

    # 资金确认矩阵
    for r in all_results:
        flow = _fund_flow_matrix(r["trend_score"], r["rs_score"])
        r["fund_flow"] = flow["flow"]
        r["fund_flow_desc"] = flow["desc"]
        r["action"] = flow["action"]

    # 融合消息面评分
    for r in all_results:
        news_score = 0
        if news_scores and r["name"] in news_scores:
            news_score = news_scores[r["name"]]["score"]

        r["news_score"] = news_score

        # 板块综合评分: news*0.4 + trend*0.3 + rs*0.3
        r["sector_score"] = round(
            news_score * 0.4 + r["trend_score"] * 0.3 + r["rs_score"] * 0.3, 2
        )

        # 分区
        if r["sector_score"] >= 0.8:
            r["zone"] = "strong"
        elif r["sector_score"] >= 0.3:
            r["zone"] = "watch"
        elif r["sector_score"] >= -0.3:
            r["zone"] = "neutral"
        else:
            r["zone"] = "avoid"

    # 按sector_score降序
    all_results.sort(key=lambda x: x["sector_score"], reverse=True)

    # benchmark
    spy_price = round(spy_df["Close"].iloc[-1], 2)
    spy_20d = round((spy_df["Close"].iloc[-1] / spy_df["Close"].iloc[-21] - 1) * 100, 2)
    spy_60d = round((spy_df["Close"].iloc[-1] / spy_df["Close"].iloc[-61] - 1) * 100, 2)

    return {
        "benchmark": {
            "spy_price": spy_price,
            "spy_return_20d": spy_20d,
            "spy_return_60d": spy_60d,
        },
        "sectors": all_results,
    }


def get_sector_stocks(sector_name: str) -> list[str]:
    """
    获取板块内所有成分股ticker
    从 company_info.csv 按 Sector 字段筛选
    """
    ci = pd.read_csv(ROOT / "data" / "company_info.csv")
    stocks = ci[ci["Sector"] == sector_name]["Symbol"].tolist()
    return stocks


def get_strong_sectors(sector_result: dict, min_score: float = 0.3) -> list[str]:
    """
    从板块分析结果中提取强势板块和关注板块
    返回板块名列表（用于第三层个股扫描的过滤条件）
    """
    return [
        s["name"] for s in sector_result["sectors"]
        if s["sector_score"] >= min_score and s["zone"] in ("strong", "watch")
    ]
```

### 2.4 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| ETF趋势用 `ma_alignment + consensus` 双确认 | 两个都要同向 | 单一指标容易误判，双确认更可靠 |
| RS用20日+60日加权 | 0.6/0.4 | 20日反映近期，60日反映中期，近期权重更大 |
| RS排名分3档而非连续值 | 1-3/4-8/9-11 | 排名比绝对值更稳定，不受市场整体波动影响 |
| 资金确认矩阵4+1种状态 | inflow/outflow/distribution/rotation/neutral | 覆盖所有趋势×RS组合，每种有明确操作建议 |
| `sector_score`融合news | news*0.4+trend*0.3+rs*0.3 | 消息面权重最高，因为消息驱动资金流向 |
| 板块分4区(strong/watch/neutral/avoid) | ≥0.8/≥0.3/≥-0.3/<-0.3 | 避免模糊地带，操作明确 |

---

## 3. 三维集成入口设计 (`src/three_dim_analyzer.py`)

### 3.1 核心问题

前两层的输出怎么传给第三层？第三层是现有代码，不能改它的内部逻辑，只能在**调用入口**加过滤。

### 3.2 集成方式：装饰器模式

不改 `screener.analyze_ticker()` 的任何代码，而是在外面包一层：

```python
# src/three_dim_analyzer.py

import pandas as pd
from src.news_analysis import analyze_news
from src.sector_rotation import analyze_sectors, get_sector_stocks, get_strong_sectors
from src.screener import analyze_ticker
from src.data_loader import fetch_data
from src.strategies import get_latest_consensus

# 个股评分映射: 把 consensus 的 aggregate_score 映射到 -1~+1
def _stock_score_from_consensus(consensus: dict) -> float:
    """将多策略共识的 aggregate_score 映射到 [-1, +1]"""
    score = consensus.get("aggregate_score", 0)
    # aggregate_score 范围约 -40~+40 (20策略 × ±2)
    # 归一化
    normalized = max(-1.0, min(1.0, score / 20.0))
    return normalized


def full_scan(news_text: str = "", top_n: int = 10,
              min_score: float = 1.0, period: str = "6mo") -> dict:
    """
    三维全市场扫描（核心入口）

    流程:
    1. 消息面: news_text → analyze_news() → 11板块news_score
    2. 板块轮动: news_score → analyze_sectors() → 11板块sector_score
    3. 个股扫描: 在strong/watch板块内 → analyze_ticker() → stock_score
    4. 三维融合: final = news*0.25 + sector*0.35 + stock*0.40
    5. 过滤+排序 → 返回TOP N
    """
    # ── Layer 1: 消息面 ──
    news_result = analyze_news(news_text) if news_text else None
    news_scores = news_result["sectors"] if news_result else None

    # ── Layer 2: 板块轮动 ──
    sector_result = analyze_sectors(news_scores)
    strong_sectors = get_strong_sectors(sector_result, min_score=0.3)

    # 如果没有新闻输入，纯靠板块ETF做技术分析
    if not strong_sectors:
        strong_sectors = [
            s["name"] for s in sector_result["sectors"]
            if s["sector_score"] >= 0
        ]

    # ── Layer 3: 个股扫描 ──
    recommendations = []

    for sector_name in strong_sectors:
        # 获取板块内所有成分股
        tickers = get_sector_stocks(sector_name)

        # 板块评分（所有该板块股票共享）
        sector_info = next(
            s for s in sector_result["sectors"] if s["name"] == sector_name
        )
        sector_score = sector_info["sector_score"]
        sector_news = sector_info.get("news_score", 0)

        # 遍历板块内个股
        for ticker in tickers:
            try:
                # 复用现有 analyze_ticker
                result = analyze_ticker(ticker, period=period, hold_profile="medium")
                if result is None:
                    continue

                # 个股共识评分
                stock_score = _stock_score_from_consensus(
                    result.get("consensus", {})
                )

                # 三维融合
                final_score = (
                    sector_news * 0.25
                    + sector_score * 0.35
                    + stock_score * 0.40
                )

                # 一票否决: 消息面强利空 或 板块资金流出
                if sector_news <= -2 or sector_score <= -1:
                    continue

                if final_score < min_score:
                    continue

                recommendations.append({
                    "ticker": ticker,
                    "company": result.get("company", ""),
                    "sector": sector_name,
                    "price": result.get("price", 0),
                    "scores": {
                        "news": sector_news,
                        "sector": sector_score,
                        "stock": round(stock_score, 2),
                        "final": round(final_score, 2),
                    },
                    "verdict": _verdict(final_score),
                    "top_strategies": result.get("top_strategies", []),
                    "active_signals": result.get("active_signals", []),
                    "suggestion": _suggestion(final_score, result),
                })

            except Exception:
                continue

    # 排序+截取
    recommendations.sort(key=lambda x: x["scores"]["final"], reverse=True)
    recommendations = recommendations[:top_n]

    return {
        "news_analysis": news_result,
        "sector_snapshot": sector_result,
        "strong_sectors": strong_sectors,
        "recommendations": recommendations,
        "summary": {
            "total_scanned": sum(
                len(get_sector_stocks(s)) for s in strong_sectors
            ),
            "passed_filter": len(recommendations),
            "strongest_sector": sector_result["sectors"][0]["name"]
                if sector_result["sectors"] else None,
            "weakest_sector": sector_result["sectors"][-1]["name"]
                if sector_result["sectors"] else None,
        },
    }


def _verdict(score: float) -> dict:
    """三维评分 → 判断"""
    if score >= 2.0:
        return {"label": "strong_bullish", "cn": "强偏多"}
    elif score >= 1.0:
        return {"label": "bullish", "cn": "偏多"}
    elif score >= -1.0:
        return {"label": "neutral", "cn": "中性"}
    elif score >= -2.0:
        return {"label": "bearish", "cn": "偏空"}
    else:
        return {"label": "strong_bearish", "cn": "强偏空"}


def _suggestion(score: float, stock_result: dict) -> str:
    """生成操作建议"""
    if score >= 2.0:
        return "三维共振，可考虑短线做多，止损-4%，止盈+8%"
    elif score >= 1.0:
        return "两个维度支持，可关注，等待入场信号"
    elif score >= -1.0:
        return "信号矛盾，观望"
    else:
        return "偏空，回避"
```

### 3.3 集成关键点

**不改现有代码，只加入口过滤**：

```
之前 (screener.py):
    analyze_ticker("NVDA")          ← 随便输入，不管板块

现在 (three_dim_analyzer.py):
    1. news_analysis → 命中 Technology 板块
    2. sector_rotation → Technology 是强势板块 (sector_score=1.0)
    3. get_sector_stocks("Technology") → ["AAPL", "MSFT", "NVDA", ...]
    4. 对每只跑 analyze_ticker()     ← 还是调原来的函数，没改
    5. 但只在强势板块里跑，不在弱势板块跑
```

**数据怎么在三层之间传递**：

```
Layer 1 输出                Layer 2 输入                Layer 3 输入
─────────────              ─────────────              ─────────────
news_result = {            analyze_sectors(           get_sector_stocks(
  "sectors": {               news_scores=                 "Technology")
    "Technology": {            news_result["sectors"])   → ["AAPL","MSFT",...]
      "score": 2.0           )
    }                       sector_result = {
  }                           "sectors": [              analyze_ticker(ticker)
}                               {"name":"Technology",   ← 现有函数，不改
                                 "sector_score": 1.0,
                                 "zone": "strong"}
                              ]
                            }
```

---

## 4. 资金轮动逻辑详解（用户核心关注）

### 4.1 为什么"板块ETF排名 = 资金流向"

美股11个SPDR板块ETF覆盖了标普500几乎全部市值。资金不会凭空消失，只会在板块间移动：

```
场景: AI热潮来袭
    │
    ├── 资金涌入 XLK (科技) → XLK 超额收益 +6.8% (60日)
    │   └── RS排名 #1 → rs_score = +1
    │
    ├── 资金从 XLU (公用) 撤出 → XLU 超额收益 -4.2%
    │   └── RS排名 #11 → rs_score = -1
    │
    └── SPY 总涨幅 +5.3% (资金总盘子没变，只是重新分配)
```

### 4.2 资金确认矩阵的实战意义

| 状态 | 含义 | 实战场景 |
|------|------|---------|
| **inflow** (趋势↑+RS强) | 资金正在流入且趋势确认 | XLK持续放量上涨，超SPY 6% → 选科技股 |
| **distribution** (趋势↑+RS弱) | 价格在涨但已经跑输大盘 | XLK涨了但只涨2%，SPY涨5% → 主力在出货 |
| **rotation** (趋势↓+RS强) | 近期回调但长期仍然强势 | XLK近5天跌3%但60日仍超SPY 8% → 回调买入机会 |
| **outflow** (趋势↓+RS弱) | 资金正在流出且趋势确认 | XLU连续下跌，60日跑输SPY 5% → 坚决不碰 |

### 4.3 为什么不需要看资金流数据

| 传统资金流数据 | 本方案 |
|---------------|--------|
| 需要Level2逐笔数据 | 只需日线收盘价 |
| 需要实时API | 本地CSV即可 |
| 大单/小单分类复杂 | ETF超额收益直接反映净效果 |
| 容易被算法混淆 | 价格是所有资金行为的最终结果 |
| 黑客松环境搞不到 | 任何人都能复现 |

**一句话**：所有买单减去所有卖单 = 价格变化。ETF的超额收益已经包含了所有资金行为的信息，不需要再拆开看。

---

## 5. 模块间依赖关系

```
three_dim_analyzer.py (入口)
    │
    ├── 调用 news_analysis.py
    │   ├── 读取 data/event_sector_mapping.json (关键词)
    │   ├── 读取 data/company_info.csv (TF-IDF语料)
    │   └── 输出 news_score[11板块]
    │
    ├── 调用 sector_rotation.py
    │   ├── 读取 data/sector_etf_prices.csv (ETF行情)
    │   ├── 调用 patterns.add_indicators()     ← 现有
    │   ├── 调用 patterns.detect_all()         ← 现有
    │   ├── 调用 strategies.get_latest_consensus() ← 现有
    │   ├── 调用 indicators.ma_alignment()     ← 现有
    │   ├── 融合 news_score → sector_score[11板块]
    │   └── 输出 strong_sectors (板块名列表)
    │
    └── 调用 screener.analyze_ticker()          ← 现有，不改
        ├── 调用 data_loader.fetch_data()       ← 现有
        ├── 调用 factors.add_all_factors()      ← 现有
        ├── 调用 strategies.backtest_all()      ← 现有
        └── 输出 stock_score (个股评分)
```

### 新增 vs 复用 总结

| 代码 | 状态 | 行数估计 |
|------|------|---------|
| `news_analysis.py` | **全新** | ~200行 |
| `sector_rotation.py` | **全新** | ~250行 |
| `three_dim_analyzer.py` | **全新**（入口编排） | ~150行 |
| `patterns.py` | 不改 | - |
| `strategies.py` | 不改 | - |
| `indicators.py` | 不改 | - |
| `data_loader.py` | 不改（ETF加载新写在了sector_rotation里） | - |
| `screener.py` | 不改 | - |
| `api.py` | 加3个路由 | +60行 |
| `web/index.html` | 加3个Tab | +200行 |

**核心原则：新代码只做新的事，现有代码一个字不改。** 三个新模块通过"调用现有函数 + 在入口加过滤"的方式集成，不侵入原有逻辑。

---

## 6. 数据缺口对模块的影响

| 缺口 | 影响哪个模块 | 严重程度 | 解决方案 |
|------|------------|---------|---------|
| 缺XLC通信服务ETF | sector_rotation | 中（11变10板块） | yfinance下载，2018年至今 |
| ETF数据停在2月20日 | sector_rotation | 高（4个月缺口） | yfinance全量更新 |
| VIX数据停在2月20日 | 风控模块 | 低（先用着） | yfinance更新 |
| 个股K线缺65只 | three_dim_analyzer | 中（部分股票扫不到） | yfinance补下载 |
| 个股K线停在2月20日 | screener | 高（信号过时） | yfinance增量更新 |

**Phase 0（写代码前）必须先做的数据准备**：
1. 补XLC历史数据 → sector_etf_prices.csv
2. 全量更新ETF+VIX到最新日期
3. 增量更新个股K线到最新日期
4. 补齐缺失的65只个股K线
