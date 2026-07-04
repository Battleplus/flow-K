# FLOW AI — API 接口文档

> 基地址：`http://localhost:5000`  
> 全部接口为 JSON 格式（`Content-Type: application/json`），POST 请求需传 JSON body。  
> 启动方式：`python src/api.py`

---

## 目录

- [1. 页面与静态资源](#1-页面与静态资源)
- [2. 股票列表与基础分析](#2-股票列表与基础分析)
- [3. 策略引擎](#3-策略引擎)
- [4. 天级与日内数据](#4-天级与日内数据)
- [5. 因子数据](#5-因子数据)
- [6. 三维分析](#6-三维分析)
- [7. 形态搜索](#7-形态搜索)
- [附录：枚举值定义](#附录枚举值定义)

---

## 1. 页面与静态资源

### 1.1 `GET /` — 仪表盘首页

返回 `web/index.html` 渲染的主仪表盘页面。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| 无 | — | — | — |

**返回**：HTML 页面

---

### 1.2 `GET /report` — 回测报告页

返回 `outputs/backtest_report.html` 静态报告页面。

**返回**：HTML 页面

---

### 1.3 `GET /api/chart` — 获取图表图片

返回 `matplotlib` 生成的 PNG 图表。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `path` | string | 是 | 图片文件的绝对路径 |

**返回**：`image/png`

---

## 2. 股票列表与基础分析

### 2.1 `GET /api/tickers` — 获取可用股票列表

从 `company_info.csv` 与 `SP500_Historical_Data.csv` 交叉获取可用股票。

| 参数 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `q` | string | 否 | 无 | 搜索关键词（匹配 symbol 或 company name） |

**返回示例**：
```json
{
  "count": 200,
  "tickers": [
    {"symbol": "NVDA", "name": "NVIDIA Corporation", "sector": "Information Technology"},
    {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Information Technology"}
  ]
}
```

---

### 2.2 `GET /api/analyze/<ticker>` — 综合分析

对指定股票执行完整分析管线：数据加载 → 因子计算 → 信号检测 → 策略回测 → 图表生成。

| 参数 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `ticker` | path | 是 | — | 股票代码（如 `NVDA`） |
| `period` | query | 否 | `1y` | 数据周期：`1mo/3mo/6mo/1y/2y/3y/5y/max` |
| `mode` | query | 否 | `multi` | 图表模式：`multi`（多面板）/ `single`（单面板） |

**返回字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `ticker` | string | 股票代码 |
| `period` | string | 数据周期 |
| `mode` | string | 图表模式 |
| `data_start` | string | 数据起始日期 |
| `data_end` | string | 数据结束日期 |
| `records` | int | 数据记录数 |
| `factor_count` | int | 计算的因子数量 |
| `analysis` | object | 趋势/斜率/支撑压力等分析结果 |
| `signals` | object | 信号汇总（见下） |
| `chart_path` | string | 图表 PNG 绝对路径 |

**`signals` 结构**：
```json
{
  "score": 3,
  "verdict": "偏多",
  "bullish_count": 5,
  "bearish_count": 2,
  "neutral_count": 3,
  "active": [
    {"name": "均线金叉", "type": "bullish", "date": "2025-06-15", "description": "..."}
  ]
}
```

---

### 2.3 `GET /api/signals/<ticker>` — 信号汇总

返回当前活跃的技术信号列表。

| 参数 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `ticker` | path | 是 | — | 股票代码 |
| `period` | query | 否 | `1y` | 数据周期 |

**返回示例**：
```json
{
  "ticker": "NVDA",
  "period": "1y",
  "signals": [
    {"name": "ma_golden_cross", "type": "bullish", "category": "曲线/MA", "date": "2025-06-10"}
  ]
}
```

---

## 3. 策略引擎

### 3.1 `GET /api/strategies` — 列出全部策略

返回 20 个策略的元信息。

**返回示例**：
```json
{
  "count": 20,
  "strategies": {
    "ma_ribbon": {
      "name": "均线多头排列",
      "category": "趋势跟踪",
      "description": "MA5 > MA10 > MA20 > MA60 多头排列，RSI 未超买",
      "factors_used": ["ma5", "ma10", "ma20", "ma60", "rsi_14", "volume_ratio", "trend_slope_ma20"],
      "stop_loss_pct": 0.06,
      "take_profit_pct": 0.12
    }
  }
}
```

---

### 3.2 `GET /api/backtest/<ticker>` — Walk-Forward 回测（主接口）

**核心接口**。返回 Walk-Forward 样本外回测结果（主）+ 传统全量回测（参考），用于对比检测过拟合。

| 参数 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `ticker` | path | 是 | — | 股票代码 |
| `hold` | query | 否 | `medium` | 持仓周期：`short`(1-15天) / `medium`(30-180天) / `long`(180-360天) |
| `period` | query | 否 | 自动 | 自动根据 hold 选择：short→1y, medium→2y, long→3y |
| `strategy` | query | 否 | 无 | 指定单个策略 ID（留空则全部20个策略参与） |
| `capital` | query | 否 | `100000` | 初始资金（美元） |
| `worst_case` | query | 否 | `true` | 是否使用保守估计（信号日取最低价买入） |
| `limit` | query | 否 | `0` | 限价百分比（0=不限价） |

**返回字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `ticker` | string | 股票代码 |
| `period` | string | 数据周期 |
| `hold_profile` | string | 持仓周期标识 |
| `hold_label` | string | 持仓周期中文标签 |
| `worst_case` | bool | 是否保守估计 |
| `n_data_days` | int | 数据总天数 |
| `data_start` | string | 数据起始日期 |
| `data_end` | string | 数据结束日期 |
| `walk_forward` | object | **Walk-Forward 样本外结果（主）** |
| `in_sample_reference` | object | 传统全量回测结果（参考） |
| `buy_hold_return` | float | 买入持有收益率 |
| `overfit_gap` | float | 过拟合差距（样本内收益 - 样本外收益） |

**`walk_forward` 结构**：
```json
{
  "total_return": -0.0065,        // 样本外总收益率
  "annual_return": -0.0043,       // 年化收益率
  "sharpe_ratio": -0.02,          // 夏普比率
  "max_drawdown": -0.15,          // 最大回撤
  "win_rate": 0.42,               // 胜率
  "total_trades": 12,             // 总交易次数
  "n_folds": 3,                   // 滚动折数
  "in_sample_return": 0.3804,     // 样本内平均收益率
  "buy_hold_return": 0.4523,      // 同期买入持有收益
  "overfit_ratio": 5.85,          // 过拟合比率（>3 为严重过拟合）
  "folds": [                      // 各折详情
    {
      "fold_id": 1,
      "train_start": "2023-07-01",
      "train_end": "2024-07-01",
      "test_start": "2024-07-02",
      "test_end": "2024-10-01",
      "selected_strategies": ["macd_divergence", "bb_mean_reversion", ...],
      "in_sample_return": 0.35,
      "out_sample_return": -0.02,
      "out_sample_sharpe": -0.1,
      "out_sample_max_dd": -0.08,
      "out_sample_trades": 4,
      "out_sample_win_rate": 0.25
    }
  ],
  "equity_curve": [...]           // 拼接的样本外权益曲线
}
```

**`in_sample_reference` 结构**：
```json
{
  "strategies": {
    "macd_divergence": {
      "total_return": 1.03,
      "sharpe_ratio": 1.2,
      "max_drawdown": -0.15,
      "win_rate": 0.55,
      "total_trades": 8
    }
  },
  "ranked_by_sharpe": [
    {"strategy_id": "macd_divergence", "name": "MACD背离", "sharpe_ratio": 1.2, "total_return": 1.03}
  ]
}
```

---

### 3.3 `GET /api/consensus/<ticker>` — 多策略共识信号

对 20 个策略的信号做共识投票，返回当前综合判定和近 60 天趋势。

| 参数 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `ticker` | path | 是 | — | 股票代码 |
| `period` | query | 否 | `1y` | 数据周期 |

**返回示例**：
```json
{
  "ticker": "NVDA",
  "consensus": {
    "date": "2025-07-04",
    "aggregate_score": 0.35,
    "bullish_ratio": 0.6,
    "consensus": "偏多",
    "active_strategies": ["ma_ribbon", "ema_cross", "volume_breakout"]
  },
  "trend": [
    {"date": "2025-06-01", "score": 0.1, "ratio": 0.5},
    {"date": "2025-06-02", "score": 0.2, "ratio": 0.55}
  ]
}
```

**共识判定规则**：
| aggregate_score | consensus |
|-----------------|-----------|
| ≥ 1.0 | 强烈偏多 |
| 0.3 ~ 1.0 | 偏多 |
| -0.3 ~ 0.3 | 中性 |
| -1.0 ~ -0.3 | 偏空 |
| ≤ -1.0 | 强烈偏空 |

---

### 3.4 `GET /api/portfolio/<ticker>` — 组合回测

以并行分仓模式运行策略组合回测，返回 Walk-Forward 组合结果（主）+ 传统组合（参考）+ 市场状态。

| 参数 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `ticker` | path | 是 | — | 股票代码 |
| `hold` | query | 否 | `medium` | 持仓周期 |
| `period` | query | 否 | 自动 | 自动根据 hold 选择 |
| `worst_case` | query | 否 | `true` | 保守估计 |
| `limit` | query | 否 | `0` | 限价百分比 |

**返回字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `hold_params` | object | 持仓参数（hold_min, hold_max, stop_loss_pct, take_profit_pct） |
| `walk_forward` | object | WF 组合样本外结果 |
| `in_sample_reference` | object | 传统组合参考（含 4 种组合方式结果） |
| `market_regime` | string | 市场状态（见附录） |
| `buy_hold_return` | float | 买入持有收益 |
| `overfit_gap` | float | 过拟合差距 |

**`in_sample_reference.portfolios` 包含 4 种组合方式**：
- `equal_weight`：等权分配
- `sharpe_weighted`：夏普比率加权
- `regime_aware`：市场状态驱动
- `vol_target`：波动率目标

---

### 3.5 `GET /api/strategy_chart/<ticker>` — 策略信号标注图

生成标注了多个策略 BUY/SELL 信号的 K 线图。

| 参数 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `ticker` | path | 是 | — | 股票代码 |
| `period` | query | 否 | `1y` | 数据周期 |
| `strategy` | query | 否 | 无 | 指定策略 ID（留空则显示全部活跃策略信号） |

**返回示例**：
```json
{
  "ticker": "NVDA",
  "chart_path": "C:/Users/.../outputs/NVDA_strategy_chart.png",
  "strategies_shown": ["macd_divergence", "bb_mean_reversion", "ema_cross"]
}
```

---

### 3.6 `GET /api/walkforward/<ticker>` — Walk-Forward 独立验证

独立的 Walk-Forward 滚动验证接口，支持对比模式（传统回测 vs WF）。

| 参数 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `ticker` | path | 是 | — | 股票代码 |
| `period` | query | 否 | `2y` | 数据周期 |
| `hold` | query | 否 | `medium` | 持仓周期 |
| `top_k` | query | 否 | `5` | 每折选出的策略数量 |
| `compare` | query | 否 | `true` | 是否返回对比模式 |

**对比模式返回**（`compare=true`）：
```json
{
  "in_sample": 0.38,         // 传统全量回测收益
  "out_of_sample": -0.0065,  // WF 样本外收益
  "buy_hold": 0.4523,        // 买入持有收益
  "overfit_gap": 0.3865,     // 过拟合差距
  "overfit_ratio": 5.85,     // 过拟合比率
  "fold_details": [...]      // 各折详情
}
```

**非对比模式返回**（`compare=false`）：完整 `WalkForwardResult` 字典，字段同 `/api/backtest` 的 `walk_forward`。

---

## 4. 天级与日内数据

### 4.1 `GET /api/daily/<ticker>` — 天级 K 线数据

返回包含 OHLCV + 全部因子列的天级数据。

| 参数 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `ticker` | path | 是 | — | 股票代码 |
| `period` | query | 否 | `1y` | 数据周期 |
| `limit` | query | 否 | 无 | 限制返回条数（取最后 N 条） |
| `fields` | query | 否 | 全部 | 逗号分隔的字段名（如 `date,close,volume,rsi_14`） |

**返回示例**：
```json
{
  "ticker": "NVDA",
  "period": "1y",
  "total_records": 252,
  "returned_records": 252,
  "available_fields": ["date", "open", "high", "low", "close", "volume", "ma5", "rsi_14", ...],
  "data": [
    {"date": "2024-07-01", "open": 120.5, "high": 122.0, "low": 119.8, "close": 121.2, "volume": 3000000, "rsi_14": 55.3}
  ]
}
```

---

### 4.2 `GET /api/daily_summary/<ticker>` — 单日综合摘要

返回最新交易日（或指定日期）的 OHLCV、因子、信号、策略共识。

| 参数 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `ticker` | path | 是 | — | 股票代码 |
| `period` | query | 否 | `1y` | 数据周期（用于计算因子） |
| `date` | query | 否 | 最新 | 指定日期（YYYY-MM-DD） |

**返回字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `ohlcv` | object | {open, high, low, close, volume} |
| `change` | float | 涨跌额 |
| `change_pct` | float | 涨跌幅 |
| `factor_count` | int | 因子数量 |
| `factors` | object | {因子名: {category, description, value}} |
| `signals` | object | 活跃信号 |
| `strategy_consensus` | object | 策略共识 |

---

### 4.3 `GET /api/market_snapshot` — 大盘指数+板块快照

返回主要指数和板块 ETF 的实时快照。

**返回示例**：
```json
{
  "timestamp": "2025-07-04T18:00:00",
  "indices": {
    "SPX": {"name": "S&P 500", "price": 5500.0, "change_pct": 0.5},
    "VIX": {"name": "Volatility Index", "price": 12.5, "change_pct": -2.1}
  },
  "sectors": [
    {"ticker": "XLK", "name": "Technology", "price": 220.0, "change_pct": 1.2},
    {"ticker": "XLE", "name": "Energy", "price": 95.0, "change_pct": -0.5}
  ]
}
```

**覆盖指数**：^GSPC(SPX), ^NDX(NDX), ^DJI(DJI), ^RUT(RUT), ^VIX(VIX)  
**覆盖板块**：11 只 SPDR 板块 ETF（XLK/XLF/XLE/XLI/XLY/XLP/XLV/XLB/XLU/XLRE/XLC）

---

### 4.4 `GET /api/intraday/<ticker>` — 日内分时数据

返回日内分时 K 线数据（通过 yfinance 实时获取）。

| 参数 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `ticker` | path | 是 | — | 股票代码 |
| `interval` | query | 否 | `5m` | 分时粒度：`1m/2m/5m/15m/30m/60m/90m` |
| `period` | query | 否 | `1d` | 数据周期：`1d/5d/1mo` |

**返回字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `current_price` | float | 当前价格 |
| `prev_close` | float | 前收价 |
| `change` | float | 涨跌额 |
| `change_pct` | float | 涨跌幅 |
| `data_count` | int | 数据条数 |
| `data` | array | [{datetime, open, high, low, close, volume}] |

> 缓存 TTL：120 秒

---

### 4.5 `GET /api/quote/<ticker>` — 实时报价

返回单只股票的完整实时报价信息。

**返回字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `price` | float | 当前价格 |
| `change` / `change_pct` | float | 涨跌额/幅 |
| `prev_close` / `open` | float | 前收/开盘 |
| `day_high` / `day_low` | float | 日内最高/最低 |
| `volume` / `avg_volume` | int | 成交量/均量 |
| `market_cap` | float | 市值 |
| `pe_ratio` | float | 市盈率 |
| `52w_high` / `52w_low` | float | 52周最高/最低 |
| `bid` / `ask` | float | 买一/卖一 |
| `market_state` | string | 市场状态（`OPEN`/`CLOSED`/`PRE`/`POST`） |

> 缓存 TTL：30 秒

---

### 4.6 `GET /api/quotes` — 批量实时报价

| 参数 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `tickers` | query | 是 | — | 逗号分隔的股票代码，最多 20 个 |

**返回示例**：
```json
{
  "count": 3,
  "quotes": {
    "NVDA": {"price": 121.2, "change_pct": 0.5, ...},
    "AAPL": {"price": 210.0, "change_pct": -0.3, ...}
  },
  "timestamp": "2025-07-04T18:00:00"
}
```

---

## 5. 因子数据

### 5.1 `GET /api/factors/<ticker>` — 获取最新因子值

返回最新交易日的全部因子值（40 个因子）。

| 参数 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `ticker` | path | 是 | — | 股票代码 |
| `period` | query | 否 | `1y` | 数据周期（用于计算因子） |

**返回示例**：
```json
{
  "ticker": "NVDA",
  "date": "2025-07-04",
  "factors": {
    "rsi_14": {"category": "动量", "description": "14日相对强弱指标", "value": 55.3},
    "macd_hist": {"category": "动量", "description": "MACD柱状图", "value": 0.5},
    "bb_upper": {"category": "波动率", "description": "布林带上轨", "value": 130.0}
  }
}
```

**因子类别**：趋势(10) / 波动率(9) / 动量(9) / 成交量(4) / 结构(7) / 自适应(4)

---

## 6. 三维分析

### 6.1 `POST /api/news_analyze` — 消息面分析（第一层）

输入新闻文本，输出 11 个板块的情绪评分。

**请求 Body**：
```json
{
  "text": "NVIDIA发布新一代AI芯片，性能提升3倍"
}
```

**返回字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `sector_scores` | object | {板块名: 评分(-2~+2)} |
| `direction` | string | 整体方向：`bullish`/`bearish`/`neutral` |
| `summary` | string | 分析摘要 |
| `keyword_result` | object | 关键词匹配结果（0.4 权重） |
| `tfidf_result` | object | TF-IDF 语义匹配结果（0.6 权重） |
| `top_sectors` | array | 评分最高的板块列表 |

**双路融合机制**：
- 关键词匹配（权重 0.4）：17 组预设关键词映射
- TF-IDF 语义匹配（权重 0.6）：基于 470 只股票 BusinessSummary 训练的 2000 维向量，余弦相似度

---

### 6.2 `GET /api/sector_snapshot` — 板块 ETF 轮动快照（第二层）

返回 11 只 SPDR 板块 ETF 的趋势、相对强度、资金流向。

**返回字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `sectors` | array | 各板块详情列表 |
| `top_sectors` | array | 资金流入板块（flow_score ≥ 1） |
| `weak_sectors` | array | 资金流出板块（flow_score ≤ -1） |
| `rotation_signals` | array | 轮动信号（early_rotation / potential_reversal / distribution_risk） |
| `market_state` | string | 市场状态（见附录） |
| `bullish_count` / `bearish_count` | int | 看多/看空板块数 |

**单板块结构**：
```json
{
  "sector": "Technology",
  "etf": "XLK",
  "trend_score": 1,
  "trend_label": "上升趋势",
  "ma_alignment": "多头排列",
  "consensus_direction": "偏多",
  "rs_score": 0.8,
  "rs_short_score": 0.6,
  "flow_score": 1.5,
  "flow_label": "资金流入",
  "flow_direction": "流入",
  "price": 220.0,
  "change_5d": 2.5,
  "signals": [...]
}
```

**资金流向评分矩阵**（趋势 × 相对强度）：

| | RS 强 (+1) | RS 中 (0) | RS 弱 (-1) |
|---|---|---|---|
| 趋势上 (+1) | +2 强流入 | +1 流入 | 0 中性 |
| 趋势平 (0) | +1 流入 | 0 中性 | -1 流出 |
| 趋势下 (-1) | 0 中性 | -1 流出 | -2 强流出 |

---

### 6.3 `POST /api/three_dim_scan` — 三维全链路扫描（第三层）

消息面 → 板块轮动 → 个股 K 线，三维评分排序输出推荐股票。

**请求 Body**：
```json
{
  "news_text": "NVIDIA发布新AI芯片，数据中心需求激增",
  "period": "6mo",
  "top_n": 10
}
```

| 参数 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `news_text` | string | 是 | — | 新闻文本 |
| `period` | string | 否 | `6mo` | 个股分析周期 |
| `top_n` | int | 否 | `10` | 返回前 N 只（上限 20） |

**返回字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `news_analysis` | object | 消息面分析结果 |
| `sector_snapshot` | object | 板块轮动快照 |
| `target_sectors` | array | 目标板块（消息面 ∩ 资金流入） |
| `veto_sectors` | array | 否决板块（消息面利空 ∩ 资金流出） |
| `stocks` | array | 推荐股票列表（按 final_score 降序） |
| `total_scanned` | int | 扫描股票总数 |
| `summary` | object | 汇总统计 |

**单只股票结构**：
```json
{
  "ticker": "NVDA",
  "sector": "Technology",
  "news_score": 1.5,        // 消息面评分（0.25 权重）
  "sector_score": 1.2,      // 板块评分（0.35 权重）
  "stock_score": 0.8,       // 个股评分（0.40 权重）
  "final_score": 1.13,      // 三维综合评分
  "consensus": "偏多"
}
```

**三维权重**：`final_score = news_score × 0.25 + sector_score × 0.35 + stock_score × 0.40`

**一票否决机制**：若股票所在板块同时满足「消息面利空 + 资金流出」，则该股票被否决，不进入推荐列表。

---

### 6.4 `GET /api/three_dim/<ticker>` — 单股三维详情

返回单只股票的三维评分详情。

| 参数 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `ticker` | path | 是 | — | 股票代码 |
| `period` | query | 否 | `6mo` | 分析周期 |

**返回字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `news_score` | float | 消息面评分 |
| `sector_score` | float | 板块评分 |
| `stock_score` | float | 个股评分 |
| `final_score` | float | 综合评分 |
| `stock_info` | object | 个股详情（共识/策略/回测） |
| `sector_info` | object | 所在板块详情 |

---

### 6.5 `GET /api/recommendations` — 获取推荐结果

返回预生成的三维扫描推荐结果（来自 `outputs/stock_recommendations.json`）。

**返回**：预扫描的 JSON 数据，结构同 `/api/three_dim_scan` 的返回。

---

### 6.6 `POST /api/recommendations/run` — 触发三维全板块扫描

异步触发全板块三维扫描，生成新的推荐结果。

**请求 Body**：
```json
{
  "news_text": "美联储宣布降息50个基点",
  "top_n": 10
}
```

**返回字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `generated_at` | string | 生成时间 |
| `news_text` | string | 输入新闻 |
| `sector_snapshot` | object | 板块快照 |
| `all_results` | array | 全部扫描结果 |
| `by_sector` | object | 按板块分组的结果 |
| `top_10` | array | 前 10 推荐 |

---

## 7. 形态搜索

### 7.1 `GET /api/pattern_search/<ticker>` — 形态相似搜索

在历史数据或跨股票中搜索与当前走势相似的时段。

| 参数 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `ticker` | path | 是 | — | 股票代码 |
| `period` | query | 否 | `1y` | 数据周期 |
| `window` | query | 否 | `30` | 匹配窗口长度（天） |
| `top_n` | query | 否 | `5` | 返回前 N 个相似结果 |
| `type` | query | 否 | `historical` | `historical`（历史相似）/ `cross_stock`（跨股相似） |

**返回字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `ticker` | string | 股票代码 |
| `type` | string | 搜索类型 |
| `similar_periods` / `similar_stocks` | array | 相似结果列表 |
| `pattern_features` | object | 当前走势特征 |

**相似结果结构**：
```json
{
  "correlation": 0.85,
  "start_date": "2023-01-15",
  "end_date": "2023-02-15",
  "future_return_5d": 3.2,
  "future_return_10d": 5.1
}
```

**算法**：z-score 归一化 + Pearson 相关系数（可选 DTW 动态时间规整）。

---

## 附录：枚举值定义

### 持仓周期（hold_profile）

| 值 | 标签 | hold_min | hold_max | 止损 | 止盈 |
|----|------|----------|----------|------|------|
| `short` | 短线 (1-15天) | 1天 | 15天 | 4% | 8% |
| `medium` | 中线 (30-180天,无止损) | 30天 | 180天 | 无 | 无 |
| `long` | 长线 (180-360天,无止损) | 180天 | 360天 | 无 | 无 |

### Walk-Forward 窗口配置

| hold_profile | 训练窗口 | 测试窗口 | 最少数据 | 典型折数 |
|--------------|----------|----------|----------|----------|
| `short` | 120天 | 30天 | 180天 | ~12 折 |
| `medium` | 252天 | 63天 | 400天 | ~3 折 |
| `long` | 360天 | 90天 | 540天 | ~2 折 |

### 策略信号枚举（Signal）

| 值 | 含义 |
|----|------|
| `STRONG_SELL` (-2) | 强烈卖出 |
| `SELL` (-1) | 卖出 |
| `HOLD` (0) | 持有 |
| `BUY` (+1) | 买入 |
| `STRONG_BUY` (+2) | 强烈买入 |

### 市场状态（market_regime）

| 值 | 含义 |
|----|------|
| `risk_on` | 风险偏好（牛市） |
| `risk_off` | 避险模式（熊市） |
| `cautious_bull` | 谨慎看多 |
| `cautious_bear` | 谨慎看空 |
| `neutral` | 中性 |

### 20 个策略 ID 速查

| 类别 | 策略 ID |
|------|---------|
| 趋势跟踪 (5) | `ma_ribbon`, `ema_cross`, `trend_breakout`, `ma_pullback`, `trend_acceleration` |
| 动量反转 (5) | `rsi_extreme`, `bb_mean_reversion`, `stochastic_cross`, `macd_divergence`, `williams_r_reversal` |
| 成交量确认 (3) | `volume_breakout`, `obv_divergence`, `vwap_mean_reversion` |
| 波动率突破 (3) | `bb_squeeze`, `keltner_breakout`, `atr_band` |
| 多重确认 (4) | `triple_confirm_long`, `triple_confirm_short`, `ichimoku_cloud`, `full_monty` |

### 数据周期映射

| period 参数 | 交易日数 |
|-------------|----------|
| `1mo` | 30 |
| `3mo` | 90 |
| `6mo` | 180 |
| `1y` | 365 |
| `2y` | 730 |
| `3y` | 1095 |
| `5y` | 1825 |
| `max` | 99999 |

### HTTP 状态码

| 状态码 | 含义 |
|--------|------|
| 200 | 成功 |
| 400 | 请求参数错误 |
| 404 | 股票代码不存在或数据不足 |
| 500 | 服务器内部错误（数据加载失败、因子计算异常等） |
