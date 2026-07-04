# FLOW AI — 数据来源文档

> 本文档描述 FLOW AI 交易分析平台的全部数据来源、数据格式、加载机制和依赖关系。

---

## 目录

- [1. 数据架构总览](#1-数据架构总览)
- [2. 本地数据文件](#2-本地数据文件)
- [3. 外部数据源](#3-外部数据源)
- [4. 数据加载机制](#4-数据加载机制)
- [5. 因子体系](#5-因子体系)
- [6. 策略体系](#6-策略体系)
- [7. 信号系统](#7-信号系统)
- [8. Walk-Forward 验证数据流](#8-walk-forward-验证数据流)
- [9. 三维分析数据流](#9-三维分析数据流)
- [10. 数据更新与维护](#10-数据更新与维护)

---

## 1. 数据架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                      数据输入层                              │
│                                                             │
│  ┌─────────────────────┐    ┌──────────────────────────┐   │
│  │   本地 CSV 文件      │    │   yfinance (外部 API)     │   │
│  │                     │    │                          │   │
│  │  SP500_Historical   │    │  实时报价                 │   │
│  │  company_info       │    │  日内分时                 │   │
│  │  sector_etf_prices  │    │  市场指数                 │   │
│  │  vix_data           │    │  本地数据不足时补充下载    │   │
│  └────────┬────────────┘    └───────────┬──────────────┘   │
│           │                             │                   │
│           └──────────┬──────────────────┘                   │
│                      ▼                                      │
│             ┌────────────────┐                              │
│             │  data_loader   │  统一数据加载层               │
│             │  fetch_data()  │  本地优先 → yfinance 补充     │
│             └───────┬────────┘                              │
│                     ▼                                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                    数据处理层                         │  │
│  │                                                      │  │
│  │  factors.py     →  40 个技术因子计算                  │  │
│  │  patterns.py    →  34 个 K 线信号检测                 │  │
│  │  strategies.py  →  20 个策略信号生成                  │  │
│  │  walkforward.py →  Walk-Forward 滚动验证              │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 数据优先级

1. **本地 CSV** — 优先读取（历史日线数据，覆盖 2000-2026，472 只 S&P500 股票）
2. **yfinance 补充** — 当本地数据覆盖不足 period 的 40% 时，从 yfinance 下载补齐
3. **yfinance 实时** — 实时报价、日内分时、市场指数快照完全依赖 yfinance

---

## 2. 本地数据文件

### 2.1 `SP500_Historical_Data.csv` — S&P500 历史日线

| 属性 | 值 |
|------|-----|
| 路径 | 项目根目录 |
| 大小 | ~142 MB |
| 格式 | CSV |
| 股票数 | 472 只 |
| 时间范围 | 2000-01-01 ~ 2026-07 |
| 每股票约 | ~6,500 条记录 |

**字段定义**：

| 列名 | 类型 | 说明 |
|------|------|------|
| `Ticker` | string | 股票代码（如 `NVDA`） |
| `Date` | string | 日期（`YYYY-MM-DD`） |
| `Open` | float | 开盘价 |
| `High` | float | 最高价 |
| `Low` | float | 最低价 |
| `Close` | float | 收盘价 |
| `Adj Close` | float | 复权收盘价 |
| `Volume` | int | 成交量 |

**数据特点**：
- 多股票合并在同一文件，通过 `Ticker` 列区分
- 已包含复权价格（Adj Close）
- 个别股票数据起始时间不同（如 META/TSLA 上市较晚）

---

### 2.2 `data/company_info.csv` — 公司信息

| 属性 | 值 |
|------|-----|
| 路径 | `data/` 目录 |
| 记录数 | 5,029 家公司 |
| 来源 | Wikipedia S&P500 列表 + yfinance 补充 |

**字段定义**：

| 列名 | 类型 | 说明 |
|------|------|------|
| `Symbol` | string | 股票代码 |
| `Company` | string | 公司全称 |
| `Sector` | string | 板块（11 大板块） |
| `Sub-Industry` | string | 子行业 |
| `Headquarters` | string | 总部地址 |
| `DateAdded` | string | 加入 S&P500 日期 |
| `CIK` | string | SEC CIK 编号 |
| `Founded` | string | 成立年份 |
| `BusinessSummary` | string | 业务描述（yfinance 获取，用于 TF-IDF 语义匹配） |

**11 大板块**：Information Technology / Health Care / Financials / Consumer Discretionary / Communication Services / Industrials / Consumer Staples / Energy / Real Estate / Utilities / Materials

---

### 2.3 `data/sector_etf_prices.csv` — 板块 ETF 价格

| 属性 | 值 |
|------|-----|
| 路径 | `data/` 目录 |
| 覆盖 | 11 只 SPDR 板块 ETF + SPY + QQQ |
| 时间范围 | 截至 2026-07-02 |

**包含的 ETF**：

| Ticker | 板块 | ETF 名称 |
|--------|------|----------|
| XLK | Information Technology | Technology Select Sector SPDR |
| XLF | Financials | Financial Select Sector SPDR |
| XLE | Energy | Energy Select Sector SPDR |
| XLI | Industrials | Industrial Select Sector SPDR |
| XLY | Consumer Discretionary | Consumer Discretionary Select Sector SPDR |
| XLP | Consumer Staples | Consumer Staples Select Sector SPDR |
| XLV | Health Care | Health Care Select Sector SPDR |
| XLB | Materials | Materials Select Sector SPDR |
| XLU | Utilities | Utilities Select Sector SPDR |
| XLRE | Real Estate | Real Estate Select Sector SPDR |
| XLC | Communication Services | Communication Services Select Sector SPDR |
| SPY | 大盘基准 | SPDR S&P 500 ETF |
| QQQ | 科技基准 | Invesco QQQ Trust |

**字段定义**：`Ticker, Sector, Date, Open, High, Low, Close, Adj Close, Volume`

---

### 2.4 `data/vix_data.csv` — VIX 波动率指数

| 属性 | 值 |
|------|-----|
| 路径 | `data/` 目录 |
| Ticker | `^VIX` |
| 用途 | 市场状态识别、风险偏好判断 |

**字段定义**：`Ticker, Date, Open, High, Low, Close, Adj Close, Volume`

---

### 2.5 `data/event_sector_mapping.json` — 事件关键词映射

| 属性 | 值 |
|------|-----|
| 路径 | `data/` 目录 |
| 用途 | 消息面分析的关键词匹配规则 |

**结构**：
```json
{
  "version": "1.0",
  "description": "事件关键词到板块ETF的映射表",
  "sector_etf_map": {
    "Technology": "XLK",
    "Energy": "XLE",
    ...
  },
  "event_keywords": [
    {
      "keywords": ["AI芯片", "GPU", "半导体", "数据中心"],
      "sector": "Information Technology",
      "etf": "XLK",
      "direction": "bullish",
      "weight": 1.5,
      "note": "AI/半导体利好"
    },
    ...  // 共 17 组
  ]
}
```

---

### 2.6 `data/tfidf_model.pkl` — TF-IDF 语义模型

| 属性 | 值 |
|------|-----|
| 路径 | `data/` 目录 |
| 格式 | Python Pickle |
| 训练数据 | `company_info.csv` 中 470 只 S&P500 股票的 BusinessSummary |
| 向量维度 | 2,000 |
| ngram | 1-2 |
| 用途 | 消息面分析的语义匹配路径 |

**模型结构**：
```python
{
    "vectorizer": TfidfVectorizer,      # scikit-learn TF-IDF 向量化器
    "sector_vectors": {                 # 板块 → 平均向量
        "Information Technology": np.ndarray,
        ...
    },
    "company_sectors": {                # 股票 → 板块映射
        "NVDA": "Information Technology",
        ...
    }
}
```

---

### 2.7 `outputs/stock_recommendations.json` — 预生成推荐

| 属性 | 值 |
|------|-----|
| 路径 | `outputs/` 目录 |
| 生成方式 | `python src/full_scan.py` |
| 用途 | `/api/recommendations` 的数据源 |

**结构**：
```json
{
  "generated_at": "2025-07-04T12:00:00",
  "news_text": "...",
  "sector_snapshot": {...},
  "all_results": [...],
  "by_sector": {...},
  "top_10": [...]
}
```

---

## 3. 外部数据源

### 3.1 yfinance

**概述**：Yahoo Finance 的非官方 Python API，本项目唯一的外部数据源。

**依赖版本**：`yfinance >= 1.5.0`

**使用场景**：

| 场景 | 函数 | 说明 |
|------|------|------|
| 历史日线补充 | `yf.Ticker(t).history(period, interval)` | 本地 CSV 数据不足时 |
| 实时报价 | `yf.Ticker(t).info` / `yf.Ticker(t).fast_info` | `/api/quote` 和 `/api/quotes` |
| 日内分时 | `yf.Ticker(t).history(period, interval)` | `/api/intraday`，interval=1m/5m/15m/30m/60m |
| 市场指数 | `yf.Ticker("^GSPC").fast_info` | `/api/market_snapshot` |

**覆盖的市场指数**：

| 代码 | 名称 |
|------|------|
| `^GSPC` | S&P 500 |
| `^NDX` | Nasdaq 100 |
| `^DJI` | Dow Jones Industrial Average |
| `^RUT` | Russell 2000 |
| `^VIX` | CBOE Volatility Index |

### 3.2 缓存策略

| 数据类型 | 缓存位置 | TTL | 说明 |
|----------|----------|-----|------|
| 实时报价 | 内存 | 30 秒 | 避免频繁请求 yfinance |
| 日内分时 | 内存 | 120 秒 | 分时数据延迟可接受 |
| 历史日线 | 本地 CSV | 永久 | 手动通过 `update_data.py` 更新 |

---

## 4. 数据加载机制

### 4.1 核心函数 `fetch_data(ticker, period, interval)`

位于 `src/data_loader.py`，是全部数据访问的统一入口。

```
fetch_data(ticker, period="1y", interval="1d")
    │
    ├─ 1. 读取本地 SP500_Historical_Data.csv
    │     ├─ 按 Ticker 过滤
    │     ├─ 按 period 截取日期范围
    │     └─ 返回 DataFrame
    │
    ├─ 2. 检查覆盖率
    │     └─ 如果本地数据覆盖天数 < period 的 40%
    │         └─ 调用 yfinance 下载补充
    │
    └─ 3. 完全回退
          └─ 如果本地 CSV 不存在或无该股票
              └─ yfinance 直接下载
```

### 4.2 Period 映射

| period 参数 | 目标交易日数 |
|-------------|-------------|
| `1mo` | 30 |
| `3mo` | 90 |
| `6mo` | 180 |
| `1y` | 365 |
| `2y` | 730 |
| `3y` | 1095 |
| `5y` | 1825 |
| `max` | 99999 |

### 4.3 Walk-Forward 持仓周期自动选 period

| hold_profile | 自动 period | 原因 |
|--------------|------------|------|
| `short` | `1y` | 训练窗口 120 天 + 多折测试 |
| `medium` | `2y` | 训练窗口 252 天 + 3 折测试 |
| `long` | `3y` | 训练窗口 360 天 + 2 折测试 |

### 4.4 其他数据加载函数

| 函数 | 说明 |
|------|------|
| `fetch_intraday(ticker, interval, period)` | 日内分时数据，直接调用 yfinance |
| `fetch_quote(ticker)` | 单股实时报价，调用 yfinance fast_info |
| `fetch_market_indices()` | 大盘指数 + 板块 ETF 快照，批量调用 yfinance |
| `load_company_info()` | 加载 `company_info.csv` |
| `load_sector_etfs()` | 加载 `sector_etf_prices.csv` |
| `load_vix()` | 加载 `vix_data.csv` |

---

## 5. 因子体系

位于 `src/factors.py`，共 **40 个因子**，分 6 大类。

### 5.1 趋势因子（10 个）

| 因子 | 说明 | 计算方式 |
|------|------|----------|
| `ma5` | 5日均线 | Close.rolling(5).mean() |
| `ma10` | 10日均线 | Close.rolling(10).mean() |
| `ma20` | 20日均线 | Close.rolling(20).mean() |
| `ma60` | 60日均线 | Close.rolling(60).mean() |
| `ema12` | 12日指数均线 | Close.ewm(span=12).mean() |
| `ema26` | 26日指数均线 | Close.ewm(span=26).mean() |
| `ema50` | 50日指数均线 | Close.ewm(span=50).mean() |
| `ema100` | 100日指数均线 | Close.ewm(span=100).mean() |
| `trend_slope_ma20` | MA20斜率 | 线性回归斜率 |
| `ma_ribbon_score` | 均线排列评分 | MA5/10/20/60 多空排列打分 |

### 5.2 波动率因子（9 个）

| 因子 | 说明 | 计算方式 |
|------|------|----------|
| `bb_upper` | 布林带上轨 | MA20 + 2σ |
| `bb_mid` | 布林带中轨 | MA20 |
| `bb_lower` | 布林带下轨 | MA20 - 2σ |
| `keltner_upper` | Keltner上轨 | EMA20 + 2×ATR |
| `keltner_mid` | Keltner中轨 | EMA20 |
| `keltner_lower` | Keltner下轨 | EMA20 - 2×ATR |
| `atr_upper_20` | ATR上轨 | Close + 2×ATR(20) |
| `atr_lower_20` | ATR下轨 | Close - 2×ATR(20) |
| `hist_vol_20` | 20日历史波动率 | std(log_returns) × √252 |

### 5.3 动量因子（9 个）

| 因子 | 说明 | 计算方式 |
|------|------|----------|
| `rsi_14` | 14日RSI | 相对强弱指标 |
| `rsi_28` | 28日RSI | 长周期RSI |
| `stochastic_k` | Stochastic %K | (Close - LowLow) / (HighHigh - LowLow) × 100 |
| `stochastic_d` | Stochastic %D | %K 的 3 日均线 |
| `macd_line` | MACD线 | EMA12 - EMA26 |
| `macd_signal` | MACD信号线 | MACD线的 9 日 EMA |
| `macd_hist` | MACD柱状图 | MACD线 - 信号线 |
| `roc_10` | 10日变动率 | (Close - Close.shift(10)) / Close.shift(10) |
| `williams_r` | Williams %R | (HighHigh - Close) / (HighHigh - LowLow) × (-100) |

### 5.4 成交量因子（4 个）

| 因子 | 说明 | 计算方式 |
|------|------|----------|
| `vwap` | 成交量加权均价 | Σ(Price × Volume) / ΣVolume |
| `obv` | 能量潮 | 累积成交量指标 |
| `volume_ratio` | 量比 | 当日量 / MA(20日量) |
| `ad_line` | 累积/派发线 | CLV × Volume 累积 |

### 5.5 结构因子（7 个）

| 因子 | 说明 | 计算方式 |
|------|------|----------|
| `ichimoku_conv` | 一目均衡转换线 | (9日高 + 9日低) / 2 |
| `ichimoku_base` | 一目均衡基准线 | (26日高 + 26日低) / 2 |
| `ichimoku_span_a` | 先行带A | (转换线 + 基准线) / 2 |
| `ichimoku_span_b` | 先行带B | (52日高 + 52日低) / 2 |
| `donchian_high` | 唐奇安上轨 | 20日最高价 |
| `donchian_low` | 唐奇安下轨 | 20日最低价 |
| `pivot_pp` | 枢轴点 | (High + Low + Close) / 3 |

### 5.6 自适应因子（4 个）

| 因子 | 说明 | 计算方式 |
|------|------|----------|
| `adx_14` | 14日ADX | 趋势强度指标（不分方向） |
| `di_plus` | +DI | 上升方向指标 |
| `di_minus` | -DI | 下降方向指标 |
| `trend_strength` | 趋势强度评分 | ADX + DI 综合评分 |

### 5.7 未来函数防护

以下因子在计算时使用 **rolling 窗口** 而非全局统计，确保不引入未来数据（look-ahead bias）：

| 因子 | 原实现（有未来函数） | 修正后（无未来函数） |
|------|---------------------|---------------------|
| `bb_squeeze` 中的带宽分位 | `bbw.quantile(0.1)` 全局分位数 | `bbw.rolling(120).quantile(0.1)` |
| `detect_market_regime` 中的带宽归一化 | `bbw_series.min()/max()` 全局极值 | rolling 窗口归一化 |

---

## 6. 策略体系

位于 `src/strategies.py`，共 **20 个策略**，按 5 大类组织。

### 6.1 持仓周期预设（HOLD_PROFILES）

| Profile | hold_min | hold_max | 止损 | 止盈 | 标签 |
|---------|----------|----------|------|------|------|
| `short` | 1天 | 15天 | 4% | 8% | 短线 (1-15天) |
| `medium` | 30天 | 180天 | 无 | 无 | 中线 (30-180天,无止损) |
| `long` | 180天 | 360天 | 无 | 无 | 长线 (180-360天,无止损) |

### 6.2 策略信号枚举

| 值 | 含义 | 数值 |
|----|------|------|
| `STRONG_SELL` | 强烈卖出 | -2 |
| `SELL` | 卖出 | -1 |
| `HOLD` | 持有 | 0 |
| `BUY` | 买入 | +1 |
| `STRONG_BUY` | 强烈买入 | +2 |

### 6.3 策略完整清单

#### A. 趋势跟踪（5 个）

| ID | 名称 | 核心因子 | 止损 | 止盈 |
|----|------|----------|------|------|
| `ma_ribbon` | 均线多头排列 | MA5/10/20/60 + RSI + 量比 + 斜率 | 6% | 12% |
| `ema_cross` | EMA交叉 | EMA12/26/50 | 6% | 10% |
| `trend_breakout` | 趋势突破 | 布林带 + ADX + 量比 | 7% | 15% |
| `ma_pullback` | 均线回踩 | MA5/10/20/60 + RSI | 5% | 10% |
| `trend_acceleration` | 趋势加速度 | MA20斜率变化 | 6% | 12% |

#### B. 动量反转（5 个）

| ID | 名称 | 核心因子 | 止损 | 止盈 |
|----|------|----------|------|------|
| `rsi_extreme` | RSI超买超卖 | RSI(14) | 6% | 8% |
| `bb_mean_reversion` | 布林带回归 | 布林带 + RSI | 5% | 8% |
| `stochastic_cross` | Stochastic双线交叉 | %K / %D | 5% | 8% |
| `macd_divergence` | MACD背离 | MACD柱状图 | 6% | 12% |
| `williams_r_reversal` | Williams %R反转 | Williams %R | 5% | 8% |

#### C. 成交量确认（3 个）

| ID | 名称 | 核心因子 | 止损 | 止盈 |
|----|------|----------|------|------|
| `volume_breakout` | 放量突破 | MA排列 + 量比 | 7% | 15% |
| `obv_divergence` | OBV背离 | OBV | 5% | 10% |
| `vwap_mean_reversion` | VWAP回归 | VWAP + 量比 | 5% | 8% |

#### D. 波动率突破（3 个）

| ID | 名称 | 核心因子 | 止损 | 止盈 |
|----|------|----------|------|------|
| `bb_squeeze` | 布林带收缩突破 | 布林带 + 量比 | 8% | 18% |
| `keltner_breakout` | Keltner突破 | Keltner通道 + ADX | 7% | 15% |
| `atr_band` | ATR轨道触及 | ATR上下轨 + RSI | 6% | 10% |

#### E. 多重确认（4 个）

| ID | 名称 | 核心因子 | 止损 | 止盈 |
|----|------|----------|------|------|
| `triple_confirm_long` | 三重确认做多 | MA + RSI + ADX/DI | 6% | 15% |
| `triple_confirm_short` | 三重确认做空 | MA + RSI + ADX/DI | 6% | 12% |
| `ichimoku_cloud` | 一目均衡云 | Ichimoku四线 | 7% | 15% |
| `full_monty` | 全维度确认 | MA + RSI + MACD + 量比 + ADX | 7% | 15% |

### 6.4 回测引擎参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `initial_capital` | $100,000 | 初始资金 |
| `commission` | 0.1% | 单边手续费 |
| `worst_case` | true | 保守估计（信号日最低价买入） |
| `stop_loss_pct` | 策略定义 | 止损百分比 |
| `take_profit_pct` | 策略定义 | 止盈百分比 |
| `hold_days_min` | 持仓周期定义 | 最短持仓天数 |
| `hold_days_max` | 持仓周期定义 | 最长持仓天数 |

---

## 7. 信号系统

位于 `src/patterns.py`，共 **34 个 K 线信号**，分 5 大类。

### 7.1 趋势线信号（6 个）

| 信号 | 类型 | 说明 |
|------|------|------|
| `trendline_break_up` | 看多 | 突破上升趋势线 |
| `trendline_break_down` | 看空 | 跌破上升趋势线 |
| `trendline_bounce_up` | 看多 | 上升趋势线反弹 |
| `trendline_bounce_down` | 看空 | 下降趋势线反弹 |
| `channel_top_touch` | 看空 | 触及通道上轨 |
| `channel_bottom_touch` | 看多 | 触及通道下轨 |

### 7.2 斜率信号（6 个）

| 信号 | 类型 | 说明 |
|------|------|------|
| `slope_accelerating_up` | 看多 | 上升斜率加速 |
| `slope_decelerating_up` | 看空 | 上升斜率减速 |
| `slope_accelerating_down` | 看空 | 下降斜率加速 |
| `slope_decelerating_down` | 看多 | 下降斜率减速 |
| `slope_divergence_bullish` | 看多 | 斜率看多背离 |
| `slope_divergence_bearish` | 看空 | 斜率看空背离 |

### 7.3 曲线/MA 信号（8 个）

| 信号 | 类型 | 说明 |
|------|------|------|
| `ma_golden_cross` | 看多 | 均线金叉 |
| `ma_death_cross` | 看空 | 均线死叉 |
| `ma_convergence` | 中性 | 均线收敛 |
| `ma_divergence_bullish` | 看多 | 均线看多发散 |
| `ma_divergence_bearish` | 看空 | 均线看空发散 |
| `ma_ribbon_bullish` | 看多 | 均线多头排列 |
| `ma_ribbon_bearish` | 看空 | 均线空头排列 |
| `bb_squeeze_long` / `bb_squeeze_short` | 看多/看空 | 布林带收缩 |

### 7.4 经典 K 线形态（10 个）

| 信号 | 类型 | 说明 |
|------|------|------|
| `dark_cloud_cover` | 看空 | 乌云盖顶 |
| `piercing_line` | 看多 | 刺透形态 |
| `morning_star` | 看多 | 晨星 |
| `evening_star` | 看空 | 黄昏星 |
| `hammer` | 看多 | 锤子线 |
| `shooting_star` | 看空 | 射击之星 |
| `bullish_engulfing` | 看多 | 看多吞没 |
| `bearish_engulfing` | 看空 | 看空吞没 |
| `three_white_soldiers` | 看多 | 三白兵 |
| `three_black_crows` | 看空 | 三乌鸦 |

### 7.5 增强版信号（4 个）

| 信号 | 类型 | 说明 |
|------|------|------|
| `big_bull_breakout` | 看多 | 大阳突破 |
| `support_bounce` | 看多 | 支撑反弹 |
| `reversal_after_decline` | 看多 | 跌后反转 |

---

## 8. Walk-Forward 验证数据流

### 8.1 核心流程

```
输入: DataFrame (含 OHLCV + 因子), hold_profile
  │
  ├─ 1. 预计算全部 20 个策略的信号序列 (一次性)
  │
  ├─ 2. 根据数据长度自适应窗口
  │     short  → train=120, test=30
  │     medium → train=252, test=63
  │     long   → train=360, test=90
  │
  ├─ 3. 滚动循环
  │     for each fold:
  │       ├─ 训练窗口 [t, t+train]
  │       │   └─ 对全部 20 策略跑回测 → 按夏普排序 → 选 Top K
  │       │
  │       ├─ 测试窗口 [t+train, t+train+test]
  │       │   └─ 对选出的 K 个策略纯回测 (不重新选策略)
  │       │
  │       ├─ 并行分仓: 每策略分得 1/K 资金
  │       └─ 滑动窗口: t += test
  │
  ├─ 4. 拼接所有测试段权益曲线 (复利)
  │
  └─ 5. 计算统计指标
        ├─ 样本外总收益/年化/夏普/回撤/胜率
        ├─ 样本内平均收益
        ├─ 买入持有收益
        └─ 过拟合比率 = |样本内收益 / 样本外收益|
```

### 8.2 过拟合比率解读

| 比率范围 | 含义 |
|----------|------|
| < 1.5 | 较稳健，样本内外差异小 |
| 1.5 ~ 3.0 | 存在一定过拟合 |
| > 3.0 | 严重过拟合，策略不可靠 |

### 8.3 与传统回测的对比

| 维度 | 传统回测 | Walk-Forward |
|------|----------|-------------|
| 数据使用 | 全量数据选策略 + 全量数据回测 | 训练窗口选策略 + 测试窗口回测 |
| 数据重叠 | 完全重叠（自己考自己） | 严格分离 |
| 过拟合风险 | 极高（系统性高估） | 较低（贴近真实） |
| 典型结果 | TSLA中线 +114% | TSLA中线 -13.7% |

---

## 9. 三维分析数据流

### 9.1 全链路流程

```
输入: 新闻文本 + period + top_n
  │
  ├─ 第一层: 消息面分析 (news_analysis.py)
  │   ├─ 路径1: 关键词匹配 (0.4 权重)
  │   │   └─ event_sector_mapping.json → 17组关键词 → 板块评分
  │   │
  │   ├─ 路径2: TF-IDF 语义匹配 (0.6 权重)
  │   │   └─ tfidf_model.pkl → 余弦相似度 → 板块评分
  │   │
  │   └─ 融合: sector_scores = kw_score × 0.4 + tfidf_score × 0.6
  │       输出: 11 板块评分 (-2 ~ +2), 方向, top_sectors
  │
  ├─ 第二层: 板块 ETF 轮动 (sector_rotation.py)
  │   ├─ 趋势评分: etf_trend_score() → -1/0/+1
  │   ├─ 相对强度: relative_strength() → RS vs SPY
  │   ├─ 资金流向: flow_judgment() → 趋势 × RS 矩阵
  │   └─ 输出: flow_score (-2 ~ +2), rotation_signals, market_state
  │
  ├─ 目标板块 = 消息面利好 ∩ 资金流入
  ├─ 否决板块 = 消息面利空 ∩ 资金流出
  │
  ├─ 第三层: 个股 K 线扫描 (screener.py)
  │   ├─ 对目标板块成分股逐只分析
  │   │   ├─ fetch_data → add_all_factors → detect_all
  │   │   ├─ 策略共识 → consensus_direction
  │   │   └─ 回测 → avg_sharpe_top5, pf_return
  │   │
  │   └─ 输出: stock_score (个股维度评分)
  │
  └─ 三维综合评分
      final_score = news_score × 0.25 + sector_score × 0.35 + stock_score × 0.40
      │
      └─ 一票否决: 否决板块内的股票直接排除
```

### 9.2 三维权重设计逻辑

| 维度 | 权重 | 设计逻辑 |
|------|------|----------|
| 消息面 | 0.25 | 信息时效性强但噪音大，权重最低 |
| 板块轮动 | 0.35 | 资金流向是中期驱动力，权重中等 |
| 个股 K 线 | 0.40 | 最终落脚点是个股，权重最高 |

### 9.3 市场状态识别

`detect_market_regime()` 综合以下信号判断市场状态：

| 信号 | 来源 | 权重 |
|------|------|------|
| MA20 趋势方向 | 均线斜率 | 高 |
| 布林带宽度归一化 | 波动率状态 | 中 |
| ADX 趋势强度 | 趋势/震荡判断 | 中 |
| VIX 水平 | 市场恐慌度 | 低 |

**输出状态**：

| 状态 | 条件 | 含义 |
|------|------|------|
| `risk_on` | MA↑ + ADX>25 + VIX<15 | 牛市，风险偏好 |
| `risk_off` | MA↓ + ADX>25 + VIX>25 | 熊市，避险模式 |
| `cautious_bull` | MA↑ + ADX<20 | 谨慎看多 |
| `cautious_bear` | MA↓ + ADX<20 | 谨慎看空 |
| `neutral` | 混合信号 | 中性 |

---

## 10. 数据更新与维护

### 10.1 数据更新脚本

| 脚本 | 用途 | 命令 |
|------|------|------|
| `src/update_data.py` | 更新 ETF/VIX 价格到最新 | `python src/update_data.py` |
| `src/download_company_info.py` | 重新爬取 S&P500 公司列表 | `python src/download_company_info.py` |
| `src/add_summaries.py` | 为 company_info 添加 BusinessSummary | `python src/add_summaries.py` |
| `src/fill_summaries.py` | 补充缺失的 BusinessSummary | `python src/fill_summaries.py` |
| `src/full_scan.py` | 全板块三维扫描 → recommendations.json | `python src/full_scan.py` |

### 10.2 数据更新频率建议

| 数据 | 更新频率 | 说明 |
|------|----------|------|
| `SP500_Historical_Data.csv` | 季度 | 季度更新即可，日常用 yfinance 补充 |
| `sector_etf_prices.csv` | 月度 | 板块轮动分析需要近期数据 |
| `vix_data.csv` | 月度 | 市场状态判断需要近期数据 |
| `company_info.csv` | 半年 | S&P500 成分股变动频率低 |
| `tfidf_model.pkl` | 半年 | 跟随 company_info 更新 |
| `event_sector_mapping.json` | 按需 | 关键词规则人工维护 |

### 10.3 依赖清单

| 依赖 | 版本要求 | 用途 |
|------|----------|------|
| `pandas` | >= 2.0.0 | 数据处理核心 |
| `numpy` | >= 1.24.0 | 数值计算 |
| `matplotlib` | >= 3.7.0 | 图表绘制 |
| `mplfinance` | >= 0.12.0 | K 线图绘制 |
| `yfinance` | >= 1.5.0 | 外部数据源 |
| `flask` | >= 3.0.0 | Web API 服务 |
| `scipy` | >= 1.10.0 | 统计计算 |
| `scikit-learn` | 未列入 requirements.txt | TF-IDF 向量化（消息面分析必需） |
| `fastdtw` | 可选 | 形态搜索的 DTW 算法（可选） |

> **注意**：`scikit-learn` 未列入 `requirements.txt`，但 `news_analysis.py` 依赖它。首次部署时需手动安装：`pip install scikit-learn`

---

## 附录：项目文件结构

```
flow/
├── SP500_Historical_Data.csv          # S&P500 历史日线 (142MB)
├── requirements.txt                   # Python 依赖
├── README.md                          # 项目概述
├── KLINE_STRATEGY_FRAMEWORK.md        # K线策略框架文档
├── THREE_DIM_FRAMEWORK.md             # 三维分析框架文档
├── THREE_DIM_ARCHITECTURE.md          # 三维架构详细文档
├── SCREENER_ARCHITECTURE.md           # 筛选器架构文档
├── MODULE_DESIGN.md                   # 模块设计文档
├── docs/
│   ├── API_REFERENCE.md               # API 接口文档
│   └── DATA_SOURCES.md                # 数据来源文档（本文档）
├── data/
│   ├── company_info.csv               # 公司信息 (5029家)
│   ├── sector_etf_prices.csv          # 板块ETF价格
│   ├── vix_data.csv                   # VIX波动率
│   ├── event_sector_mapping.json      # 事件关键词映射
│   └── tfidf_model.pkl                # TF-IDF模型
├── src/
│   ├── api.py                         # Flask API + 仪表盘 (27路由)
│   ├── analyzer.py                    # CLI 分析入口
│   ├── data_loader.py                 # 数据加载模块
│   ├── indicators.py                  # 技术指标
│   ├── patterns.py                    # K线信号 (34个)
│   ├── factors.py                     # 因子计算 (40个)
│   ├── chart.py                       # 图表生成
│   ├── strategies.py                  # 策略引擎 (20个)
│   ├── walkforward.py                 # Walk-Forward 滚动验证
│   ├── strategy_portfolio.py          # 策略组合+市场状态
│   ├── pattern_search.py              # 形态搜索
│   ├── stats.py                       # 统计模块
│   ├── report.py                      # 报告生成
│   ├── screener.py                    # 批量筛选器
│   ├── three_dim_analyzer.py          # 三维分析入口
│   ├── news_analysis.py               # 消息面分析 (第一层)
│   ├── sector_rotation.py             # 板块ETF轮动 (第二层)
│   ├── full_scan.py                   # 全板块扫描脚本
│   ├── update_data.py                 # 数据更新脚本
│   ├── download_company_info.py       # 公司信息下载
│   ├── add_summaries.py               # 摘要补充
│   └── fill_summaries.py              # 摘要填充
├── web/
│   └── index.html                     # 仪表盘前端 (单文件)
├── outputs/
│   ├── backtest_report.html           # 回测报告页
│   ├── walkforward_report.html        # WF验证报告
│   ├── stock_recommendations.json     # 推荐结果
│   └── *.png                          # 图表输出
└── .workbuddy/
    └── memory/                        # 项目记忆
        ├── MEMORY.md                  # 长期记忆
        └── YYYY-MM-DD.md              # 每日工作日志
```
