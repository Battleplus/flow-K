# Flow-K: K-line Trend, Slope & Signal Analysis Tool

A focused technical analysis tool that detects **trend lines**, **slope dynamics**, **MA curve patterns**, and **candlestick reversal patterns** from real-time stock data, then visualizes everything with professional charts.

## What It Does

- Fetches real-time OHLCV data via **yfinance** or local S&P500 CSV
- Computes MA5/MA10/MA20/MA60 and their **slopes** (linear regression angle)
- Detects **trend lines** (diagonal support/resistance) and price breakouts/bounces
- Identifies **slope changes**: acceleration, deceleration, bullish/bearish divergence
- Analyzes **MA curves**: golden/death cross, convergence/divergence, bullish/bearish alignment, Bollinger squeeze
- Detects **classic candlestick patterns** (enhanced with confirmation): 乌云盖顶, 刺透, 晨星, 黄昏之星, 锤头, 射击之星, 吞没, 白三兵, 三只乌鸦, etc.
- Generates a **composite signal score** (看多/看空/中性) based on all detected factors
- Generates professional **candlestick charts** with trend lines, MA curves, signal markers

## Quick Start

```bash
pip install -r requirements.txt

# CLI
python src/analyzer.py NVDA

# Backtest a specific strategy
python src/analyzer.py NVDA --strategy full_monty

# Web dashboard
python src/api.py
# Open http://localhost:5000
```

## CLI Usage

```bash
python src/analyzer.py NVDA              # Default: 1 year daily
python src/analyzer.py NVDA --period 6mo  # 6 months
python src/analyzer.py 0700.HK            # HK stock
python src/analyzer.py AAPL --mas 5,10,20,60
```

## Web Dashboard

```bash
python src/api.py
```

Interactive dashboard with:
- Stock ticker input + period selector
- Overall trend card (上升 / 下降 / 横盘)
- MA alignment card (多头排列 / 空头排列 / 交织)
- Signal score card with bullish/bearish verdict
- Slope panel showing each MA's slope %, direction, and current value
- Support & resistance panel
- Recent golden/death cross list
- **Active signals panel** showing triggered trend/slope/curve/pattern signals
- Professional candlestick chart with colored MA overlays, trend lines, and signal markers

## Signal System

The system now registers **34 signals** across 4 categories:

| Category | Examples | Count |
|----------|----------|-------|
| **Trend Lines** | trendline_break_up, trendline_bounce_up, channel_top_touch | 6 |
| **Slope Dynamics** | slope_accelerating_up, slope_decelerating_up, bullish/bearish divergence | 6 |
| **Curve / MA** | ma_golden_cross, ma_convergence, ma_divergence, bb_squeeze | 8 |
| **Candlestick Patterns** | morning_star, bearish_engulfing, hammer, three_white_soldiers | 10 |
| **Enhanced Classic** | big_bull_breakout, support_bounce, reversal_after_decline | 4 |

Each signal includes direction (bullish/bearish/neutral) and strength score, and contributes to a composite verdict.

## Strategy Engine (NEW)

The system now treats every indicator "line" as a **factor** and combines multiple factors into actionable **strategies**.

- **20 strategies** across 5 categories: Trend Following, Momentum Reversal, Volume Confirmation, Volatility Breakout, Multi-Confirmation
- Each strategy emits **BUY / STRONG_BUY / SELL / STRONG_SELL** signals
- Built-in backtest engine with stop-loss, take-profit, and max-hold-days
- Strategy ranking by Sharpe ratio, return, win rate, max drawdown, profit factor
- **Multi-strategy consensus** aggregates all 20 strategies into a single score

Example CLI output:

```bash
$ python src/analyzer.py AAPL --period 6mo

🏆 Top 5 Strategies by Sharpe:
  1. Stochastic双线交叉   收益 11.4%  夏普 2.06  胜率 67%  交易3次
  2. OBV背离             收益  8.5%  夏普 1.06  胜率 75%  交易4次
  3. 均线多头排列         收益  5.7%  夏普 1.03  胜率100%  交易2次

🗳 多策略共识: 偏空 (评分: -2)
```

## Web Dashboard

```bash
python src/api.py
```

Interactive dashboard with:
- Stock ticker input + period selector
- Overall trend card (上升 / 下降 / 横盘)
- MA alignment card (多头排列 / 空头排列 / 交织)
- Signal score card with bullish/bearish verdict
- **Strategy panel** showing backtest leaderboard + multi-strategy consensus
- **Strategy signal chart** showing BUY/SELL markers for top-performing strategies
- Slope panel, support/resistance, pattern search, factor table
- Professional candlestick charts with 30+ indicator lines and multi-panel mode

## Project Structure

```
flow/
├── requirements.txt
├── README.md
├── KLINE_STRATEGY_FRAMEWORK.md
├── data/
│   ├── sector_etf_prices.csv
│   ├── event_sector_mapping.json
│   └── vix_data.csv
├── src/
│   ├── analyzer.py          # CLI entry point
│   ├── api.py               # Flask API + dashboard server
│   ├── data_loader.py       # yfinance / local CSV data loading
│   ├── indicators.py        # MA, slope, trend, crosses, support/resistance
│   ├── patterns.py          # Trend lines, slope, curves, candlestick patterns
│   ├── factors.py           # 85+ factor calculation (MA, volatility, momentum, volume, structure, adaptive)
│   ├── pattern_search.py    # Historical / cross-stock pattern similarity search
│   ├── strategies.py        # Multi-factor strategy engine + backtest framework
│   ├── stats.py             # Signal forward-return statistics
│   ├── report.py            # Markdown report generation
│   └── chart.py             # mplfinance candlestick charts with multi-panel support
└── web/
    └── index.html           # Dashboard frontend
```

## Dependencies

- Python >= 3.10
- pandas, numpy, scipy
- matplotlib, mplfinance
- yfinance
- flask

## License

MIT
