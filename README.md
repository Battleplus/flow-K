# Flow-K: K-line Trend & Slope Analysis Tool

A focused technical analysis tool that fetches real-time stock data, calculates multi-period moving average slopes, identifies trend direction, and visualizes everything with professional candlestick charts.

## What It Does

- Fetches real-time OHLCV data via **yfinance** (no local CSV needed)
- Computes MA5/MA10/MA20/MA60 and their **slopes** (linear regression angle)
- Judges **trend direction**: rising / falling / flat for each MA
- Detects **MA alignment**: bullish (多头排列) / bearish (空头排列) / choppy
- Finds recent **MA crosses** (golden cross / death cross)
- Identifies **support & resistance** levels
- Generates professional **candlestick charts** with colored MA curves using mplfinance

## Quick Start

```bash
pip install -r requirements.txt

# CLI
python src/analyzer.py NVDA

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
- Overall trend card (bullish / bearish / sideways)
- MA alignment card (bullish alignment / bearish alignment)
- Slope panel showing each MA''s slope %, direction, and current value
- Support & resistance panel
- Recent golden/death cross list
- Professional candlestick chart with colored MA overlays

## Project Structure

```
flow/
├── requirements.txt
├── README.md
├── src/
│   ├── analyzer.py          # CLI entry point
│   ├── api.py               # Flask API + dashboard server
│   ├── data_loader.py       # yfinance data fetching
│   ├── indicators.py        # MA, slope, trend, crosses, support/resistance
│   └── chart.py             # mplfinance candlestick charts
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
