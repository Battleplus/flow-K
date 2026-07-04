# Flow-K: Statistical K-line Strategy Research Tool

A focused K-line (candlestick) pattern research tool for the FLOW AI Hackathon Trader track. Define visually recognizable candlestick signals, scan historical data for occurrences, and compute forward returns, win rates, and drawdowns &mdash; all backed by hard statistics rather than black-box predictions.

## Core Philosophy

> Turn a trader&rsquo;s visual K-line reading experience into testable, statistical, and explainable rules.

- **Rule-based signals**: 4 classic candlestick patterns with clear, auditable conditions
- **Statistical validation**: Scan every historical occurrence, compute 1d/3d/5d/10d/20d forward performance
- **No prediction, just statistics**: Signal appears &rarr; check how it performed historically &rarr; inform current decision

## Quick Start

```bash
pip install -r requirements.txt

# CLI analysis
python src/analyzer.py NVDA

# Web dashboard
python src/api.py
# Open http://localhost:5000
```

## CLI Usage

```bash
# Analyze a single stock (default 3-year lookback)
python src/analyzer.py NVDA

# Named parameter style
python src/analyzer.py --ticker AAPL

# Specific signals only
python src/analyzer.py NVDA --signals big_bull_breakout,support_bounce

# Batch analyze multiple stocks
python src/analyzer.py --batch AAPL,MSFT,GOOGL

# List all available tickers
python src/analyzer.py --list

# Custom lookback period
python src/analyzer.py NVDA --lookback 1

# Skip chart generation
python src/analyzer.py NVDA --no-chart
```

## Web Dashboard

The dashboard provides an interactive UI for exploring K-line signals across all S&P 500 stocks.

```bash
python src/api.py
```

Features:
- Dropdown stock selector with all 472 S&P 500 tickers
- Real-time overview cards (latest close, triggered signals, data range)
- Composite judgment banner (bullish / neutral / bearish)
- Signal statistics table with win rates, average returns, and max drawdown
- Interactive K-line chart with signal markers and MA overlays
- Forward 5-day return distribution charts per signal

## Four Core Signals

| Signal | Description | Direction |
|--------|-------------|-----------|
| **Big Bull Breakout** | Daily return &gt; 3%, close breaks above 20-day high, volume confirmation | Bullish |
| **Support Bounce** | Low near 20-day low, close &gt; open, volume above 80% MA | Bullish |
| **Long Lower Shadow** | Lower shadow &gt; 2&times; body, close &gt; open | Bullish |
| **Reversal After Decline** | 3-day decline followed by close above yesterday&rsquo;s high, volume spike | Bullish |

## Project Structure

```
flow/
├── SP500_Historical_Data.csv   # Historical OHLCV data (self-provided)
├── requirements.txt            # pandas, numpy, matplotlib, flask
├── .gitignore
├── README.md
├── src/
│   ├── analyzer.py             # CLI entry point
│   ├── api.py                  # Flask web API + dashboard server
│   ├── data_loader.py          # Data loading & column normalization
│   ├── patterns.py             # Candlestick signal definitions & indicators
│   ├── stats.py                # Forward return statistics engine
│   ├── chart.py                # K-line chart + return distribution plots
│   └── report.py               # Markdown report generation
└── web/
    └── index.html              # Dashboard frontend (zero JS dependencies)
```

## Output

Running the analyzer produces:

```
outputs/
├── charts/
│   ├── NVDA_kline_signals.png          # K-line chart with signals + MAs
│   └── NVDA_signal_xxx_fwd5d_dist.png  # Per-signal 5-day return distribution
└── report_NVDA.md                      # Full Markdown analysis report
```

## Data Requirements

- CSV with columns: `Ticker`, `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`
- Must include adjusted close prices (`Adj Close`) to account for stock splits
- Should include delisted stocks to avoid survivorship bias
- Default file: `SP500_Historical_Data.csv` in the project root

## Dependencies

- Python &ge; 3.10
- pandas &ge; 2.0
- numpy &ge; 1.24
- matplotlib &ge; 3.7
- flask &ge; 3.0

## License

MIT
