"""数据加载模块 - 本地 CSV 优先，yfinance 备用"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "SP500_Historical_Data.csv"

COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
REQUIRED = ["Ticker", "Date"] + COLUMNS


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名"""
    df = df.copy()
    renames = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("ticker", "symbol"):
            renames[c] = "Ticker"
        elif cl in ("date", "datetime"):
            renames[c] = "Date"
        elif cl == "open":
            renames[c] = "Open"
        elif cl == "high":
            renames[c] = "High"
        elif cl == "low":
            renames[c] = "Low"
        elif cl == "close":
            renames[c] = "Close"
        elif cl == "volume":
            renames[c] = "Volume"
        elif "adj" in cl and "close" in cl:
            renames[c] = "Adj Close"
    df = df.rename(columns=renames)
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]
    return df


def fetch_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """加载数据：优先本地 CSV，失败则 yfinance"""
    t = ticker.strip().upper()

    # 尝试本地 CSV
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH, parse_dates=["Date"])
        df = _normalize_columns(df)
        sub = df[df["Ticker"] == t].copy()
        if not sub.empty:
            sub = sub.sort_values("Date").reset_index(drop=True)
            # 根据 period 截取
            period_map = {"1y": 365, "6mo": 180, "3mo": 90, "1mo": 30, "5y": 1825, "max": 99999}
            days = period_map.get(period, 365)
            sub = sub[sub["Date"] >= sub["Date"].max() - pd.Timedelta(days=days)]
            sub["Date"] = sub["Date"].dt.date
            return sub[REQUIRED]

    # Fallback: yfinance
    import yfinance as yf
    stock = yf.Ticker(t)
    df = stock.history(period=period, interval=interval)
    if df.empty:
        raise ValueError(f"未获取到数据: {t}")
    df = df[COLUMNS].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df.reset_index()
    df["Date"] = df["Date"].dt.date
    return df
