"""数据加载模块 - 读取 SP500_Historical_Data.csv，统一列名，按 ticker 过滤"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = ROOT / "SP500_Historical_Data.csv"
PROCESSED_DATA = ROOT / "data" / "processed" / "prices.parquet"

REQUIRED_COLS = ["Ticker", "Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]


def load_raw() -> pd.DataFrame:
    """加载原始 CSV 并统一列名"""
    if not RAW_DATA.exists():
        raise FileNotFoundError(f"未找到数据文件: {RAW_DATA}")
    df = pd.read_csv(RAW_DATA, parse_dates=["Date"])
    # 统一列名
    renames = {}
    for c in df.columns:
        cl = c.strip()
        if cl.lower() in ("ticker", "symbol"):
            renames[c] = "Ticker"
        elif cl.lower() in ("date", "datetime"):
            renames[c] = "Date"
        elif cl.lower() in ("open",):
            renames[c] = "Open"
        elif cl.lower() in ("high",):
            renames[c] = "High"
        elif cl.lower() in ("low",):
            renames[c] = "Low"
        elif cl.lower() in ("close",):
            renames[c] = "Close"
        elif "adj" in cl.lower() and "close" in cl.lower():
            renames[c] = "Adj Close"
        elif cl.lower() in ("volume",):
            renames[c] = "Volume"
    df = df.rename(columns=renames)
    # 确保有 Adj Close
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]
    return df[REQUIRED_COLS]


def load_ticker(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """过滤单个股票，按日期排序，重置索引"""
    sub = df[df["Ticker"] == ticker].copy()
    sub = sub.sort_values("Date").reset_index(drop=True)
    if sub.empty:
        raise ValueError(f"未找到股票: {ticker}")
    return sub


def list_tickers(df: pd.DataFrame) -> list:
    """列出所有可用的 ticker"""
    return sorted(df["Ticker"].unique().tolist())
