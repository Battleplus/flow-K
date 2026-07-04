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
            period_map = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "3y": 1095, "5y": 1825, "max": 99999}
            days = period_map.get(period, 365)
            cutoff = sub["Date"].max() - pd.Timedelta(days=days)
            sub_cut = sub[sub["Date"] >= cutoff].copy()
            # CSV 数据覆盖不足时，用 yfinance 补充
            if len(sub_cut) < days * 0.4 and days > 365:
                try:
                    import yfinance as yf
                    stock = yf.Ticker(t)
                    yf_df = stock.history(period=period, interval="1d")
                    if not yf_df.empty and len(yf_df) > len(sub_cut):
                        yf_df = yf_df[["Open", "High", "Low", "Close", "Volume"]].copy()
                        yf_df.index = pd.to_datetime(yf_df.index)
                        yf_df["Date"] = yf_df.index.date
                        yf_df["Ticker"] = t
                        return yf_df[REQUIRED].sort_values("Date").reset_index(drop=True)
                except Exception:
                    pass
            sub_cut["Date"] = sub_cut["Date"].dt.date
            return sub_cut[REQUIRED]

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


# ═══════════════════════════════════════════════════════════════
# 日内数据 & 实时报价
# ═══════════════════════════════════════════════════════════════

import time
from functools import lru_cache

# 简单的内存缓存，避免频繁请求 yfinance
_QUOTE_CACHE: dict = {}
_QUOTE_CACHE_TTL = 30  # 实时报价缓存 30 秒
_INTRADAY_CACHE: dict = {}
_INTRADAY_CACHE_TTL = 120  # 日内数据缓存 2 分钟


def fetch_intraday(ticker: str, period: str = "1d", interval: str = "5m") -> dict:
    """
    获取日内分时数据
    
    参数:
        ticker:   股票代码
        period:   时间范围 (1d/5d/1mo)
        interval: K线粒度 (1m/2m/5m/15m/30m/60m/90m/1h)
    
    返回:
        {
            "ticker": "AAPL",
            "period": "1d",
            "interval": "5m",
            "current_price": 195.50,
            "prev_close": 194.80,
            "change": 0.70,
            "change_pct": 0.36,
            "data": [{"datetime": "2026-07-04 09:35", "open": ..., "high": ..., "low": ..., "close": ..., "volume": ...}, ...]
        }
    """
    import yfinance as yf
    
    t = ticker.strip().upper()
    cache_key = f"{t}_{period}_{interval}"
    
    # 检查缓存
    if cache_key in _INTRADAY_CACHE:
        ts, data = _INTRADAY_CACHE[cache_key]
        if time.time() - ts < _INTRADAY_CACHE_TTL:
            return data
    
    stock = yf.Ticker(t)
    
    # 获取日内数据
    df = stock.history(period=period, interval=interval)
    if df.empty:
        raise ValueError(f"未获取到日内数据: {t}")
    
    # 前一交易日收盘价
    prev_close = None
    try:
        info = stock.fast_info
        prev_close = getattr(info, "previous_close", None) or getattr(info, "regular_market_previous_close", None)
    except Exception:
        pass
    
    # 如果没有 fast_info，用前一天收盘
    if prev_close is None and period == "1d":
        try:
            hist_2d = stock.history(period="2d", interval="1d")
            if len(hist_2d) >= 2:
                prev_close = float(hist_2d["Close"].iloc[-2])
        except Exception:
            pass
    
    # 当前价
    current_price = float(df["Close"].iloc[-1])
    
    # 计算涨跌
    if prev_close and prev_close > 0:
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100
    else:
        change = 0.0
        change_pct = 0.0
        prev_close = current_price
    
    # 构建数据点列表
    data_points = []
    for idx, row in df.iterrows():
        dt = idx if hasattr(idx, "strftime") else pd.Timestamp(idx)
        data_points.append({
            "datetime": dt.strftime("%Y-%m-%d %H:%M") if hasattr(dt, "strftime") else str(dt)[:16],
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
            "volume": int(row["Volume"]),
        })
    
    result = {
        "ticker": t,
        "period": period,
        "interval": interval,
        "current_price": round(current_price, 2),
        "prev_close": round(float(prev_close), 2) if prev_close else None,
        "change": round(change, 2),
        "change_pct": round(change_pct, 2),
        "data_count": len(data_points),
        "data": data_points,
        "data_start": data_points[0]["datetime"] if data_points else None,
        "data_end": data_points[-1]["datetime"] if data_points else None,
    }
    
    # 写入缓存
    _INTRADAY_CACHE[cache_key] = (time.time(), result)
    
    return result


def fetch_quote(ticker: str) -> dict:
    """
    获取实时报价快照
    
    返回:
        {
            "ticker": "AAPL",
            "name": "Apple Inc.",
            "price": 195.50,
            "change": 0.70,
            "change_pct": 0.36,
            "prev_close": 194.80,
            "open": 194.90,
            "day_high": 196.00,
            "day_low": 194.50,
            "volume": 45678900,
            "avg_volume": 55000000,
            "market_cap": 3000000000000,
            "pe_ratio": 35.2,
            "52w_high": 210.00,
            "52w_low": 165.00,
            "bid": 195.45,
            "ask": 195.55,
            "currency": "USD",
            "exchange": "NASDAQ",
            "timestamp": "2026-07-04 16:42:00",
        }
    """
    import yfinance as yf
    
    t = ticker.strip().upper()
    
    # 检查缓存
    if t in _QUOTE_CACHE:
        ts, data = _QUOTE_CACHE[t]
        if time.time() - ts < _QUOTE_CACHE_TTL:
            return data
    
    stock = yf.Ticker(t)
    
    try:
        info = stock.info
    except Exception:
        info = {}
    
    try:
        fast = stock.fast_info
    except Exception:
        fast = None
    
    # 价格
    price = None
    if fast:
        price = getattr(fast, "last_price", None) or getattr(fast, "regular_market_price", None)
    if price is None:
        price = info.get("regularMarketPrice") or info.get("currentPrice")
    if price is None:
        # Fallback: 用最近一条日线收盘价
        try:
            hist = stock.history(period="1d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
        except Exception:
            price = 0.0
    
    price = float(price) if price else 0.0
    
    # 前收盘
    prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
    if prev_close is None and fast:
        prev_close = getattr(fast, "previous_close", None)
    if prev_close is None:
        prev_close = price
    
    prev_close = float(prev_close) if prev_close else price
    
    change = price - prev_close
    change_pct = (change / prev_close * 100) if prev_close else 0.0
    
    # 日内范围
    day_high = info.get("dayHigh") or info.get("regularMarketDayHigh")
    day_low = info.get("dayLow") or info.get("regularMarketDayLow")
    if fast:
        day_high = getattr(fast, "day_high", None) or day_high
        day_low = getattr(fast, "day_low", None) or day_low
    
    # 今开
    today_open = info.get("open") or info.get("regularMarketOpen")
    if fast:
        today_open = getattr(fast, "open", None) or today_open
    
    result = {
        "ticker": t,
        "name": info.get("longName") or info.get("shortName", ""),
        "price": round(price, 2),
        "change": round(change, 2),
        "change_pct": round(change_pct, 2),
        "prev_close": round(prev_close, 2),
        "open": round(float(today_open), 2) if today_open else None,
        "day_high": round(float(day_high), 2) if day_high else None,
        "day_low": round(float(day_low), 2) if day_low else None,
        "volume": info.get("volume") or info.get("regularMarketVolume", 0),
        "avg_volume": info.get("averageVolume") or info.get("averageDailyVolume10Day", 0),
        "market_cap": info.get("marketCap", 0),
        "pe_ratio": info.get("trailingPE") or info.get("forwardPE"),
        "52w_high": info.get("fiftyTwoWeekHigh"),
        "52w_low": info.get("fiftyTwoWeekLow"),
        "bid": info.get("bid"),
        "ask": info.get("ask"),
        "currency": info.get("currency", "USD"),
        "exchange": info.get("exchange", ""),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "market_state": info.get("marketState", "UNKNOWN"),
    }
    
    # 写入缓存
    _QUOTE_CACHE[t] = (time.time(), result)
    
    return result


def fetch_market_indices() -> dict:
    """
    获取主要市场指数快照
    
    返回:
        {
            "timestamp": "...",
            "indices": {
                "SPX": {"name": "S&P 500", "price": ..., "change_pct": ...},
                "NDX": {"name": "NASDAQ 100", ...},
                "DJI": {"name": "Dow Jones", ...},
                "RUT": {"name": "Russell 2000", ...},
                "VIX": {"name": "VIX", ...},
            },
            "sectors": [
                {"ticker": "XLF", "name": "金融", "price": ..., "change_pct": ...},
                ...
            ]
        }
    """
    import yfinance as yf
    
    # 主要指数
    index_map = {
        "^GSPC": "SPX",
        "^NDX": "NDX",
        "^DJI": "DJI",
        "^RUT": "RUT",
        "^VIX": "VIX",
    }
    
    index_names = {
        "SPX": "S&P 500",
        "NDX": "NASDAQ 100",
        "DJI": "道琼斯工业",
        "RUT": "Russell 2000",
        "VIX": "波动率指数",
    }
    
    # SPDR 板块 ETF
    sector_etfs = {
        "XLF": "金融",
        "XLK": "科技",
        "XLE": "能源",
        "XLV": "医疗",
        "XLI": "工业",
        "XLY": "可选消费",
        "XLP": "必选消费",
        "XLB": "原材料",
        "XLU": "公用事业",
        "XLRE": "房地产",
        "XLC": "通信服务",
    }
    
    indices = {}
    sectors = []
    
    # 批量获取
    all_tickers = list(index_map.keys()) + list(sector_etfs.keys())
    
    try:
        data = yf.download(all_tickers, period="2d", group_by="ticker", threads=True, progress=False)
    except Exception:
        # 逐个获取
        data = None
    
    for yf_ticker, short_name in index_map.items():
        try:
            if data is not None and yf_ticker in data.columns.levels[0]:
                close = float(data[yf_ticker]["Close"].dropna().iloc[-1])
                if len(data[yf_ticker]["Close"].dropna()) >= 2:
                    prev = float(data[yf_ticker]["Close"].dropna().iloc[-2])
                    chg_pct = (close - prev) / prev * 100
                else:
                    chg_pct = 0.0
            else:
                tick = yf.Ticker(yf_ticker)
                hist = tick.history(period="2d")
                close = float(hist["Close"].iloc[-1])
                prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else close
                chg_pct = (close - prev) / prev * 100
            
            indices[short_name] = {
                "name": index_names.get(short_name, short_name),
                "price": round(close, 2),
                "change_pct": round(chg_pct, 2),
            }
        except Exception:
            continue
    
    for etf, name_cn in sector_etfs.items():
        try:
            if data is not None and etf in data.columns.levels[0]:
                close = float(data[etf]["Close"].dropna().iloc[-1])
                if len(data[etf]["Close"].dropna()) >= 2:
                    prev = float(data[etf]["Close"].dropna().iloc[-2])
                    chg_pct = (close - prev) / prev * 100
                else:
                    chg_pct = 0.0
            else:
                tick = yf.Ticker(etf)
                hist = tick.history(period="2d")
                close = float(hist["Close"].iloc[-1])
                prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else close
                chg_pct = (close - prev) / prev * 100
            
            sectors.append({
                "ticker": etf,
                "name": name_cn,
                "price": round(close, 2),
                "change_pct": round(chg_pct, 2),
            })
        except Exception:
            continue
    
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "indices": indices,
        "sectors": sectors,
    }
