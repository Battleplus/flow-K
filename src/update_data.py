"""
数据补齐脚本：下载XLC + 更新ETF/VIX到最新
Phase 0 - Three Dimensional Framework
"""

import pandas as pd
import yfinance as yf
import os
import time
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# ============================================================
# ETF 定义：11 SPDR板块 + SPY + QQQ + XLC
# ============================================================
ETF_DEFS = {
    'SPY':  ('S&P 500 Index',            '2000-01-01'),
    'QQQ':  ('Nasdaq 100',               '2018-06-01'),
    'XLB':  ('Materials',                '2000-01-01'),
    'XLC':  ('Communication Services',   '2018-06-20'),
    'XLE':  ('Energy',                   '2000-01-01'),
    'XLF':  ('Financials',               '2000-01-01'),
    'XLI':  ('Industrials',              '2000-01-01'),
    'XLK':  ('Technology',               '2000-01-01'),
    'XLP':  ('Consumer Staples',         '2000-01-01'),
    'XLRE': ('Real Estate',              '2015-10-08'),
    'XLU':  ('Utilities',                '2000-01-01'),
    'XLV':  ('Healthcare',               '2000-01-01'),
    'XLY':  ('Consumer Discretionary',   '2000-01-01'),
}


def download_single(ticker, start_date=None):
    """下载单只ETF/指数的完整历史，如果start_date为空则全量"""
    try:
        if start_date:
            df = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
        else:
            df = yf.download(ticker, start='2000-01-01', progress=False, auto_adjust=True)

        if df.empty:
            print(f"  {ticker}: 无数据返回")
            return None

        # 统一列名
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        df['Ticker'] = ticker
        print(f"  {ticker}: {len(df)}行, {df['Date'].iloc[0]} ~ {df['Date'].iloc[-1]}")
        return df
    except Exception as e:
        print(f"  {ticker}: 错误 - {e}")
        return None


def update_etfs():
    """更新所有ETF数据：已有ETF追加新数据，XLC全量下载"""
    print("=" * 60)
    print("📊 更新 ETF 数据")
    print("=" * 60)

    # 读取现有数据
    etf_path = os.path.join(DATA_DIR, 'sector_etf_prices.csv')
    existing = pd.read_csv(etf_path)
    existing_tickers = set(existing['Ticker'].unique())
    print(f"现有ETF: {sorted(existing_tickers)}")
    print(f"最新日期: {existing['Date'].max()}")

    new_rows = []

    for ticker, (sector, start) in ETF_DEFS.items():
        if ticker in existing_tickers:
            # 已有 → 只下载2026-02-21之后的新数据
            print(f"\n[{ticker}] 已有，追加增量...")
            df = download_single(ticker, start_date='2026-02-21')
            if df is not None:
                df['Sector'] = sector
                # 只保留 2026-02-21 之后的
                df = df[df['Date'] >= '2026-02-21']
                if len(df) > 0:
                    new_rows.append(df)
                else:
                    print(f"  无新数据（可能已是最新）")
        else:
            # 没有 → 全量下载
            print(f"\n[{ticker}] 🆕 新ETF，全量下载...")
            df = download_single(ticker, start_date=start)
            if df is not None:
                df['Sector'] = sector
                new_rows.append(df)

        time.sleep(1.5)  # 防止yfinance限速

    if new_rows:
        new_data = pd.concat(new_rows, ignore_index=True)
        # 列顺序统一
        cols = ['Ticker', 'Sector', 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        new_data = new_data[[c for c in cols if c in new_data.columns]]

        # 合并+去重
        merged = pd.concat([existing, new_data], ignore_index=True)
        merged = merged.drop_duplicates(subset=['Ticker', 'Date'], keep='last')
        merged = merged.sort_values(['Ticker', 'Date']).reset_index(drop=True)

        merged.to_csv(etf_path, index=False)
        print(f"\n✅ ETF 数据已更新：{merged['Ticker'].nunique()}只, {len(merged):,}行")
        for t in sorted(merged['Ticker'].unique()):
            sub = merged[merged['Ticker'] == t]
            print(f"  {t}: {sub['Date'].min()} ~ {sub['Date'].max()}, {len(sub):,}行")
    else:
        print("\n⚠️ 无新数据追加")


def update_vix():
    """更新VIX数据"""
    print("\n" + "=" * 60)
    print("📊 更新 VIX 数据")
    print("=" * 60)

    vix_path = os.path.join(DATA_DIR, 'vix_data.csv')
    existing = pd.read_csv(vix_path)
    print(f"现有: {len(existing):,}行, {existing['Date'].min()} ~ {existing['Date'].max()}")

    # 下载增量
    df = download_single('^VIX', start_date='2026-02-21')
    if df is not None:
        df['Ticker'] = 'VIX'
        new_data = df[df['Date'] >= '2026-02-21']

        if len(new_data) > 0:
            cols = ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            new_data = new_data[[c for c in cols if c in new_data.columns]]

            merged = pd.concat([existing, new_data], ignore_index=True)
            merged = merged.drop_duplicates(subset=['Ticker', 'Date'], keep='last')
            merged = merged.sort_values('Date').reset_index(drop=True)

            merged.to_csv(vix_path, index=False)
            print(f"✅ VIX 已更新：{len(merged):,}行, {merged['Date'].min()} ~ {merged['Date'].max()}")
        else:
            print("⚠️ 无新数据")
    else:
        print("❌ VIX下载失败")


def show_summary():
    """显示数据更新后的摘要"""
    print("\n" + "=" * 60)
    print("📋 数据更新完成摘要")
    print("=" * 60)

    etf = pd.read_csv(os.path.join(DATA_DIR, 'sector_etf_prices.csv'))
    vix = pd.read_csv(os.path.join(DATA_DIR, 'vix_data.csv'))

    print(f"\nETF: {etf['Ticker'].nunique()}只, {len(etf):,}行")
    print(f"日期: {etf['Date'].min()} ~ {etf['Date'].max()}")

    sector_etfs = [t for t in sorted(etf['Ticker'].unique()) if t.startswith('X')]
    print(f"\n板块ETF ({len(sector_etfs)}只): {sector_etfs}")

    print(f"\nVIX: {len(vix):,}行, {vix['Date'].min()} ~ {vix['Date'].max()}")

    # 检查是否所有数据都更新到今天
    today = datetime.now().strftime('%Y-%m-%d')
    etf_latest = etf['Date'].max()
    vix_latest = vix['Date'].max()
    print(f"\n数据时效: ETF最新={etf_latest}, VIX最新={vix_latest}, 今天={today}")
    if etf_latest < today:
        print(f"⚠️ ETF数据落后 {(datetime.strptime(today, '%Y-%m-%d') - datetime.strptime(etf_latest, '%Y-%m-%d')).days} 天")


if __name__ == '__main__':
    update_etfs()
    update_vix()
    show_summary()
