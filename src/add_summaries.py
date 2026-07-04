"""Add business summaries from yfinance to existing company_info.csv."""
import pandas as pd
import yfinance as yf
import time
import sys

FILE = 'data/company_info.csv'

df = pd.read_csv(FILE)
tickers = df['Symbol'].tolist()
total = len(tickers)

print(f"Adding business summaries for {total} tickers (3s delay each)")
print(f"Estimated time: ~{total * 3 / 60:.0f} minutes")

summaries = {}
for i, ticker in enumerate(tickers):
    try:
        info = yf.Ticker(ticker).info
        summary = info.get('longBusinessSummary', '') or info.get('businessSummary', '')
        summaries[ticker] = summary
        status = 'OK' if summary else 'empty'
        print(f"  [{i+1}/{total}] {ticker}: {status} ({len(summary)} chars)", flush=True)
    except Exception as e:
        print(f"  [{i+1}/{total}] {ticker}: FAILED - {str(e)[:60]}", flush=True)
        summaries[ticker] = ''

    if i < total - 1:
        time.sleep(3)

    if (i + 1) % 25 == 0:
        df['BusinessSummary'] = df['Symbol'].map(lambda x: summaries.get(x, ''))
        df.to_csv(FILE, index=False)
        print(f"  >> Progress saved ({i+1}/{total})", flush=True)

df['BusinessSummary'] = df['Symbol'].map(lambda x: summaries.get(x, ''))
df.to_csv(FILE, index=False)
got = sum(1 for v in summaries.values() if v)
print(f"\nDone! {got}/{total} summaries obtained.")
