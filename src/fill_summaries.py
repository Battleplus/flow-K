"""Fill missing BusinessSummary entries in company_info.csv using yfinance.
Skips tickers that already have a valid summary, only downloads missing ones.
"""
import csv
import time
import sys
import os

FILE = 'data/company_info.csv'

# 1. Read CSV and identify missing summaries
rows = []
missing_tickers = []
has_count = 0
with open(FILE, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for r in reader:
        bs = r.get('BusinessSummary', '').strip()
        if bs and bs != 'nan' and len(bs) > 10:
            has_count += 1
            rows.append(r)
        else:
            missing_tickers.append(r['Symbol'])
            rows.append(r)

total = len(rows)
need = len(missing_tickers)
print(f"Total records: {total}")
print(f"Already have summary: {has_count}")
print(f"Need to download: {need}")
print(f"Tickers to download: {missing_tickers[:10]}...")
print()

if need == 0:
    print("All summaries already downloaded! Nothing to do.")
    sys.exit(0)

import yfinance as yf

summaries = {}  # ticker -> summary
success = 0
failed = []

print(f"Starting download with 2s delay per ticker...")
print(f"Estimated time: ~{need * 2 / 60:.0f} minutes")
print()

start_time = time.time()

for i, ticker in enumerate(missing_tickers):
    try:
        info = yf.Ticker(ticker).info
        summary = info.get('longBusinessSummary', '') or info.get('businessSummary', '')
        summaries[ticker] = summary.strip() if summary else ''
        if summary:
            success += 1
            print(f"  [{i+1}/{need}] {ticker}: OK ({len(summary)} chars)")
        else:
            print(f"  [{i+1}/{need}] {ticker}: no summary available")
            failed.append(ticker)
    except Exception as e:
        print(f"  [{i+1}/{need}] {ticker}: ERROR - {str(e)[:80]}")
        summaries[ticker] = ''
        failed.append(ticker)

    # Rate limit
    if i < need - 1:
        time.sleep(2)

    # Save progress every 25 tickers (and on final)
    if (i + 1) % 25 == 0 or i == need - 1:
        # Merge summaries into rows
        for row in rows:
            symbol = row['Symbol']
            if symbol in summaries:
                row['BusinessSummary'] = summaries[symbol]

        with open(FILE, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        elapsed = (time.time() - start_time) / 60
        print(f"  >> Progress saved: {i+1}/{need} done ({elapsed:.1f} min elapsed)")
        sys.stdout.flush()

elapsed = (time.time() - start_time) / 60
print(f"\n{'='*60}")
print(f"Download complete!")
print(f"  Total downloaded: {success}/{need} ({success/need*100:.1f}%)")
print(f"  Failed: {len(failed)}")
print(f"  Total with summary: {has_count + success}/{total}")
print(f"  Time: {elapsed:.1f} minutes")
if failed:
    print(f"  Failed tickers: {failed[:20]}...")
