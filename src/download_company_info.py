"""
Download S&P 500 company info.
Step 1: Scrape Wikipedia for the full S&P 500 list (Sector, Sub-Industry, Headquarters).
Step 2: Use yfinance with 2s delay to add BusinessSummary.
"""
import csv
import time
import sys

# --- Step 1: Wikipedia scrape ---
import urllib.request
import ssl

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

def fetch_wiki_table():
    """Fetch S&P 500 company table from Wikipedia."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    req = urllib.request.Request(WIKI_URL, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, context=ctx, timeout=30)
    html = resp.read().decode("utf-8")

    # Parse the first table (constituents table)
    rows = []
    in_table = False
    current_row = []
    current_cell = []
    in_cell = False

    i = 0
    while i < len(html):
        if html[i:i+7] == "<table":
            in_table = True
        elif html[i:i+8] == "</table>" and in_table:
            break
        elif html[i:i+3] == "<tr" and in_table:
            current_row = []
        elif html[i:i+5] == "</tr>" and in_table:
            if current_row:
                rows.append(current_row)
        elif html[i:i+3] == "<td" and in_table:
            in_cell = True
            current_cell = []
            # skip to end of <td ...>
            while i < len(html) and html[i] != '>':
                i += 1
        elif html[i:i+5] == "</td>" and in_table and in_cell:
            in_cell = False
            # clean HTML tags from cell
            cell = ''.join(current_cell)
            # Remove nested tags
            import re
            cell = re.sub(r'<[^>]+>', '', cell)
            cell = cell.replace('&amp;', '&').replace('\n', ' ').strip()
            current_row.append(cell)
        elif in_cell and in_table:
            current_cell.append(html[i])
        i += 1

    return rows

def parse_wiki_data():
    """Use pandas to parse Wikipedia table (more reliable)."""
    import pandas as pd

    tables = pd.read_html(WIKI_URL, match="Symbol")
    df = tables[0]

    # Standardize columns
    print(f"Wikipedia columns: {list(df.columns)}")

    # Select relevant columns
    col_map = {}
    for col in df.columns:
        col_lower = str(col).lower()
        if 'symbol' in col_lower or col_lower == 'ticker':
            col_map[col] = 'Symbol'
        elif 'security' in col_lower or col_lower == 'company':
            col_map[col] = 'Company'
        elif 'gics sector' in col_lower or col_lower == 'sector':
            col_map[col] = 'Sector'
        elif 'gics sub' in col_lower or 'sub-industry' in col_lower or 'sub industry' in col_lower:
            col_map[col] = 'Sub-Industry'
        elif 'headquarters' in col_lower or 'location' in col_lower:
            col_map[col] = 'Headquarters'
        elif 'date added' in col_lower:
            col_map[col] = 'DateAdded'
        elif 'cik' in col_lower:
            col_map[col] = 'CIK'
        elif 'founded' in col_lower:
            col_map[col] = 'Founded'

    df = df.rename(columns=col_map)

    # Keep only standard columns
    keep = [c for c in ['Symbol', 'Company', 'Sector', 'Sub-Industry', 'Headquarters', 'DateAdded', 'CIK', 'Founded'] if c in df.columns]
    df = df[keep]

    # Clean symbol (remove dots, convert to standard)
    df['Symbol'] = df['Symbol'].astype(str).str.strip()
    df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)  # BRK.B -> BRK-B

    # Remove duplicates
    df = df.drop_duplicates(subset=['Symbol'])

    print(f"Total companies from Wikipedia: {len(df)}")
    print(f"Sectors: {df['Sector'].nunique()}")
    print(df['Sector'].value_counts())

    return df

# --- Step 2: yfinance business summaries (rate-limited) ---
def add_business_summaries(df, output_path):
    """Add BusinessSummary column using yfinance with 2s delay."""
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not available, skipping BusinessSummary")
        df['BusinessSummary'] = ''
        df.to_csv(output_path, index=False)
        return df

    tickers = df['Symbol'].tolist()
    total = len(tickers)
    summaries = {}
    failed = []

    print(f"\nDownloading business summaries for {total} tickers (2s delay each)...")
    print(f"Estimated time: ~{total * 2 / 60:.0f} minutes")

    for i, ticker in enumerate(tickers):
        try:
            t = yf.Ticker(ticker)
            info = t.info
            summary = info.get('longBusinessSummary', info.get('businessSummary', ''))
            summaries[ticker] = summary
            if summary:
                print(f"  [{i+1}/{total}] {ticker}: OK ({len(summary)} chars)")
            else:
                print(f"  [{i+1}/{total}] {ticker}: no summary")
                failed.append(ticker)
        except Exception as e:
            print(f"  [{i+1}/{total}] {ticker}: FAILED - {e}")
            summaries[ticker] = ''
            failed.append(ticker)

        # Rate limit: 2 seconds between calls
        if i < total - 1:
            time.sleep(2)

        # Save progress every 50 tickers
        if (i + 1) % 50 == 0:
            df['BusinessSummary'] = df['Symbol'].map(lambda x: summaries.get(x, ''))
            df.to_csv(output_path, index=False)
            print(f"  >> Progress saved ({i+1}/{total})")

    df['BusinessSummary'] = df['Symbol'].map(lambda x: summaries.get(x, ''))
    df.to_csv(output_path, index=False)

    print(f"\nDone! {total - len(failed)} summaries obtained, {len(failed)} failed.")
    if failed:
        print(f"Failed tickers: {failed[:20]}...")
    return df

# --- Main ---
if __name__ == '__main__':
    output_path = sys.argv[1] if len(sys.argv) > 1 else 'data/company_info.csv'

    print("=" * 60)
    print("Step 1: Fetch S&P 500 company list from Wikipedia")
    print("=" * 60)

    df = parse_wiki_data()
    df['BusinessSummary'] = ''  # placeholder

    # Save immediately (Wikipedia data is complete)
    df.to_csv(output_path, index=False)
    print(f"\nWikipedia data saved to {output_path}")

    # Step 2: Add business summaries (takes ~15 min with 2s delay)
    print("\n" + "=" * 60)
    print("Step 2: Add business summaries from yfinance (2s delay)")
    print("=" * 60)

    df = add_business_summaries(df, output_path)

    print(f"\nFinal file: {output_path}")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
