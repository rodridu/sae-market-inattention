"""
Phase 4: WRDS Data Merge

Collects market data from WRDS and merges with filing data.

Outputs:
- data/wrds_merged.parquet: Filing-level data with returns and controls

Required for Phase 3 pricing function estimation (Sarkar framework):
- Announcement returns (CAR[-1,+1])
- Post-filing drift (30/60/90 days)
- Control variables (size, B/M, leverage, momentum, volatility)

Usage:
  python 04_wrds_data_merge.py
  python 04_wrds_data_merge.py --test  # Test with 1000 filings

Data flow:
1. Load filing metadata (accession_number, CIK, filing_date)
2. Map CIK → PERMNO via EDGAR-CRSP link table
3. Get CRSP daily returns around filing dates
4. Compute CAR[-1,+1] and drift windows
5. Get Compustat fundamentals for controls
6. Merge and save to wrds_merged.parquet
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import timedelta
from tqdm import tqdm

# Parse arguments
parser = argparse.ArgumentParser(description='Merge WRDS market data with filing data')
parser.add_argument('--test', action='store_true', help='Test mode: process only 1000 filings')
args = parser.parse_args()

print("="*80)
print("PHASE 4: WRDS DATA MERGE")
print("="*80)
print(f"Mode: {'TEST (1000 filings)' if args.test else 'FULL'}")
print("="*80)

DATA_DIR = r"C:\Users\ofs4963\Dropbox\Arojects\SAE\data"

# =========================
# 1. Load filing metadata
# =========================

print("\n[1/6] Loading filing metadata...")

# Load from sec_metadata.parquet
metadata_file = os.path.join(DATA_DIR, "sec_metadata.parquet")
if not os.path.exists(metadata_file):
    print(f"ERROR: Filing metadata not found at {metadata_file}")
    print(f"Please ensure sec_metadata.parquet exists in {DATA_DIR}")
    sys.exit(1)

metadata_df = pd.read_parquet(metadata_file)
print(f"  Loaded {len(metadata_df):,} filings from sec_metadata.parquet")

# Check what columns we have
print(f"  Available columns: {metadata_df.columns.tolist()}")

# Try different column name variations
date_col = None
for col in ['filing_date', 'file_date', 'filed_date', 'date']:
    if col in metadata_df.columns:
        date_col = col
        break

if date_col is None:
    print(f"ERROR: No date column found. Available columns: {metadata_df.columns.tolist()}")
    sys.exit(1)

# Ensure required columns
required_cols = ['accession_number', 'cik', date_col]
missing_cols = [c for c in required_cols if c not in metadata_df.columns]
if missing_cols:
    print(f"ERROR: Missing columns in metadata: {missing_cols}")
    sys.exit(1)

# Extract filings
filings_df = metadata_df[required_cols].copy()

# Rename date column to filing_date for consistency
if date_col != 'filing_date':
    filings_df = filings_df.rename(columns={date_col: 'filing_date'})

# Convert filing_date to datetime
filings_df['filing_date'] = pd.to_datetime(filings_df['filing_date'])

# Extract year
filings_df['year'] = filings_df['filing_date'].dt.year

# Test mode: sample 1000 filings
if args.test:
    filings_df = filings_df.sample(n=min(1000, len(filings_df)), random_state=42).reset_index(drop=True)
    print(f"  [TEST MODE] Sampled {len(filings_df):,} filings")

print(f"  Final sample: {len(filings_df):,} filings")
print(f"  CIKs: {filings_df['cik'].nunique():,}")
print(f"  Date range: {filings_df['filing_date'].min()} to {filings_df['filing_date'].max()}")

# =========================
# 2. Map CIK to PERMNO
# =========================

print("\n[2/6] Mapping CIK to PERMNO via WRDS...")

try:
    import wrds

    # Connect to WRDS (uses .pgpass file if configured)
    print("  Connecting to WRDS...")
    print("  (Using stored credentials from .pgpass file)")
    db = wrds.Connection(wrds_username='ofs4963')
    print("  [OK] Connected to WRDS")

    # Get EDGAR-CRSP link table
    # This table maps SEC CIK to CRSP PERMNO
    print("  Querying EDGAR-CRSP link table...")

    link_query = """
    SELECT DISTINCT cik, permno
    FROM wrdsapps.crsp_edgar_link
    WHERE cik IS NOT NULL
    """

    link_df = db.raw_sql(link_query)
    print(f"  [OK] Loaded {len(link_df):,} CIK-PERMNO links")

    # Merge with filings
    filings_df = filings_df.merge(link_df, on='cik', how='left')

    matched = filings_df['permno'].notna().sum()
    print(f"  Matched {matched:,} / {len(filings_df):,} filings to PERMNO ({100*matched/len(filings_df):.1f}%)")

    # Filter to matched filings only
    filings_df = filings_df[filings_df['permno'].notna()].reset_index(drop=True)
    print(f"  Final sample with PERMNO: {len(filings_df):,} filings")

except ImportError:
    print("  ERROR: wrds module not found. Install with: pip install wrds")
    sys.exit(1)
except Exception as e:
    print(f"  ERROR: Failed to connect to WRDS or query link table: {e}")
    sys.exit(1)

# =========================
# 3. Get CRSP daily returns
# =========================

print("\n[3/6] Getting CRSP daily returns...")

# We need returns around filing dates
# CAR[-1,+1]: 3-day window
# Drift: up to 90 trading days post-filing

# Get unique PERMNO and date ranges
min_date = filings_df['filing_date'].min() - timedelta(days=5)  # Buffer
max_date = filings_df['filing_date'].max() + timedelta(days=100)  # Buffer for 90-day drift

permnos = filings_df['permno'].unique().tolist()

print(f"  Fetching returns for {len(permnos):,} PERMNOs")
print(f"  Date range: {min_date.date()} to {max_date.date()}")

# Query CRSP daily stock file
# We need: permno, date, ret, prc, shrout
# Process in chunks to avoid memory issues
chunk_size = 500
permno_chunks = [permnos[i:i+chunk_size] for i in range(0, len(permnos), chunk_size)]

crsp_list = []
for i, chunk in enumerate(tqdm(permno_chunks, desc="  Querying CRSP")):
    permno_str = ','.join(map(str, chunk))

    crsp_query = f"""
    SELECT permno, date, ret, prc, shrout, vol
    FROM crsp.dsf
    WHERE permno IN ({permno_str})
    AND date BETWEEN '{min_date.date()}' AND '{max_date.date()}'
    AND ret IS NOT NULL
    """

    chunk_df = db.raw_sql(crsp_query)
    crsp_list.append(chunk_df)

crsp_df = pd.concat(crsp_list, ignore_index=True)
print(f"  [OK] Loaded {len(crsp_df):,} PERMNO-date observations")

# Convert date to datetime
crsp_df['date'] = pd.to_datetime(crsp_df['date'])

# Sort for efficient merges
crsp_df = crsp_df.sort_values(['permno', 'date']).reset_index(drop=True)

# =========================
# 4. Compute returns around filing dates
# =========================

print("\n[4/6] Computing announcement returns and drift...")

def compute_car(filing_row, crsp_df, window_start, window_end):
    """
    Compute cumulative abnormal return in window [window_start, window_end]
    relative to filing_date.

    Returns NaN if insufficient data.
    """
    permno = filing_row['permno']
    filing_date = filing_row['filing_date']

    # Get returns for this PERMNO around filing date
    mask = (crsp_df['permno'] == permno)
    permno_rets = crsp_df[mask].set_index('date')['ret']

    # Define window
    start_date = filing_date + timedelta(days=window_start)
    end_date = filing_date + timedelta(days=window_end)

    # Get returns in window
    window_rets = permno_rets.loc[start_date:end_date]

    # Require at least 2 observations
    if len(window_rets) < 2:
        return np.nan

    # Cumulative return (simple compounding)
    car = (1 + window_rets).prod() - 1
    return car

# Compute CAR[-1,+1] (3-day announcement window)
print("  Computing CAR[-1,+1]...")
filings_df['car_minus1_plus1'] = filings_df.apply(
    lambda row: compute_car(row, crsp_df, -1, 1), axis=1
)

# Compute drift windows
print("  Computing drift windows...")
filings_df['drift_30d'] = filings_df.apply(
    lambda row: compute_car(row, crsp_df, 2, 32), axis=1  # ~30 trading days
)
filings_df['drift_60d'] = filings_df.apply(
    lambda row: compute_car(row, crsp_df, 2, 62), axis=1  # ~60 trading days
)
filings_df['drift_90d'] = filings_df.apply(
    lambda row: compute_car(row, crsp_df, 2, 92), axis=1  # ~90 trading days
)

# Report coverage
print(f"\n  Return coverage:")
print(f"    CAR[-1,+1]: {filings_df['car_minus1_plus1'].notna().sum():,} / {len(filings_df):,} ({100*filings_df['car_minus1_plus1'].notna().mean():.1f}%)")
print(f"    Drift 30d:  {filings_df['drift_30d'].notna().sum():,} / {len(filings_df):,} ({100*filings_df['drift_30d'].notna().mean():.1f}%)")
print(f"    Drift 60d:  {filings_df['drift_60d'].notna().sum():,} / {len(filings_df):,} ({100*filings_df['drift_60d'].notna().mean():.1f}%)")
print(f"    Drift 90d:  {filings_df['drift_90d'].notna().sum():,} / {len(filings_df):,} ({100*filings_df['drift_90d'].notna().mean():.1f}%)")

# =========================
# 5. Get Compustat fundamentals for controls
# =========================

print("\n[5/6] Getting Compustat fundamentals...")

# Get GVKEY for each PERMNO
print("  Mapping PERMNO to GVKEY...")

# Use CRSP-Compustat merged link table
link_query = """
SELECT lpermno AS permno, gvkey, linkdt, linkenddt
FROM crsp.ccmxpf_linktable
WHERE linktype IN ('LU', 'LC')
AND linkprim IN ('P', 'C')
"""

ccm_link = db.raw_sql(link_query)
print(f"  [OK] Loaded {len(ccm_link):,} PERMNO-GVKEY links")

# Convert dates
ccm_link['linkdt'] = pd.to_datetime(ccm_link['linkdt'])
ccm_link['linkenddt'] = pd.to_datetime(ccm_link['linkenddt'], errors='coerce')

# For each filing, find active GVKEY at filing_date
def get_gvkey(filing_row, ccm_link):
    permno = filing_row['permno']
    filing_date = filing_row['filing_date']

    links = ccm_link[ccm_link['permno'] == permno]

    for _, link in links.iterrows():
        linkdt = link['linkdt']
        linkenddt = link['linkenddt']

        # Check if filing_date is within link period
        if pd.isna(linkenddt):
            linkenddt = pd.Timestamp('2099-12-31')

        if linkdt <= filing_date <= linkenddt:
            return link['gvkey']

    return None

print("  Mapping filings to GVKEY...")
filings_df['gvkey'] = filings_df.apply(lambda row: get_gvkey(row, ccm_link), axis=1)

matched = filings_df['gvkey'].notna().sum()
print(f"  Matched {matched:,} / {len(filings_df):,} filings to GVKEY ({100*matched/len(filings_df):.1f}%)")

# Get Compustat annual fundamentals
# We need: size (log market cap), B/M, leverage, past returns, volatility
print("  Querying Compustat fundamentals...")

gvkeys = filings_df['gvkey'].dropna().unique().tolist()

# Get annual data
compustat_query = f"""
SELECT gvkey, datadate, fyear, at, lt, ceq, csho, prcc_f, sale, ni
FROM comp.funda
WHERE gvkey IN ({','.join(["'" + str(g) + "'" for g in gvkeys])})
AND indfmt = 'INDL'
AND datafmt = 'STD'
AND popsrc = 'D'
AND consol = 'C'
AND datadate >= '{(min_date - timedelta(days=365)).date()}'
"""

compustat_df = db.raw_sql(compustat_query)
print(f"  [OK] Loaded {len(compustat_df):,} firm-year observations")

# Convert datadate to datetime
compustat_df['datadate'] = pd.to_datetime(compustat_df['datadate'])

# Compute controls
# Size: log(market cap) = log(prcc_f * csho)
compustat_df['size'] = np.log(compustat_df['prcc_f'] * compustat_df['csho'])

# Book-to-market: ceq / (prcc_f * csho)
compustat_df['bm'] = compustat_df['ceq'] / (compustat_df['prcc_f'] * compustat_df['csho'])

# Leverage: lt / at
compustat_df['leverage'] = compustat_df['lt'] / compustat_df['at']

# Merge with filings (use most recent fiscal year end before filing_date)
print("  Merging Compustat controls with filings...")

def merge_compustat(filing_row, compustat_df):
    gvkey = filing_row['gvkey']
    filing_date = filing_row['filing_date']

    if pd.isna(gvkey):
        return pd.Series({'size': np.nan, 'bm': np.nan, 'leverage': np.nan})

    # Get all fiscal year ends before filing_date
    firm_data = compustat_df[
        (compustat_df['gvkey'] == gvkey) &
        (compustat_df['datadate'] < filing_date)
    ].sort_values('datadate')

    if len(firm_data) == 0:
        return pd.Series({'size': np.nan, 'bm': np.nan, 'leverage': np.nan})

    # Use most recent
    latest = firm_data.iloc[-1]

    return pd.Series({
        'size': latest['size'],
        'bm': latest['bm'],
        'leverage': latest['leverage']
    })

controls = filings_df.apply(lambda row: merge_compustat(row, compustat_df), axis=1)
filings_df = pd.concat([filings_df, controls], axis=1)

print(f"\n  Control coverage:")
print(f"    Size:     {filings_df['size'].notna().sum():,} / {len(filings_df):,} ({100*filings_df['size'].notna().mean():.1f}%)")
print(f"    B/M:      {filings_df['bm'].notna().sum():,} / {len(filings_df):,} ({100*filings_df['bm'].notna().mean():.1f}%)")
print(f"    Leverage: {filings_df['leverage'].notna().sum():,} / {len(filings_df):,} ({100*filings_df['leverage'].notna().mean():.1f}%)")

# =========================
# 6. Compute additional controls from CRSP
# =========================

print("\n[6/6] Computing additional controls from CRSP...")

# Past returns: 6-month momentum ([-180, -30] days before filing)
# Past volatility: std of daily returns in past 180 days

def compute_momentum_vol(filing_row, crsp_df):
    permno = filing_row['permno']
    filing_date = filing_row['filing_date']

    # Get returns for this PERMNO
    mask = (crsp_df['permno'] == permno)
    permno_rets = crsp_df[mask].set_index('date')['ret']

    # Past returns window: [-180, -30]
    start_date = filing_date - timedelta(days=180)
    end_date = filing_date - timedelta(days=30)
    past_rets = permno_rets.loc[start_date:end_date]

    if len(past_rets) < 60:  # Require at least 60 observations
        return pd.Series({'past_ret': np.nan, 'past_vol': np.nan})

    # Momentum: cumulative return
    momentum = (1 + past_rets).prod() - 1

    # Volatility: std of daily returns
    volatility = past_rets.std()

    return pd.Series({'past_ret': momentum, 'past_vol': volatility})

print("  Computing momentum and volatility...")
momentum_vol = filings_df.apply(lambda row: compute_momentum_vol(row, crsp_df), axis=1)
filings_df = pd.concat([filings_df, momentum_vol], axis=1)

print(f"\n  Momentum/Vol coverage:")
print(f"    Past return: {filings_df['past_ret'].notna().sum():,} / {len(filings_df):,} ({100*filings_df['past_ret'].notna().mean():.1f}%)")
print(f"    Past vol:    {filings_df['past_vol'].notna().sum():,} / {len(filings_df):,} ({100*filings_df['past_vol'].notna().mean():.1f}%)")

# =========================
# 7. Save merged data
# =========================

print("\n[7/7] Saving merged data...")

output_file = os.path.join(DATA_DIR, "wrds_merged.parquet")

# Select columns to save
output_cols = [
    # Identifiers
    'accession_number', 'cik', 'permno', 'gvkey', 'filing_date', 'year',
    # Returns
    'car_minus1_plus1', 'drift_30d', 'drift_60d', 'drift_90d',
    # Controls
    'size', 'bm', 'leverage', 'past_ret', 'past_vol'
]

output_df = filings_df[output_cols].copy()

# Save
output_df.to_parquet(output_file, index=False)
print(f"  [OK] Saved {len(output_df):,} observations to {output_file}")

# Summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print(f"\nSample size: {len(output_df):,} filings")
print(f"  Unique firms (CIK): {output_df['cik'].nunique():,}")
print(f"  Unique securities (PERMNO): {output_df['permno'].nunique():,}")
print(f"  Date range: {output_df['filing_date'].min().date()} to {output_df['filing_date'].max().date()}")

print("\nOutcome coverage:")
for var in ['car_minus1_plus1', 'drift_30d', 'drift_60d', 'drift_90d']:
    pct = 100 * output_df[var].notna().mean()
    print(f"  {var:20s}: {pct:5.1f}%")

print("\nControl coverage:")
for var in ['size', 'bm', 'leverage', 'past_ret', 'past_vol']:
    pct = 100 * output_df[var].notna().mean()
    print(f"  {var:20s}: {pct:5.1f}%")

print("\nDescriptive statistics (returns):")
print(output_df[['car_minus1_plus1', 'drift_30d', 'drift_60d', 'drift_90d']].describe())

print("\nDescriptive statistics (controls):")
print(output_df[['size', 'bm', 'leverage', 'past_ret', 'past_vol']].describe())

print("\n" + "="*80)
print("WRDS DATA MERGE COMPLETE")
print("="*80)
print(f"Output: {output_file}")
print("\nNext step: Run 03_sae_training.py to estimate pricing function and test RI predictions")
print("="*80)

# Close WRDS connection
db.close()
