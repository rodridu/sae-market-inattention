"""
Fetch control variables from WRDS (CRSP + Compustat)

This script fetches standard control variables for asset pricing regressions:
- Size: log(market cap)
- Book-to-Market: book equity / market equity
- Leverage: total debt / total assets
- Past returns: prior 12-month return (momentum)
- Past volatility: std dev of daily returns

Usage:
    python fetch_controls_wrds.py

Requires:
    - WRDS account with access to CRSP and Compustat
    - wrds package: pip install wrds
    - outcomes_wrds.parquet with permno and filing_date
"""

import pandas as pd
import numpy as np
import wrds
import os
from datetime import datetime, timedelta
from tqdm import tqdm

# Configuration
DATA_DIR = r"C:\Users\ofs4963\Dropbox\Arojects\SAE\data"
OUTCOMES_FILE = os.path.join(DATA_DIR, "outcomes_wrds.parquet")
OUTPUT_FILE = os.path.join(DATA_DIR, "controls_wrds.parquet")

print("=" * 60)
print("FETCHING CONTROL VARIABLES FROM WRDS")
print("=" * 60)

# Load outcomes data to get list of (permno, date) pairs
print("\n1. Loading outcomes data...")
outcomes_df = pd.read_parquet(OUTCOMES_FILE)
print(f"   Loaded {len(outcomes_df):,} filings")

# Filter to rows with valid permno
outcomes_df = outcomes_df[outcomes_df['permno'].notna()].copy()
print(f"   {len(outcomes_df):,} filings with valid PERMNO")

# Extract unique (permno, year-month) pairs
outcomes_df['year'] = outcomes_df['filing_date'].dt.year
outcomes_df['month'] = outcomes_df['filing_date'].dt.month
unique_firm_months = outcomes_df[['permno', 'year', 'month']].drop_duplicates()
print(f"   {len(unique_firm_months):,} unique firm-month combinations")

# Date range
min_date = outcomes_df['filing_date'].min()
max_date = outcomes_df['filing_date'].max()
print(f"   Date range: {min_date.date()} to {max_date.date()}")

# Connect to WRDS
print("\n2. Connecting to WRDS...")
print("   (This will prompt for username/password if not cached)")
db = wrds.Connection()
print("   Connected successfully")

# =========================
# 3. Fetch CRSP monthly data
# =========================
print("\n3. Fetching CRSP monthly data...")
print("   Variables: price, shares outstanding, returns")

# CRSP monthly stock file
# We need data from 12 months before min_date to 3 months after max_date
# (for 12-month momentum and post-filing drift)
start_date = (min_date - pd.DateOffset(months=13)).strftime('%Y-%m-%d')
end_date = (max_date + pd.DateOffset(months=4)).strftime('%Y-%m-%d')

crsp_query = f"""
    SELECT
        permno,
        date,
        prc,
        shrout,
        ret,
        vol
    FROM crsp.msf
    WHERE date >= '{start_date}'
        AND date <= '{end_date}'
        AND permno IN ({','.join(map(str, outcomes_df['permno'].unique().astype(int)))})
    ORDER BY permno, date
"""

print(f"   Querying CRSP from {start_date} to {end_date}...")
crsp_monthly = db.raw_sql(crsp_query)
print(f"   Retrieved {len(crsp_monthly):,} firm-month observations")

# Clean CRSP data
crsp_monthly['prc'] = crsp_monthly['prc'].abs()  # prices are negative if average of bid/ask
crsp_monthly['mktcap'] = crsp_monthly['prc'] * crsp_monthly['shrout']  # in thousands
crsp_monthly['year'] = pd.to_datetime(crsp_monthly['date']).dt.year
crsp_monthly['month'] = pd.to_datetime(crsp_monthly['date']).dt.month

# =========================
# 4. Fetch CRSP daily data for volatility
# =========================
print("\n4. Fetching CRSP daily returns for volatility calculation...")
print("   (This may take a few minutes for large datasets)")

# For volatility, we need 60 trading days before each filing
# We'll compute rolling volatility for each month
crsp_daily_query = f"""
    SELECT
        permno,
        date,
        ret
    FROM crsp.dsf
    WHERE date >= '{start_date}'
        AND date <= '{end_date}'
        AND permno IN ({','.join(map(str, outcomes_df['permno'].unique().astype(int)))})
        AND ret IS NOT NULL
    ORDER BY permno, date
"""

print(f"   Querying CRSP daily data...")
crsp_daily = db.raw_sql(crsp_daily_query)
print(f"   Retrieved {len(crsp_daily):,} firm-day observations")

# Compute 60-day rolling volatility
print("   Computing rolling volatility...")
crsp_daily['date'] = pd.to_datetime(crsp_daily['date'])
crsp_daily = crsp_daily.sort_values(['permno', 'date'])

volatility_list = []
for permno, group in tqdm(crsp_daily.groupby('permno'), desc="   Computing volatility"):
    group = group.sort_values('date').set_index('date')
    # 60-day rolling std
    vol = group['ret'].rolling(window=60, min_periods=20).std()
    vol_monthly = vol.resample('M').last()  # Take end-of-month value
    vol_df = pd.DataFrame({
        'permno': permno,
        'date': vol_monthly.index,
        'past_vol': vol_monthly.values * np.sqrt(252)  # Annualize
    })
    volatility_list.append(vol_df)

volatility_df = pd.concat(volatility_list, ignore_index=True)
volatility_df['year'] = volatility_df['date'].dt.year
volatility_df['month'] = volatility_df['date'].dt.month
volatility_df = volatility_df[['permno', 'year', 'month', 'past_vol']]

# =========================
# 5. Compute momentum (past 12-month return)
# =========================
print("\n5. Computing momentum (past 12-month returns)...")
crsp_monthly = crsp_monthly.sort_values(['permno', 'date'])

momentum_list = []
for permno, group in tqdm(crsp_monthly.groupby('permno'), desc="   Computing momentum"):
    group = group.sort_values('date').copy()
    # 12-month cumulative return (skip most recent month)
    group['ret_shift'] = group['ret'].shift(1)  # Lag 1 month
    group['past_ret'] = group['ret_shift'].rolling(window=12, min_periods=6).apply(
        lambda x: (1 + x).prod() - 1, raw=False
    )
    momentum_list.append(group[['permno', 'year', 'month', 'past_ret']])

momentum_df = pd.concat(momentum_list, ignore_index=True)

# =========================
# 6. Fetch Compustat fundamentals
# =========================
print("\n6. Fetching Compustat fundamentals...")
print("   Variables: total assets, book equity, total debt")

# We need annual data - link CRSP permno to Compustat gvkey
# First get the CCM link table
print("   Fetching CRSP-Compustat link table...")
ccm_query = """
    SELECT
        lpermno as permno,
        gvkey,
        linkdt,
        linkenddt
    FROM crsp.ccmxpf_linktable
    WHERE linktype IN ('LU', 'LC')
        AND linkprim IN ('P', 'C')
"""
ccm = db.raw_sql(ccm_query)
print(f"   Retrieved {len(ccm):,} PERMNO-GVKEY links")

# Fetch Compustat annual fundamentals
comp_query = f"""
    SELECT
        gvkey,
        datadate,
        fyear,
        at,     -- total assets
        ceq,    -- common equity
        lt,     -- total liabilities
        dlc,    -- debt in current liabilities
        dltt,   -- long-term debt
        seq,    -- stockholders equity
        pstkrv, -- preferred stock redemption value
        pstkl,  -- preferred stock liquidating value
        pstk,   -- preferred stock par value
        txditc  -- deferred taxes and investment tax credit
    FROM comp.funda
    WHERE datadate >= '{start_date}'
        AND datadate <= '{end_date}'
        AND indfmt = 'INDL'
        AND datafmt = 'STD'
        AND popsrc = 'D'
        AND consol = 'C'
    ORDER BY gvkey, datadate
"""

print(f"   Querying Compustat from {start_date} to {end_date}...")
compustat = db.raw_sql(comp_query)
print(f"   Retrieved {len(compustat):,} firm-year observations")

# =========================
# 7. Compute book equity (following Fama-French)
# =========================
print("\n7. Computing book equity (Fama-French methodology)...")

# Book equity = SEQ + TXDITC - Preferred Stock
# Preferred stock = PSTKRV, or PSTKL if missing, or PSTK if both missing
compustat['ps'] = compustat['pstkrv'].fillna(compustat['pstkl']).fillna(compustat['pstk']).fillna(0)
compustat['be'] = compustat['seq'].fillna(0) + compustat['txditc'].fillna(0) - compustat['ps']

# Total debt = DLC + DLTT
compustat['total_debt'] = compustat['dlc'].fillna(0) + compustat['dltt'].fillna(0)

# Keep relevant columns
compustat = compustat[['gvkey', 'datadate', 'fyear', 'at', 'be', 'total_debt']].copy()
compustat['datadate'] = pd.to_datetime(compustat['datadate'])

print(f"   Computed book equity for {compustat['be'].notna().sum():,} observations")

# =========================
# 8. Merge CCM to link Compustat to CRSP
# =========================
print("\n8. Merging Compustat with CRSP via CCM link...")

# Convert date types
ccm['linkdt'] = pd.to_datetime(ccm['linkdt'])
ccm['linkenddt'] = pd.to_datetime(ccm['linkenddt'])
ccm['linkenddt'] = ccm['linkenddt'].fillna(pd.Timestamp('2099-12-31'))

# Merge compustat with ccm
comp_ccm = compustat.merge(ccm, on='gvkey', how='left')

# Filter to valid link dates
comp_ccm = comp_ccm[
    (comp_ccm['datadate'] >= comp_ccm['linkdt']) &
    (comp_ccm['datadate'] <= comp_ccm['linkenddt'])
].copy()

print(f"   Matched {len(comp_ccm):,} observations to PERMNO")

# For each firm-year, we use accounting data from fiscal year ending in calendar year t-1
# to compute controls for filings in calendar year t
comp_ccm['year'] = comp_ccm['datadate'].dt.year + 1  # Apply to next calendar year

# =========================
# 9. Merge all control variables
# =========================
print("\n9. Merging all control variables...")

# Start with outcomes to get the right (permno, year, month) combinations
controls = outcomes_df[['accession_number', 'permno', 'filing_date', 'year', 'month']].copy()

print(f"   Starting with {len(controls):,} filings")

# Merge CRSP monthly (size)
controls = controls.merge(
    crsp_monthly[['permno', 'year', 'month', 'mktcap']],
    on=['permno', 'year', 'month'],
    how='left'
)
print(f"   After CRSP merge: {controls['mktcap'].notna().sum():,} with market cap")

# Merge momentum
controls = controls.merge(
    momentum_df[['permno', 'year', 'month', 'past_ret']],
    on=['permno', 'year', 'month'],
    how='left'
)
print(f"   After momentum merge: {controls['past_ret'].notna().sum():,} with past returns")

# Merge volatility
controls = controls.merge(
    volatility_df[['permno', 'year', 'month', 'past_vol']],
    on=['permno', 'year', 'month'],
    how='left'
)
print(f"   After volatility merge: {controls['past_vol'].notna().sum():,} with volatility")

# Merge Compustat (annual data - match on permno and year only)
controls = controls.merge(
    comp_ccm[['permno', 'year', 'at', 'be', 'total_debt']].drop_duplicates(['permno', 'year']),
    on=['permno', 'year'],
    how='left'
)
print(f"   After Compustat merge: {controls['be'].notna().sum():,} with book equity")

# =========================
# 10. Compute final control variables
# =========================
print("\n10. Computing final control variables...")

# Size: log(market cap in millions)
controls['size'] = np.log(controls['mktcap'] / 1000)  # Convert thousands to millions, then log

# Book-to-Market: book equity (millions) / market cap (millions)
controls['mktcap_millions'] = controls['mktcap'] / 1000
controls['bm'] = controls['be'] / controls['mktcap_millions']

# Leverage: total debt / total assets
controls['leverage'] = controls['total_debt'] / controls['at']

# Keep only final control variables
final_controls = controls[[
    'accession_number',
    'permno',
    'filing_date',
    'year',
    'size',
    'bm',
    'leverage',
    'past_ret',
    'past_vol'
]].copy()

# =========================
# 11. Quality checks and summary
# =========================
print("\n11. Data quality checks...")

print(f"\nFinal sample: {len(final_controls):,} filings")
print(f"\nData coverage:")
print(f"  Size:         {final_controls['size'].notna().sum():,} ({100*final_controls['size'].notna().mean():.1f}%)")
print(f"  Book-to-Mkt:  {final_controls['bm'].notna().sum():,} ({100*final_controls['bm'].notna().mean():.1f}%)")
print(f"  Leverage:     {final_controls['leverage'].notna().sum():,} ({100*final_controls['leverage'].notna().mean():.1f}%)")
print(f"  Past Return:  {final_controls['past_ret'].notna().sum():,} ({100*final_controls['past_ret'].notna().mean():.1f}%)")
print(f"  Past Vol:     {final_controls['past_vol'].notna().sum():,} ({100*final_controls['past_vol'].notna().mean():.1f}%)")

print(f"\nSummary statistics:")
print(final_controls[['size', 'bm', 'leverage', 'past_ret', 'past_vol']].describe())

# Winsorize extreme values
print(f"\nWinsorizing at 1st and 99th percentiles...")
for col in ['bm', 'leverage', 'past_ret', 'past_vol']:
    if col in final_controls.columns:
        p01 = final_controls[col].quantile(0.01)
        p99 = final_controls[col].quantile(0.99)
        final_controls[col] = final_controls[col].clip(p01, p99)

# =========================
# 12. Save to file
# =========================
print(f"\n12. Saving to {OUTPUT_FILE}...")
final_controls.to_parquet(OUTPUT_FILE, index=False)
print(f"   Saved {len(final_controls):,} rows")

# Close WRDS connection
db.close()
print("\n" + "=" * 60)
print("CONTROL VARIABLES SUCCESSFULLY FETCHED")
print("=" * 60)
print(f"\nOutput file: {OUTPUT_FILE}")
print(f"Ready to merge with outcomes_wrds.parquet by accession_number")
