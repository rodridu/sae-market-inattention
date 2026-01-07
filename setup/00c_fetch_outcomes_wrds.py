"""
Fetch Outcome Variables from WRDS

This script fetches market-based and analyst-based outcomes for 10-K filings
to use as dependent variables in the SAE analysis.

Data Sources (all from WRDS):
1. CRSP: Stock returns for announcement effects and drift
2. I/B/E/S: Analyst forecast revisions and dispersion
3. Compustat: Fundamentals for control variables

Outcomes Generated:
- car_minus1_plus1: Cumulative abnormal return [-1, +1] around filing
- drift_30d, drift_60d, drift_90d: Post-filing drift
- volume_ratio: Trading volume spike around filing
- volatility_ratio: Volatility increase around filing
- forecast_revision: Change in analyst EPS forecasts
- forecast_dispersion: Standard deviation of analyst forecasts

Prerequisites:
1. Install wrds library: pip install wrds
2. Set up WRDS credentials (or have .pgpass file)

Usage:
    python 00c_fetch_outcomes_wrds.py
    python 00c_fetch_outcomes_wrds.py --test  # Test with 100 filings
"""

import wrds
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
import argparse
from tqdm import tqdm

# ============================================
# Configuration
# ============================================

DATA_DIR = Path(r"C:\Users\ofs4963\Dropbox\Arojects\SAE\data")
METADATA_FILE = DATA_DIR / "sec_metadata.parquet"
OUTPUT_FILE = DATA_DIR / "outcomes_wrds.parquet"

# ============================================
# Helper Functions
# ============================================

def connect_wrds():
    """Connect to WRDS database."""
    print("\nConnecting to WRDS...")
    try:
        db = wrds.Connection()
        print("[OK] Connected to WRDS")
        return db
    except Exception as e:
        print(f"ERROR connecting to WRDS: {e}")
        print("\nMake sure you have:")
        print("  1. Installed wrds: pip install wrds")
        print("  2. Set up credentials: https://wrds-www.wharton.upenn.edu/pages/support/programming-wrds/programming-python/python-wrds-cloud/")
        return None

def get_permno_from_ticker(db, tickers_df):
    """
    Map tickers to CRSP PERMNOs using CRSP stock header.

    Args:
        db: WRDS connection
        tickers_df: DataFrame with ticker, filing_date columns

    Returns:
        DataFrame with ticker, permno, start_date, end_date
    """
    print("\nMapping tickers to CRSP PERMNOs...")

    # Get unique tickers
    unique_tickers = tickers_df['ticker'].unique().tolist()
    tickers_str = "','".join([t for t in unique_tickers if t != 'UNKNOWN'])

    query = f"""
    SELECT ticker, permno, namedt as start_date, nameenddt as end_date
    FROM crsp.stocknames
    WHERE ticker IN ('{tickers_str}')
    AND shrcd IN (10, 11)  -- Common shares only
    ORDER BY ticker, namedt
    """

    permno_mapping = db.raw_sql(query)
    print(f"[OK] Found {len(permno_mapping):,} ticker-permno mappings")

    return permno_mapping

def fetch_crsp_returns(db, filings_df, window_before=1, window_after=1, drift_days=[30, 60, 90]):
    """
    Fetch CRSP returns around filing dates.

    Computes:
    - Abnormal returns (stock return - market return)
    - CAR around announcement
    - Post-filing drift
    - Volume and volatility ratios

    Args:
        db: WRDS connection
        filings_df: DataFrame with permno, filing_date
        window_before: Days before filing for CAR
        window_after: Days after filing for CAR
        drift_days: List of days for measuring drift

    Returns:
        DataFrame with returns metrics
    """
    print("\nFetching CRSP returns...")
    print(f"  Filing dates: {filings_df['filing_date'].min()} to {filings_df['filing_date'].max()}")

    # Get date range
    min_date = filings_df['filing_date'].min() - timedelta(days=window_before + 30)
    max_date = filings_df['filing_date'].max() + timedelta(days=max(drift_days) + 30)

    # Get unique PERMNOs
    permnos = filings_df['permno'].unique().tolist()
    permno_str = ','.join(map(str, permnos))

    print(f"  Fetching returns for {len(permnos):,} stocks from {min_date} to {max_date}")

    # Fetch daily returns
    query = f"""
    SELECT a.permno, a.date, a.ret, a.vol, a.shrout, a.prc,
           b.vwretd as mkt_ret
    FROM crsp.dsf a
    LEFT JOIN crsp.dsi b ON a.date = b.date
    WHERE a.permno IN ({permno_str})
    AND a.date >= '{min_date}'
    AND a.date <= '{max_date}'
    AND a.ret IS NOT NULL
    ORDER BY a.permno, a.date
    """

    returns_df = db.raw_sql(query)

    # Convert returns to numeric
    returns_df['ret'] = pd.to_numeric(returns_df['ret'], errors='coerce')
    returns_df['mkt_ret'] = pd.to_numeric(returns_df['mkt_ret'], errors='coerce')

    # Calculate abnormal returns
    returns_df['abret'] = returns_df['ret'] - returns_df['mkt_ret']

    print(f"[OK] Fetched {len(returns_df):,} daily returns")

    # Calculate metrics for each filing
    results = []
    for _, filing in tqdm(filings_df.iterrows(), total=len(filings_df), desc="Computing metrics"):
        permno = filing['permno']
        filing_date = filing['filing_date']

        # Get returns for this stock
        stock_returns = returns_df[returns_df['permno'] == permno].copy()
        stock_returns['date'] = pd.to_datetime(stock_returns['date'])

        # CAR around announcement
        car_start = filing_date - timedelta(days=window_before)
        car_end = filing_date + timedelta(days=window_after)
        car_window = stock_returns[(stock_returns['date'] >= car_start) &
                                   (stock_returns['date'] <= car_end)]

        car = car_window['abret'].sum() if len(car_window) > 0 else np.nan

        # Post-filing drift
        drift_metrics = {}
        for days in drift_days:
            drift_start = filing_date + timedelta(days=2)  # Start after CAR window
            drift_end = filing_date + timedelta(days=days)
            drift_window = stock_returns[(stock_returns['date'] >= drift_start) &
                                        (stock_returns['date'] <= drift_end)]
            drift_metrics[f'drift_{days}d'] = drift_window['abret'].sum() if len(drift_window) > 0 else np.nan

        # Volume ratio (filing day / average 30 days before)
        volume_before = stock_returns[(stock_returns['date'] >= filing_date - timedelta(days=30)) &
                                     (stock_returns['date'] < filing_date)]
        volume_filing = stock_returns[stock_returns['date'] == filing_date]

        if len(volume_before) > 0 and len(volume_filing) > 0:
            avg_vol_before = volume_before['vol'].mean()
            vol_filing = volume_filing['vol'].iloc[0]
            volume_ratio = vol_filing / avg_vol_before if avg_vol_before > 0 else np.nan
        else:
            volume_ratio = np.nan

        # Volatility ratio (std dev in [-1,+1] / std dev 30 days before)
        vol_before = stock_returns[(stock_returns['date'] >= filing_date - timedelta(days=30)) &
                                  (stock_returns['date'] < filing_date)]
        vol_around = car_window

        if len(vol_before) > 0 and len(vol_around) > 0:
            std_before = vol_before['abret'].std()
            std_around = vol_around['abret'].std()
            volatility_ratio = std_around / std_before if std_before > 0 else np.nan
        else:
            volatility_ratio = np.nan

        # Store results
        result = {
            'permno': permno,
            'filing_date': filing_date,
            'car_minus1_plus1': car,
            'volume_ratio': volume_ratio,
            'volatility_ratio': volatility_ratio,
            **drift_metrics
        }
        results.append(result)

    return pd.DataFrame(results)

def fetch_ibes_forecasts(db, filings_df):
    """
    Fetch I/B/E/S analyst forecast data.

    Computes:
    - Forecast revision: Change in consensus EPS forecast after filing
    - Forecast dispersion: Std dev of analyst forecasts

    Args:
        db: WRDS connection
        filings_df: DataFrame with ticker, filing_date

    Returns:
        DataFrame with analyst forecast metrics
    """
    print("\nFetching I/B/E/S analyst forecasts...")

    # Get unique tickers
    tickers = filings_df['ticker'].unique().tolist()
    tickers = [t for t in tickers if t != 'UNKNOWN']
    tickers_str = "','".join(tickers)

    # Get date range
    min_date = filings_df['filing_date'].min() - timedelta(days=90)
    max_date = filings_df['filing_date'].max() + timedelta(days=90)

    print(f"  Fetching forecasts for {len(tickers):,} tickers from {min_date} to {max_date}")

    # Fetch summary statistics (consensus forecasts)
    query = f"""
    SELECT ticker, statpers as forecast_date, fpedats as fiscal_end,
           meanest as mean_forecast, stdev as forecast_stdev, numest as num_analysts
    FROM ibes.statsum_epsus
    WHERE ticker IN ('{tickers_str}')
    AND statpers >= '{min_date}'
    AND statpers <= '{max_date}'
    AND fpi = '1'  -- Annual forecasts
    AND measure = 'EPS'
    ORDER BY ticker, statpers
    """

    try:
        ibes_df = db.raw_sql(query)
        print(f"[OK] Fetched {len(ibes_df):,} forecast observations")
    except Exception as e:
        print(f"ERROR fetching I/B/E/S data: {e}")
        return pd.DataFrame()

    # Calculate forecast revisions for each filing
    results = []
    for _, filing in tqdm(filings_df.iterrows(), total=len(filings_df), desc="Computing forecast metrics"):
        ticker = filing['ticker']
        filing_date = filing['filing_date']

        if ticker == 'UNKNOWN':
            continue

        # Get forecasts for this ticker
        ticker_forecasts = ibes_df[ibes_df['ticker'] == ticker].copy()
        ticker_forecasts['forecast_date'] = pd.to_datetime(ticker_forecasts['forecast_date'])

        if len(ticker_forecasts) == 0:
            continue

        # Forecast before filing (within 30 days before)
        forecast_before = ticker_forecasts[
            (ticker_forecasts['forecast_date'] >= filing_date - timedelta(days=30)) &
            (ticker_forecasts['forecast_date'] < filing_date)
        ].sort_values('forecast_date').tail(1)

        # Forecast after filing (within 30 days after)
        forecast_after = ticker_forecasts[
            (ticker_forecasts['forecast_date'] > filing_date) &
            (ticker_forecasts['forecast_date'] <= filing_date + timedelta(days=30))
        ].sort_values('forecast_date').head(1)

        if len(forecast_before) > 0 and len(forecast_after) > 0:
            mean_before = forecast_before['mean_forecast'].iloc[0]
            mean_after = forecast_after['mean_forecast'].iloc[0]

            # Forecast revision (percentage change)
            if mean_before != 0:
                forecast_revision = (mean_after - mean_before) / abs(mean_before)
            else:
                forecast_revision = np.nan

            # Dispersion after filing
            forecast_dispersion = forecast_after['forecast_stdev'].iloc[0]
            num_analysts = forecast_after['num_analysts'].iloc[0]
        else:
            forecast_revision = np.nan
            forecast_dispersion = np.nan
            num_analysts = np.nan

        result = {
            'ticker': ticker,
            'filing_date': filing_date,
            'forecast_revision': forecast_revision,
            'forecast_dispersion': forecast_dispersion,
            'num_analysts': num_analysts
        }
        results.append(result)

    return pd.DataFrame(results)

# ============================================
# Main Pipeline
# ============================================

def main(test_mode=False):
    """
    Main execution pipeline.

    Args:
        test_mode: If True, process only 100 filings for testing
    """
    print("="*80)
    print("FETCHING OUTCOME VARIABLES FROM WRDS")
    print("="*80)

    # Connect to WRDS
    db = connect_wrds()
    if db is None:
        return

    # Load filing metadata
    print(f"\nLoading filing metadata from {METADATA_FILE}...")
    metadata_df = pd.read_parquet(METADATA_FILE)

    # Filter to filings with dates and tickers
    metadata_df = metadata_df[
        (metadata_df['filing_date'].notna()) &
        (metadata_df['ticker'] != 'UNKNOWN')
    ].copy()

    print(f"Loaded {len(metadata_df):,} filings with valid dates and tickers")

    if test_mode:
        metadata_df = metadata_df.sample(n=min(100, len(metadata_df)), random_state=42)
        print(f"\nTEST MODE: Using {len(metadata_df)} filings")

    # Step 1: Map tickers to PERMNOs
    permno_mapping = get_permno_from_ticker(db, metadata_df)

    # Merge with filings (match on ticker and date range)
    metadata_df['filing_date'] = pd.to_datetime(metadata_df['filing_date'])
    permno_mapping['start_date'] = pd.to_datetime(permno_mapping['start_date'])
    permno_mapping['end_date'] = pd.to_datetime(permno_mapping['end_date'])

    # For each filing, find the permno valid at that date
    filings_with_permno = []
    for _, filing in metadata_df.iterrows():
        ticker = filing['ticker']
        filing_date = filing['filing_date']

        matching = permno_mapping[
            (permno_mapping['ticker'] == ticker) &
            (permno_mapping['start_date'] <= filing_date) &
            (permno_mapping['end_date'] >= filing_date)
        ]

        if len(matching) > 0:
            filing_with_permno = filing.to_dict()
            filing_with_permno['permno'] = matching.iloc[0]['permno']
            filings_with_permno.append(filing_with_permno)

    filings_df = pd.DataFrame(filings_with_permno)
    print(f"\n[OK] Matched {len(filings_df):,} filings to CRSP PERMNOs")

    # Step 2: Fetch CRSP returns
    crsp_outcomes = fetch_crsp_returns(db, filings_df)

    # Step 3: Fetch I/B/E/S forecasts
    ibes_outcomes = fetch_ibes_forecasts(db, filings_df)

    # Step 4: Merge outcomes
    print("\nMerging outcomes...")
    outcomes_df = filings_df.merge(
        crsp_outcomes,
        on=['permno', 'filing_date'],
        how='left'
    )

    if len(ibes_outcomes) > 0:
        outcomes_df = outcomes_df.merge(
            ibes_outcomes,
            on=['ticker', 'filing_date'],
            how='left'
        )

    # Save outcomes
    print(f"\nSaving outcomes to {OUTPUT_FILE}...")
    outcomes_df.to_parquet(OUTPUT_FILE, index=False)

    # Summary
    print("\n" + "="*80)
    print("OUTCOME SUMMARY")
    print("="*80)
    print(f"Total filings: {len(outcomes_df):,}")
    print(f"\nMarket-based outcomes:")
    print(f"  CAR[-1,+1]: {outcomes_df['car_minus1_plus1'].notna().sum():,} ({outcomes_df['car_minus1_plus1'].notna().sum()/len(outcomes_df)*100:.1f}%)")
    print(f"  Drift (30d): {outcomes_df['drift_30d'].notna().sum():,} ({outcomes_df['drift_30d'].notna().sum()/len(outcomes_df)*100:.1f}%)")
    print(f"  Volume ratio: {outcomes_df['volume_ratio'].notna().sum():,} ({outcomes_df['volume_ratio'].notna().sum()/len(outcomes_df)*100:.1f}%)")

    if 'forecast_revision' in outcomes_df.columns:
        print(f"\nAnalyst-based outcomes:")
        print(f"  Forecast revision: {outcomes_df['forecast_revision'].notna().sum():,} ({outcomes_df['forecast_revision'].notna().sum()/len(outcomes_df)*100:.1f}%)")
        print(f"  Forecast dispersion: {outcomes_df['forecast_dispersion'].notna().sum():,} ({outcomes_df['forecast_dispersion'].notna().sum()/len(outcomes_df)*100:.1f}%)")

    print(f"\nOutput file: {OUTPUT_FILE}")
    print("\nNext step: Re-run 03_sae_training.py with real outcomes to get meaningful Lasso results!")

    # Close WRDS connection
    db.close()
    print("\n[OK] WRDS connection closed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch outcome variables from WRDS")
    parser.add_argument('--test', action='store_true', help='Test mode: process only 100 filings')
    args = parser.parse_args()

    main(test_mode=args.test)
