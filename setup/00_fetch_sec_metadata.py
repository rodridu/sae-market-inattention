"""
Phase 0: Fetch Company Metadata from SEC EDGAR

This script fetches company names, tickers, and filing dates from SEC EDGAR APIs
to enrich the sentences dataset with proper metadata.

SEC Data Sources:
1. Company Tickers: https://www.sec.gov/files/company_tickers.json
   Maps CIK → Ticker, Company Name

2. Submissions API: https://data.sec.gov/submissions/CIK{cik}.json
   Contains all filings with dates for each company

SEC Rate Limit: 10 requests per second
User-Agent required for all requests

Usage:
    python 00_fetch_sec_metadata.py
    python 00_fetch_sec_metadata.py --test  # Test with first 100 CIKs only
"""

import requests
import pandas as pd
import numpy as np
from pathlib import Path
import time
import argparse
from tqdm import tqdm
import json

# ============================================
# Configuration
# ============================================

DATA_DIR = Path(r"C:\Users\ofs4963\Dropbox\Arojects\SAE\data")
SENTENCES_FILE = DATA_DIR / "sentences_sae_train.parquet"
OUTPUT_FILE = DATA_DIR / "sec_metadata.parquet"
CACHE_DIR = DATA_DIR / "sec_cache"

# SEC requires User-Agent header
# Replace with your actual contact info
SEC_HEADERS = {
    'User-Agent': 'Academic Research Project yourname@university.edu',
    'Accept-Encoding': 'gzip, deflate',
    'Host': 'data.sec.gov'
}

SEC_RATE_LIMIT = 10  # requests per second
RATE_LIMIT_DELAY = 1.0 / SEC_RATE_LIMIT  # 0.1 seconds between requests

# ============================================
# Step 1: Fetch Company Tickers
# ============================================

def fetch_company_tickers():
    """
    Fetch CIK to ticker mapping from SEC.

    Returns:
        DataFrame with columns: cik, ticker, company_name
    """
    print("\nFetching company tickers from SEC...")
    url = "https://www.sec.gov/files/company_tickers.json"

    # Use generic user agent for this endpoint (different domain)
    headers = {'User-Agent': 'Academic Research Project yourname@university.edu'}

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Convert to DataFrame
        # Format: {0: {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
        records = []
        for key, value in data.items():
            records.append({
                'cik': int(value['cik_str']),
                'ticker': value['ticker'],
                'company_name': value['title']
            })

        df = pd.DataFrame(records)
        print(f"[OK] Fetched {len(df):,} company tickers")
        return df

    except Exception as e:
        print(f"ERROR fetching company tickers: {e}")
        return None

# ============================================
# Step 2: Fetch Filing Metadata per CIK
# ============================================

def fetch_filing_metadata_for_cik(cik, headers, cache_dir=None):
    """
    Fetch all filings for a specific CIK from SEC Submissions API.

    Args:
        cik: Company CIK (integer)
        headers: HTTP headers for SEC request
        cache_dir: Optional directory to cache responses

    Returns:
        DataFrame with columns: accession_number, filing_date, form_type
        Or None if request fails
    """
    cik_str = str(cik).zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_str}.json"

    # Check cache first
    if cache_dir:
        cache_file = cache_dir / f"CIK{cik_str}.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                data = json.load(f)
                filings = data['filings']['recent']
                return pd.DataFrame({
                    'accession_number': filings['accessionNumber'],
                    'filing_date': filings['filingDate'],
                    'form_type': filings['form']
                })

    # Fetch from SEC
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Cache response
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f"CIK{cik_str}.json"
            with open(cache_file, 'w') as f:
                json.dump(data, f)

        # Extract recent filings
        filings = data['filings']['recent']
        return pd.DataFrame({
            'accession_number': filings['accessionNumber'],
            'filing_date': filings['filingDate'],
            'form_type': filings['form']
        })

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            # CIK not found - this is normal for some old/invalid CIKs
            return None
        else:
            print(f"  HTTP error for CIK {cik}: {e}")
            return None
    except Exception as e:
        print(f"  Error fetching CIK {cik}: {e}")
        return None

def fetch_all_filing_metadata(ciks, headers, cache_dir=None, test_mode=False):
    """
    Fetch filing metadata for all CIKs with rate limiting.

    Args:
        ciks: List of CIK integers
        headers: HTTP headers for SEC requests
        cache_dir: Optional directory to cache responses
        test_mode: If True, only process first 100 CIKs

    Returns:
        DataFrame with columns: cik, accession_number, filing_date, form_type
    """
    if test_mode:
        ciks = ciks[:100]
        print(f"\nTEST MODE: Processing only {len(ciks)} CIKs")

    print(f"\nFetching filing metadata for {len(ciks):,} CIKs...")
    print(f"Rate limit: {SEC_RATE_LIMIT} requests/second")
    print(f"Estimated time: ~{len(ciks) / SEC_RATE_LIMIT / 60:.1f} minutes")

    all_filings = []
    failed_ciks = []

    for cik in tqdm(ciks, desc="Fetching CIK metadata"):
        # Fetch metadata for this CIK
        filings_df = fetch_filing_metadata_for_cik(cik, headers, cache_dir)

        if filings_df is not None and len(filings_df) > 0:
            filings_df['cik'] = cik
            all_filings.append(filings_df)
        else:
            failed_ciks.append(cik)

        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)

    # Combine all filings
    if all_filings:
        combined_df = pd.concat(all_filings, ignore_index=True)
        print(f"\n[OK] Fetched {len(combined_df):,} filings for {len(all_filings):,} CIKs")
        if failed_ciks:
            print(f"[WARN] Failed to fetch {len(failed_ciks):,} CIKs (likely not found in SEC database)")
        return combined_df
    else:
        print("\nERROR: No filings fetched")
        return None

# ============================================
# Step 3: Merge with Sentences Dataset
# ============================================

def enrich_sentences_with_metadata(sentences_file, company_tickers_df, filing_metadata_df, output_file):
    """
    Enrich sentences dataset with company tickers and filing dates.

    Args:
        sentences_file: Path to sentences parquet file
        company_tickers_df: DataFrame with cik, ticker, company_name
        filing_metadata_df: DataFrame with cik, accession_number, filing_date
        output_file: Path to save enriched metadata
    """
    print(f"\nLoading sentences from {sentences_file}...")
    sentences_df = pd.read_parquet(sentences_file)
    print(f"Loaded {len(sentences_df):,} sentences")

    # Get unique (cik, accession_number) pairs
    print("\nExtracting unique documents...")
    unique_docs = sentences_df[['cik', 'accession_number']].drop_duplicates()
    print(f"Found {len(unique_docs):,} unique documents")

    # Merge with company tickers
    print("\nMerging with company tickers...")
    enriched = unique_docs.merge(
        company_tickers_df,
        on='cik',
        how='left'
    )

    # Check merge success
    matched_tickers = enriched['ticker'].notna().sum()
    print(f"  Matched {matched_tickers:,} / {len(enriched):,} documents ({matched_tickers/len(enriched)*100:.1f}%)")

    # Merge with filing metadata
    print("\nMerging with filing dates...")
    enriched = enriched.merge(
        filing_metadata_df[['accession_number', 'filing_date', 'form_type']],
        on='accession_number',
        how='left'
    )

    # Check merge success
    matched_dates = enriched['filing_date'].notna().sum()
    print(f"  Matched {matched_dates:,} / {len(enriched):,} documents ({matched_dates/len(enriched)*100:.1f}%)")

    # Fill missing tickers with 'UNKNOWN'
    enriched['ticker'] = enriched['ticker'].fillna('UNKNOWN')
    enriched['company_name'] = enriched['company_name'].fillna('Unknown Company')

    # Convert filing_date to datetime
    enriched['filing_date'] = pd.to_datetime(enriched['filing_date'], errors='coerce')

    # Save enriched metadata
    print(f"\nSaving enriched metadata to {output_file}...")
    enriched.to_parquet(output_file, index=False)
    print(f"[OK] Saved {len(enriched):,} records")

    # Summary statistics
    print("\n" + "="*60)
    print("METADATA ENRICHMENT SUMMARY")
    print("="*60)
    print(f"Total documents: {len(enriched):,}")
    print(f"Documents with ticker: {(enriched['ticker'] != 'UNKNOWN').sum():,}")
    print(f"Documents with filing_date: {enriched['filing_date'].notna().sum():,}")
    print(f"Unique companies: {enriched['ticker'].nunique():,}")
    print(f"Date range: {enriched['filing_date'].min()} to {enriched['filing_date'].max()}")

    return enriched

# ============================================
# Main Execution
# ============================================

def main(test_mode=False):
    """
    Main execution pipeline.

    Args:
        test_mode: If True, process only first 100 CIKs for testing
    """
    print("="*60)
    print("PHASE 0: FETCH SEC METADATA")
    print("="*60)

    # Load sentences to get list of CIKs we need
    print(f"\nLoading sentences from {SENTENCES_FILE}...")
    sentences_df = pd.read_parquet(SENTENCES_FILE)
    unique_ciks = sorted(sentences_df['cik'].unique())
    print(f"Found {len(unique_ciks):,} unique CIKs to fetch")

    # Step 1: Fetch company tickers
    company_tickers_df = fetch_company_tickers()
    if company_tickers_df is None:
        print("ERROR: Failed to fetch company tickers. Exiting.")
        return

    # Step 2: Fetch filing metadata for each CIK
    filing_metadata_df = fetch_all_filing_metadata(
        unique_ciks,
        SEC_HEADERS,
        cache_dir=CACHE_DIR,
        test_mode=test_mode
    )

    if filing_metadata_df is None:
        print("ERROR: Failed to fetch filing metadata. Exiting.")
        return

    # Step 3: Merge with sentences dataset
    enriched_metadata = enrich_sentences_with_metadata(
        SENTENCES_FILE,
        company_tickers_df,
        filing_metadata_df,
        OUTPUT_FILE
    )

    print("\n" + "="*60)
    print("PHASE 0 COMPLETE")
    print("="*60)
    print(f"\nOutput file: {OUTPUT_FILE}")
    print("\nNext step: Merge this metadata with your sentences using:")
    print("  sentences_enriched = sentences.merge(sec_metadata, on=['cik', 'accession_number'])")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch SEC metadata for company tickers and filing dates")
    parser.add_argument('--test', action='store_true', help='Test mode: process only first 100 CIKs')
    args = parser.parse_args()

    main(test_mode=args.test)
