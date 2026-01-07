"""
Merge SEC metadata (tickers, company names, filing dates) with sentences dataset.

This script enriches the sentences_sae_train.parquet file with metadata
fetched from SEC EDGAR in Phase 0.

Input:
- data/sentences_sae_train.parquet (1M sentences with 'UNKNOWN' tickers)
- data/sec_metadata.parquet (company tickers, names, filing dates)

Output:
- data/sentences_sae_train_enriched.parquet (sentences with real metadata)

Usage:
    python 00b_merge_metadata.py
"""

import pandas as pd
from pathlib import Path

# ============================================
# Configuration
# ============================================

DATA_DIR = Path(r"C:\Users\ofs4963\Dropbox\Arojects\SAE\data")
SENTENCES_FILE = DATA_DIR / "sentences_sae_train.parquet"
METADATA_FILE = DATA_DIR / "sec_metadata.parquet"
OUTPUT_FILE = DATA_DIR / "sentences_sae_train_enriched.parquet"

# ============================================
# Main Merge
# ============================================

def main():
    print("="*60)
    print("MERGING SEC METADATA WITH SENTENCES")
    print("="*60)

    # Load sentences
    print(f"\nLoading sentences from {SENTENCES_FILE}...")
    sentences_df = pd.read_parquet(SENTENCES_FILE)
    print(f"  Loaded {len(sentences_df):,} sentences")
    print(f"  Columns: {list(sentences_df.columns)}")

    # Load metadata
    print(f"\nLoading SEC metadata from {METADATA_FILE}...")
    metadata_df = pd.read_parquet(METADATA_FILE)
    print(f"  Loaded {len(metadata_df):,} metadata records")
    print(f"  Columns: {list(metadata_df.columns)}")

    # Drop old ticker column from sentences (will be replaced with SEC data)
    print("\nPreparing for merge...")
    if 'ticker' in sentences_df.columns:
        print("  Dropping old 'ticker' column from sentences")
        sentences_df = sentences_df.drop('ticker', axis=1)

    # Merge on cik and accession_number
    print("\nMerging sentences with metadata on ['cik', 'accession_number']...")
    enriched_df = sentences_df.merge(
        metadata_df[['cik', 'accession_number', 'ticker', 'company_name', 'filing_date', 'form_type']],
        on=['cik', 'accession_number'],
        how='left'
    )

    # Check merge results
    print("\nMerge Results:")
    print(f"  Total sentences: {len(enriched_df):,}")
    print(f"  Sentences with ticker: {enriched_df['ticker'].notna().sum():,} ({enriched_df['ticker'].notna().sum()/len(enriched_df)*100:.1f}%)")
    print(f"  Sentences with company_name: {enriched_df['company_name'].notna().sum():,} ({enriched_df['company_name'].notna().sum()/len(enriched_df)*100:.1f}%)")
    print(f"  Sentences with filing_date: {enriched_df['filing_date'].notna().sum():,} ({enriched_df['filing_date'].notna().sum()/len(enriched_df)*100:.1f}%)")
    print(f"  Sentences with form_type: {enriched_df['form_type'].notna().sum():,} ({enriched_df['form_type'].notna().sum()/len(enriched_df)*100:.1f}%)")

    # Fill missing tickers with 'UNKNOWN'
    enriched_df['ticker'] = enriched_df['ticker'].fillna('UNKNOWN')
    enriched_df['company_name'] = enriched_df['company_name'].fillna('Unknown Company')

    # Show sample of enriched data
    print("\nSample of enriched data:")
    sample_cols = ['accession_number', 'cik', 'ticker', 'company_name', 'filing_date', 'year', 'item_type']
    print(enriched_df[sample_cols].head(10))

    # Save enriched dataset
    print(f"\nSaving enriched sentences to {OUTPUT_FILE}...")
    enriched_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"[OK] Saved {len(enriched_df):,} enriched sentences")

    # Summary statistics
    print("\n" + "="*60)
    print("ENRICHMENT SUMMARY")
    print("="*60)
    print(f"Input sentences: {len(sentences_df):,}")
    print(f"Output sentences: {len(enriched_df):,}")
    print(f"Unique companies: {enriched_df['ticker'].nunique():,}")
    print(f"Companies with real tickers: {(enriched_df['ticker'] != 'UNKNOWN').sum() / len(enriched_df) * 100:.1f}%")
    print(f"Filing date coverage: {enriched_df['filing_date'].notna().sum() / len(enriched_df) * 100:.1f}%")
    print(f"\nNew columns added:")
    print(f"  - ticker (company stock symbol)")
    print(f"  - company_name (full company name)")
    print(f"  - filing_date (exact filing date)")
    print(f"  - form_type (e.g., 10-K, 10-Q)")
    print(f"\nOutput file: {OUTPUT_FILE}")

    # Update other related files
    print("\n" + "="*60)
    print("UPDATING RELATED FILES")
    print("="*60)

    # Also update the novelty and relevance files if they exist
    novelty_file = DATA_DIR / "novelty_cln_sae.parquet"
    relevance_file = DATA_DIR / "relevance_kmnz_sae.parquet"

    if novelty_file.exists():
        print(f"\nUpdating {novelty_file.name}...")
        novelty_df = pd.read_parquet(novelty_file)
        if 'ticker' in novelty_df.columns:
            novelty_df = novelty_df.drop('ticker', axis=1)
        novelty_enriched = novelty_df.merge(
            metadata_df[['cik', 'accession_number', 'ticker', 'company_name', 'filing_date']],
            on=['cik', 'accession_number'],
            how='left'
        )
        novelty_enriched['ticker'] = novelty_enriched['ticker'].fillna('UNKNOWN')
        novelty_enriched['company_name'] = novelty_enriched['company_name'].fillna('Unknown Company')
        novelty_enriched.to_parquet(novelty_file, index=False)
        print(f"  [OK] Updated {novelty_file.name}")

    if relevance_file.exists():
        print(f"\nUpdating {relevance_file.name}...")
        relevance_df = pd.read_parquet(relevance_file)
        if 'ticker' in relevance_df.columns:
            relevance_df = relevance_df.drop('ticker', axis=1)
        relevance_enriched = relevance_df.merge(
            metadata_df[['cik', 'accession_number', 'ticker', 'company_name', 'filing_date']],
            on=['cik', 'accession_number'],
            how='left'
        )
        relevance_enriched['ticker'] = relevance_enriched['ticker'].fillna('UNKNOWN')
        relevance_enriched['company_name'] = relevance_enriched['company_name'].fillna('Unknown Company')
        relevance_enriched.to_parquet(relevance_file, index=False)
        print(f"  [OK] Updated {relevance_file.name}")

    print("\n" + "="*60)
    print("MERGE COMPLETE")
    print("="*60)
    print(f"\nYou can now use {OUTPUT_FILE.name} for your analysis!")
    print("\nNext steps:")
    print("  1. Update 03_sae_training.py to use sentences_sae_train_enriched.parquet")
    print("  2. Re-run SAE training if you want the enriched metadata in final results")
    print("  3. Use company_name and filing_date for analysis and visualization")

if __name__ == "__main__":
    main()
