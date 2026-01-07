"""
Stratified Sampling for SAE Training

Three-stage sampling strategy to ensure balanced representation:
  Stage 1a: Max sentences per (firm, year, item) - ensures each filing is treated equally
  Stage 1b: Max sentences per firm - prevents single firms from dominating
  Stage 2:  Max sentences per (year, item) - controls overall dataset size

This addresses the concern that firms with long filing histories should not be
under-sampled compared to firms with short histories. Each 10-K filing is a
separate disclosure event and should receive equal treatment.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

# ============================================
# Configuration
# ============================================

DATA_DIR = Path(r"C:\Users\ofs4963\Dropbox\Arojects\SAE\data")

# Input: All sentences from Phase 1
SENTENCE_FILE = DATA_DIR / "sentences.parquet"

# Output: Sampled sentences for SAE training
SENTENCE_SAMPLE_OUTPUT = DATA_DIR / "sentences_sampled.parquet"

# Default sampling parameters
# Updated based on empirical data: Item1A avg=273, Item7 avg=267, Item1 avg=254
DEFAULT_SENTENCES_PER_FIRM_YEAR_ITEM = 300  # Max sentences per filing (covers ~100% of typical filings)
DEFAULT_SENTENCES_PER_FIRM = 5000            # Max sentences per firm (across all years)
DEFAULT_MAX_PER_YEAR_ITEM = 20000            # Max sentences per (year, item)
RANDOM_STATE = 42

# ============================================
# Stratified Sampling Function
# ============================================

def stratified_sampling(
    sentence_df,
    sentences_per_firm_year_item=DEFAULT_SENTENCES_PER_FIRM_YEAR_ITEM,
    sentences_per_firm=DEFAULT_SENTENCES_PER_FIRM,
    max_per_year_item=DEFAULT_MAX_PER_YEAR_ITEM,
    random_state=RANDOM_STATE,
):
    """
    Three-stage stratified sampling for balanced representation.

    Stage 1a: Sample up to X sentences per (firm, year, item)
      → Ensures each 10-K filing is treated equally
      → Addresses temporal balance: firms with 20 years of filings don't get
        under-sampled compared to firms with 2 years

    Stage 1b: Cap at Y sentences per firm (across all years)
      → Prevents any single firm from dominating the dataset
      → Ensures firm-level diversity

    Stage 2: Cap at Z sentences per (year, item)
      → Controls overall dataset size
      → Maintains temporal and item-type coverage

    Args:
        sentence_df: Full sentence DataFrame with columns:
                     [accession_number, cik, year, item_type, sentence_id, text, ...]
        sentences_per_firm_year_item: Max sentences per filing (default: 500)
        sentences_per_firm: Max sentences per firm across all years (default: 5000)
        max_per_year_item: Max sentences per (year, item) cell (default: 30000)
        random_state: Random seed for reproducibility

    Returns:
        Sampled DataFrame ready for embedding generation
    """

    print("\n" + "="*60)
    print("THREE-STAGE STRATIFIED SAMPLING")
    print("="*60)
    print(f"\nInput: {len(sentence_df):,} sentences")
    print(f"  Firms: {sentence_df['cik'].nunique():,}")
    print(f"  Years: {sentence_df['year'].min():.0f} - {sentence_df['year'].max():.0f}")
    print(f"  Items: {', '.join(sorted(sentence_df['item_type'].unique()))}")

    print(f"\nSampling parameters:")
    print(f"  Stage 1a: Max {sentences_per_firm_year_item:,} sentences per (firm, year, item)")
    print(f"  Stage 1b: Max {sentences_per_firm:,} sentences per firm (total)")
    print(f"  Stage 2:  Max {max_per_year_item:,} sentences per (year, item)")

    # ========================================
    # Stage 1a: Per-filing sampling
    # ========================================
    print(f"\n{'='*60}")
    print("STAGE 1a: Firm-Year-Item Sampling")
    print(f"{'='*60}")
    print(f"Sampling up to {sentences_per_firm_year_item:,} sentences per filing...")

    firm_year_item_samples = []
    n_filings_processed = 0
    n_filings_sampled = 0

    for (cik, year, item), group in tqdm(
        sentence_df.groupby(['cik', 'year', 'item_type'], dropna=False),
        desc="Stage 1a: Sampling filings"
    ):
        n_filings_processed += 1
        n_filing = len(group)
        n_sample = min(n_filing, sentences_per_firm_year_item)

        if n_sample < n_filing:
            sampled = group.sample(n=n_sample, random_state=random_state)
            n_filings_sampled += 1
        else:
            sampled = group

        firm_year_item_samples.append(sampled)

    stage1a_df = pd.concat(firm_year_item_samples, ignore_index=True)

    print(f"\nStage 1a results:")
    print(f"  Filings processed: {n_filings_processed:,}")
    print(f"  Filings down-sampled: {n_filings_sampled:,} ({n_filings_sampled/n_filings_processed*100:.1f}%)")
    print(f"  Sentences after Stage 1a: {len(stage1a_df):,}")
    print(f"  Reduction: {len(sentence_df) - len(stage1a_df):,} ({(1 - len(stage1a_df)/len(sentence_df))*100:.1f}%)")

    # ========================================
    # Stage 1b: Per-firm cap
    # ========================================
    print(f"\n{'='*60}")
    print("STAGE 1b: Firm-Level Cap")
    print(f"{'='*60}")
    print(f"Capping at {sentences_per_firm:,} sentences per firm...")

    firm_samples = []
    n_firms_capped = 0

    for cik, group in tqdm(
        stage1a_df.groupby('cik', dropna=False),
        desc="Stage 1b: Capping firms"
    ):
        n_firm = len(group)
        n_sample = min(n_firm, sentences_per_firm)

        if pd.isna(cik):
            # Handle missing CIKs: take small sample
            n_sample = min(len(group), 100)

        if n_sample < n_firm:
            sampled = group.sample(n=n_sample, random_state=random_state)
            n_firms_capped += 1
        else:
            sampled = group

        firm_samples.append(sampled)

    stage1b_df = pd.concat(firm_samples, ignore_index=True)

    print(f"\nStage 1b results:")
    print(f"  Firms processed: {stage1a_df['cik'].nunique():,}")
    print(f"  Firms capped: {n_firms_capped:,} ({n_firms_capped/stage1a_df['cik'].nunique()*100:.1f}%)")
    print(f"  Sentences after Stage 1b: {len(stage1b_df):,}")
    print(f"  Reduction: {len(stage1a_df) - len(stage1b_df):,} ({(1 - len(stage1b_df)/len(stage1a_df))*100:.1f}%)")

    # ========================================
    # Stage 2: Year×Item balance
    # ========================================
    print(f"\n{'='*60}")
    print("STAGE 2: Year×Item Balance")
    print(f"{'='*60}")
    print(f"Capping at {max_per_year_item:,} sentences per (year, item)...")

    year_item_samples = []
    n_cells_capped = 0

    for (year, item), group in tqdm(
        stage1b_df.groupby(['year', 'item_type']),
        desc="Stage 2: Balancing year×item"
    ):
        n_cell = len(group)
        n_sample = min(n_cell, max_per_year_item)

        if n_sample < n_cell:
            sampled = group.sample(n=n_sample, random_state=random_state)
            n_cells_capped += 1
        else:
            sampled = group

        year_item_samples.append(sampled)

    final_df = pd.concat(year_item_samples, ignore_index=True)

    print(f"\nStage 2 results:")
    print(f"  (Year, Item) cells processed: {stage1b_df.groupby(['year', 'item_type']).ngroups:,}")
    print(f"  Cells capped: {n_cells_capped:,}")
    print(f"  Sentences after Stage 2: {len(final_df):,}")
    print(f"  Reduction: {len(stage1b_df) - len(final_df):,} ({(1 - len(final_df)/len(stage1b_df))*100:.1f}%)")

    # ========================================
    # Final statistics
    # ========================================
    final_df = final_df.sort_values(
        ['year', 'item_type', 'cik', 'accession_number', 'sentence_id']
    ).reset_index(drop=True)

    print(f"\n{'='*60}")
    print("FINAL SAMPLE STATISTICS")
    print(f"{'='*60}")
    print(f"\nOverall:")
    print(f"  Total sentences: {len(final_df):,}")
    print(f"  Unique firms: {final_df['cik'].nunique():,}")
    print(f"  Overall sampling rate: {len(final_df)/len(sentence_df)*100:.1f}%")

    # Sentences per firm distribution
    print(f"\nSentences per firm (distribution):")
    firm_counts = final_df[final_df['cik'].notna()].groupby('cik').size()
    if len(firm_counts) > 0:
        print(f"  Mean:   {firm_counts.mean():.1f}")
        print(f"  Median: {firm_counts.median():.1f}")
        print(f"  Min:    {firm_counts.min()}")
        print(f"  Max:    {firm_counts.max()}")
        print(f"  Std:    {firm_counts.std():.1f}")

        # Show distribution of firms by sentence count
        print(f"\n  Firms with max cap ({sentences_per_firm:,} sentences): {(firm_counts == sentences_per_firm).sum():,}")
        print(f"  Firms with < 100 sentences: {(firm_counts < 100).sum():,}")
        print(f"  Firms with 100-1000 sentences: {((firm_counts >= 100) & (firm_counts < 1000)).sum():,}")
        print(f"  Firms with 1000-5000 sentences: {((firm_counts >= 1000) & (firm_counts < sentences_per_firm)).sum():,}")

    # Temporal coverage
    print(f"\nCoverage by year:")
    year_coverage = final_df.groupby('year').size()
    print(year_coverage.to_string())

    # Item coverage
    print(f"\nCoverage by item:")
    item_coverage = final_df.groupby('item_type').size()
    print(item_coverage.to_string())

    # Year×Item coverage
    print(f"\nCoverage by (year, item):")
    year_item_coverage = final_df.groupby(['year', 'item_type']).size().unstack(fill_value=0)
    print(year_item_coverage.to_string())

    return final_df


# ============================================
# Main Execution
# ============================================

def main(
    sentences_per_firm_year_item=DEFAULT_SENTENCES_PER_FIRM_YEAR_ITEM,
    sentences_per_firm=DEFAULT_SENTENCES_PER_FIRM,
    max_per_year_item=DEFAULT_MAX_PER_YEAR_ITEM,
    force=False,
):
    """
    Run stratified sampling pipeline.

    Args:
        sentences_per_firm_year_item: Max sentences per filing
        sentences_per_firm: Max sentences per firm (total)
        max_per_year_item: Max sentences per (year, item)
        force: If True, re-run sampling even if output exists
    """

    print("="*60)
    print("STRATIFIED SAMPLING FOR SAE TRAINING")
    print("="*60)

    # Check if output already exists
    if SENTENCE_SAMPLE_OUTPUT.exists() and not force:
        print(f"\n[INFO] Output file already exists: {SENTENCE_SAMPLE_OUTPUT}")
        print(f"[INFO] Loading existing sample...")
        sample_df = pd.read_parquet(SENTENCE_SAMPLE_OUTPUT)
        print(f"[INFO] Loaded {len(sample_df):,} sentences from existing sample")
        print(f"\nTo re-run sampling, use --force flag")
        return sample_df

    # Load all sentences
    print(f"\nLoading sentences from {SENTENCE_FILE}...")
    if not SENTENCE_FILE.exists():
        print(f"\nERROR: {SENTENCE_FILE} not found!")
        print("Run 01_data_preparation.py first to generate sentences.parquet")
        return None

    sent_df = pd.read_parquet(SENTENCE_FILE)
    print(f"[OK] Loaded {len(sent_df):,} sentences from {SENTENCE_FILE.name}")

    # Run stratified sampling
    sample_df = stratified_sampling(
        sent_df,
        sentences_per_firm_year_item=sentences_per_firm_year_item,
        sentences_per_firm=sentences_per_firm,
        max_per_year_item=max_per_year_item,
        random_state=RANDOM_STATE,
    )

    # Save sampled data
    print(f"\nSaving sampled sentences to {SENTENCE_SAMPLE_OUTPUT}...")
    sample_df.to_parquet(SENTENCE_SAMPLE_OUTPUT, index=False)
    print(f"[OK] Saved {len(sample_df):,} sentences to {SENTENCE_SAMPLE_OUTPUT.name}")

    # Summary
    print("\n" + "="*60)
    print("SAMPLING COMPLETE")
    print("="*60)
    print(f"Input:  {len(sent_df):,} sentences")
    print(f"Output: {len(sample_df):,} sentences ({len(sample_df)/len(sent_df)*100:.1f}%)")
    print(f"\nOutput file: {SENTENCE_SAMPLE_OUTPUT}")
    print(f"\nNext step:")
    print(f"  Run 02_embeddings_and_features.py to generate embeddings")

    return sample_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Stratified sampling for SAE training'
    )
    parser.add_argument(
        '--sentences-per-filing',
        type=int,
        default=DEFAULT_SENTENCES_PER_FIRM_YEAR_ITEM,
        help=f'Max sentences per (firm, year, item) - i.e., per filing (default: {DEFAULT_SENTENCES_PER_FIRM_YEAR_ITEM})'
    )
    parser.add_argument(
        '--sentences-per-firm',
        type=int,
        default=DEFAULT_SENTENCES_PER_FIRM,
        help=f'Max sentences per firm across all years (default: {DEFAULT_SENTENCES_PER_FIRM})'
    )
    parser.add_argument(
        '--max-per-year-item',
        type=int,
        default=DEFAULT_MAX_PER_YEAR_ITEM,
        help=f'Max sentences per (year, item) cell (default: {DEFAULT_MAX_PER_YEAR_ITEM})'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-run sampling even if output exists'
    )

    args = parser.parse_args()

    sample_df = main(
        sentences_per_firm_year_item=args.sentences_per_filing,
        sentences_per_firm=args.sentences_per_firm,
        max_per_year_item=args.max_per_year_item,
        force=args.force,
    )
