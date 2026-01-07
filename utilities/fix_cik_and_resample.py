"""
Quick fix: Add CIK to existing sentences and re-run stratified sampling
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm

OUTPUT_DIR = Path(r"C:\Users\ofs4963\Dropbox\Arojects\SAE\data")

def extract_cik_from_accession(accession_number):
    """Extract CIK from SEC accession number format: XXXXXXXXXX-YY-ZZZZZZ"""
    if pd.isna(accession_number):
        return None
    parts = str(accession_number).strip().split('-')
    if len(parts) >= 2:
        try:
            return int(parts[0])
        except ValueError:
            return None
    return None

print("="*60)
print("FIXING CIK AND RE-SAMPLING")
print("="*60)

# Load existing sentences file
print("\nLoading sentences.parquet (this may take a minute)...")
sent_df = pd.read_parquet(OUTPUT_DIR / "sentences.parquet")
print(f"Loaded {len(sent_df):,} sentences")

# Add CIK column if it doesn't exist or is all zeros
if 'cik' not in sent_df.columns or (sent_df['cik'] == 0).all():
    print("\nExtracting CIK from accession numbers...")
    sent_df['cik'] = sent_df['accession_number'].apply(extract_cik_from_accession)

    cik_extracted = sent_df['cik'].notna().sum()
    print(f"  Successfully extracted CIK for {cik_extracted:,} / {len(sent_df):,} records ({cik_extracted/len(sent_df)*100:.1f}%)")
    print(f"  Unique firms (CIK): {sent_df['cik'].nunique():,}")

    # Save updated sentences file
    print("\nSaving updated sentences.parquet...")
    sent_df.to_parquet(OUTPUT_DIR / "sentences.parquet", index=False)
    print("  ✓ Saved")
else:
    print(f"\nCIK column already exists with {sent_df['cik'].nunique():,} unique firms")

# Now run the stratified sampling
print("\n" + "="*60)
print("CREATING STRATIFIED SAMPLE WITH FIRM-LEVEL BALANCE")
print("="*60)

def create_sae_training_sample(sentence_df, target_n=5_000_000,
                               sentences_per_firm=1000,
                               max_per_year_item=50_000,
                               random_state=42):
    """Create stratified sample with firm-level balance."""

    print(f"\n  Multi-stage stratified sampling strategy:")
    print(f"    - Stage 1: Max {sentences_per_firm:,} sentences per firm (CIK)")
    print(f"    - Stage 2: Max {max_per_year_item:,} per (year, item_type)")
    print(f"    - Stage 3: Global cap at {target_n:,}")

    # Stage 1: Sample per firm
    print(f"\n  Stage 1: Firm-level sampling...")
    firm_samples = []
    firms_with_missing_cik = sentence_df['cik'].isna().sum()

    if firms_with_missing_cik > 0:
        print(f"   WARNING: {firms_with_missing_cik:,} sentences have missing CIK")

    for cik, group in tqdm(sentence_df.groupby('cik', dropna=False), desc="  Sampling firms"):
        if pd.isna(cik):
            n_sample = min(len(group), 100)
            sampled = group.sample(n=n_sample, random_state=random_state) if n_sample < len(group) else group
        else:
            n_firm = len(group)
            n_sample = min(n_firm, sentences_per_firm)
            sampled = group.sample(n=n_sample, random_state=random_state) if n_sample < n_firm else group

        firm_samples.append(sampled)

    firm_balanced_df = pd.concat(firm_samples, ignore_index=True)
    n_firms_sampled = sentence_df['cik'].nunique()
    print(f"  After firm sampling: {len(firm_balanced_df):,} sentences from {n_firms_sampled:,} firms")

    # Stage 2: Balance by (year, item_type)
    print(f"\n  Stage 2: Year × Item sampling...")
    year_item_samples = []

    for (year, item), group in tqdm(firm_balanced_df.groupby(['year', 'item_type']),
                                     desc="  Sampling year×item cells"):
        n_cell = len(group)
        n_sample = min(n_cell, max_per_year_item)

        if n_sample < n_cell:
            sampled = group.sample(n=n_sample, random_state=random_state)
        else:
            sampled = group

        year_item_samples.append(sampled)

    sample_df = pd.concat(year_item_samples, ignore_index=True)
    print(f"  After year×item sampling: {len(sample_df):,} sentences")

    # Stage 3: Global cap if needed
    if len(sample_df) > target_n:
        print(f"\n  Stage 3: Global cap {len(sample_df):,} → {target_n:,}")
        sample_df = sample_df.sample(n=target_n, random_state=random_state)

    # Sort
    sample_df = sample_df.sort_values(['year', 'item_type', 'cik', 'accession_number', 'sentence_id'])
    sample_df = sample_df.reset_index(drop=True)

    # Report coverage
    print(f"\n  Final sample statistics:")
    print(f"    Total sentences: {len(sample_df):,}")
    print(f"    Unique firms: {sample_df['cik'].nunique():,}")
    print(f"    Sampling rate: {len(sample_df)/len(sentence_df)*100:.1f}%")

    print(f"\n  Sentences per firm (distribution):")
    firm_counts = sample_df[sample_df['cik'].notna()].groupby('cik').size()
    if len(firm_counts) > 0:
        print(f"    Mean: {firm_counts.mean():.1f}")
        print(f"    Median: {firm_counts.median():.1f}")
        print(f"    Min: {firm_counts.min()}")
        print(f"    Max: {firm_counts.max()}")

    return sample_df

# Run sampling
sae_sample_df = create_sae_training_sample(
    sent_df,
    target_n=5_000_000,
    sentences_per_firm=1000,
    max_per_year_item=50_000,
    random_state=42
)

# Save
sae_sample_output = OUTPUT_DIR / "sentences_sae_train.parquet"
sae_sample_df.to_parquet(sae_sample_output, index=False)
print(f"\n✓ Saved {len(sae_sample_df):,} sentences to {sae_sample_output.name}")

print("\n" + "="*60)
print("COMPLETE!")
print("="*60)
print(f"Updated files:")
print(f"  - sentences.parquet (now with CIK column)")
print(f"  - sentences_sae_train.parquet (firm-balanced sample)")
