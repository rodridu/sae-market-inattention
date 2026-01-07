"""
Merge WRDS control variables with outcomes data
Creates a unified dataset for regression analysis
"""

import pandas as pd
import os

DATA_DIR = r"C:\Users\ofs4963\Dropbox\Arojects\SAE\data"

print("="*60)
print("MERGING WRDS CONTROLS AND OUTCOMES")
print("="*60)

# Load controls
controls_file = os.path.join(DATA_DIR, "controls_wrds.parquet")
print(f"\n1. Loading controls from {controls_file}...")
controls_df = pd.read_parquet(controls_file)
print(f"   Loaded {len(controls_df):,} filings")
print(f"   Columns: {controls_df.columns.tolist()}")

# Load outcomes
outcomes_file = os.path.join(DATA_DIR, "outcomes_wrds.parquet")
print(f"\n2. Loading outcomes from {outcomes_file}...")
outcomes_df = pd.read_parquet(outcomes_file)
print(f"   Loaded {len(outcomes_df):,} filings")
print(f"   Columns: {outcomes_df.columns.tolist()}")

# Merge on accession_number AND permno (composite key)
print(f"\n3. Merging on (accession_number, permno)...")
merged_df = outcomes_df.merge(controls_df, on=['accession_number', 'permno'], how='inner')
print(f"   Merged: {len(merged_df):,} filing-security combinations with complete data")

# Check for duplicate columns (from merging filing_date, year, etc.)
duplicate_cols = [col for col in merged_df.columns if col.endswith('_x') or col.endswith('_y')]
if duplicate_cols:
    print(f"\n   Resolving duplicate columns: {duplicate_cols}")
    # Keep the '_x' version (from outcomes) for date columns
    for col in merged_df.columns:
        if col.endswith('_x'):
            base_col = col[:-2]
            if base_col + '_y' in merged_df.columns:
                # Keep _x, drop _y
                merged_df[base_col] = merged_df[col]
                merged_df = merged_df.drop(columns=[col, base_col + '_y'])
                print(f"     Kept {base_col} from outcomes")

# Data quality summary
print(f"\n4. Data quality summary:")
print(f"   Total filings: {len(merged_df):,}")

outcome_cols = ['car_minus1_plus1', 'drift_30d', 'drift_60d', 'drift_90d', 'forecast_revision']
control_cols = ['size', 'bm', 'leverage', 'past_ret', 'past_vol']

print(f"\n   Outcome coverage:")
for col in outcome_cols:
    if col in merged_df.columns:
        n_valid = merged_df[col].notna().sum()
        pct = 100 * n_valid / len(merged_df)
        print(f"     {col:25s}: {n_valid:7,} ({pct:5.1f}%)")

print(f"\n   Control coverage:")
for col in control_cols:
    if col in merged_df.columns:
        n_valid = merged_df[col].notna().sum()
        pct = 100 * n_valid / len(merged_df)
        print(f"     {col:25s}: {n_valid:7,} ({pct:5.1f}%)")

# Save merged data
output_file = os.path.join(DATA_DIR, "wrds_merged.parquet")
print(f"\n5. Saving to {output_file}...")
merged_df.to_parquet(output_file, index=False)
print(f"   Saved {len(merged_df):,} rows")

print("\n" + "="*60)
print("MERGE COMPLETE")
print("="*60)
print(f"Output: {output_file}")
print("Ready to use in 03_sae_training.py")
