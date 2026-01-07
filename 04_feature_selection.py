"""
Phase 4: Feature Selection with Lasso

After SAE training (Phase 3), this script:
1. Loads sentence-level data with SAE features
2. Aggregates to document-level (MEMORY-EFFICIENT with numpy)
3. Merges outcome variables
4. Runs Lasso feature selection controlling for CLN novelty and KMNZ relevance
5. Runs OLS regression with selected features

Usage:
  python 04_feature_selection.py
  python 04_feature_selection.py --outcome car_3d
  python 04_feature_selection.py --max-lasso-samples 100000

Aligned with proposal Section 4.6:
  y_fdt = α + β₁·Info^CLN + β₂·Relevance^KMNZ + Σδₖ·Aₖ + Γ'X + ε

Memory optimizations from old 04_extract_features.py:
  - Numpy-based aggregation (no wide DataFrames)
  - Chunked processing of features
  - Pre-allocated arrays for speed
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import gc
from tqdm import tqdm

# Lasso and regressions
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# =========================
# 0. Configuration
# =========================

parser = argparse.ArgumentParser(description='Feature selection with Lasso controlling for CLN/KMNZ')
parser.add_argument('--outcome', default='all',
                    help='Outcome variable: car_3d, drift_30d, drift_60d, or "all" (default: all)')
parser.add_argument('--max-lasso-samples', type=int, default=150000,
                    help='Maximum samples for Lasso (memory optimization, default: 150000)')
parser.add_argument('--cv-folds', type=int, default=5,
                    help='Cross-validation folds for Lasso (default: 5)')
parser.add_argument('--quantile', type=float, default=0.9,
                    help='Quantile threshold for high-activation frequency (default: 0.9)')
parser.add_argument('--agg-chunk-size', type=int, default=256,
                    help='Feature chunk size for aggregation (default: 256)')
parser.add_argument('--min-activation-rate', type=float, default=0.01,
                    help='Minimum activation rate to keep feature alive (default: 0.01)')
parser.add_argument('--temporal-split', action='store_true',
                    help='Use temporal train/test split (pre-split-year for selection, post for testing)')
parser.add_argument('--split-year', type=int, default=2016,
                    help='Year to split train/test (default: 2016)')

args = parser.parse_args()

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

print("\n" + "="*80)
print("PHASE 4: FEATURE SELECTION WITH LASSO")
print("="*80)

# =========================
# 1. Load sentence-level data with SAE features
# =========================

sent_file = os.path.join(DATA_DIR, "sent_df_with_sae.parquet")
print(f"\nLoading sentence-level data from {sent_file}...")

if not os.path.exists(sent_file):
    print(f"ERROR: {sent_file} not found!")
    print("Please run 03_sae_training.py first to generate SAE features.")
    sys.exit(1)

sent_df = pd.read_parquet(sent_file)
print(f"[OK] Loaded {len(sent_df):,} sentences")
print(f"     Columns: {len(sent_df.columns)}")

# Identify SAE feature columns
sae_cols = [c for c in sent_df.columns if c.startswith('h_M')]
n_features = len(sae_cols)
print(f"     SAE features: {n_features}")

if n_features == 0:
    print("ERROR: No SAE features found in data!")
    print("Expected columns starting with 'h_M' (ensemble features)")
    sys.exit(1)

# =========================
# 2. Filter dead features (memory optimization)
# =========================

print("\n" + "="*80)
print("FILTERING DEAD FEATURES")
print("="*80)

print(f"\nComputing activation rates for {n_features} features...")
activation_rates = {}
for col in tqdm(sae_cols, desc="Computing rates"):
    rate = (sent_df[col] > 0).mean()
    activation_rates[col] = rate

# Filter alive features
alive_features = [col for col, rate in activation_rates.items()
                 if rate >= args.min_activation_rate]

print(f"[OK] Alive features: {len(alive_features)} / {n_features} " +
      f"({100*len(alive_features)/n_features:.1f}%)")

if len(alive_features) < n_features:
    dead_features = set(sae_cols) - set(alive_features)
    print(f"  Dropping {len(dead_features)} dead features to save memory")
    sae_cols = alive_features
    n_features = len(sae_cols)

# =========================
# 3. Memory-efficient document aggregation (numpy-based)
# =========================

print("\n" + "="*80)
print("AGGREGATING TO DOCUMENT-LEVEL (MEMORY-EFFICIENT)")
print("="*80)

def aggregate_to_documents_efficient(sent_df, sae_cols, quantile=0.9, agg_chunk_size=256):
    """
    Memory-efficient aggregation using numpy arrays (not wide DataFrames).

    Adapted from old 04_extract_features.py for better memory performance.

    Returns:
        doc_meta: DataFrame with document metadata
        mean_features: (n_docs, n_features) array
        freq_features: (n_docs, n_features) array
    """
    n_features = len(sae_cols)

    print(f"\nAggregating {n_features} features to document level...")
    print(f"  Quantile for binary masks: {quantile}")
    print(f"  Chunk size: {agg_chunk_size}")

    # Extract activations as numpy array
    print("  Extracting activations as numpy array...")
    activations = sent_df[sae_cols].to_numpy(dtype=np.float32)
    print(f"    Activation matrix: {activations.shape} ({activations.nbytes / 1e9:.2f} GB)")

    # Compute thresholds
    print(f"  Computing {quantile} quantile for each feature...")
    thresholds = np.empty(n_features, dtype=np.float32)
    for i in tqdm(range(n_features), desc="Computing quantiles"):
        thresholds[i] = np.quantile(activations[:, i], quantile)

    # Aggregate metadata
    print("  Aggregating metadata by (accession_number, item_type)...")
    agg_dict = {
        "year": ("year", "first"),
        "cik": ("cik", "first"),
    }

    if "novelty_cln" in sent_df.columns:
        agg_dict["novelty_cln_mean"] = ("novelty_cln", "mean")
        agg_dict["novelty_cln_max"] = ("novelty_cln", "max")

    if "relevance_kmnz" in sent_df.columns:
        agg_dict["relevance_kmnz_mean"] = ("relevance_kmnz", "mean")

    doc_meta = sent_df.groupby(["accession_number", "item_type"]).agg(**agg_dict).reset_index()
    n_docs = len(doc_meta)
    doc_meta.insert(0, "doc_id", np.arange(n_docs, dtype=np.int32))

    print(f"  Found {n_docs:,} unique documents")

    # Create document index mapping FIRST (needed for weighted novelty)
    print("  Creating document index mapping...")
    sent_keys = (sent_df["accession_number"].astype(str) + "::" +
                sent_df["item_type"].astype(str)).to_numpy()
    doc_keys = (doc_meta["accession_number"].astype(str) + "::" +
               doc_meta["item_type"].astype(str)).to_numpy()
    doc_to_idx = {k: i for i, k in enumerate(doc_keys)}
    sent_doc_indices = np.fromiter((doc_to_idx[k] for k in sent_keys),
                                  count=len(sent_keys), dtype=np.int32)

    # Compute weighted novelty separately if both columns exist (memory-efficient)
    if "novelty_cln" in sent_df.columns and "relevance_kmnz" in sent_df.columns:
        print("  Computing relevance-weighted novelty (memory-efficient)...")
        # Use numpy for vectorized computation without pandas apply overhead
        novelty = sent_df['novelty_cln'].to_numpy(dtype=np.float32)
        relevance = sent_df['relevance_kmnz'].to_numpy(dtype=np.float32)

        # Compute weighted sum for each document
        novelty_weighted = np.zeros(n_docs, dtype=np.float32)
        relevance_sum = np.zeros(n_docs, dtype=np.float32)

        # Accumulate weighted novelty and relevance sum per document
        np.add.at(novelty_weighted, sent_doc_indices, novelty * relevance)
        np.add.at(relevance_sum, sent_doc_indices, relevance)

        # Normalize by relevance sum
        novelty_weighted = novelty_weighted / np.maximum(relevance_sum, 1e-8)

        doc_meta['novelty_cln_weighted'] = novelty_weighted

    # Initialize feature arrays
    print(f"  Aggregating features for {n_docs:,} documents...")
    mean_features = np.zeros((n_docs, n_features), dtype=np.float32)
    freq_features = np.zeros((n_docs, n_features), dtype=np.float32)
    doc_counts = np.zeros(n_docs, dtype=np.int32)

    # Count sentences per document
    print("  Counting sentences per document...")
    np.add.at(doc_counts, sent_doc_indices, 1)

    # Process features in chunks (memory-efficient)
    print(f"  Processing {n_features} features in chunks of {agg_chunk_size}...")
    for chunk_start in tqdm(range(0, n_features, agg_chunk_size), desc="Feature chunks"):
        chunk_end = min(chunk_start + agg_chunk_size, n_features)

        chunk_act = activations[:, chunk_start:chunk_end]  # View, no copy
        chunk_thr = thresholds[chunk_start:chunk_end]
        chunk_mask = (chunk_act > chunk_thr).astype(np.float32, copy=False)

        # Fast aggregation using np.add.at
        np.add.at(mean_features[:, chunk_start:chunk_end], sent_doc_indices, chunk_act)
        np.add.at(freq_features[:, chunk_start:chunk_end], sent_doc_indices, chunk_mask)

        del chunk_mask

    # Compute means and frequencies
    print("  Computing means / frequencies...")
    denom = np.maximum(doc_counts, 1).astype(np.float32)
    mean_features /= denom[:, None]
    freq_features /= denom[:, None]

    # Add sentence counts to metadata
    doc_meta['n_sentences'] = doc_counts

    print("[OK] Aggregation complete")
    return doc_meta, mean_features, freq_features

doc_meta, mean_features, freq_features = aggregate_to_documents_efficient(
    sent_df, sae_cols, args.quantile, args.agg_chunk_size
)

# Free memory
del sent_df
gc.collect()

# Create feature column names
mean_cols = [f"mean_{col.replace('h_', '')}" for col in sae_cols]
freq_cols = [f"freq_{col.replace('h_', '')}" for col in sae_cols]

# =========================
# 4. Merge outcome variables
# =========================

print("\n" + "="*80)
print("MERGING OUTCOME VARIABLES")
print("="*80)

outcomes_file = os.path.join(DATA_DIR, "outcomes_wrds.parquet")

if os.path.exists(outcomes_file):
    print(f"\nLoading WRDS outcomes from {outcomes_file}...")
    outcomes_df = pd.read_parquet(outcomes_file)

    # Rename CAR column to match pipeline expectations
    outcomes_df = outcomes_df.rename(columns={'car_minus1_plus1': 'car_3d'})

    # Check for duplicates
    n_before = len(outcomes_df)
    duplicates = outcomes_df['accession_number'].duplicated().sum()
    if duplicates > 0:
        print(f"  WARNING: Found {duplicates} duplicate accession_numbers in outcomes")
        print(f"  Keeping first occurrence of each accession_number...")
        outcomes_df = outcomes_df.drop_duplicates(subset='accession_number', keep='first')
        print(f"  Deduplicated: {n_before} -> {len(outcomes_df)} rows")

    # Merge on accession_number (should preserve doc_meta row count)
    n_docs_before = len(doc_meta)
    panel_df = doc_meta.merge(outcomes_df, on='accession_number', how='left', validate='m:1')
    n_docs_after = len(panel_df)

    print(f"[OK] Merged WRDS outcomes. Shape: {panel_df.shape}")
    print(f"     Rows before merge: {n_docs_before:,}")
    print(f"     Rows after merge: {n_docs_after:,}")

    if n_docs_after != n_docs_before:
        print(f"  ERROR: Row count changed after merge! This should not happen.")
        print(f"  The merge created {n_docs_after - n_docs_before} extra rows.")
        sys.exit(1)

    print(f"     CAR coverage: {panel_df['car_3d'].notna().sum():,} observations")
else:
    print(f"\nWARNING: {outcomes_file} not found. Using placeholder outcomes.")
    print("Run setup/00c_fetch_outcomes_wrds.py to fetch real market data.")
    panel_df = doc_meta.copy()

    # Create placeholder outcomes for testing
    np.random.seed(42)
    panel_df['car_3d'] = np.random.randn(len(panel_df)) * 0.05
    panel_df['drift_30d'] = np.random.randn(len(panel_df)) * 0.10
    panel_df['drift_60d'] = np.random.randn(len(panel_df)) * 0.15
    print("[OK] Created placeholder outcomes for testing")

# =========================
# 5. Feature selection with Lasso
# =========================

print("\n" + "="*80)
print("LASSO FEATURE SELECTION")
print("="*80)

# Define outcome variables
outcome_vars = {
    'car_3d': 'Announcement Return (CAR[-1,+1])',
    'drift_30d': '30-Day Post-Filing Drift',
    'drift_60d': '60-Day Post-Filing Drift'
}

if args.outcome != 'all':
    outcome_vars = {args.outcome: outcome_vars.get(args.outcome, args.outcome)}

# Load and merge standard controls
controls_file = os.path.join(DATA_DIR, "controls_wrds.parquet")
if os.path.exists(controls_file):
    print(f"\nLoading control variables from {controls_file}...")
    controls_df = pd.read_parquet(controls_file)

    # Check for duplicates
    duplicates = controls_df['accession_number'].duplicated().sum()
    if duplicates > 0:
        print(f"  WARNING: Found {duplicates} duplicate accession_numbers in controls")
        print(f"  Keeping first occurrence...")
        controls_df = controls_df.drop_duplicates(subset='accession_number', keep='first')

    # Merge controls
    n_before = len(panel_df)
    panel_df = panel_df.merge(
        controls_df[['accession_number', 'size', 'bm', 'leverage', 'past_ret', 'past_vol']],
        on='accession_number', how='left', validate='m:1'
    )
    n_after = len(panel_df)

    if n_after != n_before:
        print(f"  ERROR: Row count changed after controls merge! {n_before} -> {n_after}")
        sys.exit(1)

    controls = ['size', 'bm', 'leverage', 'past_ret', 'past_vol']
    print(f"[OK] Merged {len(controls)} control variables")
    print(f"     Coverage: {panel_df[controls].notna().all(axis=1).sum():,} complete observations")
else:
    print(f"\nWARNING: {controls_file} not found. Proceeding without controls.")
    controls = []

cln_kmnz_cols = [c for c in panel_df.columns if 'novelty_cln' in c or 'relevance_kmnz' in c]

print(f"\nFeature groups:")
print(f"  Controls: {len(controls)}")
print(f"  CLN/KMNZ: {len(cln_kmnz_cols)}")
print(f"  SAE mean features: {len(mean_cols)}")
print(f"  SAE freq features: {len(freq_cols)}")
print(f"  SAE total: {len(mean_cols) + len(freq_cols)}")

# Combine mean and freq features into panel_df (use concat to avoid fragmentation)
print("\nAdding SAE features to panel...")
mean_df = pd.DataFrame(mean_features, columns=mean_cols, index=panel_df.index)
freq_df = pd.DataFrame(freq_features, columns=freq_cols, index=panel_df.index)
panel_df = pd.concat([panel_df, mean_df, freq_df], axis=1)

# Free memory
del mean_features, freq_features, mean_df, freq_df
gc.collect()

sae_all_cols = mean_cols + freq_cols

# Run Lasso for each outcome
all_results = {}

for outcome_var, outcome_name in outcome_vars.items():
    print(f"\n{'='*80}")
    print(f"OUTCOME: {outcome_name} ({outcome_var})")
    print(f"{'='*80}")

    if outcome_var not in panel_df.columns:
        print(f"  WARNING: {outcome_var} not found in data. Skipping.")
        continue

    # Filter valid observations
    valid_mask = ~panel_df[outcome_var].isna()
    print(f"  Valid observations: {valid_mask.sum():,} / {len(panel_df):,}")

    if valid_mask.sum() < 100:
        print(f"  Skipping {outcome_var} - insufficient data")
        continue

    # Subsample if needed (memory optimization)
    if valid_mask.sum() > args.max_lasso_samples:
        print(f"  Subsampling {args.max_lasso_samples:,} observations for Lasso...")
        valid_indices = panel_df[valid_mask].sample(
            n=args.max_lasso_samples,
            random_state=42
        ).index
        data_mask = panel_df.index.isin(valid_indices)
    else:
        data_mask = valid_mask

    # Prepare data
    y = panel_df.loc[data_mask, outcome_var].values

    # Extract features and fill missing values
    if len(controls) > 0:
        X_controls = panel_df.loc[data_mask, controls].fillna(0).values
    else:
        X_controls = np.zeros((len(y), 0))

    X_cln_kmnz = panel_df.loc[data_mask, cln_kmnz_cols].fillna(0).values
    X_sae = panel_df.loc[data_mask, sae_all_cols].fillna(0).values

    print(f"  Data shapes: y={y.shape}, X_controls={X_controls.shape}, " +
          f"X_cln_kmnz={X_cln_kmnz.shape}, X_sae={X_sae.shape}")

    # Temporal split for out-of-sample validation
    if args.temporal_split and 'year' in panel_df.columns:
        print(f"\n  OUT-OF-SAMPLE VALIDATION (train: <{args.split_year}, test: >={args.split_year})")
        years = panel_df.loc[data_mask, 'year'].values
        train_idx = years < args.split_year
        test_idx = years >= args.split_year

        print(f"    Train samples: {train_idx.sum():,} ({100*train_idx.mean():.1f}%)")
        print(f"    Test samples: {test_idx.sum():,} ({100*test_idx.mean():.1f}%)")

        if train_idx.sum() < 100 or test_idx.sum() < 100:
            print(f"    ERROR: Insufficient train or test data. Skipping temporal split.")
            args.temporal_split = False

    # Standardize features
    try:
        scaler_controls = StandardScaler()
        scaler_cln_kmnz = StandardScaler()
        scaler_sae = StandardScaler()

        # Fit scalers on training data (or full data if no split)
        if args.temporal_split and 'year' in panel_df.columns:
            X_controls_scaled = scaler_controls.fit_transform(X_controls[train_idx]) if len(controls) > 0 else X_controls
            X_cln_kmnz_scaled_train = scaler_cln_kmnz.fit_transform(X_cln_kmnz[train_idx]) if len(cln_kmnz_cols) > 0 else X_cln_kmnz[train_idx]
            X_sae_scaled_train = scaler_sae.fit_transform(X_sae[train_idx])

            # Apply scalers to full data
            if len(controls) > 0:
                X_controls_scaled_full = scaler_controls.transform(X_controls)
            else:
                X_controls_scaled_full = X_controls

            X_cln_kmnz_scaled_full = scaler_cln_kmnz.transform(X_cln_kmnz) if len(cln_kmnz_cols) > 0 else X_cln_kmnz
            X_sae_scaled_full = scaler_sae.transform(X_sae)

            # Train set
            X_train = np.hstack([X_controls_scaled, X_cln_kmnz_scaled_train, X_sae_scaled_train])
            y_train = y[train_idx]

            # Full set
            X_full = np.hstack([X_controls_scaled_full, X_cln_kmnz_scaled_full, X_sae_scaled_full])
        else:
            X_controls_scaled = scaler_controls.fit_transform(X_controls) if len(controls) > 0 else X_controls
            X_cln_kmnz_scaled = scaler_cln_kmnz.fit_transform(X_cln_kmnz) if len(cln_kmnz_cols) > 0 else X_cln_kmnz
            X_sae_scaled = scaler_sae.fit_transform(X_sae)

            # Combine: [controls, CLN/KMNZ (forced), SAE (penalized)]
            X_full = np.hstack([X_controls_scaled, X_cln_kmnz_scaled, X_sae_scaled])
            X_train = X_full
            y_train = y

        print(f"  Running LassoCV with {args.cv_folds}-fold CV...")
        lasso = LassoCV(cv=args.cv_folds, n_jobs=-1, max_iter=10000, random_state=42)
        lasso.fit(X_train, y_train)

        # Extract selected SAE features
        n_controls = X_controls.shape[1]
        n_cln_kmnz = X_cln_kmnz.shape[1]
        sae_coefs = lasso.coef_[n_controls + n_cln_kmnz:]

        selected_mask = np.abs(sae_coefs) > 1e-6
        selected_sae_cols = [sae_all_cols[i] for i in range(len(sae_all_cols)) if selected_mask[i]]

        print(f"  [OK] Lasso selected {len(selected_sae_cols)} SAE features (out of {len(sae_all_cols)})")
        print(f"       Alpha: {lasso.alpha_:.6f}")

        # Compute in-sample and out-of-sample R²
        y_pred_train = lasso.predict(X_train)
        r2_train = 1 - np.var(y_train - y_pred_train) / np.var(y_train)
        print(f"       Training R²: {r2_train:.4f}")

        if args.temporal_split and 'year' in panel_df.columns:
            y_test = y[test_idx]
            X_test = X_full[test_idx]
            y_pred_test = lasso.predict(X_test)
            r2_test = 1 - np.var(y_test - y_pred_test) / np.var(y_test)
            print(f"       Test R² (out-of-sample): {r2_test:.4f}")

            # Save OOS validation results
            oos_results = pd.DataFrame({
                'outcome': [outcome_var],
                'split_year': [args.split_year],
                'n_train': [train_idx.sum()],
                'n_test': [test_idx.sum()],
                'n_features_selected': [len(selected_sae_cols)],
                'r2_train': [r2_train],
                'r2_test': [r2_test],
                'alpha': [lasso.alpha_]
            })
            oos_file = os.path.join(DATA_DIR, f"oos_validation_{outcome_var}.csv")
            oos_results.to_csv(oos_file, index=False)
            print(f"  [OK] Saved OOS results to {oos_file}")

        # Save results
        lasso_results = pd.DataFrame({
            'feature': selected_sae_cols,
            'coefficient': sae_coefs[selected_mask],
            'abs_coefficient': np.abs(sae_coefs[selected_mask])
        }).sort_values('abs_coefficient', ascending=False)

        suffix = '_oos' if args.temporal_split else ''
        output_file = os.path.join(DATA_DIR, f"lasso_results_{outcome_var}{suffix}.csv")
        lasso_results.to_csv(output_file, index=False)
        print(f"  [OK] Saved to {output_file}")

        all_results[outcome_var] = {
            'selected_features': selected_sae_cols,
            'n_selected': len(selected_sae_cols),
            'alpha': lasso.alpha_,
            'r2_train': r2_train,
            'r2_test': r2_test if args.temporal_split and 'year' in panel_df.columns else None
        }

    except Exception as e:
        print(f"  ERROR in Lasso: {e}")
        import traceback
        traceback.print_exc()
        continue

    finally:
        gc.collect()

# =========================
# 6. Save outputs
# =========================

print("\n" + "="*80)
print("SAVING OUTPUTS")
print("="*80)

# Save document-level features
doc_features_file = os.path.join(DATA_DIR, "doc_features.parquet")
panel_df.to_parquet(doc_features_file, index=False)
print(f"\n[OK] Saved document-level features to {doc_features_file}")
print(f"     Shape: {panel_df.shape}")

# =========================
# 7. Summary
# =========================

print("\n" + "="*80)
print("FEATURE SELECTION COMPLETE")
print("="*80)

print(f"\nSummary:")
for outcome_var, results in all_results.items():
    if results.get('r2_test') is not None:
        print(f"  {outcome_var}: {results['n_selected']} features selected " +
              f"(R² train={results['r2_train']:.4f}, test={results['r2_test']:.4f})")
    else:
        print(f"  {outcome_var}: {results['n_selected']} features selected " +
              f"(R²={results.get('r2_train', 'N/A'):.4f})")

print(f"\nOutputs:")
print(f"  - Document-level features: {doc_features_file}")
for outcome_var in all_results.keys():
    print(f"  - Lasso results ({outcome_var}): {os.path.join(DATA_DIR, f'lasso_results_{outcome_var}.csv')}")

print(f"\nNext steps:")
print(f"  - Run 05_interpret_features.py for LLM-based interpretation")
print(f"  - Run 06_sarkar_analysis.py for pricing function estimation")
print("="*80)
