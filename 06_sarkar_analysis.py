"""
Phase 6: Sarkar Pricing Function Analysis

Implements the representation-pricing framework from main.tex Section 4.4-4.5:
1. Aggregate sentence-level SAE activations to firm-level representations Z_{f,t}
2. Estimate pricing function w_t via 5-fold split-sample ridge regression
3. Compute priced new representation: w_t · ΔZ_{f,t}
4. Orthogonal decomposition: Z^{on} vs Z^{⊥}
5. Test rational inattention predictions on announcement returns and drift

Following Sarkar's framework: separate representation (what is disclosed) from
pricing function (what investors value), then test whether novel content orthogonal
to the pricing function shows weaker announcement reactions but stronger drift.

Usage:
  python 06_sarkar_analysis.py --sae-checkpoint data/sae_anthropic_sentences_v2/sae_final.pt
  python 06_sarkar_analysis.py --sae-checkpoint data/sae_anthropic_sentences_item1_v2/sae_final.pt --item item1
  python 06_sarkar_analysis.py --sae-checkpoint ... --forward-in-time  # Use only data up to t-1
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import json

# =========================
# 1. Load Anthropic SAE Model
# =========================

class AnthropicSAE(nn.Module):
    """
    Anthropic-style Sparse Autoencoder
    (Must match architecture from 03b_sae_anthropic_v2.py)
    """
    def __init__(self, input_dim, n_features):
        super().__init__()
        self.input_dim = input_dim
        self.n_features = n_features

        self.encoder = nn.Linear(input_dim, n_features, bias=True)
        self.decoder = nn.Linear(n_features, input_dim, bias=True)

    def forward(self, x):
        f = torch.relu(self.encoder(x))
        x_hat = self.decoder(f)
        return x_hat, f

def load_sae_model(checkpoint_path, device='cpu'):
    """Load trained SAE model from checkpoint"""
    print(f"\nLoading SAE model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint['config']
    model = AnthropicSAE(
        input_dim=config['input_dim'],
        n_features=config['n_features']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"  Input dim: {config['input_dim']}")
    print(f"  Features: {config['n_features']}")
    print(f"  Training step: {checkpoint['step']}")

    return model, config

# =========================
# 2. Extract SAE Activations
# =========================

@torch.no_grad()
def extract_sae_activations(model, embeddings, batch_size=2048, device='cpu'):
    """
    Extract SAE feature activations for all sentence embeddings

    Args:
        model: Trained AnthropicSAE model
        embeddings: (N, D) numpy array of sentence embeddings
        batch_size: Batch size for processing
        device: Device to use

    Returns:
        activations: (N, F) numpy array of feature activations
    """
    model.eval()
    n_samples = len(embeddings)
    n_features = model.n_features

    activations = np.zeros((n_samples, n_features), dtype=np.float32)

    print(f"\nExtracting SAE activations for {n_samples:,} sentences...")
    for i in tqdm(range(0, n_samples, batch_size), desc="Extracting activations"):
        batch = embeddings[i:i+batch_size]
        batch_tensor = torch.from_numpy(batch).float().to(device)

        _, f = model(batch_tensor)
        activations[i:i+batch_size] = f.cpu().numpy()

    print(f"[OK] Extracted activations: {activations.shape}")
    print(f"  Mean active features per sentence: {(activations > 0).sum(axis=1).mean():.1f}")
    print(f"  Feature activation frequency: {(activations > 0).mean(axis=0).mean():.4f}")

    return activations

# =========================
# 3. Aggregate to Firm-Level Representations
# =========================

def aggregate_to_firm_level(sent_df, activations):
    """
    Aggregate sentence-level SAE activations to firm-level representations Z_{f,t}

    Z_{f,t} = (1/N_{f,t}) * sum_{i in filing (f,t)} z_{i,t}

    Args:
        sent_df: DataFrame with columns [accession_number, cik, filing_date, sentence_id, ...]
        activations: (N, F) numpy array of sentence-level SAE activations

    Returns:
        firm_df: DataFrame with columns [accession_number, cik, filing_date, Z_0, ..., Z_F-1]
    """
    print("\nAggregating sentence activations to firm-level representations...")

    # Add activations to sentence dataframe
    n_features = activations.shape[1]
    sent_df = sent_df.copy()

    for i in range(n_features):
        sent_df[f'z_{i}'] = activations[:, i]

    # Aggregate to firm level (mean across sentences in each filing)
    feature_cols = [f'z_{i}' for i in range(n_features)]

    firm_df = sent_df.groupby(['accession_number', 'cik', 'filing_date'], as_index=False)[feature_cols].mean()

    # Rename columns to Z_i
    rename_dict = {f'z_{i}': f'Z_{i}' for i in range(n_features)}
    firm_df.rename(columns=rename_dict, inplace=True)

    print(f"[OK] Aggregated to {len(firm_df):,} firm-year filings")
    print(f"  Firms: {firm_df['cik'].nunique():,}")
    print(f"  Years: {pd.to_datetime(firm_df['filing_date']).dt.year.nunique()}")

    return firm_df

# =========================
# 4. Pricing Function Estimation (Split-Sample Ridge)
# =========================

def estimate_pricing_function_split_sample(
    firm_df,
    outcome_col='car_minus1_plus1',
    n_folds=5,
    alphas=np.logspace(-3, 3, 50),
    year_col='filing_year',
    forward_in_time=True
):
    """
    Estimate pricing function w_t via 5-fold split-sample ridge regression

    Following main.tex Section 4.4:
    - Partition firms into 5 folds (firm-level to preserve independence)
    - For each fold s, estimate ridge coefficients on other 4 folds
    - Apply to predict outcomes for fold s
    - Forward-in-time: use only data up to t-1 when analyzing year-t filings

    Args:
        firm_df: DataFrame with representation Z and outcomes
        outcome_col: Column name for outcome variable (e.g., CAR announcement)
        n_folds: Number of folds for split-sample
        alphas: Ridge regularization parameters to cross-validate
        year_col: Column name for year variable
        forward_in_time: If True, estimate w_t using only data up to t-1

    Returns:
        results_df: DataFrame with columns [accession_number, y_pred, w_t_fold, ...]
        pricing_weights: Dict mapping (year, fold) -> w_t coefficients
    """
    print(f"\n{'='*80}")
    print("PRICING FUNCTION ESTIMATION (5-FOLD SPLIT-SAMPLE RIDGE)")
    print(f"{'='*80}")
    print(f"Outcome variable: {outcome_col}")
    print(f"Forward-in-time: {forward_in_time}")
    print(f"Ridge alphas: {len(alphas)} values from {alphas.min():.1e} to {alphas.max():.1e}")

    # Identify feature columns (Z_0, Z_1, ..., Z_{F-1})
    feature_cols = [col for col in firm_df.columns if col.startswith('Z_')]
    n_features = len(feature_cols)
    print(f"Features: {n_features}")

    # Extract representations and outcomes
    Z = firm_df[feature_cols].values
    y = firm_df[outcome_col].values

    # Create firm-level folds (assign each firm to a fold)
    firms = firm_df['cik'].unique()
    np.random.seed(42)
    firm_to_fold = {firm: i % n_folds for i, firm in enumerate(np.random.permutation(firms))}
    firm_df['fold'] = firm_df['cik'].map(firm_to_fold)

    print(f"\nFold assignment:")
    for fold in range(n_folds):
        n_firms = (firm_df['fold'] == fold).sum()
        print(f"  Fold {fold}: {n_firms:,} observations")

    # Storage for predictions and pricing weights
    y_pred = np.zeros(len(firm_df))
    pricing_weights = {}

    if forward_in_time:
        # Estimate separately for each year, using only prior years
        years = sorted(firm_df[year_col].unique())
        print(f"\nEstimating pricing function for {len(years)} years...")

        for year in tqdm(years, desc="Years"):
            # Training set: all years before current year
            train_mask = firm_df[year_col] < year

            if train_mask.sum() == 0:
                print(f"  [SKIP] Year {year}: No prior data available")
                continue

            # Test set: current year
            test_mask = firm_df[year_col] == year

            # For each fold in test year, estimate on other folds in training years
            for fold in range(n_folds):
                # Test: current year, current fold
                test_idx = test_mask & (firm_df['fold'] == fold)

                if test_idx.sum() == 0:
                    continue

                # Train: prior years, other folds
                train_idx = train_mask & (firm_df['fold'] != fold)

                if train_idx.sum() < 10:
                    # Not enough training data
                    continue

                # Fit ridge regression with cross-validation
                ridge = RidgeCV(alphas=alphas, cv=3)
                ridge.fit(Z[train_idx], y[train_idx])

                # Predict on test fold
                y_pred[test_idx] = ridge.predict(Z[test_idx])

                # Store pricing weights
                pricing_weights[(year, fold)] = ridge.coef_

    else:
        # Standard cross-validation (no forward-in-time constraint)
        print(f"\nEstimating pricing function (standard CV)...")

        for fold in tqdm(range(n_folds), desc="Folds"):
            # Test: current fold
            test_idx = firm_df['fold'] == fold

            # Train: other folds
            train_idx = firm_df['fold'] != fold

            # Fit ridge regression
            ridge = RidgeCV(alphas=alphas, cv=3)
            ridge.fit(Z[train_idx], y[train_idx])

            # Predict on test fold
            y_pred[test_idx] = ridge.predict(Z[test_idx])

            # Store pricing weights (use year=-1 as placeholder for pooled)
            pricing_weights[(-1, fold)] = ridge.coef_

    # Add predictions to dataframe
    results_df = firm_df.copy()
    results_df['y_pred'] = y_pred
    results_df['y_actual'] = y

    # Compute R-squared
    valid_mask = ~np.isnan(y_pred) & ~np.isnan(y)
    if valid_mask.sum() > 0:
        ss_res = np.sum((y[valid_mask] - y_pred[valid_mask])**2)
        ss_tot = np.sum((y[valid_mask] - np.mean(y[valid_mask]))**2)
        r2 = 1 - ss_res / ss_tot
        print(f"\n[RESULTS] Split-sample R-squared: {r2:.4f}")
        print(f"  Correlation(y, y_pred): {np.corrcoef(y[valid_mask], y_pred[valid_mask])[0,1]:.4f}")
    else:
        print("\n[WARNING] No valid predictions generated")

    print(f"{'='*80}\n")

    return results_df, pricing_weights

# =========================
# 5. Orthogonal Decomposition
# =========================

def compute_orthogonal_decomposition(firm_df, pricing_weights, year_col='filing_year'):
    """
    Compute orthogonal decomposition of representations into on-pricing and off-pricing components

    Following main.tex Section 4.4:
    Z^{on}_{f,t} = P · Z_{f,t}    where P = w_t w_t^T / (w_t^T w_t)
    Z^{⊥}_{f,t} = (I - P) · Z_{f,t}

    Args:
        firm_df: DataFrame with representation Z and fold assignments
        pricing_weights: Dict mapping (year, fold) -> w_t coefficients
        year_col: Column name for year variable

    Returns:
        firm_df: DataFrame with additional columns [Z^on_0, ..., Z^⊥_0, ...]
    """
    print(f"\n{'='*80}")
    print("ORTHOGONAL DECOMPOSITION")
    print(f"{'='*80}")

    # Identify feature columns
    feature_cols = [col for col in firm_df.columns if col.startswith('Z_') and not col.startswith('Z^')]
    n_features = len(feature_cols)

    # Initialize decomposition columns
    for i in range(n_features):
        firm_df[f'Z^on_{i}'] = np.nan
        firm_df[f'Z^perp_{i}'] = np.nan

    # For each (year, fold) combination, compute decomposition
    for (year, fold), w_t in tqdm(pricing_weights.items(), desc="Computing decomposition"):
        # Get observations for this (year, fold)
        if year == -1:
            # Pooled model (no year-specific weights)
            mask = firm_df['fold'] == fold
        else:
            mask = (firm_df[year_col] == year) & (firm_df['fold'] == fold)

        if mask.sum() == 0:
            continue

        # Get representations for this subset
        Z = firm_df.loc[mask, feature_cols].values  # (N, F)

        # Compute projection matrix P = w_t w_t^T / (w_t^T w_t)
        w_t = w_t.reshape(-1, 1)  # (F, 1)
        P = (w_t @ w_t.T) / (w_t.T @ w_t)  # (F, F)

        # On-pricing component: Z^on = Z @ P^T (since we want to project each row)
        Z_on = Z @ P.T  # (N, F)

        # Off-pricing component: Z^⊥ = Z @ (I - P)^T
        Z_perp = Z @ (np.eye(n_features) - P).T  # (N, F)

        # Store in dataframe
        for i in range(n_features):
            firm_df.loc[mask, f'Z^on_{i}'] = Z_on[:, i]
            firm_df.loc[mask, f'Z^perp_{i}'] = Z_perp[:, i]

    # Verify decomposition: Z = Z^on + Z^⊥
    Z_reconstructed = firm_df[[f'Z^on_{i}' for i in range(n_features)]].values + \
                      firm_df[[f'Z^perp_{i}' for i in range(n_features)]].values
    Z_original = firm_df[feature_cols].values

    valid_mask = ~np.isnan(Z_reconstructed).any(axis=1)
    if valid_mask.sum() > 0:
        reconstruction_error = np.abs(Z_reconstructed[valid_mask] - Z_original[valid_mask]).max()
        print(f"[VERIFY] Max reconstruction error: {reconstruction_error:.2e}")
        if reconstruction_error > 1e-6:
            print(f"  [WARNING] Large reconstruction error detected!")

    print(f"[OK] Computed decomposition for {valid_mask.sum():,} observations")
    print(f"{'='*80}\n")

    return firm_df

# =========================
# 6. Compute Changes (ΔZ^on, ΔZ^⊥)
# =========================

def compute_representation_changes(firm_df, year_col='filing_year'):
    """
    Compute year-over-year changes in representations

    ΔZ_{f,t} = Z_{f,t} - Z_{f,t-1}
    ΔZ^{on}_{f,t} = Z^{on}_{f,t} - Z^{on}_{f,t-1}
    ΔZ^{⊥}_{f,t} = Z^{⊥}_{f,t} - Z^{⊥}_{f,t-1}

    Args:
        firm_df: DataFrame with representations and decompositions
        year_col: Column name for year variable

    Returns:
        firm_df: DataFrame with additional columns [ΔZ_0, ..., ΔZ^on_0, ..., ΔZ^⊥_0, ...]
    """
    print("\nComputing representation changes (ΔZ, ΔZ^on, ΔZ^⊥)...")

    # Sort by firm and year
    firm_df = firm_df.sort_values(['cik', year_col])

    # Identify feature columns
    Z_cols = [col for col in firm_df.columns if col.startswith('Z_') and not ('on' in col or 'perp' in col)]
    Zon_cols = [col for col in firm_df.columns if col.startswith('Z^on_')]
    Zperp_cols = [col for col in firm_df.columns if col.startswith('Z^perp_')]

    # Compute changes (shift within firm)
    for cols, prefix in [(Z_cols, 'ΔZ_'), (Zon_cols, 'ΔZ^on_'), (Zperp_cols, 'ΔZ^perp_')]:
        for col in cols:
            # Extract feature index
            feat_idx = col.split('_')[-1]
            delta_col = f'{prefix}{feat_idx}'

            # Compute change: current - lagged (within firm)
            firm_df[delta_col] = firm_df.groupby('cik')[col].diff()

    # Count valid changes (exclude first year for each firm)
    valid_changes = ~firm_df[[f'ΔZ_{i}' for i in range(len(Z_cols))]].isna().all(axis=1)
    print(f"[OK] Computed changes for {valid_changes.sum():,} observations")
    print(f"  Excluded first-year observations: {(~valid_changes).sum():,}")

    return firm_df

# =========================
# 7. Return Regressions (Announcement and Drift)
# =========================

def run_return_regressions(firm_df, outcome_col='car_minus1_plus1', control_cols=None):
    """
    Test rational inattention predictions via return regressions

    Following main.tex Section 4.5:
    r_{f,t} = α + β_on · ΔZ^{on}_{f,t} + β_⊥ · ΔZ^{⊥}_{f,t} + Γ X_{f,t} + ε_{f,t}

    RI predictions:
    - Announcement: |β_⊥| < |β_on| (off-pricing content ignored)
    - Drift: |γ'_⊥| > |γ'_on| (off-pricing content gradually priced)

    Args:
        firm_df: DataFrame with ΔZ^on, ΔZ^⊥, and outcomes
        outcome_col: Column name for outcome variable
        control_cols: List of control variable column names

    Returns:
        results: OLS regression results
        test_results: Dict with RI test statistics
    """
    print(f"\n{'='*80}")
    print(f"RETURN REGRESSIONS: {outcome_col}")
    print(f"{'='*80}")

    # Aggregate ΔZ^on and ΔZ^⊥ to scalar measures
    # Use L2 norm as default aggregation
    Zon_cols = [col for col in firm_df.columns if col.startswith('ΔZ^on_')]
    Zperp_cols = [col for col in firm_df.columns if col.startswith('ΔZ^perp_')]

    firm_df['ΔZ_on_norm'] = np.linalg.norm(firm_df[Zon_cols].fillna(0).values, axis=1)
    firm_df['ΔZ_perp_norm'] = np.linalg.norm(firm_df[Zperp_cols].fillna(0).values, axis=1)

    # Prepare regression data
    reg_cols = ['ΔZ_on_norm', 'ΔZ_perp_norm']
    if control_cols:
        reg_cols += control_cols

    # Filter to valid observations (non-missing outcome and regressors)
    valid_mask = firm_df[outcome_col].notna() & firm_df[reg_cols].notna().all(axis=1)

    print(f"Valid observations: {valid_mask.sum():,} / {len(firm_df):,}")

    if valid_mask.sum() < 10:
        print("[ERROR] Insufficient observations for regression")
        return None, None

    # Prepare regression
    y = firm_df.loc[valid_mask, outcome_col]
    X = firm_df.loc[valid_mask, reg_cols]
    X = sm.add_constant(X)

    # Run OLS with firm-clustered standard errors
    model = OLS(y, X)

    # Try clustered SEs if possible
    try:
        cik_clusters = firm_df.loc[valid_mask, 'cik']
        results = model.fit(cov_type='cluster', cov_kwds={'groups': cik_clusters})
        print("Using firm-clustered standard errors")
    except:
        results = model.fit()
        print("Using standard errors (clustering failed)")

    # Print results
    print("\n" + str(results.summary()))

    # Extract coefficients
    β_on = results.params['ΔZ_on_norm']
    β_perp = results.params['ΔZ_perp_norm']

    # Test RI predictions
    test_results = {
        'β_on': β_on,
        'β_perp': β_perp,
        'se_β_on': results.bse['ΔZ_on_norm'],
        'se_β_perp': results.bse['ΔZ_perp_norm'],
        'p_β_on': results.pvalues['ΔZ_on_norm'],
        'p_β_perp': results.pvalues['ΔZ_perp_norm'],
        'abs_β_on': abs(β_on),
        'abs_β_perp': abs(β_perp),
    }

    print(f"\n{'='*80}")
    print("RATIONAL INATTENTION TEST")
    print(f"{'='*80}")
    print(f"β_on (on-pricing): {β_on:.6f} (p={test_results['p_β_on']:.4f})")
    print(f"β_⊥ (off-pricing): {β_perp:.6f} (p={test_results['p_β_perp']:.4f})")
    print(f"\nPrediction: |β_⊥| < |β_on| for announcement, |γ'_⊥| > |γ'_on| for drift")
    print(f"Observed: |β_⊥| = {abs(β_perp):.6f}, |β_on| = {abs(β_on):.6f}")

    if outcome_col in ['car_minus1_plus1', 'announcement_return']:
        if abs(β_perp) < abs(β_on):
            print("✓ Consistent with RI: off-pricing content has weaker announcement effect")
        else:
            print("✗ Inconsistent with RI: off-pricing content has stronger announcement effect")

    print(f"{'='*80}\n")

    return results, test_results

# =========================
# 8. Main
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sarkar pricing function analysis")
    parser.add_argument('--sae-checkpoint', type=str, required=True,
                        help='Path to trained SAE checkpoint (e.g., sae_final.pt)')
    parser.add_argument('--unit', type=str, default='sentences', choices=['sentences', 'spans'])
    parser.add_argument('--item', type=str, default=None, choices=['item1', 'item1A', 'item7'])
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of folds for split-sample')
    parser.add_argument('--forward-in-time', action='store_true',
                        help='Estimate w_t using only data up to t-1 (default: False)')

    args = parser.parse_args()

    # Set up paths
    DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

    print("="*80)
    print("PHASE 6: SARKAR PRICING FUNCTION ANALYSIS")
    print("="*80)
    print(f"SAE checkpoint: {args.sae_checkpoint}")
    print(f"Analysis unit: {args.unit}")
    print(f"Item filter: {args.item}")
    print(f"Forward-in-time: {args.forward_in_time}")
    print("="*80)

    # Load SAE model
    model, config = load_sae_model(args.sae_checkpoint, device=args.device)

    # Load sentence data
    UNIT_FILE = os.path.join(DATA_DIR, "sentences_sampled.parquet")
    print(f"\nLoading sentences from {UNIT_FILE}...")
    sent_df = pd.read_parquet(UNIT_FILE)

    # Filter by item if specified
    if args.item:
        print(f"Filtering to {args.item}...")
        sent_df = sent_df[sent_df['item_type'] == args.item].copy()

    print(f"Loaded {len(sent_df):,} sentences")

    # Load embeddings (same process as 03b)
    print("\nLoading embeddings...")
    emb_chunks_dir = os.path.join(DATA_DIR, "sentence_embeddings_chunks")

    all_embeddings = []
    all_metadata = []

    chunk_files = sorted([f for f in os.listdir(emb_chunks_dir) if f.endswith('.npz')])
    for chunk_file in chunk_files:
        chunk_path = os.path.join(emb_chunks_dir, chunk_file)
        data = np.load(chunk_path, allow_pickle=True)
        all_embeddings.append(data['embeddings'])
        all_metadata.append(pd.DataFrame({
            'accession_number': data['accession_numbers'],
            'sentence_id': data['sentence_ids'],
            'item_type': data['item_types']
        }))
        print(f"  Loaded {chunk_file}: {len(data['embeddings']):,} embeddings")

    embeddings = np.vstack(all_embeddings)
    emb_df = pd.concat(all_metadata, ignore_index=True)
    print(f"[OK] Loaded {len(embeddings):,} embeddings")

    # Filter embeddings to match item
    if args.item:
        print(f"\nFiltering embeddings to {args.item}...")
        item_mask = emb_df['item_type'] == args.item
        embeddings = embeddings[item_mask]
        emb_df = emb_df[item_mask].copy()

    # Merge and align
    print("\nMerging sentences with embeddings...")
    merged_df = sent_df.merge(
        emb_df[['accession_number', 'sentence_id', 'item_type']],
        on=['accession_number', 'sentence_id', 'item_type'],
        how='inner'
    )

    # Reorder embeddings to match merged sentences
    emb_df['_orig_idx'] = np.arange(len(emb_df))
    merged_with_idx = merged_df.merge(
        emb_df[['accession_number', 'sentence_id', 'item_type', '_orig_idx']],
        on=['accession_number', 'sentence_id', 'item_type'],
        how='left'
    )
    idx_mapping = merged_with_idx['_orig_idx'].values
    embeddings = embeddings[idx_mapping]
    sent_df = merged_df.copy()

    # Normalize embeddings (same as training)
    D = embeddings.shape[1]
    target_norm = np.sqrt(D)
    current_norm = np.mean(np.linalg.norm(embeddings, axis=1))
    embeddings = embeddings * (target_norm / current_norm)

    # Extract SAE activations
    activations = extract_sae_activations(model, embeddings, device=args.device)

    # Aggregate to firm-level representations
    firm_df = aggregate_to_firm_level(sent_df, activations)

    # Add filing year
    firm_df['filing_date'] = pd.to_datetime(firm_df['filing_date'])
    firm_df['filing_year'] = firm_df['filing_date'].dt.year

    # Load WRDS outcomes (from setup/00c_fetch_outcomes_wrds.py)
    wrds_file = os.path.join(DATA_DIR, "outcomes_wrds.parquet")
    if os.path.exists(wrds_file):
        print(f"\nLoading WRDS market data from {wrds_file}...")
        wrds_df = pd.read_parquet(wrds_file)
        print(f"  Loaded {len(wrds_df):,} observations")

        # Merge with firm representations
        # Note: outcomes_wrds.parquet has outcome columns but no controls (size, bm, etc.)
        merge_cols = ['accession_number', 'car_minus1_plus1', 'drift_30d', 'drift_60d', 'drift_90d']
        # Only include control columns if they exist
        available_controls = [col for col in ['size', 'bm', 'leverage', 'past_ret', 'past_vol']
                              if col in wrds_df.columns]
        if available_controls:
            merge_cols.extend(available_controls)

        firm_df = firm_df.merge(
            wrds_df[merge_cols],
            on='accession_number',
            how='left'
        )
        print(f"  Merged {firm_df['car_minus1_plus1'].notna().sum():,} observations with returns data")

        # Control variables (if available)
        control_cols = available_controls if available_controls else None
    else:
        print(f"\n[WARNING] WRDS data not found at {wrds_file}")
        print("  Using synthetic returns for demonstration")
        print("  Run setup/00c_fetch_outcomes_wrds.py first to get real data!")

        # Generate synthetic data
        np.random.seed(42)
        firm_df['car_minus1_plus1'] = np.random.randn(len(firm_df)) * 0.05
        firm_df['drift_30d'] = np.random.randn(len(firm_df)) * 0.10
        firm_df['drift_60d'] = np.random.randn(len(firm_df)) * 0.15
        firm_df['drift_90d'] = np.random.randn(len(firm_df)) * 0.20

        control_cols = None

    # Estimate pricing function
    results_df, pricing_weights = estimate_pricing_function_split_sample(
        firm_df,
        outcome_col='car_minus1_plus1',
        n_folds=args.n_folds,
        forward_in_time=args.forward_in_time
    )

    # Compute orthogonal decomposition
    results_df = compute_orthogonal_decomposition(results_df, pricing_weights)

    # Compute changes
    results_df = compute_representation_changes(results_df)

    # Run return regressions
    print("\n" + "="*80)
    print("ANNOUNCEMENT RETURN REGRESSIONS")
    print("="*80)
    ann_results, ann_tests = run_return_regressions(
        results_df,
        outcome_col='car_minus1_plus1',
        control_cols=control_cols
    )

    if 'drift_30d' in results_df.columns:
        print("\n" + "="*80)
        print("DRIFT RETURN REGRESSIONS (30-day)")
        print("="*80)
        drift_results, drift_tests = run_return_regressions(
            results_df,
            outcome_col='drift_30d',
            control_cols=control_cols
        )

    # Save results
    output_dir = os.path.join(DATA_DIR, f"sarkar_analysis_{args.unit}")
    if args.item:
        output_dir += f"_{args.item}"
    os.makedirs(output_dir, exist_ok=True)

    # Save firm-level representations and decompositions
    results_df.to_parquet(os.path.join(output_dir, 'firm_representations.parquet'))
    print(f"\n[SAVE] Firm representations saved to {output_dir}/firm_representations.parquet")

    # Save test results
    test_summary = {
        'announcement': ann_tests if ann_tests else {},
        'drift_30d': drift_tests if 'drift_30d' in results_df.columns and drift_tests else {},
        'config': {
            'sae_checkpoint': args.sae_checkpoint,
            'n_folds': args.n_folds,
            'forward_in_time': args.forward_in_time,
            'n_firms': firm_df['cik'].nunique(),
            'n_observations': len(results_df)
        }
    }

    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        json.dump(test_summary, f, indent=2)
    print(f"[SAVE] Test results saved to {output_dir}/test_results.json")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nNext steps:")
    print(f"  1. Run 00d_wrds_data_merge.py to get real market data")
    print(f"  2. Re-run this script to get meaningful RI test results")
    print("="*80)
