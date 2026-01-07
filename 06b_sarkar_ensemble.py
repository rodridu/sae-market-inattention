"""
Phase 6: Sarkar Pricing Function Analysis (Ensemble Version)

Simplified implementation for ensemble k-sparse SAE features.
Works with doc_features.parquet from Phase 4.

Implements core Sarkar framework from main.tex Section 4.4-4.5:
1. Use document-level SAE features as representations Z_{f,t}
2. Estimate pricing function w_t via Ridge regression on CAR
3. Compute priced new representation: w_t · ΔZ_{f,t}
4. Orthogonal decomposition: Z^{on} vs Z^{⊥}
5. Test rational inattention predictions

Usage:
  python 06b_sarkar_ensemble.py
  python 06b_sarkar_ensemble.py --forward-in-time
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import statsmodels.api as sm
from tqdm import tqdm

# =========================
# 0. Configuration
# =========================

parser = argparse.ArgumentParser(description='Sarkar pricing function analysis (ensemble SAE)')
parser.add_argument('--forward-in-time', action='store_true',
                    help='Estimate w_t using only data up to t-1 (default: False)')
parser.add_argument('--n-folds', type=int, default=5,
                    help='Number of folds for split-sample estimation (default: 5)')
parser.add_argument('--use-mean-features', action='store_true',
                    help='Use mean features only (default: use both mean and freq)')

args = parser.parse_args()

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

print("\n" + "="*80)
print("PHASE 6: SARKAR PRICING FUNCTION ANALYSIS (ENSEMBLE SAE)")
print("="*80)

# =========================
# 1. Load document-level features
# =========================

doc_file = os.path.join(DATA_DIR, "doc_features.parquet")
print(f"\nLoading document-level features from {doc_file}...")

if not os.path.exists(doc_file):
    print(f"ERROR: {doc_file} not found!")
    print("Please run 04_feature_selection.py first.")
    sys.exit(1)

doc_df = pd.read_parquet(doc_file)
print(f"[OK] Loaded {len(doc_df):,} documents")
print(f"     Columns: {len(doc_df.columns)}")

# Identify SAE feature columns
if args.use_mean_features:
    sae_cols = [c for c in doc_df.columns if c.startswith('mean_M')]
    print(f"\nUsing mean features only: {len(sae_cols)}")
else:
    mean_cols = [c for c in doc_df.columns if c.startswith('mean_M')]
    freq_cols = [c for c in doc_df.columns if c.startswith('freq_M')]
    sae_cols = mean_cols + freq_cols
    print(f"\nUsing both mean and freq features: {len(sae_cols)}")
    print(f"  Mean features: {len(mean_cols)}")
    print(f"  Freq features: {len(freq_cols)}")

if len(sae_cols) == 0:
    print("ERROR: No SAE features found in data!")
    sys.exit(1)

# Check outcome coverage
print(f"\nOutcome coverage:")
print(f"  CAR (car_3d): {doc_df['car_3d'].notna().sum():,} / {len(doc_df):,} ({100*doc_df['car_3d'].notna().mean():.1f}%)")
print(f"  30-day drift: {doc_df['drift_30d'].notna().sum():,} / {len(doc_df):,} ({100*doc_df['drift_30d'].notna().mean():.1f}%)")
print(f"  60-day drift: {doc_df['drift_60d'].notna().sum():,} / {len(doc_df):,} ({100*doc_df['drift_60d'].notna().mean():.1f}%)")

# =========================
# 2. Prepare panel data
# =========================

print("\n" + "="*80)
print("PREPARING PANEL DATA")
print("="*80)

# Filter to observations with CAR
panel_df = doc_df[doc_df['car_3d'].notna()].copy()
print(f"\nFiltered to {len(panel_df):,} documents with CAR data")

# Extract year from filing_date
panel_df['year'] = pd.to_datetime(panel_df['filing_date']).dt.year
print(f"Year range: {panel_df['year'].min()} - {panel_df['year'].max()}")

# =========================
# 3. Estimate Pricing Function w_t
# =========================

print("\n" + "="*80)
print("ESTIMATING PRICING FUNCTION w_t")
print("="*80)

# Extract representations Z and outcome y
Z = panel_df[sae_cols].values
y = panel_df['car_3d'].values

print(f"\nRepresentation matrix Z: {Z.shape}")
print(f"Outcome vector y: {y.shape}")

# Standardize features
scaler = StandardScaler()
Z_scaled = scaler.fit_transform(Z)

# Identify CIK column (may be cik, cik_x, or cik_y from merge)
cik_col = None
for col_name in ['cik', 'cik_x', 'cik_y']:
    if col_name in panel_df.columns:
        cik_col = col_name
        break

# Create firm-level folds (assign each firm to a fold)
if cik_col is not None:
    firms = panel_df[cik_col].unique()
    np.random.seed(42)
    firm_to_fold = {firm: i % args.n_folds for i, firm in enumerate(np.random.permutation(firms))}
    panel_df['fold'] = panel_df[cik_col].map(firm_to_fold)
    print(f"\nCreated {args.n_folds} firm-level folds using {cik_col}")
else:
    # Random assignment if no CIK
    panel_df['fold'] = np.random.RandomState(42).randint(0, args.n_folds, len(panel_df))
    print(f"\nCreated {args.n_folds} random folds (no CIK available)")
    cik_col = 'doc_id'  # Use doc_id as fallback for grouping

# Storage for predictions and pricing weights
y_pred = np.zeros(len(panel_df))
pricing_weights = {}

if args.forward_in_time:
    # Estimate separately for each year, using only prior years
    print(f"\nMode: FORWARD-IN-TIME (use only data up to t-1)")
    years = sorted(panel_df['year'].unique())
    print(f"Estimating for {len(years)} years: {years[0]} - {years[-1]}")

    for year in tqdm(years, desc="Years"):
        # Training set: all years before current year
        train_mask = panel_df['year'] < year

        if train_mask.sum() < 50:
            continue

        # Test set: current year
        test_mask = panel_df['year'] == year

        # Estimate on training data, predict on test data
        for fold in range(args.n_folds):
            # Test: current year, current fold
            test_idx = test_mask & (panel_df['fold'] == fold)

            if test_idx.sum() == 0:
                continue

            # Train: prior years, other folds
            train_idx = train_mask & (panel_df['fold'] != fold)

            if train_idx.sum() < 20:
                continue

            # Fit ridge regression
            ridge = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=3)
            ridge.fit(Z_scaled[train_idx], y[train_idx])

            # Predict on test fold
            y_pred[test_idx] = ridge.predict(Z_scaled[test_idx])

            # Store pricing weights (in original scale)
            w_scaled = ridge.coef_
            w_original = w_scaled / scaler.scale_
            pricing_weights[(year, fold)] = w_original

else:
    # Standard cross-validation (no forward-in-time constraint)
    print(f"\nMode: STANDARD CV (pooled estimation)")

    for fold in tqdm(range(args.n_folds), desc="Folds"):
        # Test: current fold
        test_idx = panel_df['fold'] == fold

        # Train: other folds
        train_idx = panel_df['fold'] != fold

        # Fit ridge regression
        ridge = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=3)
        ridge.fit(Z_scaled[train_idx], y[train_idx])

        # Predict on test fold
        y_pred[test_idx] = ridge.predict(Z_scaled[test_idx])

        # Store pricing weights (in original scale)
        w_scaled = ridge.coef_
        w_original = w_scaled / scaler.scale_
        pricing_weights[(-1, fold)] = w_original

# Add predictions to panel
panel_df['car_pred'] = y_pred

# Compute R-squared
valid_mask = ~np.isnan(y_pred) & ~np.isnan(y)
if valid_mask.sum() > 0:
    ss_res = np.sum((y[valid_mask] - y_pred[valid_mask])**2)
    ss_tot = np.sum((y[valid_mask] - np.mean(y[valid_mask]))**2)
    r2 = 1 - ss_res / ss_tot
    corr = np.corrcoef(y[valid_mask], y_pred[valid_mask])[0,1]

    print(f"\n[RESULTS] Pricing Function Performance:")
    print(f"  Split-sample R-squared: {r2:.4f}")
    print(f"  Correlation(y, y_pred): {corr:.4f}")
    print(f"  RMSE: {np.sqrt(ss_res / valid_mask.sum()):.6f}")
else:
    print("\n[WARNING] No valid predictions")
    sys.exit(1)

# =========================
# 4. Compute Priced New Representation
# =========================

print("\n" + "="*80)
print("COMPUTING PRICED NEW REPRESENTATION")
print("="*80)

# Sort by CIK and filing_date to compute changes
panel_df = panel_df.sort_values([cik_col, 'filing_date'])

# Compute changes in representation (ΔZ_{f,t})
for col in sae_cols:
    panel_df[f'delta_{col}'] = panel_df.groupby(cik_col)[col].diff()

# Compute priced new representation: w_t · ΔZ_{f,t}
# For each observation, use the pricing weight from its (year, fold)
panel_df['priced_new_rep'] = 0.0

for (year_key, fold_key), w in pricing_weights.items():
    if year_key == -1:
        # Pooled model - use same weights for all
        mask = panel_df['fold'] == fold_key
    else:
        # Forward-in-time - use year-specific weights
        mask = (panel_df['year'] == year_key) & (panel_df['fold'] == fold_key)

    if mask.sum() == 0:
        continue

    # Extract ΔZ for this subset
    delta_cols = [f'delta_{col}' for col in sae_cols]
    delta_Z = panel_df.loc[mask, delta_cols].values

    # Compute w_t · ΔZ_{f,t}
    priced_new_rep = delta_Z @ w
    panel_df.loc[mask, 'priced_new_rep'] = priced_new_rep

print(f"[OK] Computed priced new representation for {(panel_df['priced_new_rep'] != 0).sum():,} observations")
print(f"  Mean: {panel_df['priced_new_rep'].mean():.6f}")
print(f"  Std: {panel_df['priced_new_rep'].std():.6f}")

# =========================
# 5. Orthogonal Decomposition
# =========================

print("\n" + "="*80)
print("ORTHOGONAL DECOMPOSITION: Z^on vs Z^perp")
print("="*80)

# For each observation, decompose dZ into:
# - dZ^on: projection onto w_t (priced component)
# - dZ^perp: orthogonal to w_t (unpriced component)

panel_df['norm_priced_squared'] = 0.0
panel_df['norm_orthogonal_squared'] = 0.0

for (year_key, fold_key), w in pricing_weights.items():
    if year_key == -1:
        mask = panel_df['fold'] == fold_key
    else:
        mask = (panel_df['year'] == year_key) & (panel_df['fold'] == fold_key)

    if mask.sum() == 0:
        continue

    # Extract ΔZ
    delta_cols = [f'delta_{col}' for col in sae_cols]
    delta_Z = panel_df.loc[mask, delta_cols].values

    # Normalize w
    w_norm = w / np.linalg.norm(w)

    # Project dZ onto w: dZ^on = (dZ · w_norm) * w_norm
    projection = (delta_Z @ w_norm)[:, None] * w_norm[None, :]

    # Orthogonal component: dZ^perp = dZ - dZ^on
    orthogonal = delta_Z - projection

    # Store squared norms
    panel_df.loc[mask, 'norm_priced_squared'] = np.sum(projection**2, axis=1)
    panel_df.loc[mask, 'norm_orthogonal_squared'] = np.sum(orthogonal**2, axis=1)

print(f"[OK] Computed orthogonal decomposition")
print(f"  Mean ||dZ^on||^2: {panel_df['norm_priced_squared'].mean():.6f}")
print(f"  Mean ||dZ^perp||^2: {panel_df['norm_orthogonal_squared'].mean():.6f}")

# =========================
# 6. Test Rational Inattention Predictions
# =========================

print("\n" + "="*80)
print("TESTING RATIONAL INATTENTION PREDICTIONS")
print("="*80)

# Main prediction (Proposal Section 5.1):
# Novel concepts orthogonal to pricing function (dZ^perp) should show:
# 1. Weaker announcement reactions (lower |CAR|)
# 2. Stronger post-filing drift

# Regression 1: CAR ~ dZ^on + dZ^perp + controls
print("\n[REGRESSION 1] CAR ~ priced + orthogonal components")
reg_df = panel_df[['car_3d', 'norm_priced_squared', 'norm_orthogonal_squared']].dropna()

X = sm.add_constant(reg_df[['norm_priced_squared', 'norm_orthogonal_squared']])
y = reg_df['car_3d']

model_car = sm.OLS(y, X).fit(cov_type='HC1')
print(model_car.summary())

# Regression 2: Drift ~ dZ^on + dZ^perp + controls
if 'drift_30d' in panel_df.columns:
    print("\n[REGRESSION 2] 30-day Drift ~ priced + orthogonal components")
    reg_df2 = panel_df[['drift_30d', 'norm_priced_squared', 'norm_orthogonal_squared']].dropna()

    X2 = sm.add_constant(reg_df2[['norm_priced_squared', 'norm_orthogonal_squared']])
    y2 = reg_df2['drift_30d']

    model_drift = sm.OLS(y2, X2).fit(cov_type='HC1')
    print(model_drift.summary())

# =========================
# 7. Save results
# =========================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save panel with Sarkar variables
output_file = os.path.join(DATA_DIR, "sarkar_analysis_results.parquet")
panel_df.to_parquet(output_file, index=False)
print(f"[OK] Saved analysis results to {output_file}")
print(f"     Shape: {panel_df.shape}")

# Save pricing weights
weights_file = os.path.join(DATA_DIR, "pricing_weights.csv")
weights_data = []
for (year_key, fold_key), w in pricing_weights.items():
    for i, col in enumerate(sae_cols):
        weights_data.append({
            'year': year_key,
            'fold': fold_key,
            'feature': col,
            'weight': w[i]
        })
weights_df = pd.DataFrame(weights_data)
weights_df.to_csv(weights_file, index=False)
print(f"[OK] Saved pricing weights to {weights_file}")

# Summary statistics
summary = {
    'n_documents': len(panel_df),
    'n_features': len(sae_cols),
    'r2_pricing': r2,
    'corr_pricing': corr,
    'mean_priced_new_rep': float(panel_df['priced_new_rep'].mean()),
    'mean_norm_priced': float(np.sqrt(panel_df['norm_priced_squared'].mean())),
    'mean_norm_orthogonal': float(np.sqrt(panel_df['norm_orthogonal_squared'].mean())),
    'forward_in_time': args.forward_in_time
}

import json
summary_file = os.path.join(DATA_DIR, "sarkar_summary.json")
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"[OK] Saved summary statistics to {summary_file}")

print("\n" + "="*80)
print("SARKAR ANALYSIS COMPLETE")
print("="*80)
print("\nKey findings:")
print(f"  1. Pricing function R-squared: {r2:.4f}")
print(f"  2. Mean priced component ||dZ^on||: {np.sqrt(panel_df['norm_priced_squared'].mean()):.4f}")
print(f"  3. Mean orthogonal component ||dZ^perp||: {np.sqrt(panel_df['norm_orthogonal_squared'].mean()):.4f}")
print(f"  4. Ratio (orthogonal/priced): {np.sqrt(panel_df['norm_orthogonal_squared'].mean() / panel_df['norm_priced_squared'].mean()):.4f}")
print("\nNext steps:")
print("  - Review regression results above")
print("  - Examine pricing_weights.csv to identify most valued concepts")
print("  - Analyze high-||dZ^perp|| documents for under-processed information")
print("="*80)
