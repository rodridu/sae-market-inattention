"""
Diagnostic v2: Load document-level aggregated features and check correlations
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

print("="*80)
print("LASSO DIAGNOSTIC V2: Document-Level Analysis")
print("="*80)

# Load document-level features
print("\nLoading document-level features...")
df = pd.read_parquet('data/doc_features.parquet')
print(f"Shape: {df.shape}")

# Identify feature types
mean_cols = [c for c in df.columns if c.startswith('mean_M')]
freq_cols = [c for c in df.columns if c.startswith('freq_M')]
cln_kmnz_cols = [c for c in df.columns if 'novelty_cln' in c or 'relevance_kmnz' in c]

print(f"\nFeature counts:")
print(f"  mean features: {len(mean_cols)}")
print(f"  freq features: {len(freq_cols)}")
print(f"  CLN/KMNZ features: {len(cln_kmnz_cols)}")

# Check outcomes
print("\n" + "="*80)
print("OUTCOME STATISTICS")
print("="*80)
for outcome in ['car_3d', 'drift_30d', 'drift_60d']:
    print(f"\n{outcome}:")
    print(f"  Mean: {df[outcome].mean():.6f}")
    print(f"  Std: {df[outcome].std():.6f}")
    print(f"  Range: [{df[outcome].min():.6f}, {df[outcome].max():.6f}]")

# Check if outcomes look like standard normal noise
print("\n" + "="*80)
print("ARE OUTCOMES REAL OR RANDOM?")
print("="*80)
print("\nChecking if outcomes are normally distributed (indicates random data):")
for outcome in ['car_3d', 'drift_30d', 'drift_60d']:
    vals = df[outcome].values
    # Standardize
    vals_std = (vals - vals.mean()) / vals.std()
    # Check if ~68% are within [-1, 1]
    pct_within_1std = np.mean(np.abs(vals_std) <= 1)
    pct_within_2std = np.mean(np.abs(vals_std) <= 2)
    print(f"\n{outcome}:")
    print(f"  % within 1 std: {pct_within_1std*100:.1f}% (expect ~68% for normal)")
    print(f"  % within 2 std: {pct_within_2std*100:.1f}% (expect ~95% for normal)")
    if 0.65 < pct_within_1std < 0.71 and 0.93 < pct_within_2std < 0.97:
        print(f"  --> Looks like RANDOM NORMAL noise")
    else:
        print(f"  --> Distribution deviates from normal")

# Correlation analysis with car_3d
print("\n" + "="*80)
print("CORRELATION WITH car_3d")
print("="*80)

y = df['car_3d'].values

# Top mean features
print("\nTop 10 mean_M features by |correlation|:")
corrs_mean = []
for col in mean_cols:
    corr = np.corrcoef(df[col].values, y)[0, 1]
    corrs_mean.append((col, corr))
corrs_mean.sort(key=lambda x: abs(x[1]), reverse=True)
for col, corr in corrs_mean[:10]:
    print(f"  {col}: {corr:.6f}")

# Top freq features
print("\nTop 10 freq_M features by |correlation|:")
corrs_freq = []
for col in freq_cols:
    corr = np.corrcoef(df[col].values, y)[0, 1]
    corrs_freq.append((col, corr))
corrs_freq.sort(key=lambda x: abs(x[1]), reverse=True)
for col, corr in corrs_freq[:10]:
    print(f"  {col}: {corr:.6f}")

# CLN/KMNZ
print("\nCLN/KMNZ correlations:")
for col in cln_kmnz_cols:
    corr = np.corrcoef(df[col].values, y)[0, 1]
    print(f"  {col}: {corr:.6f}")

# Summary statistics
all_sae_corrs = [abs(c[1]) for c in corrs_mean] + [abs(c[1]) for c in corrs_freq]
print(f"\nSAE feature correlation summary:")
print(f"  Max |corr|: {max(all_sae_corrs):.6f}")
print(f"  Median |corr|: {np.median(all_sae_corrs):.6f}")
print(f"  Mean |corr|: {np.mean(all_sae_corrs):.6f}")

# Test Lasso with different alphas
print("\n" + "="*80)
print("LASSO WITH DIFFERENT ALPHAS")
print("="*80)

# Subsample
n_sample = min(50000, len(df))
idx = np.random.choice(len(df), n_sample, replace=False)

y_sample = y[idx]
X_sae = df[mean_cols + freq_cols].iloc[idx].values
X_cln = df[cln_kmnz_cols].iloc[idx].values

# Standardize
scaler_sae = StandardScaler()
scaler_cln = StandardScaler()
X_sae_scaled = scaler_sae.fit_transform(X_sae)
X_cln_scaled = scaler_cln.fit_transform(X_cln)

X_combined = np.hstack([X_cln_scaled, X_sae_scaled])

print(f"\nSample size: {n_sample}")
print(f"Features: {X_combined.shape[1]} (CLN/KMNZ: {X_cln_scaled.shape[1]}, SAE: {X_sae_scaled.shape[1]})")

alphas = [0.00001, 0.0001, 0.0003, 0.001, 0.003, 0.01]
print("\nLasso selection counts:")
for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=1000, random_state=42)
    lasso.fit(X_combined, y_sample)
    n_total = np.sum(lasso.coef_ != 0)
    n_sae = np.sum(lasso.coef_[X_cln_scaled.shape[1]:] != 0)
    print(f"  Alpha={alpha:.5f}: {n_total} total ({n_sae} SAE, {n_total-n_sae} CLN/KMNZ)")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("\nIf correlations are all near zero and outcomes follow normal distribution,")
print("the outcomes are likely RANDOM PLACEHOLDER DATA, not real market returns.")
print("\nTo get meaningful results:")
print("  1. Collect real announcement returns from CRSP")
print("  2. Collect real post-filing drift data")
print("  3. Re-run 04_feature_selection.py")
