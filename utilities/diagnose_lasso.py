"""
Diagnostic script to understand why Lasso selected 0 features
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

print("="*80)
print("LASSO DIAGNOSTIC ANALYSIS")
print("="*80)

# Load data
print("\nLoading data...")
df = pd.read_parquet('data/doc_features.parquet')
print(f"Shape: {df.shape}")
print(f"Columns: {len(df.columns)}")

# Check outcomes
print("\n" + "="*80)
print("OUTCOME VARIABLES")
print("="*80)
for col in ['car_3d', 'drift_30d', 'drift_60d']:
    print(f"\n{col}:")
    print(f"  Mean: {df[col].mean():.6f}")
    print(f"  Std: {df[col].std():.6f}")
    print(f"  Min: {df[col].min():.6f}")
    print(f"  Max: {df[col].max():.6f}")
    print(f"  Null: {df[col].isnull().sum()}")

# Check SAE features
sae_cols = [c for c in df.columns if c.startswith('sae_')]
print(f"\n" + "="*80)
print(f"SAE FEATURES ({len(sae_cols)} total)")
print("="*80)
print(f"\nFirst 5 SAE features:")
for col in sae_cols[:5]:
    print(f"\n{col}:")
    print(f"  Mean: {df[col].mean():.6f}")
    print(f"  Std: {df[col].std():.6f}")
    print(f"  Non-zero: {(df[col] != 0).sum()} / {len(df)} ({(df[col] != 0).mean()*100:.1f}%)")

# Check CLN/KMNZ features
cln_kmnz_cols = [c for c in df.columns if 'novelty_cln' in c or 'relevance_kmnz' in c]
print(f"\n" + "="*80)
print(f"CLN/KMNZ FEATURES ({len(cln_kmnz_cols)} total)")
print("="*80)
for col in cln_kmnz_cols:
    print(f"\n{col}:")
    print(f"  Mean: {df[col].mean():.6f}")
    print(f"  Std: {df[col].std():.6f}")

# Check correlations
print("\n" + "="*80)
print("CORRELATION ANALYSIS (car_3d)")
print("="*80)

y = df['car_3d'].values
X_sae = df[sae_cols].values
X_cln_kmnz = df[cln_kmnz_cols].values

# Compute correlations
print("\nTop 10 SAE features by absolute correlation with car_3d:")
corrs = []
for i, col in enumerate(sae_cols):
    corr = np.corrcoef(df[col].values, y)[0, 1]
    corrs.append((col, corr))
corrs.sort(key=lambda x: abs(x[1]), reverse=True)
for col, corr in corrs[:10]:
    print(f"  {col}: {corr:.6f}")

print("\nCLN/KMNZ correlations with car_3d:")
for col in cln_kmnz_cols:
    corr = np.corrcoef(df[col].values, y)[0, 1]
    print(f"  {col}: {corr:.6f}")

# Test simple Lasso
print("\n" + "="*80)
print("SIMPLE LASSO TEST")
print("="*80)

# Subsample for speed
n_sample = min(10000, len(df))
idx = np.random.choice(len(df), n_sample, replace=False)

y_sample = y[idx]
X_sae_sample = X_sae[idx]
X_cln_kmnz_sample = X_cln_kmnz[idx]

# Standardize
scaler_sae = StandardScaler()
scaler_cln = StandardScaler()
X_sae_scaled = scaler_sae.fit_transform(X_sae_sample)
X_cln_scaled = scaler_cln.fit_transform(X_cln_kmnz_sample)

# Combined features
X_combined = np.hstack([X_cln_scaled, X_sae_scaled])

print(f"\nSample size: {n_sample}")
print(f"Features: {X_combined.shape[1]} (CLN/KMNZ: {X_cln_scaled.shape[1]}, SAE: {X_sae_scaled.shape[1]})")

# Try multiple alpha values
alphas = [0.0001, 0.0003, 0.001, 0.003, 0.01]
print("\nLasso results for different alphas:")
for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=1000)
    lasso.fit(X_combined, y_sample)
    n_selected = np.sum(lasso.coef_ != 0)
    n_sae_selected = np.sum(lasso.coef_[X_cln_scaled.shape[1]:] != 0)
    print(f"  Alpha={alpha:.4f}: {n_selected} total ({n_sae_selected} SAE)")

# Check if outcome is truly random
print("\n" + "="*80)
print("RANDOMNESS TEST")
print("="*80)
print("\nAre outcomes truly random?")
print("If car_3d is uniformly distributed in [0,1], it's placeholder data")
print(f"  car_3d range: [{df['car_3d'].min():.4f}, {df['car_3d'].max():.4f}]")
print(f"  Expected for uniform: [0.0, 1.0]")

# Check if it's uniform
hist, edges = np.histogram(df['car_3d'].values, bins=10)
print(f"\nHistogram (should be flat for uniform):")
for i, (count, edge) in enumerate(zip(hist, edges[:-1])):
    bar = '#' * int(count / len(df) * 100)
    print(f"  [{edge:.2f}, {edges[i+1]:.2f}): {bar} ({count})")
