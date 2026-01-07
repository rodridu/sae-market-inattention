"""
Phase 7: Paper Analysis - Regressions, Visualizations, and Tables

Creates publication-ready analysis of selected SAE features:
1. Targeted regressions using only Lasso-selected features
2. Visualizations (feature importance, distributions, time trends)
3. LaTeX regression tables

Usage:
  python 07_paper_analysis.py
  python 07_paper_analysis.py --no-viz  # Skip visualizations
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import FDR correction
from statsmodels.stats.multitest import multipletests

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'

# =========================
# 0. Configuration
# =========================

parser = argparse.ArgumentParser(description='Paper analysis with targeted regressions and visualizations')
parser.add_argument('--no-viz', action='store_true',
                    help='Skip visualizations (only run regressions and tables)')
parser.add_argument('--output-dir', type=str, default='paper_output',
                    help='Directory for output files (default: paper_output)')

args = parser.parse_args()

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, args.output_dir)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n" + "="*80)
print("PHASE 7: PAPER ANALYSIS - REGRESSIONS, VISUALIZATIONS, TABLES")
print("="*80)

# =========================
# 1. Load Data
# =========================

print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

# Load document features
doc_file = os.path.join(DATA_DIR, "doc_features.parquet")
print(f"\nLoading document features from {doc_file}...")
doc_df = pd.read_parquet(doc_file)
print(f"[OK] Loaded {len(doc_df):,} documents")

# Load Lasso results for drift_30d (28 selected features)
lasso_file = os.path.join(DATA_DIR, "lasso_results_drift_30d.csv")
print(f"\nLoading Lasso results from {lasso_file}...")
lasso_df = pd.read_csv(lasso_file)
print(f"[OK] Loaded {len(lasso_df)} selected features")

# Extract selected feature names
selected_features = lasso_df['feature'].tolist()
print(f"\nSelected features: {len(selected_features)}")
print(f"  Mean features: {sum(1 for f in selected_features if f.startswith('mean_'))}")
print(f"  Freq features: {sum(1 for f in selected_features if f.startswith('freq_'))}")

# =========================
# 2. Prepare Regression Data
# =========================

print("\n" + "="*80)
print("PREPARING REGRESSION DATA")
print("="*80)

# Create regression panel with selected features only
reg_df = doc_df[['doc_id', 'accession_number', 'item_type', 'year',
                 'cik_x', 'filing_date', 'car_3d', 'drift_30d', 'drift_60d',
                 'novelty_cln_mean', 'relevance_kmnz_mean'] + selected_features].copy()

# Filter to observations with outcomes
reg_df = reg_df[reg_df['car_3d'].notna()].copy()
print(f"\nRegression sample: {len(reg_df):,} documents")

# Standardize all features for interpretability
scaler = StandardScaler()
features_to_scale = ['novelty_cln_mean', 'relevance_kmnz_mean'] + selected_features
reg_df[features_to_scale] = scaler.fit_transform(reg_df[features_to_scale])

print(f"[OK] Standardized {len(features_to_scale)} features (mean=0, std=1)")

# =========================
# 3. Run Targeted Regressions
# =========================

print("\n" + "="*80)
print("RUNNING TARGETED REGRESSIONS")
print("="*80)

# Define outcome variables
outcomes = {
    'car_3d': 'Announcement Return (CAR[-1,+1])',
    'drift_30d': '30-Day Post-Filing Drift',
    'drift_60d': '60-Day Post-Filing Drift'
}

# Storage for results
regression_results = {}
regression_summaries = []

for outcome_var, outcome_name in outcomes.items():
    print(f"\n{'-'*80}")
    print(f"OUTCOME: {outcome_name}")
    print(f"{'-'*80}")

    # Filter to valid observations
    df_reg = reg_df[[outcome_var, 'novelty_cln_mean', 'relevance_kmnz_mean'] + selected_features].dropna()
    print(f"  N = {len(df_reg):,}")

    if len(df_reg) < 100:
        print(f"  [SKIP] Insufficient observations")
        continue

    # Model 1: Baseline (CLN + KMNZ only)
    X_base = sm.add_constant(df_reg[['novelty_cln_mean', 'relevance_kmnz_mean']])
    y = df_reg[outcome_var]

    model_base = sm.OLS(y, X_base).fit(cov_type='HC1')

    # Model 2: Full (CLN + KMNZ + Selected SAE Features)
    X_full = sm.add_constant(df_reg[['novelty_cln_mean', 'relevance_kmnz_mean'] + selected_features])
    model_full = sm.OLS(y, X_full).fit(cov_type='HC1')

    # Store results
    regression_results[outcome_var] = {
        'baseline': model_base,
        'full': model_full,
        'n_obs': len(df_reg)
    }

    # Print summary
    print(f"\n  Model 1 (Baseline): R-squared = {model_base.rsquared:.4f}")
    print(f"  Model 2 (Full):     R-squared = {model_full.rsquared:.4f}")
    print(f"  Delta R-squared:    {model_full.rsquared - model_base.rsquared:.4f}")

    # F-test for incremental R-squared
    from scipy.stats import f as f_dist
    n = len(df_reg)
    k_base = X_base.shape[1]
    k_full = X_full.shape[1]

    f_stat = ((model_full.ssr - model_base.ssr) / (k_full - k_base)) / (model_base.ssr / (n - k_base))
    f_pval = 1 - f_dist.cdf(f_stat, k_full - k_base, n - k_base)

    print(f"  F-test (SAE features): F({k_full - k_base}, {n - k_base}) = {f_stat:.2f}, p = {f_pval:.4f}")

    regression_summaries.append({
        'outcome': outcome_name,
        'n_obs': len(df_reg),
        'r2_baseline': model_base.rsquared,
        'r2_full': model_full.rsquared,
        'delta_r2': model_full.rsquared - model_base.rsquared,
        'f_stat': f_stat,
        'f_pval': f_pval
    })

# Save regression summary
summary_df = pd.DataFrame(regression_summaries)
summary_file = os.path.join(OUTPUT_DIR, "regression_summary.csv")
summary_df.to_csv(summary_file, index=False)
print(f"\n[OK] Saved regression summary to {summary_file}")

# =========================
# 3b. Apply FDR Multiple Testing Correction
# =========================

print("\n" + "="*80)
print("APPLYING FDR MULTIPLE TESTING CORRECTION")
print("="*80)

def apply_fdr_correction(feature_results, pval_col='p_value', alpha=0.05):
    """
    Apply Benjamini-Hochberg FDR correction to feature p-values

    Args:
        feature_results: DataFrame with feature coefficients and p-values
        pval_col: Name of p-value column
        alpha: FDR threshold (default: 0.05)

    Returns:
        DataFrame with additional columns: p_value_fdr, significant_fdr
    """
    if len(feature_results) == 0:
        return feature_results

    pvals = feature_results[pval_col].values
    reject, pvals_corrected, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')

    feature_results['p_value_fdr'] = pvals_corrected
    feature_results['significant_fdr'] = reject

    # Update significance markers
    feature_results['sig_fdr'] = feature_results.apply(
        lambda row: '***' if row['p_value_fdr'] < 0.01
                    else '**' if row['p_value_fdr'] < 0.05
                    else '*' if row['p_value_fdr'] < 0.10 else '', axis=1
    )

    return feature_results

# Apply FDR correction to each outcome
fdr_results = {}

for outcome_var, outcome_name in outcomes.items():
    if outcome_var not in regression_results:
        continue

    model_full = regression_results[outcome_var]['full']

    # Extract SAE feature coefficients and p-values
    feature_coefs = []
    for feat in selected_features:
        if feat in model_full.params.index:
            feature_coefs.append({
                'feature': feat,
                'coefficient': model_full.params[feat],
                'p_value': model_full.pvalues[feat],
                'abs_coefficient': abs(model_full.params[feat])
            })

    if len(feature_coefs) == 0:
        continue

    feature_df = pd.DataFrame(feature_coefs).sort_values('abs_coefficient', ascending=False)

    # Apply FDR correction
    feature_df = apply_fdr_correction(feature_df, pval_col='p_value', alpha=0.05)

    # Count significant features
    n_sig_uncorrected = (feature_df['p_value'] < 0.05).sum()
    n_sig_fdr = feature_df['significant_fdr'].sum()

    print(f"\n{outcome_name}:")
    print(f"  Total features: {len(feature_df)}")
    print(f"  Significant (p < 0.05, uncorrected): {n_sig_uncorrected}")
    print(f"  Significant (FDR q < 0.05): {n_sig_fdr}")
    print(f"  Reduction: {n_sig_uncorrected - n_sig_fdr} features")

    # Save FDR-corrected results
    fdr_file = os.path.join(OUTPUT_DIR, f"feature_coefficients_{outcome_var}_fdr.csv")
    feature_df.to_csv(fdr_file, index=False)
    print(f"  Saved to {fdr_file}")

    fdr_results[outcome_var] = feature_df

print(f"\n[OK] FDR correction complete for {len(fdr_results)} outcomes")

# =========================
# 3c. Joint F-Test: Drift Features on CAR
# =========================

print("\n" + "="*80)
print("TESTING DRIFT FEATURES ON CAR (JOINT SIGNIFICANCE)")
print("="*80)

print("\nRational Inattention Hypothesis:")
print("  - Drift-predictive features should NOT predict CAR")
print("  - Joint F-test: beta_drift1 = beta_drift2 = ... = beta_drift30 = 0")

# Use drift_30d features (selected by Lasso for drift prediction)
drift_features = lasso_df['feature'].tolist()

# Prepare CAR regression data with drift features FORCED in
df_car = reg_df[['car_3d', 'novelty_cln_mean', 'relevance_kmnz_mean'] + drift_features].dropna()

print(f"\nForcing {len(drift_features)} drift-predictive features into CAR regression")
print(f"  N = {len(df_car):,}")

# Run regression with drift features forced in
X_forced = sm.add_constant(df_car[['novelty_cln_mean', 'relevance_kmnz_mean'] + drift_features])
y_car = df_car['car_3d']

model_forced = sm.OLS(y_car, X_forced).fit(cov_type='HC1')

# Joint F-test for drift features
# H0: All drift feature coefficients = 0
# Construct restriction matrix
n_features = len(drift_features)
R = np.zeros((n_features, len(model_forced.params)))

# Find indices of drift features in parameter vector
for i, feat in enumerate(drift_features):
    if feat in X_forced.columns:
        R[i, X_forced.columns.get_loc(feat)] = 1

# Perform F-test
try:
    f_test_result = model_forced.f_test(R)

    # Extract F-statistic (handle different return formats)
    if isinstance(f_test_result.fvalue, np.ndarray):
        f_stat = f_test_result.fvalue[0][0] if f_test_result.fvalue.ndim > 1 else f_test_result.fvalue[0]
    else:
        f_stat = float(f_test_result.fvalue)

    f_pval = float(f_test_result.pvalue)

    print(f"\nJoint F-test Results:")
    print(f"  Testing {n_features} drift features jointly predict CAR")
    print(f"  F-statistic: {f_stat:.2f}")
    print(f"  p-value: {f_pval:.4f}")
    print(f"  R-squared (with drift features forced in): {model_forced.rsquared:.4f}")

    # Interpretation
    if f_pval > 0.10:
        print(f"\n  Interpretation: PASS - Drift features are jointly insignificant for CAR")
        print(f"  This supports rational inattention: features predict drift but NOT announcement returns")
    else:
        print(f"\n  Interpretation: WARNING - Drift features have some CAR prediction")
        print(f"  p-value < 0.10 suggests features may predict both drift AND CAR")

    # Save results
    car_test_results = pd.DataFrame({
        'test': ['Joint F-test: Drift features on CAR'],
        'n_features': [n_features],
        'f_statistic': [f_stat],
        'p_value': [f_pval],
        'r_squared': [model_forced.rsquared],
        'n_obs': [len(df_car)]
    })

    car_test_file = os.path.join(OUTPUT_DIR, "car_joint_ftest_results.csv")
    car_test_results.to_csv(car_test_file, index=False)
    print(f"\n[OK] Saved joint F-test results to {car_test_file}")

except Exception as e:
    print(f"\n  ERROR in joint F-test: {e}")
    import traceback
    traceback.print_exc()

# =========================
# 4. Create Visualizations
# =========================

if not args.no_viz:
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)

    # Visualization 1: Feature Importance (Drift vs CAR)
    print("\n[VIZ 1] Feature importance comparison (Drift vs CAR)")

    # Extract coefficients for each outcome
    coef_data = []
    for feat in selected_features:
        row = {'feature': feat}

        # CAR coefficients
        if 'car_3d' in regression_results:
            if feat in regression_results['car_3d']['full'].params.index:
                row['coef_car'] = regression_results['car_3d']['full'].params[feat]
                row['pval_car'] = regression_results['car_3d']['full'].pvalues[feat]
            else:
                row['coef_car'] = 0
                row['pval_car'] = 1

        # Drift coefficients
        if 'drift_30d' in regression_results:
            if feat in regression_results['drift_30d']['full'].params.index:
                row['coef_drift30'] = regression_results['drift_30d']['full'].params[feat]
                row['pval_drift30'] = regression_results['drift_30d']['full'].pvalues[feat]
            else:
                row['coef_drift30'] = 0
                row['pval_drift30'] = 1

        coef_data.append(row)

    coef_df = pd.DataFrame(coef_data)
    coef_df['abs_drift30'] = coef_df['coef_drift30'].abs()
    coef_df = coef_df.sort_values('abs_drift30', ascending=False)

    # Plot top 15 features
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Drift coefficients
    top_15 = coef_df.head(15)
    colors = ['red' if p < 0.05 else 'gray' for p in top_15['pval_drift30']]
    axes[0].barh(range(len(top_15)), top_15['coef_drift30'], color=colors)
    axes[0].set_yticks(range(len(top_15)))
    axes[0].set_yticklabels([f.split('_')[-1] for f in top_15['feature']], fontsize=8)
    axes[0].set_xlabel('Coefficient (standardized)', fontsize=10)
    axes[0].set_title('30-Day Drift Coefficients (Top 15)', fontsize=12)
    axes[0].axvline(0, color='black', linestyle='-', linewidth=0.5)
    axes[0].invert_yaxis()

    # Right: CAR vs Drift scatter
    axes[1].scatter(coef_df['coef_car'], coef_df['coef_drift30'],
                   alpha=0.6, s=50, color='steelblue')
    axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.5)
    axes[1].axvline(0, color='gray', linestyle='--', linewidth=0.5)
    axes[1].set_xlabel('CAR Coefficient', fontsize=10)
    axes[1].set_ylabel('30-Day Drift Coefficient', fontsize=10)
    axes[1].set_title('Feature Loadings: CAR vs Drift', fontsize=12)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved to {OUTPUT_DIR}/feature_importance.png")
    plt.close()

    # Visualization 2: R² Comparison
    print("\n[VIZ 2] Model R² comparison")

    fig, ax = plt.subplots(figsize=(10, 6))

    outcomes_list = [s['outcome'] for s in regression_summaries]
    r2_base = [s['r2_baseline'] for s in regression_summaries]
    r2_full = [s['r2_full'] for s in regression_summaries]

    x = np.arange(len(outcomes_list))
    width = 0.35

    ax.bar(x - width/2, r2_base, width, label='Baseline (CLN + KMNZ)', color='lightgray')
    ax.bar(x + width/2, r2_full, width, label='Full (+ SAE Features)', color='steelblue')

    ax.set_ylabel('R-squared', fontsize=12)
    ax.set_title('Model Performance: Baseline vs Full', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([o.replace(' ', '\n') for o in outcomes_list], fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "r2_comparison.png"), dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved to {OUTPUT_DIR}/r2_comparison.png")
    plt.close()

    # Visualization 3: Feature activation distributions
    print("\n[VIZ 3] Feature activation distributions")

    # Get top 6 features by drift coefficient
    top_6 = coef_df.head(6)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, (_, row) in enumerate(top_6.iterrows()):
        feat = row['feature']
        coef = row['coef_drift30']
        pval = row['pval_drift30']

        # Get feature values
        vals = reg_df[feat].dropna()

        axes[idx].hist(vals, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[idx].axvline(0, color='red', linestyle='--', linewidth=1)
        axes[idx].set_title(f"{feat.split('_')[-1]}\nCoef={coef:.4f}, p={pval:.3f}", fontsize=9)
        axes[idx].set_xlabel('Standardized Value', fontsize=8)
        axes[idx].set_ylabel('Frequency', fontsize=8)

    plt.suptitle('Top 6 Drift-Predictive Features: Distributions', fontsize=14, y=1.00)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "feature_distributions.png"), dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved to {OUTPUT_DIR}/feature_distributions.png")
    plt.close()

    # Visualization 4: Time trends in feature usage
    print("\n[VIZ 4] Time trends in feature usage")

    # Compute mean activation by year for top 6 features
    reg_df['year_int'] = reg_df['year'].astype(int)
    yearly_means = reg_df.groupby('year_int')[top_6['feature'].tolist()].mean()

    fig, ax = plt.subplots(figsize=(12, 6))

    for feat in top_6['feature']:
        ax.plot(yearly_means.index, yearly_means[feat], marker='o', label=feat.split('_')[-1], linewidth=2)

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Mean Feature Activation (standardized)', fontsize=12)
    ax.set_title('Time Trends: Top 6 Drift-Predictive Features', fontsize=14)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "time_trends.png"), dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved to {OUTPUT_DIR}/time_trends.png")
    plt.close()

    print(f"\n[OK] All visualizations saved to {OUTPUT_DIR}/")

# =========================
# 5. Create LaTeX Regression Tables
# =========================

print("\n" + "="*80)
print("CREATING LATEX REGRESSION TABLES")
print("="*80)

# Table 1: Main results (Baseline vs Full for all outcomes)
print("\n[TABLE 1] Main regression results")

# Collect models in order: baseline CAR, full CAR, baseline drift30, full drift30, baseline drift60, full drift60
models_for_table = []
model_names = []

for outcome_var, outcome_name in outcomes.items():
    if outcome_var in regression_results:
        models_for_table.append(regression_results[outcome_var]['baseline'])
        models_for_table.append(regression_results[outcome_var]['full'])
        model_names.append(f"{outcome_name}\n(Baseline)")
        model_names.append(f"{outcome_name}\n(Full)")

# Create table with summary_col
table1 = summary_col(
    models_for_table,
    model_names=model_names,
    stars=True,
    float_format='%.4f',
    info_dict={
        'N': lambda x: f"{int(x.nobs):,}",
        'R²': lambda x: f"{x.rsquared:.4f}",
        'Adj. R²': lambda x: f"{x.rsquared_adj:.4f}"
    }
)

# Save LaTeX
table1_file = os.path.join(OUTPUT_DIR, "table1_main_results.tex")
with open(table1_file, 'w') as f:
    f.write(table1.as_latex())
print(f"  [OK] Saved to {table1_file}")

# Also save as text
table1_txt = os.path.join(OUTPUT_DIR, "table1_main_results.txt")
with open(table1_txt, 'w') as f:
    f.write(str(table1))
print(f"  [OK] Saved to {table1_txt}")

# Table 2: Feature coefficients (top 15 features, drift only)
print("\n[TABLE 2] Top 15 feature coefficients (30-day drift)")

if 'drift_30d' in regression_results:
    model_drift = regression_results['drift_30d']['full']

    # Extract coefficients for selected features
    feature_coefs = []
    for feat in selected_features:
        if feat in model_drift.params.index:
            feature_coefs.append({
                'Feature': feat.split('_')[-1],  # Extract neuron ID
                'Type': 'Mean' if feat.startswith('mean_') else 'Freq',
                'Coefficient': model_drift.params[feat],
                'Std Error': model_drift.bse[feat],
                't-stat': model_drift.tvalues[feat],
                'p-value': model_drift.pvalues[feat],
                'Sig': '***' if model_drift.pvalues[feat] < 0.01 else ('**' if model_drift.pvalues[feat] < 0.05 else ('*' if model_drift.pvalues[feat] < 0.10 else ''))
            })

    feature_coefs_df = pd.DataFrame(feature_coefs)
    feature_coefs_df['abs_coef'] = feature_coefs_df['Coefficient'].abs()
    feature_coefs_df = feature_coefs_df.sort_values('abs_coef', ascending=False).head(15)
    feature_coefs_df = feature_coefs_df.drop('abs_coef', axis=1)

    # Save as CSV
    table2_csv = os.path.join(OUTPUT_DIR, "table2_feature_coefficients.csv")
    feature_coefs_df.to_csv(table2_csv, index=False)
    print(f"  [OK] Saved to {table2_csv}")

    # Create LaTeX version
    table2_tex = os.path.join(OUTPUT_DIR, "table2_feature_coefficients.tex")
    with open(table2_tex, 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Top 15 SAE Features Predicting 30-Day Drift}\n")
        f.write("\\label{tab:features}\n")
        f.write("\\begin{tabular}{lllrrrrl}\n")
        f.write("\\hline\\hline\n")
        f.write("Feature & Type & Coefficient & Std Error & t-stat & p-value & \\\\\n")
        f.write("\\hline\n")

        for _, row in feature_coefs_df.iterrows():
            f.write(f"{row['Feature']} & {row['Type']} & {row['Coefficient']:.4f} & {row['Std Error']:.4f} & {row['t-stat']:.2f} & {row['p-value']:.4f} & {row['Sig']} \\\\\n")

        f.write("\\hline\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\begin{tablenotes}\n")
        f.write("\\small\n")
        f.write("\\item Notes: All features are standardized (mean=0, std=1). *** p<0.01, ** p<0.05, * p<0.10.\n")
        f.write("\\end{tablenotes}\n")
        f.write("\\end{table}\n")

    print(f"  [OK] Saved to {table2_tex}")

# Table 3: Summary statistics
print("\n[TABLE 3] Summary statistics")

summary_stats = reg_df[['car_3d', 'drift_30d', 'drift_60d',
                         'novelty_cln_mean', 'relevance_kmnz_mean']].describe()

table3_csv = os.path.join(OUTPUT_DIR, "table3_summary_stats.csv")
summary_stats.to_csv(table3_csv)
print(f"  [OK] Saved to {table3_csv}")

# =========================
# 6. Final Summary
# =========================

print("\n" + "="*80)
print("PAPER ANALYSIS COMPLETE")
print("="*80)

print(f"\nOutput directory: {OUTPUT_DIR}/")
print("\nGenerated files:")
print("  1. regression_summary.csv - Model performance summary")
print("  2. table1_main_results.tex/txt - Main regression table")
print("  3. table2_feature_coefficients.csv/tex - Top features table")
print("  4. table3_summary_stats.csv - Summary statistics")

if not args.no_viz:
    print("  5. feature_importance.png - Feature loadings visualization")
    print("  6. r2_comparison.png - Model R² comparison")
    print("  7. feature_distributions.png - Top features distributions")
    print("  8. time_trends.png - Feature usage over time")

print("\n" + "="*80)
print("KEY FINDINGS FOR PAPER")
print("="*80)

for summary in regression_summaries:
    print(f"\n{summary['outcome']}:")
    print(f"  N = {summary['n_obs']:,}")
    print(f"  Baseline R-squared = {summary['r2_baseline']:.4f}")
    print(f"  Full R-squared = {summary['r2_full']:.4f}")
    print(f"  Delta R-squared = {summary['delta_r2']:.4f}")
    print(f"  F-test: F = {summary['f_stat']:.2f}, p = {summary['f_pval']:.4f}")

    if summary['f_pval'] < 0.01:
        print(f"  *** SAE features add significant explanatory power (p < 0.01)")
    elif summary['f_pval'] < 0.05:
        print(f"  ** SAE features add significant explanatory power (p < 0.05)")
    elif summary['f_pval'] < 0.10:
        print(f"  * SAE features add marginal explanatory power (p < 0.10)")
    else:
        print(f"  SAE features do not add significant explanatory power")

print("\n" + "="*80)
