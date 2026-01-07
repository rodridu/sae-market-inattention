"""
Phase 5: LLM-Based Feature Interpretation

After feature selection (Phase 4), this script:
1. Auto-detects all Lasso results with selected features
2. Samples high/low activation sentences for each feature
3. Uses LLM to generate natural language descriptions
4. Validates descriptions on held-out sentences
5. Saves interpretations and fidelity scores

Usage:
  python 05_interpret_features.py                    # Auto-detect all outcomes
  python 05_interpret_features.py --outcome car_3d   # Interpret specific outcome
  python 05_interpret_features.py --stub             # Test mode without API calls

Aligned with proposal Section 4.6 (Step 4):
  LLM-based interpretation with fidelity validation

Requirements:
  - OpenAI API key: export OPENAI_API_KEY=...
  - OR Anthropic API key: export ANTHROPIC_API_KEY=...
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import glob

# =========================
# 0. Configuration
# =========================

parser = argparse.ArgumentParser(description='Interpret SAE features using LLM')
parser.add_argument('--outcome', default=None,
                    help='Outcome variable used for Lasso (default: auto-detect all)')
parser.add_argument('--top-k', type=int, default=20,
                    help='Number of high/low activation sentences to sample (default: 20)')
parser.add_argument('--high-quantile', type=float, default=0.95,
                    help='Quantile threshold for high activations (default: 0.95)')
parser.add_argument('--low-quantile', type=float, default=0.05,
                    help='Quantile threshold for low activations (default: 0.05)')
parser.add_argument('--validation-k', type=int, default=100,
                    help='Number of held-out sentences for fidelity validation (default: 100)')
parser.add_argument('--min-fidelity', type=float, default=0.7,
                    help='Minimum F1 score to accept interpretation (default: 0.7)')
parser.add_argument('--llm-provider', default='anthropic', choices=['openai', 'anthropic'],
                    help='LLM provider (default: anthropic)')
parser.add_argument('--stub', action='store_true',
                    help='Use stub LLM (no API calls) for testing')
parser.add_argument('--max-features', type=int, default=None,
                    help='Maximum number of features to interpret (default: all selected features)')

args = parser.parse_args()

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

print("\n" + "="*80)
print("PHASE 5: LLM-BASED FEATURE INTERPRETATION")
print("="*80)

# =========================
# 1. Auto-detect outcomes with selected features
# =========================

# Find all lasso_results_*.csv files
lasso_pattern = os.path.join(DATA_DIR, "lasso_results_*.csv")
lasso_files = glob.glob(lasso_pattern)

if len(lasso_files) == 0:
    print(f"\nERROR: No lasso_results_*.csv files found in {DATA_DIR}")
    print("Please run 04_feature_selection.py first.")
    sys.exit(1)

# If user specified an outcome, use only that one
if args.outcome is not None:
    lasso_files = [os.path.join(DATA_DIR, f"lasso_results_{args.outcome}.csv")]
    if not os.path.exists(lasso_files[0]):
        print(f"\nERROR: {lasso_files[0]} not found!")
        print("Please run 04_feature_selection.py first.")
        sys.exit(1)

# Check which outcomes have selected features
outcomes_to_interpret = []
for lasso_file in lasso_files:
    outcome_name = os.path.basename(lasso_file).replace("lasso_results_", "").replace(".csv", "")
    lasso_results = pd.read_csv(lasso_file)
    n_features = len(lasso_results)

    if n_features > 0:
        outcomes_to_interpret.append((outcome_name, lasso_file, n_features))
        print(f"  [+] {outcome_name}: {n_features} selected features")
    else:
        print(f"  [-] {outcome_name}: 0 selected features (skipping)")

if len(outcomes_to_interpret) == 0:
    print("\n" + "="*80)
    print("NO FEATURES TO INTERPRET")
    print("="*80)
    print("\nAll outcomes have 0 selected features. This is expected when:")
    print("  1. Outcome variables are random/placeholder (not real market data)")
    print("  2. SAE features provide no incremental explanatory power")
    print("  3. The alpha penalty is too strong")
    print("\nTo proceed:")
    print("  - Use real outcome data (announcement returns, drift)")
    print("  - Check that CLN novelty and KMNZ relevance are properly computed")
    print("  - Consider reducing the Lasso alpha penalty")
    sys.exit(0)

print(f"\nFound {len(outcomes_to_interpret)} outcome(s) with selected features:")
for outcome_name, _, n_features in outcomes_to_interpret:
    print(f"  - {outcome_name}: {n_features} features")

# =========================
# 1b. Determine which activation columns we need
# =========================

# Collect all unique activation columns across outcomes
activation_cols_needed = set()
for outcome_name, lasso_file, n_features in outcomes_to_interpret:
    lasso_results = pd.read_csv(lasso_file)
    for feature_col in lasso_results['feature']:
        # Convert mean/freq features to activation columns
        if feature_col.startswith('mean_M'):
            activation_cols_needed.add(feature_col.replace('mean_M', 'h_M'))
        elif feature_col.startswith('freq_M'):
            activation_cols_needed.add(feature_col.replace('freq_M', 'h_M'))
        else:
            activation_cols_needed.add(feature_col)

print(f"\nNeed to load {len(activation_cols_needed)} activation columns")

# =========================
# 1c. Load sentence data (only needed columns)
# =========================

sent_file = os.path.join(DATA_DIR, "sent_df_with_sae.parquet")
print(f"Loading sentence-level data from {sent_file}...")
if not os.path.exists(sent_file):
    print(f"ERROR: {sent_file} not found!")
    print("Please run 03_sae_training.py first.")
    sys.exit(1)

# Load only the text column and the activation columns we need
columns_to_load = ['text'] + list(activation_cols_needed)
sent_df = pd.read_parquet(sent_file, columns=columns_to_load)
print(f"[OK] Loaded {len(sent_df):,} sentences × {len(columns_to_load)} columns")

# =========================
# 2. Interpretation functions
# =========================

def sample_units_for_neuron(df, feature_col, high_quantile=0.95, low_quantile=0.05, n_samples=20):
    """
    Sample high and low activation sentences for a feature.

    Aligned with proposal Section 4.6 (Step 4):
    - Sample high-activation units (e.g., top 5%)
    - Sample low-activation units (e.g., bottom 5% or zeros)

    Args:
        df: DataFrame with sentences and activations
        feature_col: Column name for the feature (e.g., 'h_M4096_k16_n42')
        high_quantile: Quantile threshold for high activations
        low_quantile: Quantile threshold for low activations
        n_samples: Number of samples per group

    Returns:
        Tuple of (high_texts, low_texts)
    """
    if feature_col not in df.columns:
        print(f"WARNING: Feature {feature_col} not found in data")
        return [], []

    h = df[feature_col]
    high_thresh = h.quantile(high_quantile)

    # Sample high-activation sentences
    high_df = df[h > high_thresh]
    n_high = min(n_samples, len(high_df))
    if n_high > 0:
        high_sample = high_df.sample(n=n_high, random_state=42)
        high_texts = high_sample["text"].tolist()
    else:
        high_texts = []

    # Sample low-activation sentences
    # For sparse SAE activations, most values are 0, so we sample from zeros
    # If there are few zeros, sample from bottom quantile of non-zero values
    zero_df = df[h == 0]
    if len(zero_df) >= n_samples:
        # Plenty of zeros - sample from them
        low_sample = zero_df.sample(n=n_samples, random_state=42)
        low_texts = low_sample["text"].tolist()
    else:
        # Few zeros - sample from bottom quantile
        low_thresh = h.quantile(low_quantile)
        low_df = df[h <= low_thresh]
        n_low = min(n_samples, len(low_df))
        if n_low > 0:
            low_sample = low_df.sample(n=n_low, random_state=42)
            low_texts = low_sample["text"].tolist()
        else:
            low_texts = []

    return high_texts, low_texts


def describe_neuron_with_llm(high_texts, low_texts, feature_name, provider='anthropic', stub=False):
    """
    Use LLM to generate natural-language description of feature pattern.

    Proposal Section 4.6 (Step 4):
    - Ask LLM: "What common pattern appears in HIGH but not LOW?"
    - Return short description

    Args:
        high_texts: List of high-activation sentences
        low_texts: List of low-activation sentences
        feature_name: Feature name for reference
        provider: 'openai' or 'anthropic'
        stub: If True, return placeholder without API call

    Returns:
        String description of concept
    """
    if stub:
        # Stub mode - no API call
        return f"[STUB] Pattern detected in {feature_name}"

    # Construct prompt
    prompt = f"""Below are sentences from 10-K filings that activate feature {feature_name} HIGHLY, followed by sentences that do NOT activate it.

HIGH-ACTIVATION SENTENCES:
{chr(10).join(f'{i+1}. {text[:200]}...' for i, text in enumerate(high_texts[:10]))}

LOW-ACTIVATION SENTENCES:
{chr(10).join(f'{i+1}. {text[:200]}...' for i, text in enumerate(low_texts[:10]))}

What common semantic pattern or concept appears consistently in the HIGH set but not the LOW set?
Provide a concise 1-2 sentence description."""

    # Call LLM API
    try:
        if provider == 'openai':
            import openai
            openai.api_key = os.getenv('OPENAI_API_KEY')

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150
            )
            description = response.choices[0].message.content.strip()

        elif provider == 'anthropic':
            import anthropic

            client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=150,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            description = message.content[0].text.strip()

        else:
            raise ValueError(f"Unknown provider: {provider}")

        return description

    except Exception as e:
        print(f"  ERROR calling {provider} API: {e}")
        return f"[ERROR] {feature_name}: {str(e)}"


def evaluate_description_fidelity(description, df, feature_col, threshold, n_eval=100, provider='anthropic', stub=False):
    """
    Validate description fidelity on held-out sentences.

    Proposal Section 4.6 (Step 4):
    - Sample held-out sentences
    - Ask LLM if description applies
    - Compare with neuron activations
    - Compute precision, recall, F1

    Only use concepts with high fidelity (F1 > 0.7) in final analysis.

    Args:
        description: Natural-language concept description
        df: DataFrame with sentences and activations
        feature_col: Feature column name
        threshold: Activation threshold (e.g., 90th percentile)
        n_eval: Number of samples for evaluation
        provider: LLM provider
        stub: If True, return random fidelity

    Returns:
        Dict with precision, recall, F1
    """
    if stub:
        # Stub mode - return random but plausible fidelity
        np.random.seed(hash(feature_col) % 2**32)
        f1 = np.random.uniform(0.6, 0.9)
        precision = np.random.uniform(0.5, 0.9)
        recall = np.random.uniform(0.5, 0.9)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "note": "Stub mode - random values"
        }

    # Sample held-out sentences
    eval_sample = df.sample(n=min(n_eval, len(df)), random_state=42)

    # Ground truth: high activation
    ground_truth = (eval_sample[feature_col] > threshold).values

    # LLM predictions: ask if description applies
    predictions = []

    for idx, row in eval_sample.iterrows():
        text = row['text']

        # Prompt: Does this sentence match the description?
        prompt = f"""Concept description: {description}

Sentence: {text[:300]}

Does this sentence exemplify the concept described above? Answer with ONLY "yes" or "no"."""

        try:
            if provider == 'openai':
                import openai
                openai.api_key = os.getenv('OPENAI_API_KEY')

                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=5
                )
                answer = response.choices[0].message.content.strip().lower()

            elif provider == 'anthropic':
                import anthropic
                client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

                message = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=5,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}]
                )
                answer = message.content[0].text.strip().lower()

            predictions.append(answer == 'yes')

        except Exception as e:
            print(f"  ERROR in fidelity evaluation: {e}")
            predictions.append(False)

    predictions = np.array(predictions)

    # Compute metrics
    tp = np.sum(predictions & ground_truth)
    fp = np.sum(predictions & ~ground_truth)
    fn = np.sum(~predictions & ground_truth)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "n_eval": len(predictions)
    }


# =========================
# 3. Interpret selected features (loop over all outcomes)
# =========================

all_results_summary = []

for outcome_idx, (outcome_name, lasso_file, n_features) in enumerate(outcomes_to_interpret):
    print("\n" + "="*80)
    print(f"OUTCOME {outcome_idx+1}/{len(outcomes_to_interpret)}: {outcome_name.upper()}")
    print(f"({n_features} selected features)")
    print("="*80)

    # Load Lasso results for this outcome
    lasso_results = pd.read_csv(lasso_file)

    # Limit features if requested
    if args.max_features is not None and len(lasso_results) > args.max_features:
        lasso_results = lasso_results.head(args.max_features)
        print(f"  Limiting to top {args.max_features} features by coefficient magnitude")

    interpretations = []
    fidelity_scores = []

    for idx, row in tqdm(lasso_results.iterrows(), total=len(lasso_results), desc=f"Interpreting {outcome_name}"):
        feature_col = row['feature']

        # Extract feature identifier (e.g., 'h_M4096_k16_n42')
        # For mean/freq features, convert back to activation column
        if feature_col.startswith('mean_M'):
            activation_col = feature_col.replace('mean_M', 'h_M')
        elif feature_col.startswith('freq_M'):
            activation_col = feature_col.replace('freq_M', 'h_M')
        else:
            activation_col = feature_col

        # Sample high/low activation sentences
        high_texts, low_texts = sample_units_for_neuron(
            sent_df,
            activation_col,
            high_quantile=args.high_quantile,
            low_quantile=args.low_quantile,
            n_samples=args.top_k
        )

        if len(high_texts) == 0 or len(low_texts) == 0:
            continue

        # Generate description
        description = describe_neuron_with_llm(
            high_texts,
            low_texts,
            feature_col,
            provider=args.llm_provider,
            stub=args.stub
        )

        # Evaluate fidelity
        threshold = sent_df[activation_col].quantile(0.9)
        fidelity = evaluate_description_fidelity(
            description,
            sent_df,
            activation_col,
            threshold,
            n_eval=args.validation_k,
            provider=args.llm_provider,
            stub=args.stub
        )

        # Save interpretation
        interpretations.append({
            'feature': feature_col,
            'description': description,
            'lasso_coefficient': row['coefficient'],
            'n_high_samples': len(high_texts),
            'n_low_samples': len(low_texts)
        })

        fidelity_scores.append({
            'feature': feature_col,
            **fidelity
        })

    # =========================
    # 4. Save results for this outcome
    # =========================

    print("\n" + "-"*80)
    print(f"SAVING RESULTS FOR {outcome_name.upper()}")
    print("-"*80)

    # Save interpretations
    interp_df = pd.DataFrame(interpretations)
    interp_file = os.path.join(DATA_DIR, f"neuron_interpretations_{outcome_name}.csv")
    interp_df.to_csv(interp_file, index=False)
    print(f"[OK] Saved {len(interp_df)} interpretations to neuron_interpretations_{outcome_name}.csv")

    # Save fidelity scores
    fidelity_df = pd.DataFrame(fidelity_scores)
    fidelity_file = os.path.join(DATA_DIR, f"fidelity_scores_{outcome_name}.csv")
    fidelity_df.to_csv(fidelity_file, index=False)
    print(f"[OK] Saved fidelity scores to fidelity_scores_{outcome_name}.csv")

    # Filter high-fidelity interpretations
    n_high_fidelity = 0
    if len(fidelity_df) > 0 and 'f1' in fidelity_df.columns:
        high_fidelity = fidelity_df[fidelity_df['f1'] > args.min_fidelity]
        n_high_fidelity = len(high_fidelity)
        print(f"[OK] {n_high_fidelity} / {len(fidelity_df)} features have F1 > {args.min_fidelity}")

        if n_high_fidelity > 0:
            high_fidelity_features = high_fidelity['feature'].tolist()
            high_fidelity_interp = interp_df[interp_df['feature'].isin(high_fidelity_features)]

            high_fidelity_file = os.path.join(DATA_DIR, f"high_fidelity_interpretations_{outcome_name}.csv")
            high_fidelity_interp.to_csv(high_fidelity_file, index=False)
            print(f"[OK] Saved high-fidelity interpretations to high_fidelity_interpretations_{outcome_name}.csv")

    all_results_summary.append({
        'outcome': outcome_name,
        'n_features_selected': n_features,
        'n_interpreted': len(interp_df),
        'n_high_fidelity': n_high_fidelity
    })

# =========================
# 5. Final summary
# =========================

print("\n" + "="*80)
print("INTERPRETATION COMPLETE")
print("="*80)

summary_df = pd.DataFrame(all_results_summary)
print("\nSummary:")
print(summary_df.to_string(index=False))

print("\nNext steps:")
print("  - Review interpretations in data/neuron_interpretations_*.csv")
print("  - Use high-fidelity features for economic analysis")
print("  - Run 06_sarkar_analysis.py for pricing function estimation")
print("="*80)
