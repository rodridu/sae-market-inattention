"""
Phase 2b: CLN Novelty Computation (N_fdt)
Computes sentence-level information measure using LLM priors over EDGAR text.

Based on Costello-Levy-Nikolaev framework:
- Information is defined as surprisal under a firm-specific LLM prior
- Prior is built from historical EDGAR filings (all forms: 10-K, 10-Q, 8-K, exhibits)
- Token-level surprisal is aggregated to sentence level

Proposal alignment (Section 4.2):
For each sentence s in filing d for firm f at time t:
  N_fdt = (1 / |tokens in s|) * Σ_τ I(τ)
where I(τ) = -log P(τ | prior_context, θ_f)

This is a FRAMEWORK IMPLEMENTATION with placeholders for:
1. Firm-specific LLM fine-tuning
2. Token-level probability computation
3. Surprisal aggregation

Full implementation requires:
- Access to historical EDGAR corpus per firm
- GPU resources for LLM fine-tuning
- Storage for firm-specific model checkpoints
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Optional: uncomment when implementing full CLN approach
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================
# Configuration
# ============================================

DATA_DIR = Path(r"C:\Users\ofs4963\Dropbox\Arojects\SAE\data")
OUTPUT_DIR = DATA_DIR

# Use SAE training sample for development
SENTENCE_FILE = DATA_DIR / "sentences_sampled.parquet"
NOVELTY_OUTPUT = DATA_DIR / "novelty_cln_sae.parquet"

# CLN parameters
BASE_MODEL = "gpt2"  # or "distilgpt2" for faster training
SURPRISAL_AGGREGATION = "mean"  # or "sum", "max"

# ============================================
# 1. Load Historical Filings for Prior Construction
# ============================================

def load_historical_filings(cik, current_date):
    """
    Load all historical EDGAR filings for a given firm up to current_date.

    For full CLN implementation, this would:
    1. Query EDGAR database for all prior 10-K, 10-Q, 8-K, exhibits
    2. Concatenate into training corpus
    3. Return text for fine-tuning firm-specific prior

    Args:
        cik: Firm identifier
        current_date: Only use filings before this date

    Returns:
        String of concatenated historical text
    """
    # PLACEHOLDER: In production, query EDGAR database
    # Example structure:
    # SELECT text FROM filings
    # WHERE cik = {cik} AND year < {current_year}
    # ORDER BY year

    # For now, return empty (will use base model as prior)
    return ""

# ============================================
# 2. Firm-Specific Prior Training
# ============================================

def get_or_train_firm_prior(cik, historical_text, base_model_name=BASE_MODEL):
    """
    Get or train firm-specific language model prior.

    Full CLN approach:
    1. Check if firm-specific model checkpoint exists
    2. If not: fine-tune base LLM on firm's historical EDGAR text
    3. Save checkpoint for reuse
    4. Return model and tokenizer

    Args:
        cik: Firm identifier
        historical_text: Concatenated historical filings
        base_model_name: Base model for fine-tuning

    Returns:
        (model, tokenizer) or None if using base model
    """
    # PLACEHOLDER: Full implementation would:
    #
    # model_path = OUTPUT_DIR / f"firm_priors/{cik}/"
    # if model_path.exists():
    #     model = AutoModelForCausalLM.from_pretrained(model_path)
    #     tokenizer = AutoTokenizer.from_pretrained(model_path)
    # else:
    #     # Fine-tune on historical_text
    #     model = AutoModelForCausalLM.from_pretrained(base_model_name)
    #     tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    #     # ... training loop with historical_text ...
    #     model.save_pretrained(model_path)
    #     tokenizer.save_pretrained(model_path)
    #
    # return model, tokenizer

    print(f"   Using base model prior (firm-specific training not implemented)")
    return None, None

# ============================================
# 3. Token-Level Surprisal Computation
# ============================================

def compute_token_surprisal(text, model, tokenizer):
    """
    Compute token-level surprisal: I(τ) = -log P(τ | context).

    Args:
        text: Sentence text
        model: Language model
        tokenizer: Tokenizer

    Returns:
        Array of token-level surprisals
    """
    # PLACEHOLDER: Full implementation would:
    #
    # inputs = tokenizer(text, return_tensors="pt")
    # with torch.no_grad():
    #     outputs = model(**inputs, labels=inputs["input_ids"])
    #     # outputs.loss is cross-entropy = negative log likelihood
    #     token_logprobs = -outputs.loss
    #     surprisals = -token_logprobs.cpu().numpy()
    # return surprisals

    # For now: return random surprisals (placeholder)
    n_tokens = len(text.split())
    return np.random.gamma(shape=2.0, scale=1.0, size=n_tokens)

# ============================================
# 4. Sentence-Level Novelty Aggregation
# ============================================

def aggregate_surprisal_to_sentence(token_surprisals, method=SURPRISAL_AGGREGATION):
    """
    Aggregate token-level surprisals to sentence-level novelty N_fdt.

    Proposal (Section 4.2): N_fdt = (1 / |tokens|) * Σ I(τ)

    Args:
        token_surprisals: Array of token-level surprisals
        method: Aggregation method ("mean", "sum", "max")

    Returns:
        Scalar sentence-level novelty
    """
    if method == "mean":
        return np.mean(token_surprisals)
    elif method == "sum":
        return np.sum(token_surprisals)
    elif method == "max":
        return np.max(token_surprisals)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

# ============================================
# 5. Simplified Novelty Proxy (Placeholder)
# ============================================

def compute_simple_novelty_proxy(text, firm_avg_length, firm_vocab_diversity):
    """
    Simplified novelty proxy using text statistics (placeholder for full CLN).

    Uses heuristics that correlate with information content:
    - Length deviation from firm average
    - Vocabulary diversity (unique words / total words)
    - Presence of numbers, capitalized terms (proxies for specific disclosure)

    Args:
        text: Sentence text
        firm_avg_length: Average sentence length for this firm
        firm_vocab_diversity: Average vocab diversity for this firm

    Returns:
        Scalar pseudo-novelty score
    """
    words = text.split()
    n_words = len(words)

    # Length novelty
    length_novelty = abs(n_words - firm_avg_length) / (firm_avg_length + 1)

    # Vocabulary diversity
    vocab_diversity = len(set(words)) / (n_words + 1)
    vocab_novelty = abs(vocab_diversity - firm_vocab_diversity)

    # Specific content markers
    n_numbers = sum(1 for w in words if any(c.isdigit() for c in w))
    n_caps = sum(1 for w in words if w[0].isupper() and len(w) > 1)
    specificity = (n_numbers + n_caps) / (n_words + 1)

    # Combine
    novelty_proxy = 0.4 * length_novelty + 0.3 * vocab_novelty + 0.3 * specificity

    return novelty_proxy

# ============================================
# 6. Compute Novelty for All Sentences
# ============================================

def compute_novelty_all_sentences(sent_df, method="proxy"):
    """
    Compute novelty for all sentences.

    Args:
        sent_df: DataFrame with sentences
        method: "cln" (full CLN) or "proxy" (simplified)

    Returns:
        DataFrame with added 'novelty_cln' column
    """
    print("\nComputing CLN novelty measures...")

    if method == "cln":
        print("   Method: Full CLN (firm-specific priors)")
        print("   WARNING: This requires substantial compute resources")

        # Group by firm and process
        novelty_scores = []
        for cik in tqdm(sent_df['cik'].unique(), desc="Firms"):
            firm_sents = sent_df[sent_df['cik'] == cik]

            for idx, row in firm_sents.iterrows():
                # Load historical filings
                hist_text = load_historical_filings(cik, row['year'])

                # Get or train firm prior
                model, tokenizer = get_or_train_firm_prior(cik, hist_text)

                # Compute token surprisals
                token_surprisals = compute_token_surprisal(row['text'], model, tokenizer)

                # Aggregate to sentence level
                novelty = aggregate_surprisal_to_sentence(token_surprisals)
                novelty_scores.append(novelty)

        sent_df['novelty_cln'] = novelty_scores

    elif method == "proxy":
        print("   Method: Simplified proxy (text statistics)")
        print("   NOTE: Replace with full CLN implementation for production")

        # Compute sentence length if not present
        if 'sentence_length' not in sent_df.columns:
            sent_df['sentence_length'] = sent_df['text'].apply(lambda x: len(str(x).split()))

        # Compute firm-level baselines
        firm_stats = sent_df.groupby('cik').agg({
            'sentence_length': 'mean',
            'text': lambda x: np.mean([len(set(str(t).split())) / (len(str(t).split()) + 1) for t in x])
        }).rename(columns={'text': 'vocab_diversity'})

        # Merge back
        sent_df = sent_df.merge(firm_stats, left_on='cik', right_index=True,
                                suffixes=('', '_firm_avg'))

        # Compute novelty proxy
        novelty_scores = []
        for idx, row in tqdm(sent_df.iterrows(), total=len(sent_df), desc="Computing novelty"):
            novelty = compute_simple_novelty_proxy(
                row['text'],
                row['sentence_length_firm_avg'],
                row['vocab_diversity']
            )
            novelty_scores.append(novelty)

        sent_df['novelty_cln'] = novelty_scores

        # Drop temporary columns
        sent_df = sent_df.drop(['sentence_length_firm_avg', 'vocab_diversity'], axis=1)

    else:
        raise ValueError(f"Unknown method: {method}")

    return sent_df

# ============================================
# Main Execution
# ============================================

def main(method="proxy"):
    """
    Main execution pipeline.

    Args:
        method: "cln" or "proxy"
    """
    print("="*60)
    print("PHASE 2b: CLN NOVELTY COMPUTATION")
    print("="*60)

    # Load sentence data
    print(f"\nLoading sentence data from {SENTENCE_FILE}...")
    sent_df = pd.read_parquet(SENTENCE_FILE)
    print(f"Loaded {len(sent_df):,} sentences")

    # Compute novelty
    sent_df_with_novelty = compute_novelty_all_sentences(sent_df, method=method)

    # Save
    NOVELTY_OUTPUT.parent.mkdir(exist_ok=True, parents=True)
    sent_df_with_novelty.to_parquet(NOVELTY_OUTPUT, index=False)
    print(f"\n✓ Saved novelty measures to {NOVELTY_OUTPUT}")

    # Summary statistics
    print("\n" + "="*60)
    print("PHASE 2b COMPLETE")
    print("="*60)
    print(f"Sentences with novelty: {len(sent_df_with_novelty):,}")
    print(f"\nNovelty statistics:")
    print(sent_df_with_novelty['novelty_cln'].describe())

    if method == "proxy":
        print("\n" + "!"*60)
        print("IMPORTANT: Current implementation uses simplified proxy")
        print("For production research:")
        print("  1. Implement firm-specific LLM fine-tuning")
        print("  2. Compute true token-level surprisals")
        print("  3. Consider using CLN's implementation if available")
        print("!"*60)

    print(f"\nNext steps:")
    print(f"  - Run 02c_relevance_kmnz.py to compute relevance R_fdt")
    print(f"  - Then run 03_sae_training.py")

    return sent_df_with_novelty

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Compute CLN novelty measures')
    parser.add_argument('--method', default='proxy', choices=['cln', 'proxy'],
                        help='Method: "cln" for full implementation, "proxy" for simplified')
    args = parser.parse_args()

    sent_df_with_novelty = main(method=args.method)
