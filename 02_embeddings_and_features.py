"""
Phase 2: Stratified Sampling + Embeddings + Baseline Features

- Loads all sentences.parquet
- Performs 3-stage stratified sampling (firm-year-item + firm cap + year×item balance)
- Generates embeddings in chunks and saves .npz files
- Computes baseline features on a capped subset

Outputs:
  - sentences_sampled.parquet (~2-3M sentences)
  - sentence_embeddings_index.csv + sentence_embeddings_chunk_*.npz
  - baseline_features.parquet (on at most 100k sampled sentences)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ============================================
# Configuration
# ============================================

DATA_DIR = Path(r"C:\Users\ofs4963\Dropbox\Arojects\SAE\data")
OUTPUT_DIR = DATA_DIR
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Input: All sentences from Phase 1
SENTENCE_FILE = DATA_DIR / "sentences.parquet"

# Sampled sentences (output from run_stratified_sampling.py or created here)
SENTENCE_SAMPLE_OUTPUT = DATA_DIR / "sentences_sampled.parquet"

# Embedding chunks directory and index file
EMBEDDING_CHUNKS_DIR = DATA_DIR / "sentence_embeddings_chunks"
EMBEDDING_CHUNKS_DIR.mkdir(exist_ok=True, parents=True)
EMBEDDING_INDEX_FILE = DATA_DIR / "sentence_embeddings_index.csv"

# Baseline features (subset)
FEATURES_OUTPUT = DATA_DIR / "baseline_features.parquet"

# ============================================
# 1. Stratified Sampling (using run_stratified_sampling.py)
# ============================================

def load_sampled_sentences():
    """
    Load pre-generated sampled sentences from Phase 1.

    Returns:
        DataFrame with sampled sentences, or None if not found
    """
    if not SENTENCE_SAMPLE_OUTPUT.exists():
        print(f"\n[ERROR] Sampled sentences not found: {SENTENCE_SAMPLE_OUTPUT.name}")
        print(f"[INFO] Please run the sampling script first:")
        print(f"[INFO]   python run_stratified_sampling.py")
        print(f"[INFO] Then re-run this script")
        return None

    print(f"\n[INFO] Found sampled sentences: {SENTENCE_SAMPLE_OUTPUT.name}")
    print(f"[INFO] Loading sampled data...")
    sample_df = pd.read_parquet(SENTENCE_SAMPLE_OUTPUT)
    print(f"[OK] Loaded {len(sample_df):,} sentences from {SENTENCE_SAMPLE_OUTPUT.name}")
    return sample_df


# ============================================
# 2. Embeddings (chunked, CPU-friendly)
# ============================================


def generate_embeddings_in_chunks(
    df,
    model_name="all-MiniLM-L6-v2",
    batch_size=256,
    chunk_size=100_000,
    out_dir=EMBEDDING_CHUNKS_DIR,
    index_file=EMBEDDING_INDEX_FILE,
):
    """
    Generate sentence embeddings in chunks and save them to disk.

    Each chunk file:
      - sentence_embeddings_chunk_XXXX.npz
      - arrays: embeddings, accession_numbers, sentence_ids, item_types, ciks

    Also writes an index CSV with chunk metadata.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers not installed.")
        print("Install with: pip install sentence-transformers")
        print("\nFalling back to random embeddings for testing...")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        n = len(df)
        n_chunks = (n + chunk_size - 1) // chunk_size
        meta_records = []

        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = min((chunk_idx + 1) * chunk_size, n)
            chunk = df.iloc[start:end]

            emb_chunk = np.random.randn(len(chunk), 384).astype("float32")

            chunk_path = out_dir / f"sentence_embeddings_chunk_{chunk_idx:04d}.npz"
            np.savez_compressed(
                chunk_path,
                embeddings=emb_chunk,
                accession_numbers=chunk["accession_number"].values,
                sentence_ids=chunk["sentence_id"].values,
                item_types=chunk["item_type"].values,
                ciks=chunk["cik"].values,
            )

            meta_records.append(
                {
                    "chunk_idx": chunk_idx,
                    "start": start,
                    "end": end,
                    "n": end - start,
                    "path": str(chunk_path),
                }
            )

        meta_df = pd.DataFrame(meta_records)
        meta_df.to_csv(index_file, index=False)
        print(f"\n[OK] Wrote random embedding index to {index_file}")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading embedding model: {model_name} (CPU)")
    model = SentenceTransformer(model_name)

    n = len(df)
    n_chunks = (n + chunk_size - 1) // chunk_size
    print(
        f"Generating embeddings for {n:,} sentences "
        f"in {n_chunks} chunks of {chunk_size}..."
    )

    meta_records = []

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min((chunk_idx + 1) * chunk_size, n)
        chunk = df.iloc[start:end]

        print(f"\n[Chunk {chunk_idx + 1}/{n_chunks}] Sentences {start}–{end - 1}")
        emb_chunk = model.encode(
            chunk["text"].tolist(),
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        chunk_path = out_dir / f"sentence_embeddings_chunk_{chunk_idx:04d}.npz"
        np.savez_compressed(
            chunk_path,
            embeddings=emb_chunk,
            accession_numbers=chunk["accession_number"].values,
            sentence_ids=chunk["sentence_id"].values,
            item_types=chunk["item_type"].values,
            ciks=chunk["cik"].values,
        )
        print(f"[OK] Saved chunk to {chunk_path}")

        meta_records.append(
            {
                "chunk_idx": chunk_idx,
                "start": start,
                "end": end,
                "n": end - start,
                "path": str(chunk_path),
            }
        )

    meta_df = pd.DataFrame(meta_records)
    meta_df.to_csv(index_file, index=False)
    print(f"\n[OK] Wrote embedding index to {index_file}")


# ============================================
# 3. Baseline Text Features (subset)
# ============================================


def compute_readability_features(text):
    """Compute basic readability metrics."""
    words = text.split()
    word_count = len(words)
    sentence_count = max(
        text.count(".") + text.count("!") + text.count("?"),
        1,
    )
    avg_word_length = np.mean([len(w) for w in words]) if words else 0

    # Approximate syllables (very rough)
    syllables_approx = sum([max(1, len(w) / 3) for w in words])
    flesch_score = (
        206.835
        - 1.015 * (word_count / sentence_count)
        - 84.6 * (syllables_approx / max(word_count, 1))
    )

    # FOG index
    complex_words = sum([1 for w in words if len(w) > 6])
    fog_index = 0.4 * (
        (word_count / sentence_count)
        + 100 * (complex_words / max(word_count, 1))
    )

    return {
        "word_count": word_count,
        "avg_word_length": avg_word_length,
        "flesch_score": flesch_score,
        "fog_index": fog_index,
        "complex_word_ratio": complex_words / max(word_count, 1),
    }


def compute_sentiment_features(text):
    """Compute simple dictionary-based sentiment scores."""
    positive_words = {
        "profit",
        "profitable",
        "profits",
        "success",
        "successful",
        "growth",
        "growing",
        "increase",
        "increased",
        "gain",
        "gains",
        "improve",
        "improved",
        "improvement",
        "strong",
        "strength",
    }

    negative_words = {
        "loss",
        "losses",
        "decline",
        "declined",
        "decrease",
        "decreased",
        "risk",
        "risks",
        "uncertainty",
        "uncertain",
        "litigation",
        "adverse",
        "adversely",
        "negative",
        "impairment",
        "failure",
    }

    words = set(text.lower().split())
    pos_count = len(words & positive_words)
    neg_count = len(words & negative_words)
    total_words = len(text.split())
    sentiment_ratio = (pos_count - neg_count) / max(total_words, 1)

    return {
        "positive_word_count": pos_count,
        "negative_word_count": neg_count,
        "sentiment_ratio": sentiment_ratio,
        "sentiment_net": pos_count - neg_count,
    }


def create_baseline_features(df):
    """Compute baseline text features for all sentences in df."""
    print("\nComputing baseline text features...")
    readability_features = []
    sentiment_features = []

    for text in tqdm(df["text"], desc="Computing features"):
        readability_features.append(compute_readability_features(text))
        sentiment_features.append(compute_sentiment_features(text))

    readability_df = pd.DataFrame(readability_features)
    sentiment_df = pd.DataFrame(sentiment_features)
    result_df = pd.concat(
        [df.reset_index(drop=True), readability_df, sentiment_df],
        axis=1,
    )

    print(
        f"[OK] Added "
        f"{len(readability_df.columns) + len(sentiment_df.columns)} feature columns"
    )
    return result_df


# ============================================
# Main Execution
# ============================================


def main():
    print("=" * 60)
    print("PHASE 2: STRATIFIED SAMPLING + EMBEDDINGS")
    print("=" * 60)

    # Step 1: Load or create sampled sentences
    print("\n" + "=" * 60)
    print("STEP 1: STRATIFIED SAMPLING")
    print("=" * 60)

    sample_df = load_sampled_sentences()

    if sample_df is None:
        print("\nERROR: Sampling failed or was cancelled")
        return None, None

    print(f"\n[OK] Using {len(sample_df):,} sampled sentences")
    print(f"  Unique firms (CIK): {sample_df['cik'].nunique():,}")
    print(f"  Years: {sample_df['year'].min():.0f} - {sample_df['year'].max():.0f}")
    print(f"  Items: {', '.join(sorted(sample_df['item_type'].unique()))}")

    # Step 2: embeddings (chunked)
    print("\n" + "=" * 60)
    print("STEP 2: EMBEDDING GENERATION (CHUNKED)")
    print("=" * 60)

    generate_embeddings_in_chunks(
        sample_df,
        model_name="all-MiniLM-L6-v2",
        batch_size=256,
        chunk_size=100_000,
        out_dir=EMBEDDING_CHUNKS_DIR,
        index_file=EMBEDDING_INDEX_FILE,
    )

    # Step 3: baseline features on subset (CPU-friendly)
    print("\n" + "=" * 60)
    print("STEP 3: BASELINE FEATURES (SUBSET)")
    print("=" * 60)

    max_features_n = 100_000
    if len(sample_df) > max_features_n:
        print(
            f"Sampling {max_features_n:,} sentences out of "
            f"{len(sample_df):,} for baseline features..."
        )
        feature_df_input = sample_df.sample(
            n=max_features_n, random_state=42
        ).reset_index(drop=True)
    else:
        feature_df_input = sample_df.copy()

    sent_df_with_features = create_baseline_features(feature_df_input)
    sent_df_with_features.to_parquet(FEATURES_OUTPUT, index=False)
    print(f"[OK] Saved features to {FEATURES_OUTPUT}")

    # Summary
    print("\n" + "=" * 60)
    print("PHASE 2 COMPLETE")
    print("=" * 60)
    print(f"Stratified sample: {len(sample_df):,} sentences")
    print(f"  From {sample_df['cik'].nunique():,} firms")
    print("\nEmbeddings:")
    print(f"  - Chunks directory: {EMBEDDING_CHUNKS_DIR}")
    print(f"  - Index file:       {EMBEDDING_INDEX_FILE}")
    print("\nBaseline features:")
    print(f"  - {FEATURES_OUTPUT.name} (on {len(sent_df_with_features):,} sentences)")
    print("\nNext steps:")
    print("  - Use sentence_embeddings_index.csv to load chunked embeddings")
    print("  - Then run 03_sae_training.py using these embeddings")

    return None, sent_df_with_features


if __name__ == "__main__":
    embeddings, features = main()
