"""
Data Validation Script

Checks data structure and alignment before running SAE training.
Run this BEFORE starting the pipeline to catch mismatches early.

Usage:
  python 00_validate_data.py
"""

import os
import sys
import pandas as pd
import numpy as np

DATA_DIR = r"C:\Users\ofs4963\Dropbox\Arojects\SAE\data"

def check_file_exists(filepath, description):
    """Check if file exists and print status"""
    exists = os.path.exists(filepath)
    status = "[OK]" if exists else "[MISSING]"
    print(f"  {status} {description}")
    if not exists:
        print(f"         Expected at: {filepath}")
    return exists

def validate_sentences():
    """Validate sentence-level data"""
    print("\n" + "="*80)
    print("VALIDATING SENTENCE-LEVEL DATA")
    print("="*80)

    # Check files exist
    sent_file = os.path.join(DATA_DIR, "sentences_sae_train.parquet")
    chunks_dir = os.path.join(DATA_DIR, "sentence_embeddings_chunks")
    novelty_file = os.path.join(DATA_DIR, "novelty_cln_sae.parquet")
    relevance_file = os.path.join(DATA_DIR, "relevance_kmnz_sae.parquet")

    print("\n1. Checking files exist...")
    files_ok = all([
        check_file_exists(sent_file, "Sentences"),
        check_file_exists(chunks_dir, "Embedding chunks"),
        check_file_exists(novelty_file, "CLN novelty"),
        check_file_exists(relevance_file, "KMNZ relevance")
    ])

    if not files_ok:
        print("\n[FAIL] VALIDATION FAILED: Missing required files")
        return False

    # Load and check shapes
    print("\n2. Checking data shapes...")
    sent_df = pd.read_parquet(sent_file)
    print(f"  Sentences: {len(sent_df):,} rows")
    print(f"  Columns: {sent_df.columns.tolist()}")

    # Check chunked embeddings
    chunk_files = sorted([f for f in os.listdir(chunks_dir)
                         if f.endswith('.npz')])
    print(f"\n3. Checking embedding chunks...")
    print(f"  Found {len(chunk_files)} chunk files")

    total_emb = 0
    for chunk_file in chunk_files:
        chunk_path = os.path.join(chunks_dir, chunk_file)
        data = np.load(chunk_path, allow_pickle=True)
        emb_shape = data['embeddings'].shape
        total_emb += emb_shape[0]
        print(f"    {chunk_file}: {emb_shape[0]:,} × {emb_shape[1]}")

    print(f"\n  Total embeddings: {total_emb:,}")
    print(f"  Expected (sentences): {len(sent_df):,}")

    if total_emb != len(sent_df):
        print(f"  [FAIL] MISMATCH: {total_emb:,} embeddings vs {len(sent_df):,} sentences")
        return False
    else:
        print(f"  [OK] MATCH")

    # Check CLN/KMNZ
    print("\n4. Checking CLN novelty...")
    novelty_df = pd.read_parquet(novelty_file)
    print(f"  Rows: {len(novelty_df):,}")
    print(f"  Columns: {novelty_df.columns.tolist()}")

    print("\n5. Checking KMNZ relevance...")
    relevance_df = pd.read_parquet(relevance_file)
    print(f"  Rows: {len(relevance_df):,}")
    print(f"  Columns: {relevance_df.columns.tolist()}")

    print("\n[OK] SENTENCE-LEVEL DATA VALIDATED")
    return True

def validate_spans():
    """Validate span-level data"""
    print("\n" + "="*80)
    print("VALIDATING SPAN-LEVEL DATA")
    print("="*80)

    # Check files exist
    span_file = os.path.join(DATA_DIR, "spans_sae_train.parquet")
    emb_file = os.path.join(DATA_DIR, "span_embeddings.npz")
    novelty_file = os.path.join(DATA_DIR, "novelty_cln_spans.parquet")
    relevance_file = os.path.join(DATA_DIR, "relevance_kmnz_spans.parquet")

    print("\n1. Checking files exist...")
    files_ok = all([
        check_file_exists(span_file, "Spans"),
        check_file_exists(emb_file, "Embeddings"),
        check_file_exists(novelty_file, "CLN novelty"),
        check_file_exists(relevance_file, "KMNZ relevance")
    ])

    if not files_ok:
        print("\n[FAIL] VALIDATION FAILED: Missing required files")
        return False

    # Load and check shapes
    print("\n2. Checking data shapes...")
    span_df = pd.read_parquet(span_file)
    print(f"  Spans: {len(span_df):,} rows")
    print(f"  Columns: {span_df.columns.tolist()}")

    # Check embeddings
    print("\n3. Checking embeddings...")
    emb_data = np.load(emb_file)
    emb_shape = emb_data['embeddings'].shape
    print(f"  Embeddings: {emb_shape[0]:,} × {emb_shape[1]}")

    if emb_shape[0] != len(span_df):
        print(f"  [FAIL] MISMATCH: {emb_shape[0]:,} embeddings vs {len(span_df):,} spans")
        return False
    else:
        print(f"  [OK] MATCH")

    # Check CLN/KMNZ
    print("\n4. Checking CLN novelty...")
    novelty_df = pd.read_parquet(novelty_file)
    print(f"  Rows: {len(novelty_df):,}")

    print("\n5. Checking KMNZ relevance...")
    relevance_df = pd.read_parquet(relevance_file)
    print(f"  Rows: {len(relevance_df):,}")

    print("\n[OK] SPAN-LEVEL DATA VALIDATED")
    return True

def main():
    print("="*80)
    print("SAE PIPELINE DATA VALIDATION")
    print("="*80)
    print(f"Data directory: {DATA_DIR}")

    sentences_ok = validate_sentences()
    spans_ok = validate_spans()

    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    status_sent = "[OK] PASS" if sentences_ok else "[FAIL] FAIL"
    status_span = "[OK] PASS" if spans_ok else "[FAIL] FAIL"

    print(f"  Sentence-level: {status_sent}")
    print(f"  Span-level: {status_span}")

    if sentences_ok and spans_ok:
        print("\n[OK] ALL VALIDATIONS PASSED")
        print("\nYou can now run:")
        print("  - Sentences: python 03b_sae_anthropic.py --unit sentences")
        print("  - Spans: python 03b_sae_anthropic.py --unit spans")
        return 0
    else:
        print("\n[FAIL] VALIDATION FAILED")
        print("\nPlease fix data issues before running SAE training")
        return 1

if __name__ == "__main__":
    sys.exit(main())
