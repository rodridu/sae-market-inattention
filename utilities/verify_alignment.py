"""
Verify that embeddings and sentences are correctly aligned.

This implements the verification test from the alignment checking document.
"""

import numpy as np
import pandas as pd
import os

def verify_alignment(sent_file, emb_chunks_dir, item_filter=None):
    """
    Verify that the first n embeddings match the filtered sent_df keys exactly.

    Args:
        sent_file: path to sentences parquet file
        emb_chunks_dir: directory containing embedding chunks
        item_filter: item type to filter (e.g., 'item1', 'item1A', 'item7')

    Returns:
        bool: True if alignment is perfect, False otherwise
    """
    print("="*80)
    print("ALIGNMENT VERIFICATION TEST")
    print("="*80)

    # Load sentences
    print(f"\nLoading sentences from {sent_file}...")
    sent_df = pd.read_parquet(sent_file)
    print(f"  Total sentences: {len(sent_df):,}")

    # Filter if requested
    if item_filter:
        print(f"\nFiltering to {item_filter}...")
        sent_df = sent_df[sent_df["item_type"] == item_filter].reset_index(drop=True)
        print(f"  Filtered sentences: {len(sent_df):,}")

    n = len(sent_df)

    # Load embedding metadata (first n rows from chunks)
    print(f"\nLoading first {n:,} embedding metadata rows...")
    chunk_files = sorted([f for f in os.listdir(emb_chunks_dir) if f.endswith('.npz')])

    keys = []
    got = 0

    for fn in chunk_files:
        if got >= n:
            break

        chunk_path = os.path.join(emb_chunks_dir, fn)
        data = np.load(chunk_path, allow_pickle=True)

        m = data["embeddings"].shape[0]
        take = min(m, n - got)

        if take <= 0:
            break

        # Filter by item type if requested
        if item_filter:
            chunk_df = pd.DataFrame({
                "accession_number": data["accession_numbers"],
                "sentence_id": data["sentence_ids"],
                "item_type": data["item_types"],
                "cik": data["ciks"],
            })
            chunk_df = chunk_df[chunk_df["item_type"] == item_filter]
            keys.append(chunk_df)
            got += len(chunk_df)
        else:
            keys.append(pd.DataFrame({
                "accession_number": data["accession_numbers"][:take],
                "sentence_id": data["sentence_ids"][:take],
                "item_type": data["item_types"][:take],
                "cik": data["ciks"][:take],
            }))
            got += take

        print(f"  Loaded {fn}: {take:,} rows (total: {got:,})")

    # Concatenate embedding keys
    print(f"\nConcatenating embedding metadata...")
    emb_key = pd.concat(keys, ignore_index=True).head(n)
    print(f"  Embedding metadata rows: {len(emb_key):,}")

    # Get sentence keys
    df_key = sent_df[["accession_number", "sentence_id", "item_type", "cik"]]

    # Compare
    print(f"\nComparing keys...")
    print(f"  Sentence keys shape: {df_key.shape}")
    print(f"  Embedding keys shape: {emb_key.shape}")

    if len(emb_key) != len(df_key):
        print(f"\n  ERROR: Length mismatch! {len(emb_key)} embeddings vs {len(df_key)} sentences")
        return False

    # Check exact equality
    matches = emb_key.equals(df_key)

    if matches:
        print(f"\n  ✓ PASS: All keys match perfectly!")
        print(f"  ✓ Embeddings are correctly aligned with sentences")
        return True
    else:
        print(f"\n  ✗ FAIL: Keys do not match!")

        # Show where mismatches occur
        print("\n  Finding mismatches...")
        for col in ["accession_number", "sentence_id", "item_type", "cik"]:
            mismatch = (emb_key[col] != df_key[col]).sum()
            if mismatch > 0:
                print(f"    {col}: {mismatch:,} mismatches")

        # Show first few mismatches
        print("\n  First 5 mismatches:")
        for i in range(min(5, len(emb_key))):
            if not (emb_key.iloc[i] == df_key.iloc[i]).all():
                print(f"\n    Row {i}:")
                print(f"      Embedding: {emb_key.iloc[i].to_dict()}")
                print(f"      Sentence:  {df_key.iloc[i].to_dict()}")

        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Verify embedding-sentence alignment')
    parser.add_argument('--sent_file', default=r"C:\Users\ofs4963\Dropbox\Arojects\SAE\data\sentences_sampled.parquet")
    parser.add_argument('--emb_chunks_dir', default=r"C:\Users\ofs4963\Dropbox\Arojects\SAE\data\sentence_embeddings_chunks")
    parser.add_argument('--item', default=None, choices=[None, 'item1', 'item1A', 'item7'])

    args = parser.parse_args()

    result = verify_alignment(args.sent_file, args.emb_chunks_dir, args.item)

    print("\n" + "="*80)
    if result:
        print("VERIFICATION PASSED ✓")
    else:
        print("VERIFICATION FAILED ✗")
    print("="*80)
