"""
Test Chunked Activation Extractor
----------------------------------
Validates that the chunked extractor produces correct results and
handles memory efficiently.

Usage:
    python utilities/test_chunked_extractor.py
"""

import numpy as np
import pandas as pd
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utilities.chunked_activation_extractor import (
    ChunkedActivationExtractor,
    KSparseAutoencoder
)


def create_test_data(n_sentences=10000, embedding_dim=384, seed=42):
    """Create synthetic test data."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    embeddings = np.random.randn(n_sentences, embedding_dim).astype(np.float32)

    sent_df = pd.DataFrame({
        'sentence_id': range(n_sentences),
        'text': [f'Sentence {i}' for i in range(n_sentences)]
    })

    return sent_df, embeddings


def create_test_model(input_dim=384, hidden_dim=1024, k=16, seed=42):
    """Create and initialize test SAE model."""
    torch.manual_seed(seed)
    model = KSparseAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim, k=k)
    model.eval()
    return model


def test_memory_efficiency():
    """Test 1: Memory efficiency with large dataset."""
    print("="*80)
    print("TEST 1: Memory Efficiency")
    print("="*80)

    # Create moderately large dataset
    n_sentences = 100000
    print(f"\nCreating test data: {n_sentences:,} sentences...")
    sent_df, embeddings = create_test_data(n_sentences=n_sentences)

    # Create test model
    M = 1024
    k = 16
    print(f"Creating test SAE: M={M}, k={k}...")
    model = create_test_model(input_dim=384, hidden_dim=M, k=k)

    # Identify "stable" features (just use first 500 for testing)
    stable_indices = np.arange(500)
    print(f"Stable features: {len(stable_indices)}")

    # Test chunked extraction
    print(f"\nTesting chunked extraction...")
    extractor = ChunkedActivationExtractor(
        chunk_size=10000,  # Small chunks for testing
        batch_size=512,
        device='cpu',
        verbose=True
    )

    try:
        activations = extractor.extract_stable_activations(
            embeddings=embeddings,
            models=[model],
            stable_indices=stable_indices,
            M=M,
            k=k
        )

        print(f"\n[OK] SUCCESS: Extracted activations shape: {activations.shape}")
        print(f"   Expected: ({n_sentences}, {len(stable_indices)})")

        assert activations.shape == (n_sentences, len(stable_indices)), \
            f"Shape mismatch: {activations.shape} != ({n_sentences}, {len(stable_indices)})"

        print(f"\n[OK] Memory test PASSED")
        return True

    except Exception as e:
        print(f"\n[FAILED] Memory test FAILED: {e}")
        return False


def test_correctness():
    """Test 2: Correctness - chunked vs non-chunked."""
    print("\n" + "="*80)
    print("TEST 2: Correctness (Chunked vs Non-Chunked)")
    print("="*80)

    # Create small dataset (so non-chunked works)
    n_sentences = 5000
    print(f"\nCreating test data: {n_sentences:,} sentences...")
    sent_df, embeddings = create_test_data(n_sentences=n_sentences)

    # Create test model
    M = 512
    k = 16
    print(f"Creating test SAE: M={M}, k={k}...")
    model = create_test_model(input_dim=384, hidden_dim=M, k=k)

    # Identify stable features
    stable_indices = np.arange(200)
    print(f"Stable features: {len(stable_indices)}")

    # 1. Non-chunked (baseline)
    print(f"\n1. Computing activations (non-chunked baseline)...")
    from torch.utils.data import TensorDataset, DataLoader

    dataset = TensorDataset(torch.from_numpy(embeddings).float())
    loader = DataLoader(dataset, batch_size=512, shuffle=False)

    model.eval()
    all_z = []
    with torch.no_grad():
        for (batch,) in loader:
            _, z = model(batch)
            all_z.append(z.cpu().numpy())

    Z_full = np.vstack(all_z)
    Z_baseline = Z_full[:, stable_indices]
    print(f"   Baseline shape: {Z_baseline.shape}")

    # 2. Chunked
    print(f"\n2. Computing activations (chunked)...")
    extractor = ChunkedActivationExtractor(
        chunk_size=1000,  # Small chunks to test chunking logic
        batch_size=512,
        device='cpu',
        verbose=False
    )

    Z_chunked = extractor.extract_stable_activations(
        embeddings=embeddings,
        models=[model],
        stable_indices=stable_indices,
        M=M,
        k=k
    )
    print(f"   Chunked shape: {Z_chunked.shape}")

    # Compare
    print(f"\n3. Comparing results...")
    print(f"   Shape match: {Z_baseline.shape == Z_chunked.shape}")

    max_diff = np.max(np.abs(Z_baseline - Z_chunked))
    mean_diff = np.mean(np.abs(Z_baseline - Z_chunked))

    print(f"   Max absolute difference: {max_diff:.2e}")
    print(f"   Mean absolute difference: {mean_diff:.2e}")

    # Check if results are close (allow small numerical errors)
    is_close = np.allclose(Z_baseline, Z_chunked, rtol=1e-5, atol=1e-6)

    if is_close:
        print(f"\n[OK] Correctness test PASSED")
        return True
    else:
        print(f"\n[FAILED] Correctness test FAILED")
        print(f"   Results differ beyond tolerance!")
        return False


def test_chunk_sizes():
    """Test 3: Different chunk sizes produce same results."""
    print("\n" + "="*80)
    print("TEST 3: Chunk Size Invariance")
    print("="*80)

    n_sentences = 10000
    print(f"\nCreating test data: {n_sentences:,} sentences...")
    sent_df, embeddings = create_test_data(n_sentences=n_sentences)

    M = 512
    k = 16
    model = create_test_model(input_dim=384, hidden_dim=M, k=k)
    stable_indices = np.arange(100)

    chunk_sizes = [1000, 2500, 5000]
    results = {}

    for chunk_size in chunk_sizes:
        print(f"\nTesting chunk_size={chunk_size}...")
        extractor = ChunkedActivationExtractor(
            chunk_size=chunk_size,
            batch_size=512,
            device='cpu',
            verbose=False
        )

        Z = extractor.extract_stable_activations(
            embeddings=embeddings,
            models=[model],
            stable_indices=stable_indices,
            M=M,
            k=k
        )
        results[chunk_size] = Z
        print(f"   Shape: {Z.shape}")

    # Compare all pairs
    print(f"\nComparing results across chunk sizes...")
    all_match = True
    for i, cs1 in enumerate(chunk_sizes):
        for cs2 in chunk_sizes[i+1:]:
            is_close = np.allclose(results[cs1], results[cs2], rtol=1e-5, atol=1e-6)
            status = "[OK]" if is_close else "[FAILED]"
            print(f"   {status} chunk_size={cs1} vs {cs2}: {is_close}")
            all_match = all_match and is_close

    if all_match:
        print(f"\n[OK] Chunk size invariance test PASSED")
        return True
    else:
        print(f"\n[FAILED] Chunk size invariance test FAILED")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("CHUNKED ACTIVATION EXTRACTOR - VALIDATION TESTS")
    print("="*80)

    results = []

    # Run tests
    results.append(("Memory Efficiency", test_memory_efficiency()))
    results.append(("Correctness", test_correctness()))
    results.append(("Chunk Size Invariance", test_chunk_sizes()))

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    for test_name, passed in results:
        status = "[OK] PASSED" if passed else "[FAILED] FAILED"
        print(f"{status}: {test_name}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n" + "="*80)
        print("ALL TESTS PASSED - Chunked extractor is working correctly!")
        print("="*80)
        return 0
    else:
        print("\n" + "="*80)
        print("WARNING: SOME TESTS FAILED - Please review the output above")
        print("="*80)
        return 1


if __name__ == '__main__':
    sys.exit(main())
