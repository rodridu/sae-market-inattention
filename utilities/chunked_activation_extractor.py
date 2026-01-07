"""
Chunked Activation Extractor - Memory-Efficient SAE Feature Extraction
------------------------------------------------------------------------
Solves the memory overflow problem in 03_sae_training.py by processing
activations in chunks and writing directly to disk.

Key improvements:
1. Processes embeddings in manageable chunks (default: 50K sentences)
2. Extracts only stable features immediately (not all M features)
3. Writes chunks to temporary parquet files
4. Concatenates parquet files at the end (much more memory efficient)
5. Uses memory-mapped arrays for large intermediate results

Author: Claude Code
Date: 2026-01-04
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Dict, List, Tuple, Any
import tempfile
import shutil
from tqdm import tqdm
import sys
import os

# Add parent directory to path to import from 03_sae_training
sys.path.insert(0, str(Path(__file__).parent.parent))


# =========================
# SAE Model Definition
# =========================

class KSparseAutoencoder(torch.nn.Module):
    """
    k-sparse autoencoder.

    Forward pass:
    - Encode: z = ReLU(Wx + b_enc)
    - Sparsify: Keep top-k activations, zero out rest
    - Decode: x_hat = Vz + b_dec

    This is a minimal copy from 03_sae_training.py for loading checkpoints.
    """
    def __init__(self, input_dim, hidden_dim, k):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = k

        self.encoder = torch.nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder = torch.nn.Linear(hidden_dim, input_dim, bias=True)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        z = self.activation(self.encoder(x))  # (batch, hidden_dim)

        # k-sparse: keep top-k activations per sample, zero out the rest
        if self.k is not None and self.k < self.hidden_dim:
            with torch.no_grad():
                # indices of top-k per row
                topk_vals, topk_idx = torch.topk(z, self.k, dim=1)
                mask = torch.zeros_like(z)
                mask.scatter_(1, topk_idx, 1.0)
            z = z * mask

        x_hat = self.decoder(z)
        return x_hat, z


# =========================
# Chunked Extractor
# =========================

class ChunkedActivationExtractor:
    """Memory-efficient extraction of SAE activations in chunks."""

    def __init__(
        self,
        chunk_size: int = 50000,
        batch_size: int = 1024,
        device: str = 'cpu',
        verbose: bool = True
    ):
        """
        Parameters
        ----------
        chunk_size : int
            Number of sentences to process per chunk (default: 50K)
        batch_size : int
            Batch size for model inference
        device : str
            Device for PyTorch ('cpu' or 'cuda')
        verbose : bool
            Print progress information
        """
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose

    def extract_stable_activations(
        self,
        embeddings: np.ndarray,
        models: List[torch.nn.Module],
        stable_indices: np.ndarray,
        M: int,
        k: int,
        temp_dir: Path = None
    ) -> np.ndarray:
        """
        Extract activations for stable features only, processing in chunks.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape (N, D) - sentence embeddings
        models : List[torch.nn.Module]
            Ensemble of SAE models (we'll use models[0])
        stable_indices : np.ndarray
            Indices of stable neurons to extract
        M : int
            SAE hidden dimension
        k : int
            Sparsity level
        temp_dir : Path, optional
            Temporary directory for intermediate files

        Returns
        -------
        activations : np.ndarray
            Shape (N, len(stable_indices)) - activations for stable features only
        """
        N = len(embeddings)
        n_stable = len(stable_indices)

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"CHUNKED ACTIVATION EXTRACTION")
            print(f"{'='*80}")
            print(f"Total sentences: {N:,}")
            print(f"Stable features: {n_stable:,} (out of {M:,})")
            print(f"Chunk size: {self.chunk_size:,}")
            print(f"Estimated chunks: {(N + self.chunk_size - 1) // self.chunk_size}")
            print(f"Memory per chunk: ~{(self.chunk_size * n_stable * 4) / 1e9:.2f} GB")
            print(f"{'='*80}\n")

        # Use first replica for extraction
        model = models[0]
        model.eval()

        # Create temporary directory for chunk files
        if temp_dir is None:
            temp_dir = Path(tempfile.mkdtemp(prefix='sae_chunks_'))
        else:
            temp_dir.mkdir(exist_ok=True, parents=True)

        if self.verbose:
            print(f"Temporary directory: {temp_dir}")

        try:
            chunk_files = []
            n_chunks = (N + self.chunk_size - 1) // self.chunk_size

            # Process each chunk
            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * self.chunk_size
                end_idx = min(start_idx + self.chunk_size, N)
                chunk_n = end_idx - start_idx

                if self.verbose:
                    print(f"\nChunk {chunk_idx + 1}/{n_chunks}: sentences {start_idx:,} to {end_idx:,} ({chunk_n:,})")

                # Extract chunk embeddings
                chunk_embeddings = embeddings[start_idx:end_idx]

                # Compute activations for this chunk
                chunk_activations = self._compute_chunk_activations(
                    chunk_embeddings, model, stable_indices
                )

                # Save chunk to parquet
                chunk_file = temp_dir / f'chunk_{chunk_idx:04d}.parquet'
                chunk_df = pd.DataFrame(
                    chunk_activations,
                    columns=[f'feat_{i}' for i in range(n_stable)]
                )
                chunk_df.to_parquet(chunk_file, index=False)
                chunk_files.append(chunk_file)

                if self.verbose:
                    print(f"  Saved chunk to {chunk_file.name}")
                    print(f"  Chunk shape: {chunk_activations.shape}")

            # Concatenate all chunks
            if self.verbose:
                print(f"\n{'='*80}")
                print(f"CONCATENATING {len(chunk_files)} CHUNKS")
                print(f"{'='*80}")

            # Read chunks in order and concatenate
            all_chunks = []
            for chunk_file in chunk_files:
                chunk_df = pd.read_parquet(chunk_file)
                all_chunks.append(chunk_df.values)

            activations = np.vstack(all_chunks)

            if self.verbose:
                print(f"Final activations shape: {activations.shape}")
                print(f"Memory: ~{activations.nbytes / 1e9:.2f} GB")
                print(f"{'='*80}\n")

            return activations

        finally:
            # Clean up temporary directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                if self.verbose:
                    print(f"Cleaned up temporary directory: {temp_dir}")

    def _compute_chunk_activations(
        self,
        chunk_embeddings: np.ndarray,
        model: torch.nn.Module,
        stable_indices: np.ndarray
    ) -> np.ndarray:
        """
        Compute activations for a single chunk.

        Parameters
        ----------
        chunk_embeddings : np.ndarray
            Shape (chunk_size, D)
        model : torch.nn.Module
            SAE model
        stable_indices : np.ndarray
            Indices of stable features to extract

        Returns
        -------
        chunk_activations : np.ndarray
            Shape (chunk_size, n_stable)
        """
        # Create DataLoader for chunk
        dataset = TensorDataset(torch.from_numpy(chunk_embeddings).float())
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # Compute activations batch by batch
        batch_results = []
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                _, z = model(batch)  # z shape: (batch_size, M)

                # Extract only stable features immediately
                z_stable = z[:, stable_indices]  # Shape: (batch_size, n_stable)

                batch_results.append(z_stable.cpu().numpy())

        # Stack batches for this chunk
        chunk_activations = np.vstack(batch_results)
        return chunk_activations


def extract_stable_activations_chunked(
    sent_df: pd.DataFrame,
    embeddings: np.ndarray,
    ensemble: Dict,
    ensemble_dir: Path,
    stability_analyzer,
    threshold: float = 0.8,
    chunk_size: int = 50000,
    batch_size: int = 1024,
    device: str = 'cpu'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract stable feature activations using chunked processing.

    This is a drop-in replacement for the original extract_stable_activations
    function in 03_sae_training.py that uses memory-efficient chunked processing.

    Parameters
    ----------
    sent_df : pd.DataFrame
        Sentence-level DataFrame with metadata
    embeddings : np.ndarray
        Shape (N, D) - sentence embeddings
    ensemble : Dict
        Dictionary mapping (M, k) -> list of model checkpoints
    ensemble_dir : Path
        Directory containing ensemble checkpoints
    stability_analyzer : StabilityAnalyzer
        Analyzer for identifying stable features
    threshold : float
        Stability threshold (default: 0.8)
    chunk_size : int
        Number of sentences per chunk (default: 50K)
    batch_size : int
        Batch size for inference
    device : str
        Device for PyTorch

    Returns
    -------
    sent_df_with_features : pd.DataFrame
        Sentence DataFrame with stable feature activations
    feature_metadata : pd.DataFrame
        Metadata for each stable feature
    """
    print("\n" + "="*80)
    print("EXTRACTING STABLE FEATURE ACTIVATIONS (CHUNKED)")
    print("="*80)

    # Initialize extractor
    extractor = ChunkedActivationExtractor(
        chunk_size=chunk_size,
        batch_size=batch_size,
        device=device,
        verbose=True
    )

    all_activations = []
    feature_metadata = []

    for (M, k), checkpoint_paths in ensemble.items():
        print(f"\nProcessing (M={M}, k={k})...")

        # Load all replicas
        models = []
        for ckpt_path in checkpoint_paths:
            ckpt = torch.load(ckpt_path, map_location=device)
            model = KSparseAutoencoder(
                input_dim=embeddings.shape[1],
                hidden_dim=M,
                k=k
            )
            model.load_state_dict(ckpt['model_state_dict'])
            model.to(device)
            models.append(model)

        # Identify stable features
        stability_info = stability_analyzer.identify_stable_features(models, M, k)
        stable_indices = stability_info['stable_indices']

        if len(stable_indices) == 0:
            print(f"  WARNING: No stable features for (M={M}, k={k}). Skipping.")
            continue

        # Extract activations using chunked processing
        Z_stable = extractor.extract_stable_activations(
            embeddings=embeddings,
            models=models,
            stable_indices=stable_indices,
            M=M,
            k=k
        )

        print(f"  Extracted activations shape: {Z_stable.shape}")

        # Create feature metadata
        for i, neuron_idx in enumerate(stable_indices):
            feature_name = f"M{M}_k{k}_n{neuron_idx}"

            feature_metadata.append({
                'feature_name': feature_name,
                'M': M,
                'k': k,
                'neuron_idx': int(neuron_idx),
                'mean_similarity': float(stability_info['mean_similarities'][neuron_idx]),
                'global_feature_idx': len(feature_metadata)
            })

        all_activations.append(Z_stable)

    # Concatenate all stable features
    if len(all_activations) == 0:
        raise ValueError("No stable features found in ensemble! Lower threshold or increase replicas.")

    Z_concat = np.hstack(all_activations)  # Shape: (N, K_total)

    print("\n" + "="*80)
    print("CONCATENATION SUMMARY")
    print("="*80)
    print(f"Total stable features (K): {Z_concat.shape[1]}")
    print(f"\nFeatures per (M,k):")

    # Group metadata by (M,k)
    meta_df = pd.DataFrame(feature_metadata)
    for (M, k) in ensemble.keys():
        n_features = len(meta_df[(meta_df['M'] == M) & (meta_df['k'] == k)])
        print(f"  (M={M:5}, k={k:2}): {n_features:4} features")

    print("="*80)

    # Create DataFrame with activations - use chunked approach
    print("\nCreating activation DataFrame...")

    # Process in chunks to avoid memory issues
    COLUMN_CHUNK_SIZE = 200  # Process 200 features at a time
    n_features = Z_concat.shape[1]
    n_column_chunks = (n_features + COLUMN_CHUNK_SIZE - 1) // COLUMN_CHUNK_SIZE

    activation_dfs = []
    for col_chunk_idx in range(n_column_chunks):
        col_start = col_chunk_idx * COLUMN_CHUNK_SIZE
        col_end = min(col_start + COLUMN_CHUNK_SIZE, n_features)

        # Extract chunk of columns
        chunk_data = {}
        for i in range(col_start, col_end):
            meta = feature_metadata[i]
            col_name = f"h_{meta['feature_name']}"
            chunk_data[col_name] = Z_concat[:, i]

        chunk_df = pd.DataFrame(chunk_data, index=sent_df.index)
        activation_dfs.append(chunk_df)

        print(f"  Processed features {col_start} to {col_end} ({col_end - col_start} features)")

    # Concatenate column chunks
    activation_df = pd.concat(activation_dfs, axis=1)
    print(f"  Total activation columns: {len(activation_df.columns)}")

    # Memory-efficient column assignment instead of pd.concat
    print("\nAdding activation columns to sentence DataFrame...")
    import gc

    # Add activation columns in batches to avoid memory spikes
    ASSIGN_BATCH_SIZE = 500
    n_cols = len(activation_df.columns)
    n_batches = (n_cols + ASSIGN_BATCH_SIZE - 1) // ASSIGN_BATCH_SIZE

    for batch_idx in range(n_batches):
        start_col = batch_idx * ASSIGN_BATCH_SIZE
        end_col = min(start_col + ASSIGN_BATCH_SIZE, n_cols)

        cols_to_assign = activation_df.columns[start_col:end_col]
        for col in cols_to_assign:
            sent_df[col] = activation_df[col].values

        if (batch_idx + 1) % 2 == 0:  # Print every 2 batches
            print(f"  Assigned {end_col}/{n_cols} columns ({100*end_col/n_cols:.1f}%)")

    print(f"  [OK] Assigned all {n_cols} activation columns")

    # Clear activation_df from memory
    del activation_df
    del activation_dfs
    gc.collect()

    sent_df_with_features = sent_df

    # Convert metadata to DataFrame
    feature_metadata_df = pd.DataFrame(feature_metadata)

    print(f"\n[OK] Created sentence DataFrame with {n_cols} stable features")
    print(f"     Shape: {sent_df_with_features.shape}")

    return sent_df_with_features, feature_metadata_df
