"""
Phase 3: SAE Training and Concept Discovery

Implements sparse autoencoder training on sentence/span embeddings and concept-level analysis.

Aligned with proposal (Sections 4.4-4.6):
1. Train k-sparse autoencoder on sentence/span embeddings (unsupervised)
2. Extract neuron activations z_fdt
3. Aggregate to document level: A_k = Σ_s z_fdt,k
4. Feature selection with Lasso controlling for CLN info and KMNZ relevance
5. LLM-based neuron interpretation and fidelity validation

Core equation (proposal Section 4.6):
  y_fdt = α + β₁·Info^CLN + β₂·Relevance^KMNZ + Σ_k δₖ·A_k + Γ'X + ε

Supports both:
- Sentence-level: Finer granularity, atomic units
- Span-level: Paragraph-like units (~80 tokens), better semantic coherence

Usage:
  python 03_sae_training.py --unit sentences  # sentence-level analysis (default)
  python 03_sae_training.py --unit spans      # span-level analysis

Assumptions:
- Unit-level dataset with CLN novelty (N_fdt) and KMNZ relevance (R_fdt)
- Access to announcement returns and drift outcomes
"""

# =========================
# 0. Imports and config
# =========================

import os
import sys
import argparse
import pandas as pd
import numpy as np
import gc

from tqdm import tqdm

# Embeddings
# e.g. from sentence_transformers import SentenceTransformer
# model = SentenceTransformer("all-MiniLM-L6-v2")

# PyTorch SAE
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Lasso and regressions
from sklearn.linear_model import LassoCV
import statsmodels.api as sm

# Ensemble infrastructure imports
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import itertools
import json
from pathlib import Path

# Memory-efficient chunked activation extractor
from utilities.chunked_activation_extractor import extract_stable_activations_chunked

# Parse command-line arguments FIRST
parser = argparse.ArgumentParser(description='Train SAE on sentence or span embeddings')
parser.add_argument('--unit', default='sentences', choices=['sentences', 'spans'],
                    help='Unit of analysis: "sentences" or "spans" (default: sentences)')

# Ensemble-specific arguments
parser.add_argument('--ensemble', action='store_true',
                    help='Train multi-(M,k) ensemble instead of single SAE')
parser.add_argument('--ensemble-M', type=int, nargs='+',
                    default=[4096, 8192, 16384],
                    help='Expansion factors for ensemble grid (default: 4096 8192 16384)')
parser.add_argument('--ensemble-k', type=int, nargs='+',
                    default=[16, 32, 64],
                    help='Sparsity levels for ensemble grid (default: 16 32 64)')
parser.add_argument('--ensemble-replicas', type=int, default=5,
                    help='Bootstrap replicas per (M,k) (default: 5)')
parser.add_argument('--ensemble-threshold', type=float, default=0.8,
                    help='Stability threshold for feature selection (default: 0.8)')
parser.add_argument('--chunk-size', type=int, default=50000,
                    help='Chunk size for memory-efficient activation extraction (default: 50000)')
parser.add_argument('--ensemble-subset', type=str, default=None,
                    help='Train subset of grid for parallelization (e.g., "M4096", "k32")')

args = parser.parse_args()

UNIT = args.unit
UNIT_SINGULAR = 'sentence' if UNIT == 'sentences' else 'span'
UNIT_ID = 'sentence_id' if UNIT == 'sentences' else 'paragraph_id'

# =========================
# Ensemble Infrastructure
# =========================

@dataclass
class EnsembleConfig:
    """
    Configuration for multi-(M,k) ensemble with bootstrap replicas.

    Aligned with proposal Section 4.2:
    - Grid of expansion factors M and sparsity levels k
    - Bootstrap replicas with different initializations
    - Stability threshold for feature selection
    """
    expansion_factors: List[int] = field(default_factory=lambda: [4096, 8192, 16384])
    sparsity_levels: List[int] = field(default_factory=lambda: [16, 32, 64])
    n_replicas: int = 5
    bootstrap_sample_frac: float = 0.8  # Sample 80% of FILINGS per replica
    stability_threshold: float = 0.8    # Cosine similarity threshold
    random_seed_base: int = 42

    def get_grid(self) -> List[Tuple[int, int]]:
        """Return all (M, k) pairs in the grid"""
        return list(itertools.product(self.expansion_factors, self.sparsity_levels))

    def get_replica_seeds(self, replica_idx: int) -> Dict[str, int]:
        """
        Return seeds for this replica.

        Returns:
            Dict with 'sample_seed' for bootstrap sampling and
            'model_seed' for model initialization
        """
        return {
            'sample_seed': self.random_seed_base + replica_idx,
            'model_seed': self.random_seed_base + 1000 + replica_idx
        }

    def total_models(self) -> int:
        """Total number of models to train"""
        return len(self.get_grid()) * self.n_replicas


def bootstrap_sample_sentences(sent_df: pd.DataFrame,
                               embeddings: np.ndarray,
                               sample_frac: float = 0.8,
                               seed: int = 42) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Bootstrap sample sentences at the FILING level.

    Critical design choice: Sample FILINGS, not individual sentences.
    Rationale: Unit of analysis is the filing. Each 10-K is a separate
    disclosure event. Sampling sentences would break document structure.

    Args:
        sent_df: Sentence-level DataFrame
        embeddings: Aligned embedding matrix (N, D)
        sample_frac: Fraction of FILINGS to sample (default: 0.8)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (sampled_sent_df, sampled_embeddings)
    """
    np.random.seed(seed)

    # Sample at filing level (accession_number = unique filing ID)
    unique_filings = sent_df['accession_number'].unique()
    n_sample = int(len(unique_filings) * sample_frac)
    sampled_filings = np.random.choice(unique_filings, size=n_sample, replace=False)

    # Filter sentences from sampled filings
    mask = sent_df['accession_number'].isin(sampled_filings)
    sampled_sent_df = sent_df[mask].reset_index(drop=True)
    sampled_embeddings = embeddings[mask]

    print(f"  Bootstrap sample: {len(sampled_filings):,} filings ({100*sample_frac:.0f}%) "
          f"-> {len(sampled_sent_df):,} sentences")

    return sampled_sent_df, sampled_embeddings


print("="*60)
print(f"PHASE 3: SAE TRAINING AND CONCEPT DISCOVERY ({UNIT.upper()} LEVEL)")
print("="*60)
print(f"Analysis unit: {UNIT}")
print(f"Unit ID column: {UNIT_ID}")
print("="*60)

# =========================
# 1. Load data with CLN and KMNZ
# =========================

DATA_DIR = r"C:\Users\ofs4963\Dropbox\Arojects\SAE\data"

# Configure file paths based on unit type
if UNIT == 'sentences':
    # FIXED: Use sentences_sampled.parquet (100% match with embeddings)
    # NOT sentences_sae_train.parquet (only 10% overlap with embeddings)
    UNIT_FILE = os.path.join(DATA_DIR, "sentences_sampled.parquet")
    EMBEDDING_FILE = os.path.join(DATA_DIR, "sentence_embeddings.npz")
    NOVELTY_FILE = os.path.join(DATA_DIR, "novelty_cln_sae.parquet")
    RELEVANCE_FILE = os.path.join(DATA_DIR, "relevance_kmnz_sae.parquet")
else:  # spans
    UNIT_FILE = os.path.join(DATA_DIR, "spans_sae_train.parquet")
    EMBEDDING_FILE = os.path.join(DATA_DIR, "span_embeddings.npz")
    NOVELTY_FILE = os.path.join(DATA_DIR, "novelty_cln_spans.parquet")
    RELEVANCE_FILE = os.path.join(DATA_DIR, "relevance_kmnz_spans.parquet")

# Load unit data (sentences or spans)
print(f"\nLoading {UNIT} data from {UNIT_FILE}...")
if not os.path.exists(UNIT_FILE):
    print(f"ERROR: {UNIT} file not found at {UNIT_FILE}")
    print(f"Please run the appropriate data preparation script first:")
    if UNIT == 'sentences':
        print("  - 01_data_preparation.py")
    else:
        print("  - 01b_construct_spans.py")
    sys.exit(1)

sent_df = pd.read_parquet(UNIT_FILE)
print(f"Loaded {len(sent_df):,} {UNIT}")
print(f"Columns: {sent_df.columns.tolist()}")

# Load CLN novelty if available
if os.path.exists(NOVELTY_FILE):
    print(f"\nLoading CLN novelty measures from {NOVELTY_FILE}...")
    novelty_df = pd.read_parquet(NOVELTY_FILE)
    merge_keys = ['accession_number', UNIT_ID, 'item_type']

    # Check initial row count
    initial_rows = len(sent_df)

    sent_df = sent_df.merge(novelty_df[merge_keys + ['novelty_cln']],
                            on=merge_keys, how='left')

    # Verify no duplicate rows created
    if len(sent_df) != initial_rows:
        print(f"  WARNING: Merge created duplicates! {initial_rows} -> {len(sent_df)} rows")
        print(f"  Keeping first occurrence only...")
        sent_df = sent_df.drop_duplicates(subset=merge_keys, keep='first')
        print(f"  After dedup: {len(sent_df)} rows")

    print(f"  [OK] Merged novelty for {sent_df['novelty_cln'].notna().sum():,} {UNIT}")
else:
    print(f"  WARNING: CLN novelty file not found. Run 02b_novelty_cln{'_spans' if UNIT == 'spans' else ''}.py first.")
    print("  Using placeholder novelty = 0")
    sent_df['novelty_cln'] = 0.0

# Load KMNZ relevance if available
if os.path.exists(RELEVANCE_FILE):
    print(f"\nLoading KMNZ relevance measures from {RELEVANCE_FILE}...")
    relevance_df = pd.read_parquet(RELEVANCE_FILE)
    merge_keys = ['accession_number', UNIT_ID, 'item_type']

    # Check initial row count
    initial_rows = len(sent_df)

    sent_df = sent_df.merge(relevance_df[merge_keys + ['relevance_kmnz']],
                            on=merge_keys, how='left')

    # Verify no duplicate rows created
    if len(sent_df) != initial_rows:
        print(f"  WARNING: Merge created duplicates! {initial_rows} -> {len(sent_df)} rows")
        print(f"  Keeping first occurrence only...")
        sent_df = sent_df.drop_duplicates(subset=merge_keys, keep='first')
        print(f"  After dedup: {len(sent_df)} rows")

    print(f"  [OK] Merged relevance for {sent_df['relevance_kmnz'].notna().sum():,} {UNIT}")
else:
    print(f"  WARNING: KMNZ relevance file not found. Run 02c_relevance_kmnz{'_spans' if UNIT == 'spans' else ''}.py first.")
    print("  Using placeholder relevance = 0")
    sent_df['relevance_kmnz'] = 0.0

# =========================
# 2. Build embeddings
# =========================

def load_embeddings(embedding_file):
    """
    Load precomputed embeddings from Phase 2.
    Supports both single-file and chunked formats.
    """
    # Check for chunked embeddings first
    chunks_dir = embedding_file.replace('.npz', '_chunks')
    index_file = embedding_file.replace('.npz', '_index.csv')

    if os.path.exists(index_file) and os.path.exists(chunks_dir):
        print(f"\nLoading chunked embeddings from {chunks_dir}...")
        index_df = pd.read_csv(index_file)

        embeddings_list = []
        for _, row in index_df.iterrows():
            chunk_path = row['path']
            print(f"  Loading chunk {row['chunk_idx']}: {row['start']:,} to {row['end']:,} ({row['n']:,} embeddings)")
            chunk_data = np.load(chunk_path)
            embeddings_list.append(chunk_data['embeddings'])

        embeddings = np.vstack(embeddings_list)
        print(f"[OK] Loaded {len(embeddings):,} embeddings from {len(index_df)} chunks")
        print(f"  Embedding dimension: {embeddings.shape[1]}")
        return embeddings

    elif os.path.exists(embedding_file):
        print(f"\nLoading embeddings from {embedding_file}...")
        data = np.load(embedding_file)
        embeddings = data['embeddings']
        print(f"Loaded embeddings with shape: {embeddings.shape}")
        return embeddings

    else:
        return None

# Load precomputed embeddings
embeddings = load_embeddings(EMBEDDING_FILE)

if embeddings is None:
    print(f"ERROR: Embeddings not found at {EMBEDDING_FILE}")
    print(f"Please run 02_embeddings{'_spans' if UNIT == 'spans' else '_and_features'}.py first.")
    sys.exit(1)

# Verify alignment
if len(embeddings) != len(sent_df):
    print(f"\nERROR: Mismatch between {UNIT} count ({len(sent_df):,}) and embeddings ({len(embeddings):,})")
    print(f"This indicates a data processing issue.")
    print(f"\nDiagnostics:")
    print(f"  - {UNIT} data loaded: {len(sent_df):,} rows")
    print(f"  - Embeddings loaded: {len(embeddings):,} rows")
    print(f"  - Expected 1:1 correspondence")
    print(f"\nPlease ensure all preprocessing steps used the same dataset.")
    sys.exit(1)

sent_df["emb_idx"] = np.arange(len(sent_df))

# =========================
# 3. k-Sparse Autoencoder definition
# =========================

class KSparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, k):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = k

        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)
        self.activation = nn.ReLU()

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
# 4. Train SAE
# =========================

class EmbeddingDataset(Dataset):
    def __init__(self, emb_array):
        self.emb = torch.from_numpy(emb_array)

    def __len__(self):
        return self.emb.shape[0]

    def __getitem__(self, idx):
        return self.emb[idx]

def train_sae(embeddings, input_dim, hidden_dim=1024, k=32,
              batch_size=1024, lr=1e-3, epochs=10, device="cuda",
              random_seed=42):
    """
    Train k-sparse autoencoder with explicit random seed for reproducibility.

    Args:
        embeddings: Input embeddings (N, D)
        input_dim: Input dimension D
        hidden_dim: Number of SAE features M
        k: Sparsity level (number of active features)
        batch_size: Training batch size
        lr: Learning rate
        epochs: Number of training epochs
        device: "cuda" or "cpu"
        random_seed: Seed for model initialization and data shuffling

    Returns:
        Trained KSparseAutoencoder model
    """
    # Set all random seeds for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)

    dataset = EmbeddingDataset(embeddings)

    # Use generator with seed for reproducible shuffling
    generator = torch.Generator()
    generator.manual_seed(random_seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

    model = KSparseAutoencoder(input_dim, hidden_dim, k).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            x_hat, z = model(batch)
            loss = criterion(x_hat, batch)  # you can add extra sparsity penalties here
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        print(f"Epoch {epoch+1}/{epochs}, loss={total_loss/len(dataset):.4f}")

    return model


def train_ensemble(sent_df: pd.DataFrame,
                   embeddings: np.ndarray,
                   config: EnsembleConfig,
                   output_dir: str,
                   device: str = "cuda",
                   batch_size: int = 2048,
                   lr: float = 1e-3,
                   epochs: int = 10) -> Dict[Tuple[int, int], List]:
    """
    Train multi-(M,k) ensemble with bootstrap replicas.

    For each (M, k) in grid:
        For each replica r in [1..n_replicas]:
            1. Bootstrap sample filings
            2. Train SAE with replica-specific seed
            3. Save checkpoint with metadata

    Args:
        sent_df: Sentence-level DataFrame
        embeddings: Full embedding matrix
        config: EnsembleConfig instance
        output_dir: Base directory for checkpoints
        device: "cuda" or "cpu"
        batch_size: Training batch size
        lr: Learning rate
        epochs: Training epochs per model

    Returns:
        Dict mapping (M, k) -> List[checkpoint_paths] (one per replica)
    """
    input_dim = embeddings.shape[1]
    grid = config.get_grid()

    print("\n" + "="*80)
    print("MULTI-(M,k) ENSEMBLE TRAINING")
    print("="*80)
    print(f"Grid: {grid}")
    print(f"Replicas per (M,k): {config.n_replicas}")
    print(f"Total models: {config.total_models()}")
    print(f"Bootstrap sample fraction: {config.bootstrap_sample_frac}")
    print(f"Output directory: {output_dir}")
    print("="*80)

    # Estimate total time
    # Rough estimate: M=1024 takes ~10 min, time scales linearly with M
    max_M = max(M for M, k in grid)
    est_time_per_model = (max_M / 1024) * 10  # minutes
    total_est_time = config.total_models() * est_time_per_model
    print(f"\nEstimated training time:")
    print(f"  Per model (M={max_M}): ~{est_time_per_model:.0f} minutes")
    print(f"  Total sequential: ~{total_est_time/60:.1f} hours")
    print(f"  With 3-way parallelization: ~{total_est_time/(3*60):.1f} hours")
    print("="*80)

    # Storage for trained models
    ensemble = {}

    # Train each (M, k) combination
    for grid_idx, (M, k) in enumerate(grid):
        print(f"\n{'='*80}")
        print(f"GRID POINT {grid_idx+1}/{len(grid)}: (M={M}, k={k})")
        print(f"{'='*80}")

        models_for_mk = []

        # Train bootstrap replicas
        for replica_idx in range(config.n_replicas):
            print(f"\n--- Replica {replica_idx + 1}/{config.n_replicas} ---")

            # Get seeds for this replica
            seeds = config.get_replica_seeds(replica_idx)

            # Bootstrap sample
            sampled_df, sampled_emb = bootstrap_sample_sentences(
                sent_df, embeddings,
                sample_frac=config.bootstrap_sample_frac,
                seed=seeds['sample_seed']
            )

            # Train model
            print(f"Training SAE (M={M}, k={k}, seed={seeds['model_seed']})...")
            start_time = pd.Timestamp.now()

            model = train_sae(
                sampled_emb,
                input_dim=input_dim,
                hidden_dim=M,
                k=k,
                batch_size=batch_size,
                lr=lr,
                epochs=epochs,
                device=device,
                random_seed=seeds['model_seed']
            )

            elapsed = (pd.Timestamp.now() - start_time).total_seconds() / 60
            print(f"  Training completed in {elapsed:.1f} minutes")

            # Save checkpoint
            checkpoint_dir = os.path.join(output_dir, f"M{M}_k{k}_replica{replica_idx}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            checkpoint_path = os.path.join(checkpoint_dir, "model.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'hyperparameters': {
                    'M': M,
                    'k': k,
                    'input_dim': input_dim,
                    'epochs': epochs,
                    'lr': lr,
                    'batch_size': batch_size
                },
                'replica_info': {
                    'replica_idx': replica_idx,
                    'sample_seed': seeds['sample_seed'],
                    'model_seed': seeds['model_seed'],
                    'n_samples': len(sampled_df),
                    'n_filings': sampled_df['accession_number'].nunique()
                },
                'timestamp': pd.Timestamp.now().isoformat()
            }, checkpoint_path)

            # Save human-readable config
            config_path = os.path.join(checkpoint_dir, "config.json")
            with open(config_path, 'w') as f:
                json.dump({
                    'M': M,
                    'k': k,
                    'replica_idx': replica_idx,
                    'input_dim': input_dim,
                    'sample_seed': seeds['sample_seed'],
                    'model_seed': seeds['model_seed'],
                    'n_samples': len(sampled_df),
                    'n_filings': sampled_df['accession_number'].nunique(),
                    'training_time_minutes': elapsed,
                    'timestamp': pd.Timestamp.now().isoformat()
                }, f, indent=2)

            print(f"  [OK] Saved checkpoint to {checkpoint_dir}")

            models_for_mk.append(checkpoint_path)

        ensemble[(M, k)] = models_for_mk

        print(f"\n[OK] Completed (M={M}, k={k}): {len(models_for_mk)} replicas trained")

    print("\n" + "="*80)
    print("ENSEMBLE TRAINING COMPLETE")
    print(f"Total models trained: {sum(len(v) for v in ensemble.values())}")
    print("="*80)

    return ensemble


class StabilityAnalyzer:
    """
    Analyze stability of SAE features across bootstrap replicas.

    Following proposal Section 4.2, lines 100-101:
    "Identify features that are stable across replicas (e.g., cosine similarity
     in decoder weights above 0.8)"

    Method:
    1. Extract decoder weight vectors for each neuron (columns of W_dec)
    2. For each replica pair, compute best-match cosine similarities
    3. Average similarities across all pairs
    4. Keep neurons with mean similarity ≥ threshold
    """

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    def compute_decoder_similarities(self, models: List) -> np.ndarray:
        """
        Compute pairwise cosine similarities between decoder weight matrices.

        For each neuron j in model A, find best-matching neuron in model B.
        This handles permutation invariance (neurons can be in different orders).

        Args:
            models: List of KSparseAutoencoder models (same M, k)

        Returns:
            Similarity matrix of shape (n_replicas, n_replicas, M)
            sim[i, j, neuron] = max cosine similarity of neuron from model i to model j
        """
        n_models = len(models)
        M = models[0].hidden_dim

        print(f"    Computing decoder similarities for {n_models} replicas...")

        # Extract and normalize decoder weights
        decoders = []
        for model in models:
            # decoder.weight shape: (input_dim, hidden_dim)
            # We want columns (one per neuron), so transpose
            W_dec = model.decoder.weight.data.cpu().numpy().T  # Shape: (M, input_dim)

            # Normalize to unit length
            norms = np.linalg.norm(W_dec, axis=1, keepdims=True) + 1e-8
            W_dec_normalized = W_dec / norms

            decoders.append(W_dec_normalized)

        # Compute pairwise best-match similarities
        similarities = np.zeros((n_models, n_models, M))

        for i in range(n_models):
            for j in range(n_models):
                if i == j:
                    similarities[i, j, :] = 1.0  # Perfect self-similarity
                else:
                    # Cosine similarity matrix: (M_i, M_j)
                    cos_sim = decoders[i] @ decoders[j].T

                    # For each neuron in model i, find best match in model j
                    similarities[i, j, :] = cos_sim.max(axis=1)

        return similarities

    def identify_stable_features(self, models: List, M: int, k: int) -> Dict:
        """
        Identify stable features for a given (M, k) combination.

        Args:
            models: List of trained replicas
            M: Expansion factor
            k: Sparsity level

        Returns:
            Dict with keys:
            - 'stable_indices': Indices of stable neurons
            - 'mean_similarities': Mean similarity per neuron
            - 'n_stable': Number of stable features
            - 'threshold': Threshold used
        """
        print(f"\n  Analyzing stability for (M={M}, k={k})...")
        print(f"  Number of replicas: {len(models)}")

        # Compute similarities
        similarities = self.compute_decoder_similarities(models)

        # Mean similarity per neuron (average over all replica pairs)
        # Shape: (n_replicas, n_replicas, M) -> (M,)
        # Exclude diagonal (self-similarity) from mean
        n_models = len(models)
        sum_sim = similarities.sum(axis=(0, 1))  # Sum over replica pairs
        mean_sim = (sum_sim - n_models) / (n_models * (n_models - 1))  # Exclude diagonal

        # Identify stable neurons
        stable_mask = mean_sim >= self.threshold
        stable_indices = np.where(stable_mask)[0]

        print(f"  Similarity statistics:")
        print(f"    Min: {mean_sim.min():.3f}")
        print(f"    Mean: {mean_sim.mean():.3f}")
        print(f"    Median: {np.median(mean_sim):.3f}")
        print(f"    Max: {mean_sim.max():.3f}")
        print(f"  Stable features (≥{self.threshold}): {len(stable_indices)}/{M} "
              f"({100*len(stable_indices)/M:.1f}%)")

        return {
            'stable_indices': stable_indices,
            'mean_similarities': mean_sim,
            'n_stable': len(stable_indices),
            'threshold': self.threshold
        }

    def analyze_ensemble(self, ensemble: Dict, save_path: str = None, device: str = "cpu") -> pd.DataFrame:
        """
        Analyze stability across entire ensemble.

        Args:
            ensemble: Dict mapping (M, k) -> List[checkpoint_paths]
            save_path: Optional path to save results CSV
            device: Device to load models on

        Returns:
            DataFrame with stability statistics per (M, k)
        """
        print("\n" + "="*80)
        print("STABILITY ANALYSIS")
        print("="*80)

        results = []

        for (M, k), checkpoint_paths in ensemble.items():
            # Load models from checkpoints
            models = []
            for ckpt_path in checkpoint_paths:
                ckpt = torch.load(ckpt_path, map_location=device)
                input_dim = ckpt['hyperparameters']['input_dim']
                model = KSparseAutoencoder(input_dim=input_dim, hidden_dim=M, k=k)
                model.load_state_dict(ckpt['model_state_dict'])
                model.to(device)
                models.append(model)

            stability_info = self.identify_stable_features(models, M, k)

            results.append({
                'M': M,
                'k': k,
                'n_replicas': len(models),
                'n_features_total': M,
                'n_stable': stability_info['n_stable'],
                'pct_stable': 100 * stability_info['n_stable'] / M,
                'similarity_mean': stability_info['mean_similarities'].mean(),
                'similarity_median': np.median(stability_info['mean_similarities']),
                'similarity_std': stability_info['mean_similarities'].std(),
                'threshold': stability_info['threshold']
            })

        results_df = pd.DataFrame(results)

        print("\n" + "="*80)
        print("STABILITY SUMMARY")
        print("="*80)
        print(results_df.to_string(index=False))

        total_stable = results_df['n_stable'].sum()
        print(f"\nTotal stable features across all (M,k): {total_stable}")
        print("="*80)

        if save_path:
            results_df.to_csv(save_path, index=False)
            print(f"\n[OK] Saved stability analysis to {save_path}")

        return results_df


# DEPRECATED: This function has memory issues with large datasets
# Use extract_stable_activations_chunked from utilities.chunked_activation_extractor instead
def extract_stable_activations(sent_df: pd.DataFrame,
                               embeddings: np.ndarray,
                               ensemble: Dict,
                               stability_analyzer: StabilityAnalyzer,
                               device: str = "cuda",
                               batch_size: int = 2048) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract activations for stable features from ensemble.

    Process:
    1. For each (M, k), identify stable neurons
    2. Compute activations on FULL dataset using first replica
    3. Extract only stable neuron activations
    4. Concatenate across all (M, k) pairs

    Args:
        sent_df: Sentence-level DataFrame
        embeddings: Full embedding matrix
        ensemble: Trained ensemble models
        stability_analyzer: StabilityAnalyzer instance
        device: "cuda" or "cpu"
        batch_size: Batch size for inference

    Returns:
        Tuple of:
        - sent_df_with_features: DataFrame with stable feature activations
        - feature_metadata: DataFrame with metadata per feature
    """
    print("\n" + "="*80)
    print("EXTRACTING STABLE FEATURE ACTIVATIONS")
    print("="*80)

    # Import get_activations (defined later in file)
    # We'll use it to compute activations for each model

    all_activations = []
    feature_metadata = []

    for (M, k), models in ensemble.items():
        print(f"\nProcessing (M={M}, k={k})...")

        # Identify stable features
        stability_info = stability_analyzer.identify_stable_features(models, M, k)
        stable_indices = stability_info['stable_indices']

        if len(stable_indices) == 0:
            print(f"  WARNING: No stable features for (M={M}, k={k}). Skipping.")
            continue

        # Use first replica for extraction
        # (All replicas should give similar activations for stable features)
        model = models[0]

        # Compute activations on FULL dataset
        print(f"  Computing activations for {len(stable_indices)} stable features...")

        # Inline get_activations to avoid forward reference
        dataset = EmbeddingDataset(embeddings)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        model.eval()
        all_z = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                _, z = model(batch)
                all_z.append(z.cpu().numpy())
        Z_full = np.vstack(all_z)  # Shape: (N, M)

        # Extract only stable neurons
        Z_stable = Z_full[:, stable_indices]  # Shape: (N, n_stable)

        print(f"  Extracted activations shape: {Z_stable.shape}")

        # Create feature names and metadata
        for i, neuron_idx in enumerate(stable_indices):
            feature_name = f"M{M}_k{k}_n{neuron_idx}"

            feature_metadata.append({
                'feature_name': feature_name,
                'M': M,
                'k': k,
                'neuron_idx': int(neuron_idx),
                'mean_similarity': float(stability_info['mean_similarities'][neuron_idx]),
                'global_feature_idx': len(feature_metadata)  # For ordering
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

    # Create DataFrame with activations
    # Column names: h_M{M}_k{k}_n{neuron_idx}
    activation_cols = {f"h_{meta['feature_name']}": Z_concat[:, i]
                       for i, meta in enumerate(feature_metadata)}

    activation_df = pd.DataFrame(activation_cols, index=sent_df.index)

    # Combine with sentence metadata
    sent_df_with_features = pd.concat([
        sent_df,
        activation_df
    ], axis=1)

    feature_metadata_df = pd.DataFrame(feature_metadata)

    return sent_df_with_features, feature_metadata_df


# =========================
# 4. Train SAE (Ensemble or Single)
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
input_dim = embeddings.shape[1]

if args.ensemble:
    # ============================================================
    # ENSEMBLE MODE: Multi-(M,k) SAE Training
    # ============================================================

    print("\n" + "="*80)
    print("ENSEMBLE MODE: Multi-(M,k) SAE Training")
    print("="*80)

    # Create configuration
    config = EnsembleConfig(
        expansion_factors=args.ensemble_M,
        sparsity_levels=args.ensemble_k,
        n_replicas=args.ensemble_replicas,
        stability_threshold=args.ensemble_threshold
    )

    # Filter grid if subset specified (for parallelization)
    if args.ensemble_subset:
        original_grid = config.get_grid()

        if args.ensemble_subset.startswith('M'):
            # Filter by expansion factor
            target_M = int(args.ensemble_subset[1:])
            config.expansion_factors = [target_M]
            print(f"\n[SUBSET MODE] Training only M={target_M}")

        elif args.ensemble_subset.startswith('k'):
            # Filter by sparsity level
            target_k = int(args.ensemble_subset[1:])
            config.sparsity_levels = [target_k]
            print(f"\n[SUBSET MODE] Training only k={target_k}")

        else:
            print(f"\nWARNING: Invalid subset specification '{args.ensemble_subset}'")
            print("Use format: 'M4096' or 'k32'")

        filtered_grid = config.get_grid()
        print(f"Grid filtered: {len(original_grid)} -> {len(filtered_grid)} combinations")

    # Create output directory
    output_dir = os.path.join(DATA_DIR, "sae_ensemble")
    os.makedirs(output_dir, exist_ok=True)

    # Train ensemble
    print("\nStarting ensemble training...")
    ensemble = train_ensemble(
        sent_df=sent_df,
        embeddings=embeddings,
        config=config,
        output_dir=output_dir,
        device=device,
        batch_size=1024,  # Can adjust based on GPU memory
        lr=1e-3,
        epochs=10
    )

    # Analyze stability
    print("\nAnalyzing ensemble stability...")
    analyzer = StabilityAnalyzer(threshold=config.stability_threshold)

    stability_df = analyzer.analyze_ensemble(
        ensemble,
        save_path=os.path.join(output_dir, "stability_analysis.csv"),
        device=device
    )

    # Check if we have enough stable features
    total_stable = stability_df['n_stable'].sum()

    if total_stable < 200:
        print("\n" + "!"*80)
        print(f"WARNING: Only {total_stable} stable features found!")
        print("Recommendations:")
        print("  1. Lower stability threshold (try --ensemble-threshold 0.7)")
        print("  2. Increase replicas (try --ensemble-replicas 7)")
        print("  3. Check if models are training properly (review loss curves)")
        print("!"*80)

    elif total_stable > 2000:
        print("\n" + "*"*80)
        print(f"NOTE: {total_stable} stable features found (very high)")
        print("Consider:")
        print("  1. Raising threshold to --ensemble-threshold 0.85 for more selectivity")
        print("  2. This is not necessarily a problem - Lasso will select relevant ones")
        print("*"*80)

    # Extract stable features using memory-efficient chunked processing
    print("\nExtracting stable features from ensemble...")
    sent_df_with_features, feature_metadata = extract_stable_activations_chunked(
        sent_df=sent_df,
        embeddings=embeddings,
        ensemble=ensemble,
        ensemble_dir=Path(output_dir),
        stability_analyzer=analyzer,
        threshold=args.ensemble_threshold,
        chunk_size=args.chunk_size,  # Memory-efficient processing
        batch_size=1024,
        device=device
    )

    # Save feature metadata
    feature_metadata.to_csv(
        os.path.join(output_dir, "feature_metadata.csv"),
        index=False
    )
    print(f"\n[OK] Saved feature metadata to {os.path.join(output_dir, 'feature_metadata.csv')}")

    # Update sent_df with ensemble features
    sent_df = sent_df_with_features

    # Get feature columns for aggregation (all starting with "h_M")
    stable_feature_cols = [c for c in sent_df.columns if c.startswith('h_M')]
    hidden_dim = len(stable_feature_cols)

    print(f"\n[OK] Ensemble mode complete. Proceeding with {hidden_dim} stable features.")

else:
    # ============================================================
    # SINGLE SAE MODE (original behavior)
    # ============================================================

    print("\n" + "="*80)
    print("SINGLE SAE MODE")
    print("="*80)

    sae_model = train_sae(embeddings, input_dim=input_dim, device=device)

    # =========================
    # 5. Compute neuron activations and aggregate to firm-year
    # =========================

    def get_activations(model, embeddings, batch_size=1024, device="cuda"):
        dataset = EmbeddingDataset(embeddings)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        model.eval()
        all_z = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                _, z = model(batch)
                all_z.append(z.cpu().numpy())
        return np.vstack(all_z)  # (N, hidden_dim)

    hidden_dim = sae_model.hidden_dim
    print(f"\nExtracting activations and adding to dataframe in chunks to avoid memory issues...")

    # Process activations in neuron chunks to avoid memory spike
    NEURON_CHUNK_SIZE = 128  # Add 128 neurons at a time
    n_neuron_chunks = (hidden_dim + NEURON_CHUNK_SIZE - 1) // NEURON_CHUNK_SIZE

    # Get all activations first (unavoidable, but we'll add to df in chunks)
    Z_all = get_activations(sae_model, embeddings, device=device)

    # Add neuron columns in chunks
    for chunk_idx in range(n_neuron_chunks):
        start_j = chunk_idx * NEURON_CHUNK_SIZE
        end_j = min((chunk_idx + 1) * NEURON_CHUNK_SIZE, hidden_dim)
        print(f"  Adding neurons {start_j}-{end_j-1} ({chunk_idx+1}/{n_neuron_chunks})...")

        # Add this chunk of neurons
        for j in range(start_j, end_j):
            sent_df[f"h_{j}"] = Z_all[:, j]

        # Free memory
        if chunk_idx % 4 == 3:  # Every 4 chunks
            import gc
            gc.collect()

    # Free the large activation matrix
    del Z_all
    import gc
    gc.collect()
    print(f"  [OK] Added {hidden_dim} neuron activation columns")

# ============================================================================
# SAVE SENTENCE-LEVEL DATA WITH SAE FEATURES
# ============================================================================

print("\n" + "="*80)
print("SAVING SENTENCE-LEVEL DATA WITH SAE FEATURES")
print("="*80)

# Save sentence-level data with SAE features
output_file = os.path.join(DATA_DIR, f"sent_df_with_sae.parquet")
print(f"\nSaving sentence-level data to {output_file}...")
print(f"  Shape: {sent_df.shape}")
print(f"  Columns: {len(sent_df.columns)}")

# Get feature column counts
sae_feature_cols = [c for c in sent_df.columns if c.startswith('h_')]
print(f"  SAE features: {len(sae_feature_cols)}")

sent_df.to_parquet(output_file, index=False)
print(f"[OK] Saved to {output_file}")

print("\n" + "="*80)
print("PHASE 3 COMPLETE: SAE TRAINING AND FEATURE EXTRACTION")
print("="*80)
print(f"\nOutputs:")
if args.ensemble:
    print(f"  1. SAE ensemble checkpoints: {output_dir}/")
    print(f"  2. Stability analysis: {output_dir}/stability_analysis.csv")
    print(f"  3. Feature metadata: {output_dir}/feature_metadata.csv")
print(f"  4. Sentence-level data with SAE features: {output_file}")
print(f"\nNext steps:")
print(f"  - Run 04_feature_selection.py for Lasso feature selection")
print(f"  - Run 05_interpret_features.py for LLM-based interpretation")
print(f"  - Run 06_sarkar_analysis.py for pricing function estimation")
print("="*80)
