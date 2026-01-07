# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project on **sparse autoencoders (SAEs) for financial disclosure analysis**. The core idea is to discover novel and relevant concepts in 10-K filings by combining three methodologies:

1. **Novelty (N)**: CLN's information measure using LLM priors over EDGAR text
2. **Relevance (R)**: KMNZ's attention-based sentence importance learned from market returns
3. **Concepts (Ck)**: Interpretable SAE neurons capturing recurring textual motifs

The research question: When firms release new narrative information, which types do investors actually process, and which types are persistently under-processed?

## Key Components

### Research Documents

**`main.tex`**
Current research proposal (LaTeX format) outlining:
- Research question: Which novel semantic directions in 10-K text load weakly on the pricing function (i.e., lie orthogonal to the directions investors actually value)?
- Conceptual framework based on rational inattention theory and Sarkar's representation framework
- Methodology: Time-indexed embeddings, multi-(M,k) sparse autoencoder ensembles, pricing function estimation via split-sample ridge, orthogonal decomposition
- Main prediction: Novel concepts orthogonal to pricing function should show weaker announcement reactions but stronger post-filing drift
- Critical discussion of SAE stability via bootstrap replicas and cross-(M,k) validation
- **Key framework alignment**: Separates representation (Z_ft) from pricing weights (w_t), defines "priced new representation" as w_t · ΔZ_ft

**`papers/`**
Reference papers:
- `BFI_WP_2024-155.pdf`: KMNZ paper on relevance learning
- `paper5.pdf`: Likely CLN or related work
- `Peng et al. - 2025 - Use Sparse Autoencoders...pdf`: Methodological guidance on when to use SAEs

### Implementation Scripts

**Updated Pipeline (5 Phases)**

The implementation now follows the proposal structure with 5 phases instead of 3:

**`01_data_preparation.py`** (Phase 1)
Data extraction and preprocessing:
- Extracts text from zip files (Item 1, 1A, 7)
- Loads and validates filing metadata (CIK, ticker, filing_date)
- Performs data quality checks (empty files, duplicates, coverage)
- **Parses documents into sentences** using sentence boundary detection
- Merges metadata for firm-level analysis
- Outputs: `data/documents_consolidated.parquet`, `data/sentences.parquet`
- Run with: `python 01_data_preparation.py [--test]`

**`02_embeddings_and_features.py`** (Phase 2)
Embedding generation and baseline features:
- **Generates sentence embeddings** using sentence-transformers
- Computes readability metrics (Flesch, FOG index)
- Computes sentiment features (simple Loughran-McDonald)
- Aggregates features to document level
- Outputs: `data/sentence_embeddings.npz`, `data/baseline_features.parquet`
- Run with: `python 02_embeddings_and_features.py`

**`02b_novelty_cln.py`** (Phase 2b - NEW)
CLN novelty computation (N_fdt):
- Computes sentence-level information measure
- Two modes:
  - `--method proxy`: Fast heuristic using text statistics (default)
  - `--method cln`: Full CLN with firm-specific LLM priors (framework only)
- Outputs: `data/novelty_cln.parquet`
- Run with: `python 02b_novelty_cln.py --method proxy`

**`02c_relevance_kmnz.py`** (Phase 2c - NEW)
KMNZ relevance computation (R_fdt):
- Learns sentence-level attention weights from market reactions
- Two modes:
  - `--method proxy`: Fast heuristic using position + keywords (default)
  - `--method kmnz`: Full attention-based transformer (framework only)
- Outputs: `data/relevance_kmnz.parquet`
- Run with: `python 02c_relevance_kmnz.py --method proxy`

**`03_sae_training.py`** (Phase 3)
SAE training and analysis:
- Loads sentence embeddings from Phase 2
- **Merges CLN novelty and KMNZ relevance** (updated)
- Trains k-sparse autoencoder with top-k sparsity
- Extracts neuron activations and aggregates to document level
- **Runs Lasso feature selection controlling for CLN info and KMNZ relevance** (key update)
- **Estimates pricing function w_t via 5-fold split-sample ridge** (Sarkar framework)
- **Computes orthogonal decomposition: Z^on vs Z^⊥**
- **Tests rational inattention predictions with ΔZ^on and ΔZ^⊥**
- Identifies predictive concepts beyond baseline
- Provides neuron interpretation framework
- Run with: `python 03_sae_training.py`

**`README_EXECUTION.md`**
Complete execution guide with:
- Installation instructions
- 5-phase pipeline walkthrough (updated)
- Data schemas for sentences, novelty, relevance
- Troubleshooting tips
- Hardware recommendations
- Implementation status checklist

### File Organization

**Core Pipeline (Root Directory)**

These 8 scripts form the complete pipeline (run in order):

1. `01_data_preparation.py` - Phase 1: Data extraction
2. `02_embeddings_and_features.py` - Phase 2: Embeddings + baseline features
3. `02b_novelty_cln.py` - Phase 2b: CLN novelty
4. `02c_relevance_kmnz.py` - Phase 2c: KMNZ relevance
5. `03_sae_training.py` - Phase 3: SAE ensemble training (CURRENT - Dec 16)
6. `04_extract_features.py` - Phase 4: Feature extraction
7. `05_interpret_features.py` - Phase 5: Feature interpretation
8. `06_sarkar_analysis.py` - Phase 6: Sarkar pricing function

**Supporting Directories**

`setup/` - One-time initialization scripts (7 files):
- `00_fetch_sec_metadata.py` - Fetch company metadata from SEC
- `00c_fetch_outcomes_wrds.py` - Fetch outcome variables from WRDS
- `setup_wrds_env.py` - Configure WRDS credentials
- (+ 4 more setup scripts)

`utilities/` - Helper scripts for validation and diagnostics (9 files):
- `validate_pipeline.py` - Comprehensive pipeline status check
- `monitor_training_loss.py` - Real-time loss monitoring
- `verify_alignment.py` - Check embedding-sentence alignment
- `stratified_sampling.py` - Stratified sampling utilities
- (+ 5 more utility scripts)

`deprecated/` - Older versions and alternatives (3 files):
- `03_sae_training_test.py` - Test version with small SAE
- `03b_sae_anthropic*.py` - Alternative architectures
- **Not used in current pipeline - use 03_sae_training.py instead**

**REMOVED:** Span analysis infrastructure (4 files deleted):
- ~~`01b_construct_spans.py`~~ - Obsolete
- ~~`02_embeddings_spans.py`~~ - Obsolete
- ~~`02b_novelty_cln_spans.py`~~ - Obsolete
- ~~`02c_relevance_kmnz_spans.py`~~ - Obsolete

Span analysis was attempted but abandoned. The main pipeline uses sentence-level analysis only.

## Code Architecture

The implementation follows the proposal's empirical design:

1. **Data layer** (`sent_df`): Sentence-level DataFrame with columns:
   - `accession_number` (document d)
   - `cik` (firm f)
   - `filing_date` (time t)
   - `sentence_id` (sentence s)
   - `text`

2. **Embedding layer**: Dense representations x_fdt (384-dim default from sentence-transformers)

3. **Novelty layer**: CLN information measure N_fdt per sentence
   - Proxy: text statistics, vocab diversity, specificity
   - Full: token-level surprisal from firm-specific LLM

4. **Relevance layer**: KMNZ attention weights R_fdt per sentence
   - Proxy: position, financial keywords, item type
   - Full: learned from return-supervised transformer

5. **SAE model**: `KSparseAutoencoder` class with encoder/decoder and top-k sparsity
   - Unsupervised training on sentence embeddings
   - Neuron activations z_fdt with k-sparse constraint

6. **Aggregation layer**: Sentence-level → document-level features
   - SAE: mean_j and freq_j (mean activation and high-activation frequency)
   - Novelty: novelty_cln_mean, novelty_cln_max, novelty_cln_weighted
   - Relevance: relevance_kmnz_mean
   - Representation: Z_ft ∈ R^K (document-level concept activations)

7. **Pricing function layer** (Sarkar framework):
   - Estimate pricing weights w_t via split-sample ridge (5-fold)
   - Outcome: announcement returns (CAR[-1,+1])
   - Forward-in-time: use only data up to t-1 for year-t analysis
   - Orthogonal decomposition: Z^on = projection onto w_t, Z^⊥ = orthogonal component
   - Priced new representation: w_t · ΔZ_ft (scalar measure of priced content change)

8. **Analysis layer**:
   - Baseline regression: y = α + β₁·N + β₂·R + Γ'X + ε
   - Full model: y = α + β₁·N + β₂·R + Σδₖ·Aₖ + Γ'X + ε
   - Lasso feature selection controls for CLN and KMNZ
   - OLS with firm-clustered standard errors
   - Rational inattention tests: r = α + β_on·ΔZ^on + β_⊥·ΔZ^⊥ + ΓX + ε

9. **Interpretation layer**: LLM-based neuron description and fidelity validation (framework only)

Key design choices:
- **Sentence-level parsing**: Uses sentence boundary detection to split documents into atomic units
- **k-sparse enforcement**: Hard top-k masking in forward pass (not learned sparsity)
- **Aggregation strategy**: Both mean activation and high-activation frequency per neuron
- **Feature selection**: Controls for CLN novelty and KMNZ relevance (proposal Section 4.6)
- **Interpretation protocol**: Sample high/low quantile sentences, use LLM for pattern description, validate with held-out data

## Running the Pipeline

The project now has a complete 5-phase pipeline:

### Quick Test (~15 minutes)
```bash
python 01_data_preparation.py --test         # Process 1000 files per item
python 02_embeddings_and_features.py         # Generate embeddings
python 02b_novelty_cln.py --method proxy     # Compute novelty
python 02c_relevance_kmnz.py --method proxy  # Compute relevance
python 03_sae_training.py                    # Train SAE with CLN/KMNZ
```

### Full Production (~6-7 hours with proxies)
```bash
python 01_data_preparation.py                # Process all ~192K filings
python 02_embeddings_and_features.py         # Generate all embeddings
python 02b_novelty_cln.py --method proxy     # Compute novelty
python 02c_relevance_kmnz.py --method proxy  # Compute relevance
python 03_sae_training.py                    # Train SAE on full data
```

### Data Locations
- **Input**: `C:\Users\ofs4963\Dropbox\Arojects\esg_dei\data\`
  - `10K_item1.zip`, `10K_item1A.zip`, `10K_item7.zip`
  - `filing_metadata_v2.csv`
- **Output**: `C:\Users\ofs4963\Dropbox\Arojects\SAE\data\` (auto-created)

See `README_EXECUTION.md` for complete details.

## Research Context

**Why SAEs for this problem?**
Per Peng et al. (2025), SAEs should be used to **discover unknown concepts**, not to act on known ones. This project uses SAEs for hypothesis generation:
- Unsupervised training on sentence embeddings (no outcome access)
- Feature selection identifies neurons with incremental explanatory power **beyond CLN information and KMNZ relevance**
- LLM-based description and validation following HypotheSAEs framework
- Economic analysis via standard panel regressions

SAEs are not the final decision rule—they enumerate candidate concepts that are then interpreted and tested using CLN novelty, KMNZ relevance, and standard econometric methods.

**Main hypotheses to test:**
1. Markets under-react to specific high-novelty, low-relevance concepts (high N, low R)
2. Firms strategically position new information in low-attention areas (obfuscation)
3. Limited-attention priors (ignoring certain forms/exhibits) cause systematic misperception
4. Regulatory shocks that push novelty into high-relevance regions improve efficiency

## Dependencies

Core packages:
- `pandas`, `numpy`
- `tqdm`
- `torch` (PyTorch)
- `sklearn` (LassoCV)
- `statsmodels` (OLS with clustered SEs)
- `sentence_transformers` (for sentence embeddings)
- `pyarrow` (for parquet files)

Optional (for full CLN/KMNZ):
- `transformers`, `accelerate` (for LLM fine-tuning)
- Access to LLM API for interpretation (OpenAI, Anthropic)

## File Naming Conventions

- Research proposals: `main.tex` (LaTeX format)
- Pipeline scripts: `01_*.py`, `02_*.py`, `02b_*.py`, `02c_*.py`, `03_*.py` (numbered by execution order)
- Documentation: `CLAUDE.md` (this file), `README_EXECUTION.md` (user guide)
- References: `papers/*.pdf`
- Data outputs: `data/*.parquet`, `data/*.npz`

## Important Notes for Development

1. **Data Schema**: The project uses `accession_number` (unique filing ID) as the primary key for documents. Firm-level identifier is `cik`. Each accession_number corresponds to a specific 10-K filing.

2. **Item Types**: Three items are analyzed separately:
   - `item1`: Business description
   - `item1A`: Risk factors
   - `item7`: MD&A (Management Discussion and Analysis)

3. **Sentence vs Document Level**:
   - **Sentences are the atomic unit** for embeddings, novelty, relevance, and SAE training
   - Features are aggregated to document level for regressions
   - Each document = one item from one filing
   - Notation: f (firm), d (document), t (filing date), s (sentence)

4. **Implementation Status**:

   **✓ Fully Implemented:**
   - Sentence-level parsing and embeddings
   - Metadata merge (CIK, ticker, filing_date)
   - k-sparse autoencoder training
   - Document-level feature aggregation
   - Lasso with CLN/KMNZ controls (proposal Section 4.6)
   - CLN novelty proxy
   - KMNZ relevance proxy

   **⚠️ Framework/Placeholder:**
   - Full CLN novelty (requires firm-specific LLM priors, substantial compute)
   - Full KMNZ relevance (requires announcement returns, I/B/E/S data)
   - LLM-based neuron interpretation (stub functions)
   - Description fidelity validation (stub functions)
   - Real outcome data (returns, analyst forecasts) - currently random placeholders
   - Concept-level N_k and R_k analysis (proposal Section 4.5) - partially implemented
   - Main empirical tests (proposal Section 5.1-5.4) - not implemented

5. **Performance**:
   - Full dataset has ~192K documents → millions of sentences (varies by document length)
   - Embedding generation is the bottleneck (~3-5 hours with GPU depending on sentence count)
   - CLN/KMNZ proxy methods are fast (~10-15 min each)
   - Full CLN/KMNZ would require days of compute
   - Always test with `--test` flag first before running full pipeline

6. **Key Alignment with Proposal**:
   - Sentence-level parsing (Section 4.1) - implemented with sentence boundary detection
   - CLN novelty measure N_fdt (Section 4.2) - proxy implemented
   - KMNZ relevance measure R_fdt (Section 4.3) - proxy implemented
   - k-sparse SAE on embeddings (Section 4.4) - complete
   - Document-level aggregation (Section 4.5) - complete
   - Feature selection controlling for CLN/KMNZ (Section 4.6) - **critical update**
   - Baseline regression: y = α + β₁·Info^CLN + β₂·Rel^KMNZ + Σδₖ·Aₖ + Γ'X + ε

## Next Steps for Production

1. **Collect Real Outcome Data**:
   - Announcement returns (CRSP CAR[-1,+1])
   - Post-filing drift (30/60/90-day returns)
   - Analyst forecast revisions (I/B/E/S)
   - Trading volume, volatility

2. **Implement Full CLN Novelty**:
   - Collect historical EDGAR corpus per firm
   - Fine-tune GPT-2/LLaMA on firm-specific text
   - Compute token-level surprisal
   - Aggregate to sentence-level N_fdt

3. **Implement Full KMNZ Relevance**:
   - Collect announcement return data
   - Implement attention-based transformer
   - Train to predict returns and EPS
   - Extract attention weights as R_fdt

4. **Implement LLM Interpretation**:
   - Set up OpenAI/Anthropic API
   - Sample high/low activation sentences
   - Generate natural-language descriptions
   - Validate fidelity on held-out data

5. **Run Main Empirical Tests** (Proposal Section 5):
   - 5.1: Under-reaction to high-N, low-R concepts
   - 5.2: Positioning/obfuscation analysis
   - 5.3: Limited attention priors
   - 5.4: Regulatory shocks

## Critical Improvement Over Original Implementation

The pipeline now correctly implements the proposal's key insight:
> **SAE concepts are selected AFTER controlling for CLN information and KMNZ relevance**

This ensures that selected neurons capture semantic patterns beyond simple novelty or market attention, addressing potential concerns about whether SAEs add value over baseline measures.

Original approach: Lasso on [controls, SAE features]
**Updated approach**: Lasso on [controls, **CLN novelty, KMNZ relevance**, SAE features]

This aligns with the proposal's equation in Section 4.6 and is critical for claiming that SAE-discovered concepts have incremental explanatory power.
