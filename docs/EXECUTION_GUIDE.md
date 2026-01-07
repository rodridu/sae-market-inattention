# SAE 10-K Analysis - Execution Guide

This guide walks through the complete pipeline for the SAE-based hypothesis generation project on 10-K filings.

**Updated**: Now aligned with research proposal - uses paragraphs as fundamental unit, includes CLN novelty and KMNZ relevance.

## Prerequisites

### Required Python Packages

```bash
pip install pandas numpy torch tqdm scikit-learn statsmodels pyarrow
pip install sentence-transformers  # For embeddings
# Optional for full CLN/KMNZ implementation:
# pip install transformers accelerate
```

### Data Requirements

- **Input data location**: `C:\Users\ofs4963\Dropbox\Arojects\esg_dei\data\`
  - `10K_item1.zip` (Business description)
  - `10K_item1A.zip` (Risk factors)
  - `10K_item7.zip` (MD&A)
  - `filing_metadata_v2.csv` (Firm identifiers, filing dates, tickers)

- **Output data location**: `C:\Users\ofs4963\Dropbox\Arojects\SAE\data\` (created automatically)

## Pipeline Overview

The pipeline now has **5 phases** aligned with the research proposal:

1. **Data Preparation**: Extract items, parse into paragraphs, merge metadata
2. **Embeddings & Features**: Generate paragraph embeddings, compute baseline text features
3. **CLN Novelty**: Compute paragraph-level information measure (N_fdt)
4. **KMNZ Relevance**: Learn paragraph-level attention weights from returns (R_fdt)
5. **SAE Training & Analysis**: Train sparse autoencoder, feature selection, concept interpretation

---

## File Organization

### Core Pipeline (Root Directory)

Run these 8 scripts in order:

1. `01_data_preparation.py` - Extract and parse 10-K text
2. `02_embeddings_and_features.py` - Generate embeddings
3. `02b_novelty_cln.py` - Compute CLN novelty
4. `02c_relevance_kmnz.py` - Compute KMNZ relevance
5. `03_sae_training.py` - Train SAE ensemble
6. `04_extract_features.py` - Extract SAE features
7. `05_interpret_features.py` - Interpret features
8. `06_sarkar_analysis.py` - Pricing function analysis

### Supporting Directories

- `setup/` - One-time scripts (WRDS, metadata)
- `utilities/` - Validation and diagnostic tools
- `deprecated/` - Old versions (not used)

---

## Pipeline Execution

### Phase 1: Data Preparation (1-2 hours for full dataset)

Extract text from zip files, parse into paragraphs, merge with metadata.

**Updated**: Now uses paragraph-level parsing (not sentences) aligned with CLN/KMNZ.

**Test mode (small sample, ~5-10 minutes):**
```bash
python 01_data_preparation.py --test
```

**Full dataset:**
```bash
python 01_data_preparation.py
```

**Output files:**
- `data/documents_consolidated.parquet` - Document-level metadata
- `data/paragraphs.parquet` - **Paragraph-level dataset** (was sentences.parquet)

**Schema (aligned with proposal notation):**
- `accession_number` → document ID (d)
- `cik` → firm ID (f)
- `filing_date` → time (t)
- `paragraph_id` → paragraph index (p)
- `text` → paragraph text

**Expected stats (full dataset):**
- ~192,000 documents across all three items
- ~25 years of data (2000-2025)
- ~1-2 million paragraphs (fewer than sentences, more meaningful units)

---

### Phase 2: Stratified Sampling + Embeddings & Features (2-4 hours depending on GPU)

Performs stratified sampling, generates sentence embeddings, and computes text statistics.

**Updated**: Now includes three-stage stratified sampling before embedding generation.

#### Three-Stage Stratified Sampling

The pipeline now uses a more defensible sampling strategy to ensure balanced representation:

**Stage 1a: Firm-Year-Item Balance (Filing-level)**
- Max 500 sentences per (firm, year, item) — i.e., per 10-K filing
- Ensures each filing receives equal treatment
- Prevents under-sampling of firms with long filing histories

**Stage 1b: Firm-Level Cap**
- Max 5,000 sentences per firm (across all years)
- Prevents any single firm from dominating the dataset
- Ensures firm-level diversity

**Stage 2: Year×Item Balance**
- Max 30,000 sentences per (year, item) cell
- Controls overall dataset size
- Maintains temporal and item-type coverage

**Why this approach?**
- The unit of analysis is the filing/document, not the firm
- Each 10-K is a separate disclosure event with its own novelty and relevance
- A firm's 2024 filing shouldn't be under-sampled just because they also filed in 2005-2023
- More defensible for research: "We sample up to 500 sentences from each filing, subject to firm and year-level caps"

#### Running Phase 2

**Option 1: Automatic (recommended)**
```bash
python 02_embeddings_and_features.py
```
This will automatically run stratified sampling if needed, then generate embeddings.

**Option 2: Manual (for custom parameters)**
```bash
# First run sampling with custom parameters
python run_stratified_sampling.py --sentences-per-filing 500 --sentences-per-firm 5000 --max-per-year-item 30000

# Then run embeddings
python 02_embeddings_and_features.py
```

**Output files:**
- `data/sentences_sampled.parquet` - Stratified sample (~2-3M sentences)
- `data/sentence_embeddings_index.csv` - Index file for chunked embeddings
- `data/sentence_embeddings_chunks/` - Chunked embedding files (.npz)
- `data/baseline_features.parquet` - Text features (subset for efficiency)

**Notes:**
- Requires `sentence-transformers` package
- Uses GPU if available (recommended for speed)
- Default model: `all-MiniLM-L6-v2` (384-dim, fast)
- Alternative: `all-mpnet-base-v2` (768-dim, higher quality)
- Embeddings are generated in chunks to avoid memory issues

---

### Phase 2b: CLN Novelty Computation (NEW)

Compute paragraph-level information measure using LLM priors.

**Proposal alignment (Section 4.2)**: N_fdt = surprisal under firm-specific prior

```bash
# Simplified proxy (fast, uses text heuristics)
python 02b_novelty_cln.py --method proxy

# Full CLN (slow, requires LLM fine-tuning per firm)
python 02b_novelty_cln.py --method cln
```

**Output file:**
- `data/novelty_cln.parquet` - Paragraph-level novelty scores

**Implementation status:**
- **Proxy method**: ✓ Implemented (uses text statistics, position, vocab diversity)
- **Full CLN method**: Framework only (requires firm-specific LLM priors)

**For production research:**
1. Collect historical EDGAR corpus per firm
2. Fine-tune GPT-2 or similar on firm's past filings
3. Compute token-level surprisal for new paragraphs
4. Aggregate to paragraph-level novelty

---

### Phase 2c: KMNZ Relevance Computation (NEW)

Learn paragraph-level attention weights from announcement returns.

**Proposal alignment (Section 4.3)**: R_fdt = attention weight from return-supervised transformer

```bash
# Simplified proxy (fast, uses position + content heuristics)
python 02c_relevance_kmnz.py --method proxy

# Full KMNZ (slow, requires outcome data and transformer training)
python 02c_relevance_kmnz.py --method kmnz
```

**Output file:**
- `data/relevance_kmnz.parquet` - Paragraph-level relevance scores

**Implementation status:**
- **Proxy method**: ✓ Implemented (uses position, financial keywords, item type)
- **Full KMNZ method**: Framework only (requires announcement returns)

**For production research:**
1. Collect announcement returns (CRSP CAR[-1,+1])
2. Collect EPS surprises (I/B/E/S)
3. Train attention-based transformer to predict returns
4. Extract attention weights as paragraph-level relevance

---

### Phase 3: SAE Training & Analysis (1-2 hours)

Train k-sparse autoencoder on paragraph embeddings and perform concept-level analysis.

**Updated**: Now incorporates CLN and KMNZ in baseline regressions.

```bash
python 03_sae_training.py
```

**What this does:**
1. Loads paragraph embeddings from Phase 2
2. Merges CLN novelty (Phase 2b) and KMNZ relevance (Phase 2c)
3. Trains k-sparse autoencoder (default: k=32, hidden_dim=1024)
4. Computes neuron activations z_fdt for all paragraphs
5. Aggregates to document level: A_k, N_mean, R_mean
6. **Runs Lasso controlling for CLN info and KMNZ relevance** (key update)
7. Identifies predictive concepts beyond baseline
8. Samples paragraphs for neuron interpretation

**Proposal equation implemented (Section 4.6):**
```
y = α + β₁·Info^CLN + β₂·Relevance^KMNZ + Σ_k δₖ·A_k + Γ'X + ε
```

**Key parameters to adjust** (in `03_sae_training.py`):
- `k`: Sparsity level (default 32)
- `hidden_dim`: Number of neurons (default 1024)
- `batch_size`: Training batch size
- `epochs`: Training epochs (default 10)
- `lr`: Learning rate (default 1e-3)

---

## Quick Start (Test Mode)

To quickly validate the pipeline on a small sample:

```bash
# Phase 1: Extract 1000 files per item (~5 min)
python 01_data_preparation.py --test

# Phase 2: Stratified sampling + embeddings (~2-3 min)
# Note: Sampling runs automatically if sentences_sampled.parquet doesn't exist
python 02_embeddings_and_features.py

# Phase 2b: Compute novelty proxy (~1 min)
python 02b_novelty_cln.py --method proxy

# Phase 2c: Compute relevance proxy (~1 min)
python 02c_relevance_kmnz.py --method proxy

# Phase 3: Train SAE with CLN/KMNZ (~5 min)
python 03_sae_training.py
```

Total test time: ~15-20 minutes

---

## Full Pipeline (Production)

For the complete dataset:

```bash
# Phase 1: Extract all files and parse sentences (~2 hours)
python 01_data_preparation.py

# Phase 2: Stratified sampling + embeddings (~3-4 hours with GPU)
# Note: Three-stage sampling runs automatically before embeddings
python 02_embeddings_and_features.py

# Alternative: Run sampling separately with custom parameters
# python run_stratified_sampling.py --sentences-per-filing 500 --sentences-per-firm 5000
# python 02_embeddings_and_features.py

# Phase 2b: Compute CLN novelty
# Option A: Fast proxy method (~10 min)
python 02b_novelty_cln.py --method proxy

# Option B: Full CLN (requires substantial compute, ~days)
# python 02b_novelty_cln.py --method cln

# Phase 2c: Compute KMNZ relevance
# Option A: Fast proxy method (~10 min)
python 02c_relevance_kmnz.py --method proxy

# Option B: Full KMNZ (requires outcome data, ~hours)
# python 02c_relevance_kmnz.py --method kmnz

# Phase 3: Train SAE with full controls (~1 hour)
python 03_sae_training.py
```

Total production time (with proxies): 6-8 hours
Total production time (full CLN/KMNZ): Several days

---

## Data Schema

### Sentence-level data (`sentences.parquet` - full dataset)
| Column | Type | Description | Proposal Notation |
|--------|------|-------------|-------------------|
| accession_number | string | SEC filing identifier | d |
| cik | int64 | Firm CIK | f |
| ticker | string | Stock ticker | - |
| year | int | Fiscal year | - |
| item_type | string | Item 1, 1A, or 7 | - |
| sentence_id | int | Sentence index within document | s |
| text | string | Sentence text | - |
| sentence_length | int | Character count | - |

### Sampled sentences (`sentences_sampled.parquet` - stratified sample)
Same schema as `sentences.parquet`, but with ~2-3M sentences after three-stage stratified sampling:
- Stage 1a: ≤500 sentences per (firm, year, item)
- Stage 1b: ≤5,000 sentences per firm
- Stage 2: ≤30,000 sentences per (year, item)

### Novelty data (`novelty_cln.parquet`)
| Column | Type | Description | Proposal Notation |
|--------|------|-------------|-------------------|
| accession_number | string | Document ID | d |
| paragraph_id | int | Paragraph index | p |
| novelty_cln | float | Paragraph-level information | N_fdt |

### Relevance data (`relevance_kmnz.parquet`)
| Column | Type | Description | Proposal Notation |
|--------|------|-------------|-------------------|
| accession_number | string | Document ID | d |
| paragraph_id | int | Paragraph index | p |
| relevance_kmnz | float | Paragraph-level attention weight | R_fdt |

### Document-level features (from `03_sae_training.py`)
| Column | Type | Description |
|--------|------|-------------|
| accession_number | string | Document ID |
| year | int | Fiscal year |
| item_type | string | Item type |
| mean_j | float | Mean activation of neuron j |
| freq_j | float | Frequency of high activation for neuron j |
| novelty_cln_mean | float | Document-level novelty (mean) |
| novelty_cln_max | float | Document-level novelty (max) |
| relevance_kmnz_mean | float | Document-level relevance |
| novelty_cln_weighted | float | Relevance-weighted novelty |

### Embeddings (`paragraph_embeddings.npz`)
- `embeddings`: numpy array (N × 384 or N × 768)
- `accession_numbers`: Document identifiers
- `paragraph_ids`: Paragraph indices
- `item_types`: Item types

---

## Implementation Status

### ✓ Fully Implemented
- [x] Paragraph-level parsing (was sentence-level)
- [x] Metadata merge (CIK, ticker, filing_date)
- [x] Paragraph embeddings
- [x] k-sparse autoencoder training
- [x] Document-level feature aggregation
- [x] Lasso with CLN/KMNZ controls
- [x] CLN novelty proxy
- [x] KMNZ relevance proxy

### ⚠️ Framework/Placeholder
- [ ] Full CLN novelty (firm-specific LLM priors)
- [ ] Full KMNZ relevance (attention-based transformer)
- [ ] LLM-based neuron interpretation
- [ ] Description fidelity validation
- [ ] Real outcome data (returns, drift, analyst forecasts)
- [ ] Concept-level N_k and R_k analysis (Section 4.5)
- [ ] Main empirical tests (Section 5.1-5.4)

---

## Next Steps for Production Research

### 1. Implement Full CLN Novelty
- Collect historical EDGAR corpus for each firm
- Fine-tune LLM (GPT-2, LLaMA) on firm-specific text
- Compute token-level surprisal for new paragraphs
- Replace proxy with true information measure

### 2. Implement Full KMNZ Relevance
- Collect announcement return data (CRSP)
- Collect analyst forecast data (I/B/E/S)
- Implement attention-based transformer architecture
- Train model to predict returns and EPS
- Extract attention weights as relevance

### 3. Real Outcome Data
- Event-day returns (CAR[-1,+1])
- Post-filing drift (returns over 30/60/90 days)
- Analyst forecast revisions
- Trading volume, volatility
- Future earnings surprises

### 4. Concept-Level Analysis (Proposal Section 4.5)
- Compute N_k = E[N_fdt | z_k > 0] (concept-level novelty)
- Compute R_k = E[R_fdt | z_k > 0] (concept-level relevance)
- Classify concepts by (N_k, R_k) quadrants
- Study distribution across items and forms

### 5. LLM Interpretation & Fidelity
- Implement OpenAI/Anthropic API calls
- Sample high/low activation paragraphs
- Generate natural-language descriptions
- Validate on held-out data (precision, recall, F1)
- Only use high-fidelity concepts (F1 > 0.7)

### 6. Main Empirical Tests (Proposal Section 5)
- **5.1**: Under-reaction to high-N, low-R concepts
- **5.2**: Positioning/obfuscation by firm characteristics
- **5.3**: Limited attention priors (10-K only vs full EDGAR)
- **5.4**: Regulatory shocks (Reg S-K, new 8-K items)

---

## Troubleshooting

**"novelty_cln.parquet not found":**
- Run `python 02b_novelty_cln.py --method proxy` first
- Or SAE script will use placeholder (novelty = 0)

**"relevance_kmnz.parquet not found":**
- Run `python 02c_relevance_kmnz.py --method proxy` first
- Or SAE script will use placeholder (relevance = 0)

**Out of memory errors:**
- Reduce `batch_size` in embedding generation
- Process paragraphs in smaller chunks
- Use smaller embedding model
- Reduce SAE `hidden_dim`

**Slow embedding generation:**
- Use GPU: Install `torch` with CUDA support
- Reduce dataset size for testing
- Use faster model (MiniLM vs MPNet)

**SAE training issues:**
- Reduce `hidden_dim` if memory constrained
- Adjust sparsity `k` (higher = less sparse)
- Check embedding normalization
- Ensure CLN/KMNZ are not NaN

**No SAE features selected:**
- Check that outcome data is not all placeholders
- Try running with real returns data
- Increase SAE `hidden_dim` for more concepts
- Check that CLN/KMNZ don't perfectly predict outcome

---

## Hardware Recommendations

**Minimum (for testing):**
- 16 GB RAM
- CPU only
- ~500 GB storage
- Time: ~15 min test mode

**Recommended (for production with proxies):**
- 32 GB RAM
- NVIDIA GPU with 8+ GB VRAM (RTX 3070, 3080)
- ~1 TB storage
- Time: 6-7 hours

**Optimal (for full CLN/KMNZ):**
- 64 GB RAM
- NVIDIA GPU with 16+ GB VRAM (A100, H100, 4090)
- ~2 TB SSD storage
- Time: Several days for firm-specific LLM training

---

## Alignment with Research Proposal

This pipeline now implements the core structure from the proposal:

| Proposal Section | Implementation | Status |
|-----------------|----------------|--------|
| 4.1 Units (f, d, t, p) | Data schema in 01_data_preparation.py | ✓ Complete |
| 4.2 CLN Novelty N_fdt | 02b_novelty_cln.py | ⚠️ Proxy only |
| 4.3 KMNZ Relevance R_fdt | 02c_relevance_kmnz.py | ⚠️ Proxy only |
| 4.4 SAE Training | 03_sae_training.py | ✓ Complete |
| 4.5 Concept-level N_k, R_k | aggregate_document_features() | ⚠️ Partial |
| 4.6 Feature selection | Lasso with CLN/KMNZ controls | ✓ Complete |
| 4.6 LLM interpretation | Stub functions | ⚠️ Framework only |
| 5.1-5.4 Main tests | Not implemented | 🔴 TODO |

**Key improvement over original**: The pipeline now correctly controls for CLN information and KMNZ relevance when selecting SAE concepts, ensuring that selected neurons capture patterns beyond simple novelty or attention.
