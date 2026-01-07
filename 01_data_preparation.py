"""
Phase 1: Data Preparation & Exploration
Extracts 10-K items, performs quality checks, and creates sentence-level dataset.
Aligned with proposal: uses sentences as the fundamental unit for CLN/KMNZ analysis.
"""

import os
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import re
from collections import defaultdict

# ============================================
# Configuration
# ============================================

DATA_DIR = Path(r"C:\Users\ofs4963\Dropbox\Arojects\esg_dei\data")
OUTPUT_DIR = Path(r"C:\Users\ofs4963\Dropbox\Arojects\SAE\data")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

ZIP_FILES = {
    "item1": DATA_DIR / "10K_item1.zip",
    "item1A": DATA_DIR / "10K_item1A.zip",
    "item7": DATA_DIR / "10K_item7.zip",
}

# Metadata file location (found in data_backup_full)
METADATA_FILE = Path(r"C:\Users\ofs4963\Dropbox\Arojects\esg_dei\data_backup_full\filing_metadata_clean.csv")

# ============================================
# 1. Load Metadata
# ============================================

def convert_numeric_accession_to_sec_format(numeric_acc, cik):
    """
    Convert numeric accession number to SEC format with hyphens.

    Numeric format: CCCCCCCYYZZZZZ (CIK + Year + Sequence, no hyphens)
    SEC format: XXXXXXXXXX-YY-ZZZZZZ (10-digit CIK + Year + Sequence, with hyphens)

    Args:
        numeric_acc: Numeric accession (e.g., 109087224000040)
        cik: CIK value (e.g., 1090872)

    Returns:
        SEC format accession (e.g., '0001090872-24-000040')
    """
    if pd.isna(numeric_acc) or pd.isna(cik) or numeric_acc == 0:
        return None

    # Convert to string and remove decimal point
    acc_str = str(int(numeric_acc))
    cik_str = str(int(cik))

    # The numeric format is: CIK (variable length) + YY (2 digits) + ZZZZZZ (6 digits)
    # Last 8 digits are YYZZZZZZ
    if len(acc_str) >= 8:
        year_seq = acc_str[-8:]  # Last 8 digits: YYZZZZZZ
        year = year_seq[:2]      # First 2: YY
        sequence = year_seq[2:]  # Last 6: ZZZZZZ

        # Pad CIK to 10 digits
        cik_padded = cik_str.zfill(10)

        # Return SEC format
        return f"{cik_padded}-{year}-{sequence}"

    return None

def load_metadata():
    """Load and validate filing metadata."""
    print("Loading metadata...")
    metadata = pd.read_csv(METADATA_FILE)

    print(f"\nMetadata shape: {metadata.shape}")
    print(f"Columns: {metadata.columns.tolist()}")
    print(f"\nYear range: {metadata['fiscal_year'].min():.0f} - {metadata['fiscal_year'].max():.0f}")
    print(f"Number of unique firms (CIK): {metadata['cik'].nunique()}")
    print(f"Number of unique tickers: {metadata['ticker'].nunique()}")

    # Convert numeric accession_number to SEC format with hyphens
    print("\nConverting accession numbers to SEC format...")
    metadata['accession_number_sec'] = metadata.apply(
        lambda row: convert_numeric_accession_to_sec_format(row['accession_number'], row['cik']),
        axis=1
    )

    # Replace accession_number with the SEC format
    metadata['accession_number'] = metadata['accession_number_sec']
    metadata = metadata.drop('accession_number_sec', axis=1)

    # Report conversion success
    valid_acc = metadata['accession_number'].notna().sum()
    print(f"  Successfully converted {valid_acc:,} / {len(metadata):,} accession numbers ({valid_acc/len(metadata)*100:.1f}%)")

    # Show sample
    print("\nSample accession numbers after conversion:")
    print(metadata[metadata['accession_number'].notna()]['accession_number'].head(5).tolist())

    return metadata

# ============================================
# 2. Extract and Consolidate Text Data
# ============================================

def extract_accession_from_path(filepath):
    """Extract accession number from file path like 'item1/2020/0000001800-20-000015.txt'"""
    filename = os.path.basename(filepath)
    return filename.replace('.txt', '')

def extract_cik_from_accession(accession_number):
    """
    Extract CIK from SEC accession number.

    Format: XXXXXXXXXX-YY-ZZZZZZ
    Where X = CIK (10 digits, zero-padded)
          Y = Year (2 digits)
          Z = Sequence (6 digits)

    Examples:
        '0000009892-00-000006' → 9892
        '0001234567-20-000123' → 1234567

    Args:
        accession_number: SEC accession string

    Returns:
        Integer CIK (leading zeros removed), or None if parsing fails
    """
    if pd.isna(accession_number):
        return None

    # Handle string format
    acc_str = str(accession_number).strip()

    # Extract first part before first hyphen
    parts = acc_str.split('-')
    if len(parts) >= 2:
        cik_str = parts[0]
        # Convert to int (removes leading zeros)
        try:
            return int(cik_str)
        except ValueError:
            return None
    return None

def load_texts_from_zip(zip_path, item_name, max_files=None, test_mode=False):
    """
    Load all text files from a zip archive.

    Args:
        zip_path: Path to zip file
        item_name: Item identifier (item1, item1A, item7)
        max_files: Optional limit for testing (None = load all)
        test_mode: If True, sample from recent years (2020-2024) instead of first N files

    Returns:
        DataFrame with columns: accession_number, year, text, item_type
    """
    print(f"\nExtracting {item_name} from {zip_path.name}...")

    records = []
    skipped = 0

    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Get list of text files
        txt_files = [f for f in zf.namelist() if f.endswith('.txt')]

        if max_files:
            if test_mode:
                # In test mode, sample from recent years (2020-2024) to match metadata coverage
                recent_files = [f for f in txt_files if any(f'/{year}/' in f for year in [2020, 2021, 2022, 2023, 2024])]
                if recent_files:
                    txt_files = recent_files[:max_files]
                    print(f"  Test mode: Sampling {len(txt_files)} files from years 2020-2024")
                else:
                    # Fallback if no recent files found
                    txt_files = txt_files[:max_files]
            else:
                txt_files = txt_files[:max_files]

        for filepath in tqdm(txt_files, desc=f"Loading {item_name}"):
            try:
                # Extract year from path
                year_match = re.search(r'/(\d{4})/', filepath)
                year = int(year_match.group(1)) if year_match else None

                # Extract accession number
                accession = extract_accession_from_path(filepath)

                # Extract CIK from accession number
                cik = extract_cik_from_accession(accession)

                # Read text content
                with zf.open(filepath) as f:
                    text = f.read().decode('utf-8', errors='ignore').strip()

                # Skip empty files
                if not text:
                    skipped += 1
                    continue

                records.append({
                    'accession_number': accession,
                    'cik': cik,
                    'year': year,
                    'text': text,
                    'item_type': item_name,
                    'text_length': len(text)
                })

            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                skipped += 1
                continue

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} files, skipped {skipped} (empty or error)")

    return df

def consolidate_all_items(metadata, max_files_per_item=None, test_mode=False):
    """
    Load all three items and create consolidated dataset.

    Args:
        metadata: Metadata DataFrame with filing_date
        max_files_per_item: Optional limit for testing
        test_mode: If True, sample from recent years

    Returns:
        DataFrame with all items combined, merged with metadata
    """
    all_items = []

    for item_name, zip_path in ZIP_FILES.items():
        if not zip_path.exists():
            print(f"WARNING: {zip_path} not found, skipping...")
            continue

        item_df = load_texts_from_zip(zip_path, item_name, max_files=max_files_per_item, test_mode=test_mode)
        all_items.append(item_df)

    # Combine all items
    combined_df = pd.concat(all_items, ignore_index=True)
    print(f"\nTotal combined records: {len(combined_df)}")

    # Merge with metadata to get filing_date
    print("\nMerging with metadata to add filing_date...")
    print(f"  Records before merge: {len(combined_df):,}")

    # Prepare metadata for merge (keep only needed columns)
    metadata_cols = metadata[['accession_number', 'filing_date']].copy()

    # Merge on accession_number
    combined_df = combined_df.merge(metadata_cols, on='accession_number', how='left')

    # Report merge results
    with_filing_date = combined_df['filing_date'].notna().sum()
    print(f"  Records after merge: {len(combined_df):,}")
    print(f"  Records with filing_date: {with_filing_date:,} ({with_filing_date/len(combined_df)*100:.1f}%)")

    if with_filing_date < len(combined_df):
        missing = len(combined_df) - with_filing_date
        print(f"  WARNING: {missing:,} records missing filing_date (not in metadata)")

    return combined_df

# ============================================
# 3. Data Quality Checks
# ============================================

def data_quality_report(df, metadata=None):
    """Generate comprehensive data quality report."""
    print("\n" + "="*60)
    print("DATA QUALITY REPORT")
    print("="*60)

    # Basic stats
    print(f"\n1. BASIC STATISTICS")
    print(f"   Total records: {len(df):,}")
    print(f"   Unique accession numbers: {df['accession_number'].nunique():,}")
    print(f"   Year range: {df['year'].min():.0f} - {df['year'].max():.0f}")

    # By item type
    print(f"\n2. BY ITEM TYPE")
    item_stats = df.groupby('item_type').agg({
        'text_length': ['count', 'mean', 'median', 'min', 'max']
    }).round(0)
    print(item_stats)

    # By year
    print(f"\n3. COVERAGE BY YEAR")
    year_stats = df.groupby('year')['accession_number'].count()
    print(f"   Years with < 100 filings: {(year_stats < 100).sum()}")
    print(f"   Peak year: {year_stats.idxmax():.0f} ({year_stats.max():,} filings)")

    # Text length distribution
    print(f"\n4. TEXT LENGTH DISTRIBUTION (characters)")
    print(df['text_length'].describe().round(0))

    # Check for very short texts (potential quality issues)
    short_texts = (df['text_length'] < 100).sum()
    print(f"\n5. QUALITY FLAGS")
    print(f"   Very short texts (< 100 chars): {short_texts:,} ({short_texts/len(df)*100:.1f}%)")

    # Report CIK extraction success (CIK already extracted from accession numbers during loading)
    print(f"\n6. CIK EXTRACTION FROM ACCESSION NUMBERS")
    cik_extracted = df['cik'].notna().sum()
    print(f"   Successfully extracted CIK for {cik_extracted:,} / {len(df):,} records ({cik_extracted/len(df)*100:.1f}%)")
    print(f"   Unique firms (CIK): {df['cik'].nunique():,}")

    if cik_extracted < len(df) * 0.9:
        print(f"   WARNING: Some accession numbers failed to parse")

    # Set placeholder field for ticker
    # (Can be enriched later from SEC EDGAR master index if needed)
    df['ticker'] = 'UNKNOWN'

    return df

# ============================================
# 4. Sentence Parsing
# ============================================

def sentence_split(text, min_length=20):
    """
    Split text into sentences using punctuation boundaries.
    Sentences are the atomic unit for CLN novelty and KMNZ relevance.

    Args:
        text: Input text
        min_length: Minimum sentence length (filter out short fragments)

    Returns:
        List of sentences
    """
    # Split on sentence boundaries (.!?)
    # Handle common abbreviations and edge cases
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    # Clean and filter sentences
    cleaned_sentences = []
    for sent in sentences:
        sent = sent.strip()
        # Filter out very short fragments and non-sentences
        if len(sent) >= min_length and not re.match(r'^\d+$', sent):
            cleaned_sentences.append(sent)

    return cleaned_sentences

def create_sentence_dataset(df, max_rows=None, batch_size=10000):
    """
    Parse documents into sentences and create sentence-level dataset.
    Uses batched processing to avoid memory issues with large datasets.

    Aligned with proposal notation (Section 4.1):
    - f = firm (via accession_number)
    - d = document (accession_number)
    - t = filing date
    - s = sentence index

    Args:
        df: DataFrame with document-level text
        max_rows: Optional limit for testing
        batch_size: Number of documents to process in each batch

    Returns:
        DataFrame with sentence-level data
    """
    print("\nParsing documents into sentences...")

    if max_rows:
        df = df.head(max_rows)

    # Process in batches to avoid memory issues
    all_batches = []
    total_sentences = 0

    for batch_start in range(0, len(df), batch_size):
        batch_end = min(batch_start + batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end]

        sentence_records = []

        for idx, row in tqdm(batch_df.iterrows(),
                            total=len(batch_df),
                            desc=f"Parsing batch {batch_start//batch_size + 1}/{(len(df)-1)//batch_size + 1}",
                            leave=False):
            sentences = sentence_split(row['text'])

            for sent_idx, sentence in enumerate(sentences):
                sentence_records.append({
                    'accession_number': row['accession_number'],
                    'cik': row.get('cik'),
                    'ticker': row.get('ticker'),
                    'year': row['year'],
                    'filing_date': row.get('filing_date'),
                    'item_type': row['item_type'],
                    'sentence_id': sent_idx,
                    'text': sentence,
                    'sentence_length': len(sentence)
                })

        # Convert batch to DataFrame
        batch_sent_df = pd.DataFrame(sentence_records)
        all_batches.append(batch_sent_df)
        total_sentences += len(batch_sent_df)

        print(f"  Batch {batch_start//batch_size + 1}: {len(batch_sent_df):,} sentences from {len(batch_df):,} documents")

    # Combine all batches
    print("\nCombining batches...")
    sent_df = pd.concat(all_batches, ignore_index=True)

    print(f"Created {len(sent_df):,} sentences from {len(df):,} documents")
    print(f"Average sentences per document: {len(sent_df)/len(df):.1f}")

    return sent_df

# ============================================
# 5. Stratified Sampling for SAE Training
# ============================================

def create_sae_training_sample(sentence_df, target_n=5_000_000,
                               sentences_per_firm=1000,
                               max_per_year_item=50_000,
                               random_state=42):
    """
    Create stratified sample with firm-level balance.

    Multi-stage sampling strategy to ensure firm diversity:
    1. Stage 1 (Firm-level): Sample up to N sentences per firm
       → Prevents large firms from dominating the sample
       → Ensures broad firm coverage
    2. Stage 2 (Year×Item): Balance within each (year, item_type) cell
       → Maintains temporal and item-type coverage
    3. Stage 3 (Global cap): Limit to target_n if needed

    This ensures:
    - No single firm dominates the sample
    - Temporal coverage (year)
    - Item-type coverage (item1, 1A, 7)
    - Firm-level diversity

    Args:
        sentence_df: Full sentence DataFrame with 'cik' column
        target_n: Target sample size (default 5M)
        sentences_per_firm: Max sentences per CIK (default 1000)
        max_per_year_item: Max sentences per (year, item_type) cell
        random_state: Random seed

    Returns:
        Sampled DataFrame for SAE training
    """
    print(f"\n  Multi-stage stratified sampling strategy:")
    print(f"    - Stage 1: Max {sentences_per_firm:,} sentences per firm (CIK)")
    print(f"    - Stage 2: Max {max_per_year_item:,} per (year, item_type)")
    print(f"    - Stage 3: Global cap at {target_n:,}")

    # Stage 1: Sample per firm to prevent large-firm dominance
    print(f"\n  Stage 1: Firm-level sampling...")
    firm_samples = []
    firms_with_missing_cik = sentence_df['cik'].isna().sum()

    if firms_with_missing_cik > 0:
        print(f"   WARNING: {firms_with_missing_cik:,} sentences have missing CIK, these will be included separately")

    for cik, group in tqdm(sentence_df.groupby('cik', dropna=False), desc="  Sampling firms"):
        if pd.isna(cik):
            # Handle missing CIKs: take a small sample
            n_sample = min(len(group), 100)
            sampled = group.sample(n=n_sample, random_state=random_state) if n_sample < len(group) else group
        else:
            # Regular firm sampling
            n_firm = len(group)
            n_sample = min(n_firm, sentences_per_firm)
            sampled = group.sample(n=n_sample, random_state=random_state) if n_sample < n_firm else group

        firm_samples.append(sampled)

    # Combine firm samples
    firm_balanced_df = pd.concat(firm_samples, ignore_index=True)
    n_firms_sampled = sentence_df['cik'].nunique()
    print(f"  After firm sampling: {len(firm_balanced_df):,} sentences from {n_firms_sampled:,} firms")

    # Stage 2: Balance by (year, item_type)
    print(f"\n  Stage 2: Year × Item sampling...")
    year_item_samples = []

    for (year, item), group in tqdm(firm_balanced_df.groupby(['year', 'item_type']),
                                     desc="  Sampling year×item cells"):
        n_cell = len(group)
        n_sample = min(n_cell, max_per_year_item)

        if n_sample < n_cell:
            sampled = group.sample(n=n_sample, random_state=random_state)
        else:
            sampled = group

        year_item_samples.append(sampled)

    # Combine
    sample_df = pd.concat(year_item_samples, ignore_index=True)
    print(f"  After year×item sampling: {len(sample_df):,} sentences")

    # Stage 3: Global cap if needed
    if len(sample_df) > target_n:
        print(f"\n  Stage 3: Global cap {len(sample_df):,} → {target_n:,}")
        sample_df = sample_df.sample(n=target_n, random_state=random_state)

    # Sort and reset index
    sample_df = sample_df.sort_values(['year', 'item_type', 'cik', 'accession_number', 'sentence_id'])
    sample_df = sample_df.reset_index(drop=True)

    # Report coverage
    print(f"\n  Final sample statistics:")
    print(f"    Total sentences: {len(sample_df):,}")
    print(f"    Unique firms: {sample_df['cik'].nunique():,}")
    print(f"    Sampling rate: {len(sample_df)/len(sentence_df)*100:.1f}%")

    print(f"\n  Sentences per firm (distribution):")
    firm_counts = sample_df[sample_df['cik'].notna()].groupby('cik').size()
    if len(firm_counts) > 0:
        print(f"    Mean: {firm_counts.mean():.1f}")
        print(f"    Median: {firm_counts.median():.1f}")
        print(f"    Min: {firm_counts.min()}")
        print(f"    Max: {firm_counts.max()}")
        print(f"    Std: {firm_counts.std():.1f}")

    print(f"\n  Coverage by year:")
    year_coverage = sample_df.groupby('year').size()
    print(year_coverage.to_string())

    print(f"\n  Coverage by item:")
    item_coverage = sample_df.groupby('item_type').size()
    print(item_coverage.to_string())

    return sample_df

# ============================================
# Main Execution
# ============================================

def main(test_mode=False):
    """
    Main execution pipeline.

    Args:
        test_mode: If True, process only a small sample for testing
    """
    print("="*60)
    print("PHASE 1: DATA PREPARATION & EXPLORATION")
    print("="*60)

    # Step 1: Load metadata
    metadata = load_metadata()

    # Step 2: Extract and consolidate texts
    # Use small sample for testing, None for full dataset
    max_files = 1000 if test_mode else None

    text_df = consolidate_all_items(metadata, max_files_per_item=max_files, test_mode=test_mode)

    # Step 3: Data quality checks
    text_df_merged = data_quality_report(text_df, metadata)

    # Save consolidated document-level data (drop text column to save memory)
    doc_output = OUTPUT_DIR / "documents_consolidated.parquet"
    text_df_to_save = text_df_merged.drop('text', axis=1)
    text_df_to_save.to_parquet(doc_output, index=False)
    print(f"\n[OK] Saved document-level data to {doc_output}")
    print(f"   (Text column excluded - preserved in sentences.parquet)")

    # Step 4: Create sentence-level dataset (aligned with proposal)
    max_rows = 100 if test_mode else None
    sentence_df = create_sentence_dataset(text_df_merged, max_rows=max_rows)

    # Save sentence-level data
    sent_output = OUTPUT_DIR / "sentences.parquet"
    sentence_df.to_parquet(sent_output, index=False)
    print(f"[OK] Saved sentence-level data to {sent_output}")

    # Step 5: Create per-item sentence files for organization
    print("\nCreating per-item sentence files...")
    for item_type in sentence_df['item_type'].unique():
        item_sent_df = sentence_df[sentence_df['item_type'] == item_type]
        item_output = OUTPUT_DIR / f"sentences_{item_type}.parquet"
        item_sent_df.to_parquet(item_output, index=False)
        print(f"  - Saved {len(item_sent_df):,} sentences for {item_type} to {item_output.name}")

    # Final summary
    print("\n" + "="*60)
    print("PHASE 1 COMPLETE")
    print("="*60)
    print(f"Documents processed: {len(text_df_merged):,}")
    print(f"Sentences created: {len(sentence_df):,}")
    print(f"Unique firms (CIK): {sentence_df['cik'].nunique():,}")
    print(f"\nOutput files:")
    print(f"  - {doc_output} (document-level metadata)")
    print(f"  - {sent_output} (all sentences with CIK)")
    print(f"  - sentences_item1.parquet, item1A, item7 (per-item)")
    print(f"\nSchema aligned with proposal:")
    print(f"  - accession_number (document ID)")
    print(f"  - cik (firm ID)")
    print(f"  - filing_date (time t)")
    print(f"  - sentence_id (sentence index s)")
    print(f"\nNEXT STEP:")
    print(f"  Run 02_embeddings_and_features.py")
    print(f"  Phase 2 will handle stratified sampling + embedding")
    return text_df_merged, sentence_df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Prepare 10-K text data')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode with small sample')
    args = parser.parse_args()

    text_df, sentence_df = main(test_mode=args.test)
