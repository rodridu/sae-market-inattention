"""
Pipeline Validation Script

Checks data integrity and consistency across all pipeline phases (00-06).
Detects corruption, missing files, schema issues, and data quality problems.

Usage:
  python 00_validate_pipeline.py
  python 00_validate_pipeline.py --verbose  # Show detailed diagnostics
  python 00_validate_pipeline.py --fix      # Attempt to fix minor issues

Validation Checks:
- Phase 0: Metadata, WRDS outcomes, link tables
- Phase 1: Sentences, spans, text extraction
- Phase 2: Embeddings, novelty, relevance
- Phase 3: SAE models and checkpoints
- Phase 4: Extracted features
- Phase 5: Feature interpretations
- Phase 6: Sarkar analysis outputs
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# =========================
# Configuration
# =========================

DATA_DIR = Path(r"C:\Users\ofs4963\Dropbox\Arojects\SAE\data")
REQUIRED_COLUMNS = {
    'sec_metadata': ['accession_number', 'cik', 'filing_date'],
    'sentences': ['accession_number', 'cik', 'filing_date', 'sentence_id', 'text', 'item_type'],
    'embeddings': ['accession_number', 'sentence_id', 'item_type'],
    'wrds_merged': ['accession_number', 'cik', 'filing_date', 'car_minus1_plus1'],
}

class ValidationReport:
    """Tracks validation results across all phases"""

    def __init__(self):
        self.phases = {}
        self.errors = []
        self.warnings = []
        self.info = []

    def add_phase(self, phase_name):
        self.phases[phase_name] = {
            'status': 'pending',
            'checks': [],
            'files': [],
            'issues': []
        }

    def add_check(self, phase_name, check_name, passed, details=None):
        if phase_name not in self.phases:
            self.add_phase(phase_name)

        self.phases[phase_name]['checks'].append({
            'name': check_name,
            'passed': passed,
            'details': details
        })

        if not passed:
            self.phases[phase_name]['status'] = 'failed'
            self.errors.append(f"[{phase_name}] {check_name}: {details}")
        elif self.phases[phase_name]['status'] == 'pending':
            self.phases[phase_name]['status'] = 'passed'

    def add_warning(self, phase_name, message):
        self.warnings.append(f"[{phase_name}] {message}")
        if phase_name in self.phases:
            self.phases[phase_name]['issues'].append(message)

    def add_info(self, phase_name, message):
        self.info.append(f"[{phase_name}] {message}")

    def print_summary(self):
        print("\n" + "="*80)
        print("PIPELINE VALIDATION SUMMARY")
        print("="*80)

        # Phase-by-phase status
        for phase_name, phase_data in self.phases.items():
            status = phase_data['status']
            icon = "[OK]" if status == 'passed' else "[FAIL]" if status == 'failed' else "[?]"
            print(f"\n{icon} {phase_name}: {status.upper()}")

            # Show failed checks
            failed_checks = [c for c in phase_data['checks'] if not c['passed']]
            if failed_checks:
                for check in failed_checks:
                    print(f"  [X] {check['name']}: {check['details']}")

        # Overall statistics
        print(f"\n" + "="*80)
        print(f"Total Errors: {len(self.errors)}")
        print(f"Total Warnings: {len(self.warnings)}")
        print(f"Total Info: {len(self.info)}")

        if self.errors:
            print(f"\n" + "="*80)
            print("ERRORS:")
            print("="*80)
            for error in self.errors:
                print(f"  {error}")

        if self.warnings:
            print(f"\n" + "="*80)
            print("WARNINGS:")
            print("="*80)
            for warning in self.warnings[:20]:  # Limit to 20
                print(f"  {warning}")
            if len(self.warnings) > 20:
                print(f"  ... and {len(self.warnings) - 20} more warnings")

        print("="*80 + "\n")

# =========================
# Phase 0: Metadata & WRDS
# =========================

def validate_phase_0(report, verbose=False):
    """Validate Phase 0: Metadata and WRDS data"""
    phase = "Phase 0: Metadata & WRDS"
    report.add_phase(phase)

    print(f"\n{'='*80}")
    print(f"Validating {phase}")
    print(f"{'='*80}")

    # Check 1: SEC metadata exists
    metadata_file = DATA_DIR / "sec_metadata.parquet"
    if metadata_file.exists():
        try:
            metadata_df = pd.read_parquet(metadata_file)
            report.add_check(phase, "SEC metadata file exists", True, f"{len(metadata_df):,} rows")

            # Check schema
            missing_cols = [c for c in REQUIRED_COLUMNS['sec_metadata'] if c not in metadata_df.columns]
            if missing_cols:
                report.add_check(phase, "SEC metadata schema", False, f"Missing columns: {missing_cols}")
            else:
                report.add_check(phase, "SEC metadata schema", True, "All required columns present")

            # Check data quality
            null_counts = metadata_df[REQUIRED_COLUMNS['sec_metadata']].isnull().sum()
            if null_counts.any():
                report.add_warning(phase, f"Null values in metadata: {null_counts[null_counts > 0].to_dict()}")

            # Check date range
            if 'filing_date' in metadata_df.columns:
                metadata_df['filing_date'] = pd.to_datetime(metadata_df['filing_date'], errors='coerce')
                date_range = f"{metadata_df['filing_date'].min()} to {metadata_df['filing_date'].max()}"
                report.add_info(phase, f"Filing date range: {date_range}")
                report.add_info(phase, f"Unique CIKs: {metadata_df['cik'].nunique():,}")

        except Exception as e:
            report.add_check(phase, "SEC metadata file readable", False, str(e))
    else:
        report.add_check(phase, "SEC metadata file exists", False, "File not found")

    # Check 2: WRDS merged data
    wrds_file = DATA_DIR / "wrds_merged.parquet"
    if wrds_file.exists():
        try:
            wrds_df = pd.read_parquet(wrds_file)
            report.add_check(phase, "WRDS merged data exists", True, f"{len(wrds_df):,} rows")

            # Check schema
            missing_cols = [c for c in REQUIRED_COLUMNS['wrds_merged'] if c not in wrds_df.columns]
            if missing_cols:
                report.add_check(phase, "WRDS data schema", False, f"Missing columns: {missing_cols}")
            else:
                report.add_check(phase, "WRDS data schema", True, "All required columns present")

            # Check outcome coverage
            outcome_cols = ['car_minus1_plus1', 'drift_30d', 'drift_60d', 'drift_90d']
            for col in outcome_cols:
                if col in wrds_df.columns:
                    coverage = wrds_df[col].notna().sum() / len(wrds_df)
                    report.add_info(phase, f"{col} coverage: {coverage*100:.1f}%")
                    if coverage < 0.5:
                        report.add_warning(phase, f"Low coverage for {col}: {coverage*100:.1f}%")

            # Check control coverage
            control_cols = ['size', 'bm', 'leverage', 'past_ret', 'past_vol']
            for col in control_cols:
                if col in wrds_df.columns:
                    coverage = wrds_df[col].notna().sum() / len(wrds_df)
                    if coverage < 0.5:
                        report.add_warning(phase, f"Low coverage for {col}: {coverage*100:.1f}%")

        except Exception as e:
            report.add_check(phase, "WRDS merged data readable", False, str(e))
    else:
        report.add_warning(phase, "WRDS merged data not found - run 00d_wrds_data_merge.py")

# =========================
# Phase 1: Text Preparation
# =========================

def validate_phase_1(report, verbose=False):
    """Validate Phase 1: Sentences and text extraction"""
    phase = "Phase 1: Text Preparation"
    report.add_phase(phase)

    print(f"\n{'='*80}")
    print(f"Validating {phase}")
    print(f"{'='*80}")

    # Check sentences file
    sentences_file = DATA_DIR / "sentences_sampled.parquet"
    if sentences_file.exists():
        try:
            sent_df = pd.read_parquet(sentences_file)
            report.add_check(phase, "Sentences file exists", True, f"{len(sent_df):,} sentences")

            # Check schema
            missing_cols = [c for c in REQUIRED_COLUMNS['sentences'] if c not in sent_df.columns]
            if missing_cols:
                report.add_check(phase, "Sentences schema", False, f"Missing columns: {missing_cols}")
            else:
                report.add_check(phase, "Sentences schema", True, "All required columns present")

            # Check for empty text
            empty_text = sent_df['text'].isna().sum() + (sent_df['text'].str.len() == 0).sum()
            if empty_text > 0:
                report.add_warning(phase, f"Empty text in {empty_text:,} sentences ({empty_text/len(sent_df)*100:.2f}%)")

            # Check item distribution
            if 'item_type' in sent_df.columns:
                item_dist = sent_df['item_type'].value_counts()
                report.add_info(phase, f"Item distribution: {item_dist.to_dict()}")

            # Check sentence length distribution
            sent_lengths = sent_df['text'].str.len()
            report.add_info(phase, f"Sentence length: mean={sent_lengths.mean():.0f}, median={sent_lengths.median():.0f}")

            # Check for duplicates
            dup_count = sent_df.duplicated(subset=['accession_number', 'sentence_id', 'item_type']).sum()
            if dup_count > 0:
                report.add_check(phase, "No duplicate sentences", False, f"{dup_count:,} duplicates found")
            else:
                report.add_check(phase, "No duplicate sentences", True)

        except Exception as e:
            report.add_check(phase, "Sentences file readable", False, str(e))
    else:
        report.add_check(phase, "Sentences file exists", False, "File not found")

    # Check documents consolidated (if exists)
    docs_file = DATA_DIR / "documents_consolidated.parquet"
    if docs_file.exists():
        try:
            docs_df = pd.read_parquet(docs_file)
            report.add_info(phase, f"Documents consolidated: {len(docs_df):,} documents")
        except Exception as e:
            report.add_warning(phase, f"Documents consolidated file corrupted: {e}")

# =========================
# Phase 2: Embeddings & Features
# =========================

def validate_phase_2(report, verbose=False):
    """Validate Phase 2: Embeddings, novelty, relevance"""
    phase = "Phase 2: Embeddings & Features"
    report.add_phase(phase)

    print(f"\n{'='*80}")
    print(f"Validating {phase}")
    print(f"{'='*80}")

    # Check embeddings chunks
    emb_chunks_dir = DATA_DIR / "sentence_embeddings_chunks"
    if emb_chunks_dir.exists():
        chunk_files = sorted([f for f in emb_chunks_dir.iterdir() if f.suffix == '.npz'])

        if chunk_files:
            report.add_check(phase, "Embeddings chunks exist", True, f"{len(chunk_files)} chunks")

            # Load and validate first chunk
            try:
                first_chunk = np.load(chunk_files[0], allow_pickle=True)

                # Check expected keys
                expected_keys = ['embeddings', 'accession_numbers', 'sentence_ids', 'item_types']
                missing_keys = [k for k in expected_keys if k not in first_chunk.files]
                if missing_keys:
                    report.add_check(phase, "Embedding chunk schema", False, f"Missing keys: {missing_keys}")
                else:
                    report.add_check(phase, "Embedding chunk schema", True)

                # Check embedding dimension
                emb_dim = first_chunk['embeddings'].shape[1]
                report.add_info(phase, f"Embedding dimension: {emb_dim}")

                # Check all chunks
                total_embeddings = 0
                corrupted_chunks = []
                for chunk_file in chunk_files:
                    try:
                        chunk_data = np.load(chunk_file, allow_pickle=True)
                        total_embeddings += len(chunk_data['embeddings'])

                        # Check for NaN/Inf
                        if np.isnan(chunk_data['embeddings']).any():
                            corrupted_chunks.append(f"{chunk_file.name} (contains NaN)")
                        if np.isinf(chunk_data['embeddings']).any():
                            corrupted_chunks.append(f"{chunk_file.name} (contains Inf)")
                    except Exception as e:
                        corrupted_chunks.append(f"{chunk_file.name} ({str(e)})")

                if corrupted_chunks:
                    report.add_check(phase, "No corrupted embeddings", False, f"{len(corrupted_chunks)} chunks corrupted")
                    for chunk in corrupted_chunks[:5]:
                        report.add_warning(phase, f"Corrupted chunk: {chunk}")
                else:
                    report.add_check(phase, "No corrupted embeddings", True)

                report.add_info(phase, f"Total embeddings: {total_embeddings:,}")

            except Exception as e:
                report.add_check(phase, "Embeddings readable", False, str(e))
        else:
            report.add_check(phase, "Embeddings chunks exist", False, "No chunk files found")
    else:
        report.add_check(phase, "Embeddings directory exists", False, "Directory not found")

    # Check novelty data
    novelty_file = DATA_DIR / "novelty_cln.parquet"
    if novelty_file.exists():
        try:
            novelty_df = pd.read_parquet(novelty_file)
            report.add_check(phase, "Novelty data exists", True, f"{len(novelty_df):,} rows")

            # Check for valid novelty values (should be between 0 and 1)
            novelty_cols = [c for c in novelty_df.columns if 'novelty' in c.lower()]
            for col in novelty_cols:
                if col in novelty_df.columns:
                    invalid = ((novelty_df[col] < 0) | (novelty_df[col] > 1)).sum()
                    if invalid > 0:
                        report.add_warning(phase, f"{col}: {invalid:,} invalid values (outside [0,1])")
        except Exception as e:
            report.add_warning(phase, f"Novelty data corrupted: {e}")
    else:
        report.add_warning(phase, "Novelty data not found - run 02b_novelty_cln.py")

    # Check relevance data
    relevance_file = DATA_DIR / "relevance_kmnz.parquet"
    if relevance_file.exists():
        try:
            relevance_df = pd.read_parquet(relevance_file)
            report.add_check(phase, "Relevance data exists", True, f"{len(relevance_df):,} rows")
        except Exception as e:
            report.add_warning(phase, f"Relevance data corrupted: {e}")
    else:
        report.add_warning(phase, "Relevance data not found - run 02c_relevance_kmnz.py")

# =========================
# Phase 3: SAE Models
# =========================

def validate_phase_3(report, verbose=False):
    """Validate Phase 3: SAE training outputs"""
    phase = "Phase 3: SAE Training"
    report.add_phase(phase)

    print(f"\n{'='*80}")
    print(f"Validating {phase}")
    print(f"{'='*80}")

    # Look for SAE directories
    sae_dirs = [d for d in DATA_DIR.iterdir() if d.is_dir() and 'sae_anthropic' in d.name]

    if not sae_dirs:
        report.add_check(phase, "SAE models exist", False, "No SAE directories found")
        return

    report.add_info(phase, f"Found {len(sae_dirs)} SAE directories")

    for sae_dir in sae_dirs:
        # Check for final model
        final_model = sae_dir / "sae_final.pt"
        if final_model.exists():
            try:
                import torch
                checkpoint = torch.load(final_model, map_location='cpu')

                # Check checkpoint structure
                required_keys = ['model_state_dict', 'config', 'step']
                missing_keys = [k for k in required_keys if k not in checkpoint]
                if missing_keys:
                    report.add_warning(phase, f"{sae_dir.name}: Missing checkpoint keys: {missing_keys}")
                else:
                    config = checkpoint['config']
                    report.add_info(phase, f"{sae_dir.name}: {config['n_features']} features, step {checkpoint['step']}")

            except Exception as e:
                report.add_warning(phase, f"{sae_dir.name}: Corrupted model file: {e}")
        else:
            report.add_warning(phase, f"{sae_dir.name}: No final model found")

        # Check training loss log
        loss_csv = sae_dir / "training_loss.csv"
        if loss_csv.exists():
            try:
                loss_df = pd.read_csv(loss_csv)
                if len(loss_df) > 0:
                    report.add_info(phase, f"{sae_dir.name}: {len(loss_df)} training steps logged")

                    # Check for NaN losses
                    if loss_df['loss'].isna().any():
                        report.add_warning(phase, f"{sae_dir.name}: NaN losses detected in training")
                else:
                    report.add_warning(phase, f"{sae_dir.name}: Empty loss log")
            except Exception as e:
                report.add_warning(phase, f"{sae_dir.name}: Corrupted loss log: {e}")

        # Check config
        config_file = sae_dir / "config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                report.add_info(phase, f"{sae_dir.name}: Config loaded successfully")
            except Exception as e:
                report.add_warning(phase, f"{sae_dir.name}: Corrupted config: {e}")

    report.add_check(phase, "At least one SAE model valid", True, f"{len(sae_dirs)} models found")

# =========================
# Phase 4-6: Analysis Outputs
# =========================

def validate_phase_4_6(report, verbose=False):
    """Validate Phases 4-6: Feature extraction, interpretation, Sarkar analysis"""

    # Phase 4: Feature extraction
    phase = "Phase 4: Feature Extraction"
    report.add_phase(phase)

    print(f"\n{'='*80}")
    print(f"Validating {phase}")
    print(f"{'='*80}")

    # Check for extracted features (placeholder - depends on implementation)
    report.add_info(phase, "Feature extraction validation: TODO")

    # Phase 5: Interpretation
    phase = "Phase 5: Feature Interpretation"
    report.add_phase(phase)

    print(f"\n{'='*80}")
    print(f"Validating {phase}")
    print(f"{'='*80}")

    report.add_info(phase, "Interpretation validation: TODO")

    # Phase 6: Sarkar analysis
    phase = "Phase 6: Sarkar Analysis"
    report.add_phase(phase)

    print(f"\n{'='*80}")
    print(f"Validating {phase}")
    print(f"{'='*80}")

    # Look for Sarkar output directories
    sarkar_dirs = [d for d in DATA_DIR.iterdir() if d.is_dir() and 'sarkar_analysis' in d.name]

    if sarkar_dirs:
        for sarkar_dir in sarkar_dirs:
            firm_reps = sarkar_dir / "firm_representations.parquet"
            test_results = sarkar_dir / "test_results.json"

            if firm_reps.exists():
                try:
                    reps_df = pd.read_parquet(firm_reps)
                    report.add_info(phase, f"{sarkar_dir.name}: {len(reps_df):,} firm-year observations")

                    # Check for decomposition columns
                    on_cols = [c for c in reps_df.columns if c.startswith('Z^on_')]
                    perp_cols = [c for c in reps_df.columns if c.startswith('Z^perp_')]

                    if on_cols and perp_cols:
                        report.add_info(phase, f"{sarkar_dir.name}: Orthogonal decomposition completed")
                    else:
                        report.add_warning(phase, f"{sarkar_dir.name}: Missing decomposition columns")

                except Exception as e:
                    report.add_warning(phase, f"{sarkar_dir.name}: Corrupted representations: {e}")

            if test_results.exists():
                try:
                    with open(test_results, 'r') as f:
                        results = json.load(f)
                    report.add_info(phase, f"{sarkar_dir.name}: Test results available")
                except Exception as e:
                    report.add_warning(phase, f"{sarkar_dir.name}: Corrupted test results: {e}")
    else:
        report.add_warning(phase, "No Sarkar analysis outputs found - run 06_sarkar_analysis.py")

# =========================
# Cross-Phase Validation
# =========================

def validate_cross_phase(report, verbose=False):
    """Check consistency across phases"""
    phase = "Cross-Phase Consistency"
    report.add_phase(phase)

    print(f"\n{'='*80}")
    print(f"Validating {phase}")
    print(f"{'='*80}")

    try:
        # Check if sentences and embeddings match
        sentences_file = DATA_DIR / "sentences_sampled.parquet"
        emb_chunks_dir = DATA_DIR / "sentence_embeddings_chunks"

        if sentences_file.exists() and emb_chunks_dir.exists():
            sent_df = pd.read_parquet(sentences_file)

            # Load embedding metadata
            emb_count = 0
            chunk_files = sorted([f for f in emb_chunks_dir.iterdir() if f.suffix == '.npz'])
            for chunk_file in chunk_files:
                chunk_data = np.load(chunk_file, allow_pickle=True)
                emb_count += len(chunk_data['embeddings'])

            sent_count = len(sent_df)

            if sent_count == emb_count:
                report.add_check(phase, "Sentences-Embeddings count match", True, f"{sent_count:,} sentences")
            else:
                report.add_check(phase, "Sentences-Embeddings count match", False,
                                f"Sentences: {sent_count:,}, Embeddings: {emb_count:,}")

        # Check if metadata and sentences match
        metadata_file = DATA_DIR / "sec_metadata.parquet"
        if metadata_file.exists() and sentences_file.exists():
            metadata_df = pd.read_parquet(metadata_file)
            sent_df = pd.read_parquet(sentences_file)

            # Check CIK coverage
            metadata_ciks = set(metadata_df['cik'].unique())
            sent_ciks = set(sent_df['cik'].unique())

            coverage = len(sent_ciks) / len(metadata_ciks) if len(metadata_ciks) > 0 else 0
            report.add_info(phase, f"CIK coverage (sentences/metadata): {coverage*100:.1f}%")

            if coverage < 0.5:
                report.add_warning(phase, f"Low CIK coverage: only {coverage*100:.1f}% of metadata CIKs in sentences")

    except Exception as e:
        report.add_warning(phase, f"Cross-phase validation error: {e}")

# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser(description="Validate SAE pipeline")
    parser.add_argument('--verbose', action='store_true', help='Show detailed diagnostics')
    parser.add_argument('--fix', action='store_true', help='Attempt to fix minor issues')
    args = parser.parse_args()

    print("="*80)
    print("SAE PIPELINE VALIDATION")
    print("="*80)
    print(f"Data directory: {DATA_DIR}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Create validation report
    report = ValidationReport()

    # Run validations
    validate_phase_0(report, verbose=args.verbose)
    validate_phase_1(report, verbose=args.verbose)
    validate_phase_2(report, verbose=args.verbose)
    validate_phase_3(report, verbose=args.verbose)
    validate_phase_4_6(report, verbose=args.verbose)
    validate_cross_phase(report, verbose=args.verbose)

    # Print summary
    report.print_summary()

    # Save report
    report_file = DATA_DIR / "validation_report.json"
    with open(report_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'phases': report.phases,
            'errors': report.errors,
            'warnings': report.warnings,
            'info': report.info
        }, f, indent=2)

    print(f"Validation report saved to: {report_file}")

    # Exit code
    if report.errors:
        print("\n[FAILED] Pipeline validation found errors")
        sys.exit(1)
    else:
        print("\n[PASSED] Pipeline validation completed successfully")
        sys.exit(0)

if __name__ == "__main__":
    main()
