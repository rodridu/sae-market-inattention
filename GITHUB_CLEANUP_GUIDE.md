# GitHub Cleanup Guide

This guide shows what to keep vs delete before uploading to GitHub.

---

## ✅ KEEP (Essential Files)

### Core Pipeline Scripts
```
✓ 01_data_preparation.py
✓ 02_embeddings_and_features.py
✓ 02b_novelty_cln.py
✓ 02c_relevance_kmnz.py
✓ 03_sae_training.py
✓ 04_feature_selection.py
✓ 05_interpret_features.py
✓ 06_sarkar_analysis.py          (keep for transparency about failed approach)
✓ 06b_sarkar_ensemble.py         (keep for transparency)
✓ 07_paper_analysis.py
```

### Documentation
```
✓ README.md                       (NEW - professional showcase)
✓ LICENSE                         (NEW - MIT license)
✓ .gitignore                      (NEW - excludes data files)
✓ requirements.txt
✓ CLAUDE.md                       (explains project context for Claude Code)
```

### Documentation Folder (docs/)
```
✓ docs/EXECUTION_GUIDE.md         (copied from README_EXECUTION.md)
✓ docs/METHODOLOGY.md             (NEW - technical deep dive)
✓ docs/CHECKPOINT1_VALIDATION.md  (Phase 1 validation results)
✓ docs/OOS_VALIDATION_RESULTS.md  (Out-of-sample validation analysis)
```

### Supporting Scripts
```
✓ setup/                          (keep entire folder)
  ✓ setup/00b_merge_metadata.py
  ✓ setup/00c_fetch_outcomes_wrds.py
  ✓ setup/00d_wrds_data_merge.py

✓ utilities/                      (keep entire folder)
  ✓ utilities/validate_pipeline.py
  ✓ utilities/monitor_training_loss.py
  ✓ utilities/verify_alignment.py
  ✓ utilities/stratified_sampling.py
```

### Paper Outputs (Selective)
```
✓ paper_output/paper_draft.tex
✓ paper_output/references.bib
✓ paper_output/README_PAPER.md
✓ paper_output/*.csv              (regression results, feature tables)
✓ paper_output/CHECKPOINT1_VALIDATION.md   (will be in docs/)
✓ paper_output/OOS_VALIDATION_RESULTS.md   (will be in docs/)
```

---

## ❌ DELETE (Clutter / Development Files)

### Development Documentation (Redundant)
```
✗ BUG_FIX_SUMMARY.md              (incorporated into docs)
✗ CHANGES.md                      (version history not needed)
✗ CLEANUP_PLAN.md                 (development artifact)
✗ CLEANUP_PLAN_v2.md              (development artifact)
✗ EXECUTION_SEQUENCE.md           (superseded by docs/EXECUTION_GUIDE.md)
✗ IMPLEMENTATION_SUMMARY.md       (superseded by docs/METHODOLOGY.md)
✗ IMPROVEMENTS_FROM_SAMPLE_CODE.md
✗ MEMORY_MANAGEMENT.md            (internals, not needed)
✗ NEXT_STEPS.md                   (outdated)
✗ PIPELINE_ARCHITECTURE.md        (redundant with README)
✗ QUICK_REFERENCE.md              (redundant)
✗ README_ANTHROPIC_SAE.md         (reference material, not core)
✗ README_EXECUTION.md             (moved to docs/EXECUTION_GUIDE.md)
✗ REFACTORING_COMPLETE.md         (development log)
✗ REFACTORING_FINAL.md            (development log)
✗ REFACTORING_PLAN.md             (development log)
✗ SAMPLING_STRATEGY_UPDATE.md     (development log)
✗ SARKAR_ALIGNMENT_SUMMARY.md     (redundant with CHECKPOINT1)
✗ SOLUTION_SUMMARY.md             (development log)
✗ CLAUDE.md.tmp.*                 (temporary file)
```

### Proposal Files (Research-Specific)
```
✗ main.tex                        (original proposal, not needed for GitHub)
✗ big_proposal.md                 (original proposal)
✗ small_proposal_1st.md           (original proposal)
```

### Temporary & Log Files
```
✗ *.log                           (all log files)
✗ nul                             (empty file)
✗ temp_extracted.py               (temporary)
✗ *.tmp.*                         (all temp files)
```

### Folders to Delete
```
✗ deprecated/                     (all old code versions)
✗ sample_code/                    (reference code, not yours)
✗ slurm_jobs/                     (HPC-specific, not portable)
✗ __pycache__/                    (Python cache, in .gitignore)
✗ .claude/                        (Claude Code cache, in .gitignore)
✗ papers/                         (reference PDFs, optional)
```

### Paper Output (Large/Generated Files)
```
✗ paper_output/*.png              (large, can regenerate - but keep 1-2 samples)
✗ paper_output/*.pdf              (generated LaTeX output)
✗ paper_output/*.txt              (plain text tables, redundant)
✗ paper_output/ALIGNMENT_ANALYSIS.md       (too detailed, keep summary)
✗ paper_output/ALIGNMENT_SUMMARY.txt       (keep this one, compact)
✗ paper_output/RESULTS_SUMMARY.md          (redundant with CHECKPOINT1)
✗ paper_output/WRITING_GUIDE.md            (internal guide, not needed)
```

---

## 📦 Final Directory Structure for GitHub

```
sae-market-inattention/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── CLAUDE.md
│
├── 01_data_preparation.py
├── 02_embeddings_and_features.py
├── 02b_novelty_cln.py
├── 02c_relevance_kmnz.py
├── 03_sae_training.py
├── 04_feature_selection.py
├── 05_interpret_features.py
├── 06_sarkar_analysis.py
├── 06b_sarkar_ensemble.py
├── 07_paper_analysis.py
│
├── setup/
│   ├── 00b_merge_metadata.py
│   ├── 00c_fetch_outcomes_wrds.py
│   └── 00d_wrds_data_merge.py
│
├── utilities/
│   ├── validate_pipeline.py
│   ├── monitor_training_loss.py
│   ├── verify_alignment.py
│   └── stratified_sampling.py
│
├── docs/
│   ├── EXECUTION_GUIDE.md
│   ├── METHODOLOGY.md
│   ├── CHECKPOINT1_VALIDATION.md
│   └── OOS_VALIDATION_RESULTS.md
│
└── paper_output/
    ├── paper_draft.tex
    ├── references.bib
    ├── README_PAPER.md
    ├── ALIGNMENT_SUMMARY.txt
    ├── feature_coefficients_drift_30d_fdr.csv
    ├── oos_validation_drift_30d.csv
    ├── car_joint_ftest_results.csv
    ├── regression_summary.csv
    └── feature_importance.png (1 sample figure)
```

---

## 🚀 Cleanup Commands

### Step 1: Delete Development Documentation
```bash
cd "C:\Users\ofs4963\Dropbox\Arojects\SAE"

# Delete redundant docs
rm BUG_FIX_SUMMARY.md CHANGES.md CLEANUP_PLAN*.md
rm EXECUTION_SEQUENCE.md IMPLEMENTATION_SUMMARY.md
rm IMPROVEMENTS_FROM_SAMPLE_CODE.md MEMORY_MANAGEMENT.md
rm NEXT_STEPS.md PIPELINE_ARCHITECTURE.md QUICK_REFERENCE.md
rm README_ANTHROPIC_SAE.md README_EXECUTION.md
rm REFACTORING_*.md SAMPLING_STRATEGY_UPDATE.md
rm SARKAR_ALIGNMENT_SUMMARY.md SOLUTION_SUMMARY.md
rm *.tmp.* temp_extracted.py nul

# Delete proposal files
rm main.tex big_proposal.md small_proposal_1st.md

# Delete log files
rm *.log
```

### Step 2: Delete Folders
```bash
# Delete entire folders
rm -rf deprecated/
rm -rf sample_code/
rm -rf slurm_jobs/
rm -rf __pycache__/
rm -rf papers/  # optional - contains reference PDFs
```

### Step 3: Clean Paper Output
```bash
cd paper_output/

# Keep only essentials
# Delete redundant markdown
rm ALIGNMENT_ANALYSIS.md RESULTS_SUMMARY.md WRITING_GUIDE.md

# Delete large generated files (keep 1 sample PNG)
rm r2_comparison.png feature_distributions.png time_trends.png
# Keep feature_importance.png as sample

# Delete plain text tables (LaTeX is enough)
rm *.txt
# Restore ALIGNMENT_SUMMARY.txt (it's compact)
git checkout ALIGNMENT_SUMMARY.txt
```

### Step 4: Verify .gitignore
```bash
# Make sure data/ is ignored
cat .gitignore | grep "data/"

# Output should show:
# data/
# *.parquet
# *.npz
# *.csv (but we keep specific CSVs in paper_output/)
```

---

## ✅ Pre-Upload Checklist

Before `git push`:

1. **Remove sensitive information**:
   - [ ] No API keys or credentials
   - [ ] No absolute paths (C:\Users\...)
   - [ ] No personal email addresses (use placeholder)

2. **Update placeholder info**:
   - [ ] Replace `[Your Name]` in README.md
   - [ ] Replace `yourusername` in GitHub links
   - [ ] Replace `your.email@example.com`

3. **Test README**:
   - [ ] All links work (no 404s)
   - [ ] Code blocks have correct syntax highlighting
   - [ ] Badges/shields render correctly

4. **Verify .gitignore**:
   - [ ] Data files excluded
   - [ ] Large files excluded (>100 MB)
   - [ ] Sensitive files excluded

5. **Final check**:
   - [ ] Total repo size < 100 MB (without data/)
   - [ ] All Python files have docstrings
   - [ ] requirements.txt is up to date

---

## 📊 Expected Repo Stats

After cleanup:
- **Files**: ~30 core files (down from ~70+)
- **Size**: ~5 MB (without data/)
- **Languages**: Python 95%, Markdown 4%, TeX 1%
- **Commits**: Start fresh with clean history

---

## 🎯 GitHub Repository Settings

### Repository Name
`sae-market-inattention` or `reverse-engineer-market-attention`

### Description
"Using sparse autoencoders to discover which corporate disclosure concepts algorithmic traders systematically under-process. Built in Python with temporal validation, FDR correction, and 8 robust features predicting drift."

### Topics/Tags
```
machine-learning
finance
natural-language-processing
sparse-autoencoders
market-efficiency
textual-analysis
sec-filings
rational-inattention
pytorch
econometrics
```

### README Sections (Already Included)
✓ Badge row (Python version, license)
✓ What it does (1-2 sentences)
✓ Key finding (results up front)
✓ The approach (visual pipeline)
✓ Results (tables with numbers)
✓ Technical highlights (what you built/debugged)
✓ Quick start (installation + test run)
✓ Project structure
✓ Methodology deep dive
✓ References
✓ Author info

---

## 🚢 Ready to Ship!

Once cleaned up:
```bash
git init
git add .
git commit -m "Initial commit: SAE market inattention detection pipeline"
git branch -M main
git remote add origin git@github.com:yourusername/sae-market-inattention.git
git push -u origin main
```

Then add to your CV/application:
📎 **GitHub**: github.com/yourusername/sae-market-inattention

---

*This project showcases: hands-on ML implementation, debugging real-world data issues, pivoting when methods fail, and building production-grade research code.*
