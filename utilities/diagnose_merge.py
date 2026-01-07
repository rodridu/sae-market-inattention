"""
Quick diagnostic to identify accession number format mismatch
"""
import pandas as pd
import zipfile
import os
from pathlib import Path

# Paths
DATA_DIR = Path(r"C:\Users\ofs4963\Dropbox\Arojects\esg_dei\data")
METADATA_FILE = DATA_DIR / "filing_metadata_v2.csv"
ZIP_FILE = DATA_DIR / "10K_item1.zip"

print("="*60)
print("ACCESSION NUMBER FORMAT DIAGNOSIS")
print("="*60)

# 1. Check metadata format
print("\n1. METADATA accession_number format:")
metadata = pd.read_csv(METADATA_FILE)
print(f"   Total metadata records: {len(metadata):,}")
print(f"   Column type: {metadata['accession_number'].dtype}")
print(f"   Sample values (first 5):")
for val in metadata['accession_number'].head(5):
    print(f"     '{val}'")

# After conversion (current code)
metadata['accession_number'] = metadata['accession_number'].astype(str).str.replace('.0', '', regex=False)
print(f"\n   After str conversion & .0 removal:")
for val in metadata['accession_number'].head(5):
    print(f"     '{val}'")

# 2. Check zip file format
print("\n2. ZIP FILE accession_number format:")
with zipfile.ZipFile(ZIP_FILE, 'r') as zf:
    txt_files = [f for f in zf.namelist() if f.endswith('.txt')][:5]
    print(f"   Sample filenames (first 5):")
    for filepath in txt_files:
        filename = os.path.basename(filepath)
        accession = filename.replace('.txt', '')
        print(f"     Path: {filepath}")
        print(f"     Accession: '{accession}'")
        print()

# 3. Check for any matches
print("\n3. CHECKING FOR MATCHES:")
with zipfile.ZipFile(ZIP_FILE, 'r') as zf:
    txt_files = [f for f in zf.namelist() if f.endswith('.txt')][:100]
    zip_accessions = [os.path.basename(f).replace('.txt', '') for f in txt_files]

metadata_accessions = set(metadata['accession_number'].values)
zip_accessions_set = set(zip_accessions)

matches = metadata_accessions.intersection(zip_accessions_set)
print(f"   Metadata unique accessions (total): {len(metadata_accessions):,}")
print(f"   ZIP accessions (sample of 100): {len(zip_accessions_set)}")
print(f"   Matches in sample: {len(matches)}")

if len(matches) > 0:
    print(f"\n   Example matches:")
    for val in list(matches)[:3]:
        print(f"     '{val}'")
else:
    print("\n   ❌ NO MATCHES FOUND!")
    print("\n   Trying to identify the pattern difference:")

    # Compare formats
    sample_meta = list(metadata_accessions)[:3]
    sample_zip = zip_accessions[:3]

    print(f"\n   Metadata samples:")
    for val in sample_meta:
        print(f"     '{val}' (length: {len(str(val))})")

    print(f"\n   ZIP samples:")
    for val in sample_zip:
        print(f"     '{val}' (length: {len(val)})")

    # Check if it's just dashes vs underscores or something
    print(f"\n   Checking transformations:")
    meta_val = str(sample_meta[0])
    zip_val = sample_zip[0]

    print(f"     Metadata: '{meta_val}'")
    print(f"     ZIP:      '{zip_val}'")
    print(f"     With dashes removed from META: '{meta_val.replace('-', '')}'")
    print(f"     With dashes removed from ZIP:  '{zip_val.replace('-', '')}'")

print("\n" + "="*60)
