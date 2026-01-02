#!/usr/bin/env python3
"""
Prepare MIMIC-IV Data for 50 ICD-10 Codes
==========================================
Processes raw MIMIC-IV files to create mimic_dx_data.csv with:
- Clinical discharge notes (text)
- Binary labels for top 50 ICD-10 diagnosis codes
- Patient/admission metadata

Input:
- MIMIC-IV-Note: discharge.csv.gz (clinical notes)
- MIMIC-IV: diagnoses_icd.csv.gz (ICD codes)
- MIMIC-IV: d_icd_diagnoses.csv.gz (ICD descriptions)

Output:
- mimic_dx_data.csv with columns: text, subject_id, hadm_id, [50 ICD code columns]
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import gzip
from tqdm.auto import tqdm

print("="*80)
print("🔧 PREPARING MIMIC-IV DATA FOR 50 ICD-10 CODES")
print("="*80)

# ============================================================================
# PATHS
# ============================================================================

BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')

# Input paths
MIMIC_NOTE_PATH = BASE_PATH / '01_Raw_Datasets/Extracted/mimic-iv-note-2.2/note'
MIMIC_HOSP_PATH = BASE_PATH / '01_Raw_Datasets/Extracted/mimic-iv-3.1/mimic-iv-3.1/hosp'

# Output path
OUTPUT_PATH = BASE_PATH / 'mimic_dx_data.csv'

print(f"\n📁 Paths:")
print(f"   Notes:      {MIMIC_NOTE_PATH}")
print(f"   Hospital:   {MIMIC_HOSP_PATH}")
print(f"   Output:     {OUTPUT_PATH}")

# ============================================================================
# STEP 1: LOAD DISCHARGE NOTES
# ============================================================================

print("\n" + "="*80)
print("📝 STEP 1: LOADING DISCHARGE NOTES")
print("="*80)

discharge_file = MIMIC_NOTE_PATH / 'discharge.csv.gz'

if not discharge_file.exists():
    raise FileNotFoundError(f"❌ Discharge notes not found: {discharge_file}")

print(f"\n📥 Loading: {discharge_file.name}")
df_notes = pd.read_csv(discharge_file, compression='gzip')

print(f"✅ Loaded {len(df_notes):,} discharge notes")
print(f"📋 Columns: {list(df_notes.columns)}")

# Keep only necessary columns
df_notes = df_notes[['subject_id', 'hadm_id', 'text']].copy()

# Remove notes without text
df_notes = df_notes.dropna(subset=['text'])
df_notes = df_notes[df_notes['text'].str.strip() != '']

print(f"✅ After filtering: {len(df_notes):,} notes with valid text")

# ============================================================================
# STEP 2: LOAD ICD DIAGNOSES
# ============================================================================

print("\n" + "="*80)
print("🏥 STEP 2: LOADING ICD DIAGNOSES")
print("="*80)

diagnoses_file = MIMIC_HOSP_PATH / 'diagnoses_icd.csv.gz'

if not diagnoses_file.exists():
    raise FileNotFoundError(f"❌ Diagnoses not found: {diagnoses_file}")

print(f"\n📥 Loading: {diagnoses_file.name}")
df_diagnoses = pd.read_csv(diagnoses_file, compression='gzip')

print(f"✅ Loaded {len(df_diagnoses):,} diagnosis records")
print(f"📋 Columns: {list(df_diagnoses.columns)}")

# Filter for ICD-10 codes only (icd_version == 10)
if 'icd_version' in df_diagnoses.columns:
    df_diagnoses_icd10 = df_diagnoses[df_diagnoses['icd_version'] == 10].copy()
    print(f"✅ Filtered to {len(df_diagnoses_icd10):,} ICD-10 diagnosis records")
else:
    df_diagnoses_icd10 = df_diagnoses.copy()
    print(f"⚠️  No icd_version column - assuming all are ICD-10")

# ============================================================================
# STEP 3: FIND TOP 50 MOST FREQUENT ICD-10 CODES
# ============================================================================

print("\n" + "="*80)
print("📊 STEP 3: FINDING TOP 50 ICD-10 CODES")
print("="*80)

# Count frequency of each ICD code
icd_counter = Counter(df_diagnoses_icd10['icd_code'])

print(f"\n🔢 Found {len(icd_counter)} unique ICD-10 codes")

# Get top 50
top_50 = icd_counter.most_common(50)
top_50_codes = [code for code, count in top_50]

print(f"\n📋 Top 50 Most Frequent ICD-10 Codes:")
print(f"{'Rank':<6} {'Code':<10} {'Frequency':<12} {'Percentage'}")
print("-" * 50)
total_diagnoses = len(df_diagnoses_icd10)
for i, (code, count) in enumerate(top_50, 1):
    pct = (count / total_diagnoses) * 100
    print(f"{i:<6} {code:<10} {count:<12,} {pct:>6.2f}%")

# ============================================================================
# STEP 4: CREATE BINARY LABEL MATRIX
# ============================================================================

print("\n" + "="*80)
print("🏷️  STEP 4: CREATING BINARY LABELS")
print("="*80)

# Filter diagnoses to top 50 codes only
df_diagnoses_top50 = df_diagnoses_icd10[
    df_diagnoses_icd10['icd_code'].isin(top_50_codes)
].copy()

print(f"✅ Filtered to {len(df_diagnoses_top50):,} diagnosis records for top 50 codes")

# Group by hadm_id (hospital admission) and create binary labels
print("\n🔄 Creating binary label matrix...")

# Initialize label dict
admission_labels = {}

for hadm_id, group in tqdm(df_diagnoses_top50.groupby('hadm_id'), desc="Processing admissions"):
    # Get all ICD codes for this admission
    codes = set(group['icd_code'])

    # Create binary vector for top 50 codes
    labels = {code: 1 if code in codes else 0 for code in top_50_codes}
    admission_labels[hadm_id] = labels

print(f"✅ Created labels for {len(admission_labels):,} admissions")

# ============================================================================
# STEP 5: MERGE NOTES WITH LABELS
# ============================================================================

print("\n" + "="*80)
print("🔗 STEP 5: MERGING NOTES WITH LABELS")
print("="*80)

# Convert label dict to DataFrame
df_labels = pd.DataFrame.from_dict(admission_labels, orient='index')
df_labels['hadm_id'] = df_labels.index
df_labels = df_labels.reset_index(drop=True)

print(f"📋 Label matrix shape: {df_labels.shape}")
print(f"   Admissions: {len(df_labels)}")
print(f"   Codes: {len(top_50_codes)}")

# Merge notes with labels
print("\n🔄 Merging...")
df_merged = df_notes.merge(df_labels, on='hadm_id', how='inner')

print(f"✅ Merged: {len(df_merged):,} notes with labels")

# ============================================================================
# STEP 6: REMOVE CODES WITH LOW FREQUENCY & QUALITY CHECKS
# ============================================================================

print("\n" + "="*80)
print("🔍 STEP 6: REMOVING LOW-FREQUENCY CODES")
print("="*80)

# Check label distribution
label_sums = df_merged[top_50_codes].sum()

# Remove codes with less than 100 samples (too rare to learn)
MIN_SAMPLES = 100
codes_to_keep = label_sums[label_sums >= MIN_SAMPLES].index.tolist()
codes_removed = [code for code in top_50_codes if code not in codes_to_keep]

if codes_removed:
    print(f"\n⚠️  Removing {len(codes_removed)} codes with < {MIN_SAMPLES} samples:")
    for code in codes_removed:
        count = label_sums[code]
        print(f"   • {code}: {count:.0f} samples (too rare)")

    top_50_codes = codes_to_keep
    print(f"\n✅ Kept {len(top_50_codes)} codes with sufficient samples")
else:
    print(f"✅ All 50 codes have >= {MIN_SAMPLES} samples")

# Update label sums after filtering
label_sums = df_merged[top_50_codes].sum()

print(f"\n📊 Label Statistics:")
print(f"   Total samples: {len(df_merged):,}")
print(f"   Total codes: {len(top_50_codes)}")
print(f"   Avg labels per sample: {df_merged[top_50_codes].sum(axis=1).mean():.2f}")
print(f"   Min labels per sample: {df_merged[top_50_codes].sum(axis=1).min():.0f}")
print(f"   Max labels per sample: {df_merged[top_50_codes].sum(axis=1).max():.0f}")

print(f"\n📈 Top 10 Code Frequencies in Final Dataset:")
for i, (code, count) in enumerate(label_sums.sort_values(ascending=False).head(10).items(), 1):
    pct = (count / len(df_merged)) * 100
    print(f"   {i:2d}. {code:8s}: {count:5.0f} ({pct:5.2f}%)")

# ============================================================================
# STEP 7: SAVE OUTPUT
# ============================================================================

print("\n" + "="*80)
print("💾 STEP 7: SAVING OUTPUT")
print("="*80)

# Ensure correct column order: text, subject_id, hadm_id, then ICD codes
columns_ordered = ['text', 'subject_id', 'hadm_id'] + top_50_codes
df_final = df_merged[columns_ordered].copy()

print(f"\n📝 Final dataset:")
print(f"   Shape: {df_final.shape}")
print(f"   Columns: {len(df_final.columns)}")
print(f"     - text")
print(f"     - subject_id, hadm_id")
print(f"     - {len(top_50_codes)} ICD code columns")

print(f"\n💾 Saving to: {OUTPUT_PATH}")
df_final.to_csv(OUTPUT_PATH, index=False)

file_size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
print(f"✅ Saved! File size: {file_size_mb:.1f} MB")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✅ DATA PREPARATION COMPLETE!")
print("="*80)

print(f"\n📦 Output Summary:")
print(f"   File:     {OUTPUT_PATH}")
print(f"   Samples:  {len(df_final):,}")
print(f"   Features: {len(df_final.columns)}")
print(f"   Size:     {file_size_mb:.1f} MB")

print(f"\n🎯 Top 50 ICD-10 Codes:")
print(f"   {', '.join(top_50_codes[:10])}, ...")

print(f"\n💡 Next Steps:")
print(f"   1. Run ShifaMind200p1.py to train Phase 1")
print(f"   2. Data will be automatically loaded from: {OUTPUT_PATH}")

print("\n" + "="*80)

# Save top 50 codes for reference
codes_file = BASE_PATH / 'top50_icd_codes.txt'
with open(codes_file, 'w') as f:
    f.write("# Top 50 Most Frequent ICD-10 Codes in MIMIC-IV\n\n")
    f.write(f"TARGET_CODES = {top_50_codes}\n\n")
    f.write("# Frequency Distribution:\n")
    for i, (code, count) in enumerate(top_50, 1):
        pct = (count / total_diagnoses) * 100
        f.write(f"# {i:2d}. {code:8s}  Count: {count:6,}  ({pct:5.2f}%)\n")

print(f"📄 Saved code list to: {codes_file}")
print("="*80)
