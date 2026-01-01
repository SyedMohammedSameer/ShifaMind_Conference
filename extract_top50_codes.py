#!/usr/bin/env python3
"""
Extract Top 50 Most Frequent ICD Codes from MIMIC-IV
Run this ONCE to generate the top 50 codes, then use them in ShifaMind200.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

# Path to MIMIC-IV diagnosis data
MIMIC_DATA_PATH = Path('/content/drive/MyDrive/ShifaMind/mimic_dx_data.csv')

print("="*80)
print("🔍 EXTRACTING TOP 50 MOST FREQUENT ICD CODES FROM MIMIC-IV")
print("="*80)

# Load MIMIC-IV data
print(f"\n📥 Loading data from: {MIMIC_DATA_PATH}")
df = pd.read_csv(MIMIC_DATA_PATH)

print(f"✅ Loaded {len(df)} clinical notes")
print(f"📋 Columns: {list(df.columns)}")

# Extract all ICD code columns (assuming they start with specific patterns like I, J, K, E, etc.)
# MIMIC-IV typically has ICD codes in separate columns
icd_columns = [col for col in df.columns if col not in ['text', 'subject_id', 'hadm_id', 'note_id']]

print(f"\n🔢 Found {len(icd_columns)} potential ICD code columns")

# Count frequency of each diagnosis across all patients
icd_counter = Counter()

for col in icd_columns:
    if col in df.columns:
        # Count positive cases (assuming binary labels)
        positive_count = df[col].sum() if df[col].dtype in [int, float] else (df[col] == 1).sum()
        if positive_count > 0:
            icd_counter[col] = positive_count

# Get top 50 most frequent codes
top_50_codes = [code for code, count in icd_counter.most_common(50)]
top_50_counts = [count for code, count in icd_counter.most_common(50)]

print("\n" + "="*80)
print("📊 TOP 50 MOST FREQUENT ICD CODES")
print("="*80)

for i, (code, count) in enumerate(zip(top_50_codes, top_50_counts), 1):
    percentage = (count / len(df)) * 100
    print(f"{i:2d}. {code:8s}  Count: {count:5d}  ({percentage:5.2f}% of patients)")

# Generate Python code for TARGET_CODES
print("\n" + "="*80)
print("📝 PYTHON CODE FOR ShifaMind200.py")
print("="*80)

print("\n# Top 50 Most Frequent ICD Codes from MIMIC-IV")
print(f"TARGET_CODES = {top_50_codes}")

# Save to file for easy copy-paste
output_file = Path('top50_icd_codes.txt')
with open(output_file, 'w') as f:
    f.write("# Top 50 Most Frequent ICD Codes from MIMIC-IV\n")
    f.write(f"# Extracted from {len(df)} clinical notes\n\n")
    f.write(f"TARGET_CODES = {top_50_codes}\n\n")
    f.write("# Frequency Distribution:\n")
    for i, (code, count) in enumerate(zip(top_50_codes, top_50_counts), 1):
        percentage = (count / len(df)) * 100
        f.write(f"# {i:2d}. {code:8s}  Count: {count:5d}  ({percentage:5.2f}% of patients)\n")

print(f"\n✅ Saved to: {output_file}")
print("\n💡 Copy the TARGET_CODES list and paste it into ShifaMind200.py at line 146")
print("="*80)
