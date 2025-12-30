#!/usr/bin/env python3
"""
================================================================================
MIMIC-IV Data Preprocessing for ShifaMind
================================================================================

This script preprocesses MIMIC-IV data to create the input CSV for ShifaMind.

Required MIMIC-IV tables:
1. mimic-iv-note/note/discharge.csv - Discharge summaries
2. mimic-iv/hosp/diagnoses_icd.csv - ICD-10 diagnosis codes

Output:
- mimic_dx_data.csv with columns: [text, J189, I5023, A419, K8000]

Usage:
    python prepare_mimic_data.py --mimic_note_path /path/to/mimic-iv-note \
                                  --mimic_hosp_path /path/to/mimic-iv/hosp \
                                  --output_path ./mimic_dx_data.csv

================================================================================
"""

import pandas as pd
import argparse
from pathlib import Path
from tqdm.auto import tqdm

# Target ICD-10 codes
TARGET_CODES = {
    'J189': 'Pneumonia, unspecified organism',
    'I5023': 'Acute on chronic systolic heart failure',
    'A419': 'Sepsis, unspecified organism',
    'K8000': 'Calculus of gallbladder with acute cholecystitis'
}

def load_discharge_notes(note_path):
    """Load discharge summaries from MIMIC-IV-Note"""
    discharge_path = Path(note_path) / 'note' / 'discharge.csv'
    print(f"\n📥 Loading discharge summaries from: {discharge_path}")

    if not discharge_path.exists():
        raise FileNotFoundError(f"Discharge notes not found: {discharge_path}")

    df_notes = pd.read_csv(discharge_path)
    print(f"✅ Loaded {len(df_notes):,} discharge summaries")

    # Keep only necessary columns
    df_notes = df_notes[['note_id', 'subject_id', 'hadm_id', 'text']]

    return df_notes

def load_diagnoses(hosp_path):
    """Load ICD-10 diagnoses from MIMIC-IV hosp"""
    diagnoses_path = Path(hosp_path) / 'diagnoses_icd.csv'
    print(f"\n📥 Loading diagnoses from: {diagnoses_path}")

    if not diagnoses_path.exists():
        raise FileNotFoundError(f"Diagnoses file not found: {diagnoses_path}")

    df_dx = pd.read_csv(diagnoses_path)
    print(f"✅ Loaded {len(df_dx):,} diagnosis records")

    # Filter for ICD-10 only
    df_dx = df_dx[df_dx['icd_version'] == 10].copy()
    print(f"   ICD-10 only: {len(df_dx):,} records")

    # Filter for target codes
    df_dx = df_dx[df_dx['icd_code'].isin(TARGET_CODES.keys())].copy()
    print(f"   Target codes only: {len(df_dx):,} records")

    return df_dx[['subject_id', 'hadm_id', 'icd_code']]

def create_labels(df_notes, df_dx):
    """Create binary labels for each target diagnosis"""
    print(f"\n🏷️  Creating binary labels...")

    # Merge notes with diagnoses
    df = df_notes.merge(df_dx, on=['subject_id', 'hadm_id'], how='left')

    # Create binary columns for each target code
    for code in TARGET_CODES.keys():
        df[code] = (df['icd_code'] == code).astype(int)

    # Group by note_id and take max (in case multiple diagnoses per admission)
    label_cols = list(TARGET_CODES.keys())
    df_grouped = df.groupby(['note_id', 'subject_id', 'hadm_id', 'text'])[label_cols].max().reset_index()

    # Remove notes with no target diagnoses
    df_final = df_grouped[df_grouped[label_cols].sum(axis=1) > 0].copy()

    print(f"✅ Created labeled dataset:")
    print(f"   Total notes with target diagnoses: {len(df_final):,}")
    print(f"\n   Label distribution:")
    for code, desc in TARGET_CODES.items():
        count = df_final[code].sum()
        pct = count / len(df_final) * 100
        print(f"   - {code} ({desc}): {count} ({pct:.1f}%)")

    # Keep only necessary columns
    df_final = df_final[['text'] + label_cols]

    return df_final

def clean_text(df):
    """Clean clinical note text"""
    print(f"\n🧹 Cleaning text...")

    # Remove very short notes
    df['text_length'] = df['text'].str.len()
    df = df[df['text_length'] > 100].copy()
    print(f"   Removed short notes (<100 chars): {len(df):,} remaining")

    # Remove duplicates
    df = df.drop_duplicates(subset=['text'])
    print(f"   Removed duplicates: {len(df):,} remaining")

    df = df.drop(columns=['text_length'])

    return df

def main():
    parser = argparse.ArgumentParser(description='Preprocess MIMIC-IV data for ShifaMind')
    parser.add_argument('--mimic_note_path', type=str, required=True,
                       help='Path to MIMIC-IV-Note directory')
    parser.add_argument('--mimic_hosp_path', type=str, required=True,
                       help='Path to MIMIC-IV hosp directory')
    parser.add_argument('--output_path', type=str, default='./mimic_dx_data.csv',
                       help='Output CSV path')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to keep (optional)')

    args = parser.parse_args()

    print("="*80)
    print("🏥 MIMIC-IV DATA PREPROCESSING FOR SHIFAMIND")
    print("="*80)

    # Load data
    df_notes = load_discharge_notes(args.mimic_note_path)
    df_dx = load_diagnoses(args.mimic_hosp_path)

    # Create labels
    df_labeled = create_labels(df_notes, df_dx)

    # Clean text
    df_clean = clean_text(df_labeled)

    # Limit samples if requested
    if args.max_samples:
        df_clean = df_clean.sample(n=min(args.max_samples, len(df_clean)), random_state=42)
        print(f"\n✂️  Sampled {len(df_clean):,} random samples")

    # Save
    output_path = Path(args.output_path)
    print(f"\n💾 Saving to: {output_path}")
    df_clean.to_csv(output_path, index=False)
    print(f"✅ Saved {len(df_clean):,} samples")

    print("\n" + "="*80)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*80)
    print(f"\nYou can now use this file with ShifaMind:")
    print(f"   Update MIMIC_DATA_PATH in phase1_v2.py to: {output_path.absolute()}")
    print(f"\nOr move it to the expected location:")
    print(f"   mv {output_path} /home/user/ShifaMind_Conference/mimic_dx_data.csv")

if __name__ == '__main__':
    main()
