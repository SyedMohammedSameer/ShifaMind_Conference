#!/usr/bin/env python3
"""
ShifaMind 042: TOP-N Filtered Concept Version

CRITICAL FIX:
Adds top-N filtering to reduce concept overload from 9,742 concepts to ~60-150.
Previously, searching for "Fever" returned 500+ concepts (Fever, Q fever, Yellow fever, etc.)
Now we select only the top-15 most relevant concepts per diagnosis.

FILTERING STRATEGY:
- Score each concept by how many search terms it matches
- Keep only top-N highest-scoring concepts per diagnosis
- A concept matching 4 terms ranks higher than one matching 1 term

Author: Mohammed Sameer Syed
Date: November 2025
Version: 042-Filtered
"""

# ============================================================================
# 1. IMPORTS
# ============================================================================

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import json
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict, Counter
import time
import pickle
import math
import re
from datetime import datetime
import matplotlib.pyplot as plt

# Seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Device: {device}")

# ============================================================================
# 2. CONFIGURATION - TARGETED APPROACH
# ============================================================================

# Paths
BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')
MIMIC_NOTES_PATH = BASE_PATH / '01_Raw_Datasets/Extracted/mimic-iv-note-2.2/note'
UMLS_META_PATH = BASE_PATH / '01_Raw_Datasets/Extracted/umls-2025AA-metathesaurus-full/2025AA/META'
ICD10_PATH = BASE_PATH / '01_Raw_Datasets/Extracted/icd10cm-CodesDescriptions-2024'
MIMIC_PATH = BASE_PATH / '01_Raw_Datasets/Extracted/mimic-iv-3.1'
DEMO_PATH = BASE_PATH / '01_Raw_Datasets/Demo_Data'
OUTPUT_PATH = BASE_PATH / '04_Results/experiments/042_filtered_concepts'
CHECKPOINT_PATH = BASE_PATH / '03_Models/checkpoints'

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)

DEMO_MODE = False

# Target diagnoses
TARGET_CODES = ['J189', 'I5023', 'A419', 'K8000']
ICD_DESCRIPTIONS = {
    'J189': 'Pneumonia, unspecified organism',
    'I5023': 'Acute on chronic systolic heart failure',
    'A419': 'Sepsis, unspecified organism',
    'K8000': 'Calculus of gallbladder with acute cholecystitis'
}

# Checkpoints
CHECKPOINT_DIAGNOSIS = CHECKPOINT_PATH / 'shifamind_042_diagnosis.pt'
CHECKPOINT_CONCEPTS = CHECKPOINT_PATH / 'shifamind_042_concepts.pt'
CHECKPOINT_FINAL = CHECKPOINT_PATH / 'shifamind_042_final.pt'

# ✨ REQUIRED MEDICAL TERMS - What we'll search for in UMLS
REQUIRED_MEDICAL_TERMS = {
    'J189': [
        # Core disease
        'Pneumonia', 'Lung infection', 'Respiratory infection',
        # Primary symptoms
        'Fever', 'Cough', 'Dyspnea', 'Shortness of breath',
        # Physical findings
        'Crackles', 'Rales', 'Rhonchi', 'Decreased breath sounds',
        # Labs/vitals
        'Tachypnea', 'Hypoxia', 'Leukocytosis', 'Elevated white blood cell',
        # Imaging
        'Pulmonary infiltrate', 'Lung consolidation',
        # Complications
        'Respiratory distress', 'Hypoxemia', 'Sputum production'
    ],
    'I5023': [
        # Core disease
        'Heart failure', 'Cardiac failure', 'Congestive heart failure', 'Cardiomyopathy',
        # Primary symptoms
        'Dyspnea', 'Shortness of breath', 'Orthopnea', 'Paroxysmal nocturnal dyspnea',
        # Physical findings
        'Edema', 'Swelling', 'Jugular venous distension', 'Pulmonary edema',
        # Cardiac signs
        'S3 gallop', 'Cardiomegaly', 'Pleural effusion',
        # Labs
        'Elevated BNP', 'B-type natriuretic peptide',
        # Other
        'Fatigue', 'Weakness', 'Pulmonary congestion'
    ],
    'A419': [
        # Core disease
        'Sepsis', 'Septicemia', 'Bacteremia', 'Systemic infection',
        # SIRS criteria
        'Fever', 'Hypothermia', 'Tachycardia', 'Tachypnea',
        # Hemodynamics
        'Hypotension', 'Shock', 'Septic shock',
        # Mental status
        'Confusion', 'Altered mental status', 'Delirium',
        # Labs
        'Leukocytosis', 'Leukopenia', 'Lactic acidosis', 'Elevated lactate',
        # Organ dysfunction
        'Organ failure', 'Multi-organ dysfunction', 'Acute kidney injury'
    ],
    'K8000': [
        # Core disease
        'Cholecystitis', 'Gallbladder inflammation', 'Acute cholecystitis',
        'Gallstones', 'Cholelithiasis',
        # Primary symptom
        'Abdominal pain', 'Right upper quadrant pain', 'Biliary colic',
        # Physical findings
        'Murphy sign', 'Abdominal tenderness',
        # Associated symptoms
        'Fever', 'Nausea', 'Vomiting',
        # Labs
        'Leukocytosis', 'Elevated white blood cell count',
        # Imaging
        'Gallbladder wall thickening', 'Pericholecystic fluid'
    ]
}

# Keywords for post-processing filter
DIAGNOSIS_KEYWORDS = {
    'J189': ['pneumonia', 'lung', 'respiratory', 'infection', 'infiltrate', 'fever', 
             'cough', 'dyspnea', 'crackles', 'sputum'],
    'I5023': ['heart', 'cardiac', 'failure', 'cardiomyopathy', 'edema', 'dyspnea', 
              'orthopnea', 'congestion'],
    'A419': ['sepsis', 'septicemia', 'bacteremia', 'infection', 'fever', 'hypotension',
             'shock', 'confusion', 'lactate'],
    'K8000': ['cholecystitis', 'gallbladder', 'gallstone', 'abdominal', 'pain', 
              'murphy', 'fever', 'nausea']
}

print("="*80)
print("SHIFAMIND 042: TOP-N FILTERED CONCEPT VERSION")
print("="*80)
print(f"\n📁 Output Directory: {OUTPUT_PATH}")
print(f"📁 Checkpoint Directory: {CHECKPOINT_PATH}")
print(f"🎯 Approach: TARGETED concept search (not random sampling)")
print(f"📋 Required terms: {sum(len(v) for v in REQUIRED_MEDICAL_TERMS.values())} medical terms")

# ============================================================================
# 3. DATA VALIDATION
# ============================================================================

def validate_data_structure():
    """Validate all required data is present"""
    
    print("\n" + "="*70)
    print("DATA VALIDATION")
    print("="*70)
    
    checks = {
        'MIMIC Notes': MIMIC_NOTES_PATH / 'discharge.csv.gz',
        'UMLS MRCONSO': UMLS_META_PATH / 'MRCONSO.RRF',
        'UMLS MRSTY': UMLS_META_PATH / 'MRSTY.RRF',
        'MIMIC Diagnoses': MIMIC_PATH / 'mimic-iv-3.1/hosp/diagnoses_icd.csv.gz',
        'Output Directory': OUTPUT_PATH,
        'Checkpoint Directory': CHECKPOINT_PATH
    }
    
    all_valid = True
    for name, path in checks.items():
        if path.exists():
            if path.is_file():
                size_mb = path.stat().st_size / (1024**2)
                print(f"   ✅ {name}: {size_mb:.1f} MB")
            else:
                print(f"   ✅ {name}: exists")
        else:
            print(f"   ❌ {name}: NOT FOUND at {path}")
            all_valid = False
    
    if not all_valid:
        raise FileNotFoundError("Missing required data files!")
    
    print(f"\n✅ All data validation checks passed")
    return True

validate_data_structure()

# ============================================================================
# 4. TARGETED UMLS LOADER
# ============================================================================

class TargetedUMLSLoader:
    """Load ONLY the medical concepts we actually need"""
    
    def __init__(self, umls_path: Path):
        self.umls_path = umls_path
        self.mrconso_path = umls_path / 'MRCONSO.RRF'
        self.mrsty_path = umls_path / 'MRSTY.RRF'
        self.mrdef_path = umls_path / 'MRDEF.RRF'
    
    def load_specific_concepts(self, required_terms: Dict[str, List[str]]):
        """Search UMLS for specific medical terms"""
        
        print("\n" + "="*70)
        print("TARGETED UMLS CONCEPT LOADING")
        print("="*70)
        
        # Flatten all required terms
        all_terms_flat = []
        for dx_code, terms_list in required_terms.items():
            all_terms_flat.extend(terms_list)
        
        # Create normalized search terms
        search_terms = set([t.lower().strip() for t in all_terms_flat])
        
        print(f"\n🎯 Searching UMLS for {len(search_terms)} specific medical terms:")
        for dx_code, terms in required_terms.items():
            print(f"   {dx_code}: {len(terms)} terms")
        
        # Search MRCONSO
        found_concepts = {}
        term_to_cuis = defaultdict(list)
        
        print("\n📖 Scanning MRCONSO (17M+ entries)...")
        print("   This will take ~30-60 seconds...")
        
        with open(self.mrconso_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in tqdm(f, desc="  Searching"):
                fields = line.strip().split('|')
                if len(fields) < 15:
                    continue
                
                cui, lang, sab, code, term = fields[0], fields[1], fields[11], fields[13], fields[14]
                
                # Only English from trusted sources
                if lang != 'ENG':
                    continue
                if sab not in ['SNOMEDCT_US', 'ICD10CM', 'MSH', 'NCI', 'MEDLINEPLUS']:
                    continue
                
                term_lower = term.lower().strip()
                
                # Check if this UMLS term matches any of our search terms
                matched_search_term = None
                for search_term in search_terms:
                    # Exact match or contains match
                    if search_term == term_lower or search_term in term_lower or term_lower in search_term:
                        matched_search_term = search_term
                        break
                
                if matched_search_term:
                    # Store concept
                    if cui not in found_concepts:
                        found_concepts[cui] = {
                            'cui': cui,
                            'preferred_name': term,
                            'terms': [term],
                            'sources': {sab: [code]},
                            'semantic_types': [],
                            'definition': ''
                        }
                    else:
                        # Add synonym
                        if term not in found_concepts[cui]['terms']:
                            found_concepts[cui]['terms'].append(term)
                        if sab not in found_concepts[cui]['sources']:
                            found_concepts[cui]['sources'][sab] = []
                        if code and code not in found_concepts[cui]['sources'][sab]:
                            found_concepts[cui]['sources'][sab].append(code)
                    
                    # Map search term to CUI
                    if cui not in term_to_cuis[matched_search_term]:
                        term_to_cuis[matched_search_term].append(cui)
        
        print(f"\n  ✅ Found {len(found_concepts)} unique concepts")
        
        # Show coverage per diagnosis
        print("\n  📊 Coverage per diagnosis:")
        dx_coverage = {}
        for dx_code, terms_list in required_terms.items():
            found_for_dx = set()
            for term in terms_list:
                term_lower = term.lower().strip()
                if term_lower in term_to_cuis:
                    found_for_dx.update(term_to_cuis[term_lower])
            dx_coverage[dx_code] = len(found_for_dx)
            print(f"    {dx_code}: {len(found_for_dx)} concepts ({len(terms_list)} terms searched)")
        
        # Load semantic types
        print("\n📋 Loading semantic types...")
        cui_to_types = self._load_semantic_types(set(found_concepts.keys()))
        
        for cui, types in cui_to_types.items():
            if cui in found_concepts:
                found_concepts[cui]['semantic_types'] = types
        
        print(f"  ✅ Added semantic types for {len(cui_to_types)} concepts")
        
        # Load definitions
        print("\n📖 Loading definitions...")
        definitions_added = self._load_definitions(found_concepts)
        print(f"  ✅ Added {definitions_added} definitions")
        
        return found_concepts, term_to_cuis, dx_coverage
    
    def _load_semantic_types(self, target_cuis: Set[str]) -> Dict[str, List[str]]:
        """Load semantic types only for found CUIs"""
        cui_to_types = defaultdict(list)
        
        with open(self.mrsty_path, 'r', encoding='utf-8') as f:
            for line in f:
                fields = line.strip().split('|')
                if len(fields) >= 2:
                    cui = fields[0]
                    if cui in target_cuis:
                        cui_to_types[cui].append(fields[1])
        
        return cui_to_types
    
    def _load_definitions(self, concepts: Dict) -> int:
        """Load definitions for found concepts"""
        if not self.mrdef_path.exists():
            return 0
        
        definitions_added = 0
        
        with open(self.mrdef_path, 'r', encoding='utf-8') as f:
            for line in f:
                fields = line.strip().split('|')
                if len(fields) >= 6:
                    cui, definition = fields[0], fields[5]
                    
                    if cui in concepts and definition:
                        if not concepts[cui]['definition']:
                            concepts[cui]['definition'] = definition
                            definitions_added += 1
        
        return definitions_added

# ============================================================================
# 4B. TOP-N CONCEPT FILTER
# ============================================================================

def filter_to_top_concepts_per_diagnosis(found_concepts, term_to_cuis, required_terms, top_n=15):
    """
    Filter to top-N most relevant concepts per diagnosis

    Strategy:
    - Score each concept by how many search terms it matches
    - For each diagnosis, keep only top-N highest-scoring concepts
    - A concept matching 4 terms ranks higher than one matching 1 term

    Args:
        found_concepts: All concepts found from UMLS search
        term_to_cuis: Mapping from search terms to CUIs
        required_terms: Dict[diagnosis_code] -> List[search_terms]
        top_n: Number of concepts to keep per diagnosis (default: 15)

    Returns:
        filtered_concepts: Dict of filtered concepts
        filtered_term_to_cuis: Updated term_to_cuis mapping
    """
    from collections import Counter

    print(f"\n🔍 Filtering to top-{top_n} concepts per diagnosis...")

    # Build diagnosis-specific concept scores
    diagnosis_concept_scores = {}

    for dx_code, terms_list in required_terms.items():
        concept_scores = Counter()

        # For each search term, give +1 score to matching concepts
        for term in terms_list:
            term_lower = term.lower().strip()
            if term_lower in term_to_cuis:
                for cui in term_to_cuis[term_lower]:
                    concept_scores[cui] += 1

        diagnosis_concept_scores[dx_code] = concept_scores

    # Select top-N per diagnosis
    filtered_whitelist = {}
    all_kept_cuis = set()

    for dx_code, concept_scores in diagnosis_concept_scores.items():
        # Get top-N by score
        top_concepts = [cui for cui, score in concept_scores.most_common(top_n)]
        filtered_whitelist[dx_code] = top_concepts
        all_kept_cuis.update(top_concepts)

        print(f"  {dx_code}: {len(top_concepts)} concepts (was: {len(concept_scores)})")

    # Filter found_concepts to only include kept CUIs
    filtered_concepts = {
        cui: info for cui, info in found_concepts.items()
        if cui in all_kept_cuis
    }

    # Filter term_to_cuis
    filtered_term_to_cuis = {}
    for term, cuis in term_to_cuis.items():
        filtered_cuis = [cui for cui in cuis if cui in all_kept_cuis]
        if filtered_cuis:
            filtered_term_to_cuis[term] = filtered_cuis

    print(f"\n  ✅ Filtered from {len(found_concepts)} to {len(filtered_concepts)} concepts")
    print(f"  ✅ Expected labels per sample: ~{len(all_kept_cuis) / 4:.0f}-{len(all_kept_cuis) / 2:.0f}")

    return filtered_concepts, filtered_term_to_cuis

# ============================================================================
# 5. LOAD TARGETED CONCEPTS
# ============================================================================

print("\n" + "="*70)
print("LOADING TARGETED CONCEPTS FROM UMLS")
print("="*70)

targeted_loader = TargetedUMLSLoader(UMLS_META_PATH)
umls_concepts_raw, term_to_cuis_raw, dx_coverage = targeted_loader.load_specific_concepts(REQUIRED_MEDICAL_TERMS)

# ✨ CRITICAL FIX: Filter to top-N concepts per diagnosis
umls_concepts, term_to_cuis = filter_to_top_concepts_per_diagnosis(
    umls_concepts_raw,
    term_to_cuis_raw,
    REQUIRED_MEDICAL_TERMS,
    top_n=15  # 15 concepts per diagnosis = ~60 total
)

print(f"\n✅ TARGETED LOADING COMPLETE (with top-N filtering):")
print(f"   Total concepts loaded: {len(umls_concepts)}")
print(f"   Strategy: Top-15 concepts per diagnosis based on term match count")

# Build ICD10 to CUI mapping from loaded concepts
icd10_to_cui = defaultdict(list)
for cui, info in umls_concepts.items():
    if 'ICD10CM' in info['sources']:
        for code in info['sources']['ICD10CM']:
            icd10_to_cui[code].append(cui)

print(f"   ICD10 mappings: {len(icd10_to_cui)}")

# ============================================================================
# 6. LOAD MIMIC-IV DATA
# ============================================================================

print("\n" + "="*70)
print("LOADING MIMIC-IV DATA")
print("="*70)

class MIMICLoader:
    """MIMIC-IV data loader"""
    
    def __init__(self, mimic_path: Path, notes_path: Path):
        self.mimic_path = mimic_path
        self.hosp_path = mimic_path / 'mimic-iv-3.1/hosp'
        self.notes_path = notes_path
    
    def load_diagnoses(self) -> pd.DataFrame:
        diag_path = self.hosp_path / 'diagnoses_icd.csv.gz'
        return pd.read_csv(diag_path, compression='gzip')
    
    def load_admissions(self) -> pd.DataFrame:
        adm_path = self.hosp_path / 'admissions.csv.gz'
        return pd.read_csv(adm_path, compression='gzip')
    
    def load_discharge_notes(self) -> pd.DataFrame:
        discharge_path = self.notes_path / 'discharge.csv.gz'
        return pd.read_csv(discharge_path, compression='gzip')

# Load ICD-10 descriptions
def load_icd10_descriptions(icd_path: Path) -> Dict[str, str]:
    codes_file = icd_path / 'icd10cm-codes-2024.txt'
    descriptions = {}
    
    if not codes_file.exists():
        print(f"   ⚠️  ICD-10 codes file not found")
        return descriptions
    
    with open(codes_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split(None, 1)
                if len(parts) == 2:
                    descriptions[parts[0]] = parts[1]
    
    return descriptions

icd_descriptions = load_icd10_descriptions(ICD10_PATH)
print(f"✅ Loaded {len(icd_descriptions)} ICD-10 descriptions")

# Load MIMIC-IV
mimic_loader = MIMICLoader(MIMIC_PATH, MIMIC_NOTES_PATH)
df_diag = mimic_loader.load_diagnoses()
df_adm = mimic_loader.load_admissions()
df_notes = mimic_loader.load_discharge_notes()

print(f"✅ Loaded MIMIC-IV data:")
print(f"   Diagnoses: {len(df_diag)}")
print(f"   Notes: {len(df_notes)}")

# Prepare dataset
def prepare_dataset(df_diag, df_adm, df_notes, icd_descriptions, target_codes, max_per_code=3000):
    print("\n🔧 Preparing dataset...")
    
    df_diag = df_diag[df_diag['icd_version'] == 10].copy()
    df_diag['icd_code'] = df_diag['icd_code'].str.replace('.', '', regex=False)
    
    text_col = 'text'
    if 'text' not in df_notes.columns:
        text_cols = [col for col in df_notes.columns if 'text' in col.lower()]
        if text_cols:
            text_col = text_cols[0]
    
    df_notes_with_diag = df_notes.merge(
        df_diag.groupby('hadm_id')['icd_code'].apply(list).reset_index(),
        on='hadm_id', how='inner'
    )
    
    df = df_notes_with_diag.rename(columns={
        'icd_code': 'icd_codes',
        text_col: 'text'
    })[['hadm_id', 'text', 'icd_codes']].copy()
    
    df['has_target'] = df['icd_codes'].apply(
        lambda codes: any(code in target_codes for code in codes)
    )
    df_filtered = df[df['has_target']].copy()
    
    df_filtered['labels'] = df_filtered['icd_codes'].apply(
        lambda codes: [1 if code in codes else 0 for code in target_codes]
    )
    
    # Balance dataset
    balanced_indices = set()
    for code in target_codes:
        code_indices = df_filtered[
            df_filtered['icd_codes'].apply(lambda x: code in x)
        ].index.tolist()
        n_samples = min(len(code_indices), max_per_code)
        selected = np.random.choice(code_indices, size=n_samples, replace=False)
        balanced_indices.update(selected)
    
    df_final = df_filtered.loc[list(balanced_indices)].reset_index(drop=True)
    df_final = df_final[df_final['text'].notnull()].reset_index(drop=True)
    
    print(f"  ✅ Dataset: {len(df_final)} samples")
    return df_final, target_codes

df_data, target_codes = prepare_dataset(
    df_diag, df_adm, df_notes, icd_descriptions, TARGET_CODES, 
    max_per_code=1000 if DEMO_MODE else 3000
)

# Train/val/test split
def get_primary_diagnosis(label_list):
    for i, val in enumerate(label_list):
        if val == 1:
            return i
    return 0

df_data['primary_dx'] = df_data['labels'].apply(get_primary_diagnosis)

df_train, df_temp = train_test_split(
    df_data, test_size=0.3, random_state=SEED, stratify=df_data['primary_dx']
)
df_val, df_test = train_test_split(
    df_temp, test_size=0.5, random_state=SEED, stratify=df_temp['primary_dx']
)

df_train = df_train.drop('primary_dx', axis=1)
df_val = df_val.drop('primary_dx', axis=1)
df_test = df_test.drop('primary_dx', axis=1)

print(f"\n📊 Split:")
print(f"  Train: {len(df_train)}")
print(f"  Val: {len(df_val)}")
print(f"  Test: {len(df_test)}")

# ============================================================================
# 7. OUTPUT MANAGER
# ============================================================================

class OutputManager:
    """Manages organized output"""
    
    def __init__(self, output_base: Path):
        self.output_base = output_base
        self.output_base.mkdir(parents=True, exist_ok=True)
        
        self.metrics_dir = self.output_base / 'metrics'
        self.figures_dir = self.output_base / 'figures'
        self.models_dir = self.output_base / 'models'
        
        for dir_path in [self.metrics_dir, self.figures_dir, self.models_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def save_metrics(self, metrics: dict, filename: str):
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        def convert_dict(d):
            if isinstance(d, dict):
                return {k: convert_dict(v) for k, v in d.items()}
            elif isinstance(d, (list, tuple)):
                return [convert_dict(item) for item in d]
            else:
                return convert_numpy(d)
        
        converted_metrics = convert_dict(metrics)
        
        filepath = self.metrics_dir / filename
        with open(filepath, 'w') as f:
            json.dump(converted_metrics, f, indent=2)
        print(f"   💾 Saved: {filepath}")
    
    def generate_report(self, results: dict):
        report_path = self.output_base / 'RESULTS_REPORT.md'
        
        with open(report_path, 'w') as f:
            f.write("# ShifaMind 042 - Top-N Filtered Concepts Results\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"- Diagnostic F1: {results.get('diagnostic_f1', 0):.4f}\n")
            if 'citation_metrics' in results:
                f.write(f"- Citation Completeness: {results['citation_metrics'].get('citation_completeness', 0):.1%}\n")
            if 'alignment_metrics' in results:
                f.write(f"- Alignment Score: {results['alignment_metrics'].get('overall_alignment', 0):.1%}\n")
        
        print(f"   📄 Report: {report_path}")

output_manager = OutputManager(OUTPUT_PATH)

# ============================================================================
# 8. FIX 1A: Post-Processing Filter
# ============================================================================

class ConceptPostProcessor:
    """Post-processing filter with realistic thresholds"""
    
    def __init__(self, concept_store, diagnosis_keywords: Dict[str, List[str]]):
        self.concept_store = concept_store
        self.diagnosis_keywords = diagnosis_keywords
    
    def filter_concepts(self, concept_scores: np.ndarray, diagnosis_code: str,
                       threshold: float = 0.5, max_concepts: int = 5) -> List[Dict]:
        keywords = self.diagnosis_keywords.get(diagnosis_code, [])
        
        valid_concepts = self.concept_store.get_concepts_for_diagnosis(diagnosis_code)
        
        keyword_filtered = self._filter_by_keywords(
            concept_scores, valid_concepts, threshold, keywords, max_concepts
        )
        
        if len(keyword_filtered) >= 3:
            return keyword_filtered[:max_concepts]
        
        return self._hybrid_filter(concept_scores, keyword_filtered, max_concepts)
    
    def _filter_by_keywords(self, concept_scores, valid_concepts, threshold, keywords, max_concepts):
        valid_indices = [
            self.concept_store.concept_to_idx[cui]
            for cui in valid_concepts.keys()
            if cui in self.concept_store.concept_to_idx
        ]
        
        results = []
        for idx in valid_indices:
            score = concept_scores[idx]
            if score <= threshold:
                continue
            
            cui = self.concept_store.idx_to_concept[idx]
            if cui not in self.concept_store.concepts:
                continue
            
            concept_info = self.concept_store.concepts[cui]
            terms_text = ' '.join(
                [concept_info['preferred_name']] + concept_info.get('terms', [])
            ).lower()
            
            if any(kw in terms_text for kw in keywords):
                results.append({
                    'idx': idx,
                    'cui': cui,
                    'name': concept_info['preferred_name'],
                    'score': float(score),
                    'semantic_types': concept_info.get('semantic_types', [])
                })
        
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        return results[:max_concepts]
    
    def _hybrid_filter(self, concept_scores, keyword_matches, max_concepts):
        results = list(keyword_matches)
        existing_cuis = set([c['cui'] for c in results])
        
        top_indices = np.argsort(concept_scores)[::-1]
        
        for idx in top_indices:
            if len(results) >= max_concepts:
                break
            if concept_scores[idx] <= 0.3:
                continue
            
            cui = self.concept_store.idx_to_concept.get(idx)
            if not cui or cui in existing_cuis:
                continue
            if cui not in self.concept_store.concepts:
                continue
            
            concept_info = self.concept_store.concepts[cui]
            results.append({
                'idx': idx,
                'cui': cui,
                'name': concept_info['preferred_name'],
                'score': float(concept_scores[idx]),
                'semantic_types': concept_info.get('semantic_types', [])
            })
            existing_cuis.add(cui)
        
        return results[:max_concepts]

print("✅ FIX 1A: Post-Processing Filter defined")

# ============================================================================
# 9. FIX 2A & 3A: Citation and Alignment Metrics
# ============================================================================

class CitationMetrics:
    def __init__(self, min_concepts_threshold: int = 3):
        self.min_concepts_threshold = min_concepts_threshold
    
    def compute_metrics(self, predicted_concepts, diagnosis_predictions):
        total_samples = len(predicted_concepts)
        samples_with_min_concepts = 0
        total_concepts = 0
        
        for concepts in predicted_concepts:
            num_concepts = len(concepts)
            total_concepts += num_concepts
            if num_concepts >= self.min_concepts_threshold:
                samples_with_min_concepts += 1
        
        return {
            'citation_completeness': samples_with_min_concepts / total_samples if total_samples > 0 else 0,
            'avg_concepts_per_sample': total_concepts / total_samples if total_samples > 0 else 0,
            'total_samples': total_samples,
            'samples_with_min_concepts': samples_with_min_concepts
        }

class AlignmentEvaluator:
    def jaccard_similarity(self, set1: Set, set2: Set) -> float:
        if len(set1) == 0 and len(set2) == 0:
            return 1.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0
    
    def compute_alignment(self, predicted_concepts, ground_truth_concepts):
        alignment_scores = []
        
        for pred_concepts, gt_concepts in zip(predicted_concepts, ground_truth_concepts):
            pred_cuis = set([c['cui'] for c in pred_concepts])
            gt_cuis = set(gt_concepts)
            score = self.jaccard_similarity(pred_cuis, gt_cuis)
            alignment_scores.append(score)
        
        return {
            'overall_alignment': np.mean(alignment_scores) if alignment_scores else 0,
            'std_alignment': np.std(alignment_scores) if alignment_scores else 0,
            'alignment_scores': alignment_scores
        }

# ============================================================================
# 10. FIX 4A: Reasoning Chain Generator
# ============================================================================

class ReasoningChainGenerator:
    def __init__(self, icd_descriptions: Dict[str, str]):
        self.icd_descriptions = icd_descriptions
    
    def generate_chain(self, diagnosis_code: str, diagnosis_confidence: float,
                      concepts: List[Dict], evidence_spans=None) -> Dict:
        chain_parts = []
        diagnosis_name = self.icd_descriptions.get(diagnosis_code, diagnosis_code)
        
        chain_parts.append(f"**{diagnosis_name}** ({diagnosis_code})")
        chain_parts.append(f"Confidence: {diagnosis_confidence:.1%}")
        chain_parts.append("")
        
        if concepts:
            chain_parts.append("**Supporting Concepts:**")
            for concept in concepts:
                chain_parts.append(f"- {concept['name']} ({concept['score']:.1%})")
        
        return {
            'explanation': "\n".join(chain_parts),
            'diagnosis_code': diagnosis_code,
            'diagnosis_name': diagnosis_name,
            'confidence': diagnosis_confidence,
            'concepts': concepts
        }

# ============================================================================
# 11. CONCEPT STORE - Using Targeted Concepts
# ============================================================================

print("\n" + "="*70)
print("BUILDING CONCEPT STORE FROM TARGETED CONCEPTS")
print("="*70)

class ConceptStore:
    """Build concept store from targeted UMLS concepts"""
    
    def __init__(self, umls_concepts: Dict, icd_to_cui: Dict):
        self.umls_concepts = umls_concepts
        self.icd_to_cui = icd_to_cui
        self.concepts = {}
        self.concept_to_idx = {}
        self.idx_to_concept = {}
    
    def build_from_targeted(self, target_codes: List[str], diagnosis_keywords: Dict):
        """Build concept store from already-loaded targeted concepts"""
        
        print(f"\n🔬 Building concept store from {len(self.umls_concepts)} targeted concepts...")
        
        # Use ALL targeted concepts - they were specifically searched for
        self.concepts = self.umls_concepts.copy()
        
        concept_list = list(self.concepts.keys())
        self.concept_to_idx = {cui: i for i, cui in enumerate(concept_list)}
        self.idx_to_concept = {i: cui for i, cui in enumerate(concept_list)}
        
        print(f"  ✅ Stored {len(self.concepts)} concepts")
        
        # Build diagnosis-concept mappings
        self._build_diagnosis_mappings(target_codes, diagnosis_keywords)
        
        return self.concepts
    
    def _build_diagnosis_mappings(self, target_codes, diagnosis_keywords):
        """Map diagnoses to relevant concepts"""
        print("\n🔗 Building diagnosis-concept mappings...")
        
        self.diagnosis_to_concepts = {}
        
        for dx_code in target_codes:
            keywords = diagnosis_keywords.get(dx_code, [])
            relevant_indices = []
            
            for cui, info in self.concepts.items():
                concept_idx = self.concept_to_idx[cui]
                terms_text = ' '.join(
                    [info['preferred_name']] + info.get('terms', [])
                ).lower()
                
                if any(kw in terms_text for kw in keywords):
                    relevant_indices.append(concept_idx)
            
            self.diagnosis_to_concepts[dx_code] = relevant_indices
            print(f"  {dx_code}: {len(relevant_indices)} relevant concepts")
    
    def get_concepts_for_diagnosis(self, diagnosis_code: str) -> Dict:
        relevant_indices = self.diagnosis_to_concepts.get(diagnosis_code, [])
        return {
            self.idx_to_concept[idx]: self.concepts[self.idx_to_concept[idx]]
            for idx in relevant_indices
        }
    
    def create_concept_embeddings(self, tokenizer, model, device):
        print("\n🧬 Creating concept embeddings...")
        
        concept_texts = []
        for cui, info in self.concepts.items():
            text = f"{info['preferred_name']}."
            if info.get('definition'):
                text += f" {info['definition'][:150]}"
            concept_texts.append(text)
        
        batch_size = 32
        all_embeddings = []
        
        model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(concept_texts), batch_size), desc="  Encoding"):
                batch = concept_texts[i:i+batch_size]
                encodings = tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=128, return_tensors='pt'
                ).to(device)
                
                outputs = model(**encodings)
                embeddings = outputs.last_hidden_state[:, 0, :]
                all_embeddings.append(embeddings.cpu())
        
        final_embeddings = torch.cat(all_embeddings, dim=0).to(device)
        print(f"  ✅ Created embeddings: {final_embeddings.shape}")
        
        return final_embeddings

# Initialize
print("\nInitializing Bio_ClinicalBERT...")
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
base_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)

concept_store = ConceptStore(umls_concepts, icd10_to_cui)
concept_store.build_from_targeted(TARGET_CODES, DIAGNOSIS_KEYWORDS)

concept_embeddings = concept_store.create_concept_embeddings(tokenizer, base_model, device)

print("\n✅ Concept store complete")

# ============================================================================
# 12. MODEL ARCHITECTURE (CRITICAL - MUST COME BEFORE TRAINING)
# ============================================================================

print("\n" + "="*70)
print("MODEL ARCHITECTURE")
print("="*70)

class EnhancedCrossAttention(nn.Module):
    """Cross-attention between clinical text and medical concepts"""
    
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, hidden_states, concept_embeddings, attention_mask=None):
        batch_size, seq_len, hidden_size = hidden_states.shape
        num_concepts = concept_embeddings.shape[0]
        
        concepts_batch = concept_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        Q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(concepts_batch).view(batch_size, num_concepts, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(concepts_batch).view(batch_size, num_concepts, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        
        gate_input = torch.cat([hidden_states, context], dim=-1)
        gate_values = torch.sigmoid(self.gate(gate_input))
        output = hidden_states + gate_values * context
        output = self.layer_norm(output)
        
        return output, attn_weights.mean(dim=1)


class ShifaMindModel(nn.Module):
    """ShifaMind: Concept-enhanced medical diagnosis prediction"""
    
    def __init__(self, base_model, num_concepts, num_classes, fusion_layers=[9, 11]):
        super().__init__()
        self.base_model = base_model
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.hidden_size = base_model.config.hidden_size
        self.fusion_layers = fusion_layers
        
        self.fusion_modules = nn.ModuleList([
            EnhancedCrossAttention(self.hidden_size, num_heads=8)
            for _ in fusion_layers
        ])
        
        self.diagnosis_head = nn.Linear(self.hidden_size, num_classes)
        self.concept_head = nn.Linear(self.hidden_size, num_concepts)
        self.diagnosis_concept_interaction = nn.Bilinear(num_classes, num_concepts, num_concepts)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, attention_mask, concept_embeddings, return_diagnosis_only=False):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        if return_diagnosis_only:
            cls_hidden = outputs.last_hidden_state[:, 0, :]
            cls_hidden = self.dropout(cls_hidden)
            diagnosis_logits = self.diagnosis_head(cls_hidden)
            return {'logits': diagnosis_logits}
        
        hidden_states = outputs.hidden_states
        current_hidden = hidden_states[-1]
        
        fusion_attentions = []
        for i, fusion_module in enumerate(self.fusion_modules):
            layer_idx = self.fusion_layers[i]
            layer_hidden = hidden_states[layer_idx]
            
            fused_hidden, attn_weights = fusion_module(
                layer_hidden, concept_embeddings, attention_mask
            )
            fusion_attentions.append(attn_weights)
            
            if i == len(self.fusion_modules) - 1:
                current_hidden = fused_hidden
        
        cls_hidden = current_hidden[:, 0, :]
        cls_hidden = self.dropout(cls_hidden)
        
        diagnosis_logits = self.diagnosis_head(cls_hidden)
        concept_logits = self.concept_head(cls_hidden)
        
        diagnosis_probs = torch.sigmoid(diagnosis_logits)
        refined_concept_logits = self.diagnosis_concept_interaction(
            diagnosis_probs, torch.sigmoid(concept_logits)
        )
        
        return {
            'logits': diagnosis_logits,
            'concept_scores': refined_concept_logits,
            'attention_weights': fusion_attentions
        }


class ClinicalDataset(Dataset):
    """Clinical text dataset"""
    
    def __init__(self, texts, labels, tokenizer, max_length=384, concept_labels=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.concept_labels = concept_labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(self.labels[idx])
        }
        
        if self.concept_labels is not None:
            item['concept_labels'] = torch.FloatTensor(self.concept_labels[idx])
        
        return item


class AlignmentLoss(nn.Module):
    """Alignment loss to enforce diagnosis-concept matching"""
    
    def __init__(self, concept_store, target_codes):
        super().__init__()
        self.concept_store = concept_store
        self.target_codes = target_codes
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, diagnosis_logits, concept_scores, diagnosis_labels, concept_labels):
        # Diagnosis loss
        diagnosis_loss = self.bce_loss(diagnosis_logits, diagnosis_labels)
        
        # Concept precision loss
        concept_precision_loss = self.bce_loss(concept_scores, concept_labels)
        
        # Confidence boost
        concept_probs = torch.sigmoid(concept_scores)
        top_k_probs = torch.topk(concept_probs, k=min(12, concept_probs.size(1)), dim=1)[0]
        confidence_loss = -torch.mean(top_k_probs)
        
        # Total loss
        total_loss = (
            0.50 * diagnosis_loss +
            0.25 * concept_precision_loss +
            0.25 * confidence_loss
        )
        
        return total_loss, {
            'diagnosis': diagnosis_loss.item(),
            'concept': concept_precision_loss.item(),
            'confidence': confidence_loss.item()
        }


# Initialize model
shifamind_model = ShifaMindModel(
    base_model=base_model,
    num_concepts=len(concept_store.concepts),
    num_classes=len(TARGET_CODES),
    fusion_layers=[9, 11]
).to(device)

print(f"  Model parameters: {sum(p.numel() for p in shifamind_model.parameters()):,}")
print("✅ Model architecture defined and initialized")

# ============================================================================
# 13. WHITELIST LABELING - Using term_to_cuis from targeted loading
# ============================================================================

print("\n" + "="*70)
print("WHITELIST LABELING FROM TARGETED CONCEPTS")
print("="*70)

class TargetedWhitelistLabeler:
    """Generate labels using concepts found during targeted loading"""
    
    def __init__(self, concept_store, term_to_cuis, required_terms):
        self.concept_store = concept_store
        self.term_to_cuis = term_to_cuis
        self.required_terms = required_terms
        self.whitelist = {}
    
    def build_whitelist(self):
        """Build whitelist from targeted search results"""
        print("\n📊 Building whitelist from search results...")
        
        for dx_code, terms_list in self.required_terms.items():
            whitelist_cuis = set()
            
            for term in terms_list:
                term_lower = term.lower().strip()
                if term_lower in self.term_to_cuis:
                    # Get CUIs that exist in concept store
                    for cui in self.term_to_cuis[term_lower]:
                        if cui in self.concept_store.concepts:
                            whitelist_cuis.add(cui)
            
            self.whitelist[dx_code] = list(whitelist_cuis)
            print(f"  {dx_code}: {len(whitelist_cuis)} concepts")
        
        total = sum(len(v) for v in self.whitelist.values())
        print(f"\n  ✅ Total whitelist concepts: {total}")
        
        return set([cui for cuis in self.whitelist.values() for cui in cuis])
    
    def generate_labels(self, diagnosis_codes: List[str]) -> List[int]:
        activated_cuis = set()
        
        for dx_code in diagnosis_codes:
            if dx_code in self.whitelist:
                activated_cuis.update(self.whitelist[dx_code])
        
        labels = []
        for cui in self.concept_store.concepts.keys():
            labels.append(1 if cui in activated_cuis else 0)
        
        return labels
    
    def generate_dataset_labels(self, df_data, cache_file=None):
        print(f"\n🏷️  Generating labels for {len(df_data)} samples...")
        
        all_labels = []
        for row in tqdm(df_data.itertuples(), total=len(df_data), desc="  Labeling"):
            labels = self.generate_labels(row.icd_codes)
            all_labels.append(labels)
        
        all_labels = np.array(all_labels)
        
        if cache_file:
            with open(cache_file, 'wb') as f:
                pickle.dump(all_labels, f)
        
        avg_labels = all_labels.sum(axis=1).mean()
        print(f"  ✅ Avg labels per sample: {avg_labels:.1f}")
        
        if 5 <= avg_labels <= 15:
            print(f"  ✅ Labels in healthy range!")
        else:
            print(f"  ⚠️  Labels outside expected range (5-15)")
        
        return all_labels


# Build labeler
labeler = TargetedWhitelistLabeler(concept_store, term_to_cuis, REQUIRED_MEDICAL_TERMS)
whitelist_concepts = labeler.build_whitelist()

print(f"\n✅ Whitelist ready: {len(whitelist_concepts)} unique concepts")

# ============================================================================
# 14. GENERATE CONCEPT LABELS (Stage 2)
# ============================================================================

print("\n" + "="*70)
print("GENERATING CONCEPT LABELS")
print("="*70)

train_concept_labels = labeler.generate_dataset_labels(
    df_train,
    cache_file=str(OUTPUT_PATH / 'concept_labels_train.pkl')
)

val_concept_labels = labeler.generate_dataset_labels(
    df_val,
    cache_file=str(OUTPUT_PATH / 'concept_labels_val.pkl')
)

test_concept_labels = labeler.generate_dataset_labels(
    df_test,
    cache_file=str(OUTPUT_PATH / 'concept_labels_test.pkl')
)

print("\n✅ All concept labels generated")

# ============================================================================
# 15. TRAINING STAGE 1: Diagnosis Head
# ============================================================================

print("\n" + "="*70)
print("STAGE 1: DIAGNOSIS HEAD TRAINING")
print("="*70)

if CHECKPOINT_DIAGNOSIS.exists():
    print(f"\n✅ Found existing checkpoint: {CHECKPOINT_DIAGNOSIS}")
    print("Skipping Stage 1 (already trained)")
    checkpoint = torch.load(CHECKPOINT_DIAGNOSIS, map_location=device)
    shifamind_model.load_state_dict(checkpoint['model_state_dict'])
else:
    print("\nPreparing data loaders...")
    train_dataset = ClinicalDataset(
        df_train['text'].tolist(),
        df_train['labels'].tolist(),
        tokenizer
    )
    val_dataset = ClinicalDataset(
        df_val['text'].tolist(),
        df_val['labels'].tolist(),
        tokenizer
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    print("\nStarting Stage 1 training...")
    optimizer = torch.optim.AdamW(shifamind_model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()

    num_training_steps = 3 * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_training_steps // 10,
        num_training_steps=num_training_steps
    )

    best_f1 = 0

    for epoch in range(3):
        print(f"\nEpoch {epoch+1}/3")

        shifamind_model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = shifamind_model(
                input_ids, attention_mask, concept_embeddings,
                return_diagnosis_only=True
            )

            loss = criterion(outputs['logits'], labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(shifamind_model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        shifamind_model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = shifamind_model(
                    input_ids, attention_mask, concept_embeddings,
                    return_diagnosis_only=True
                )

                preds = torch.sigmoid(outputs['logits']).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())

        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        pred_binary = (all_preds > 0.5).astype(int)

        macro_f1 = f1_score(all_labels, pred_binary, average='macro', zero_division=0)

        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Val Macro F1: {macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1

            torch.save({
                'model_state_dict': shifamind_model.state_dict(),
                'num_concepts': len(concept_store.concepts),
                'concept_cuis': list(concept_store.concepts.keys()),
                'concept_names': {cui: info['preferred_name'] for cui, info in concept_store.concepts.items()},
                'concept_embeddings': concept_embeddings,
                'macro_f1': best_f1
            }, CHECKPOINT_DIAGNOSIS)
            print(f"  ✅ Saved checkpoint (F1: {best_f1:.4f})")

    print(f"\n✅ Stage 1 complete. Best F1: {best_f1:.4f}")

torch.cuda.empty_cache()

# ============================================================================
# 16. TRAINING STAGE 3: Concept Head
# ============================================================================

print("\n" + "="*70)
print("STAGE 3: CONCEPT HEAD TRAINING")
print("="*70)

if CHECKPOINT_CONCEPTS.exists():
    print(f"\n✅ Found existing checkpoint: {CHECKPOINT_CONCEPTS}")
    print("Skipping Stage 3 (already trained)")
    checkpoint = torch.load(CHECKPOINT_CONCEPTS, map_location=device)
    shifamind_model.load_state_dict(checkpoint['model_state_dict'])
else:
    print("\nPreparing data loaders with concept labels...")
    train_dataset = ClinicalDataset(
        df_train['text'].tolist(),
        df_train['labels'].tolist(),
        tokenizer,
        concept_labels=train_concept_labels
    )
    val_dataset = ClinicalDataset(
        df_val['text'].tolist(),
        df_val['labels'].tolist(),
        tokenizer,
        concept_labels=val_concept_labels
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    print("\nStarting Stage 3 training...")
    optimizer = torch.optim.AdamW(shifamind_model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()

    best_concept_f1 = 0

    for epoch in range(2):
        print(f"\nEpoch {epoch+1}/2")

        shifamind_model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            concept_labels_batch = batch['concept_labels'].to(device)

            optimizer.zero_grad()

            outputs = shifamind_model(input_ids, attention_mask, concept_embeddings)

            # Concept loss
            concept_loss = criterion(outputs['concept_scores'], concept_labels_batch)

            # Confidence boost
            concept_probs = torch.sigmoid(outputs['concept_scores'])
            top_k_probs = torch.topk(concept_probs, k=min(12, concept_probs.size(1)), dim=1)[0]
            confidence_loss = -torch.mean(top_k_probs)

            loss = 0.7 * concept_loss + 0.3 * confidence_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(shifamind_model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        shifamind_model.eval()
        all_concept_preds = []
        all_concept_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                concept_labels_batch = batch['concept_labels'].to(device)

                outputs = shifamind_model(input_ids, attention_mask, concept_embeddings)

                concept_preds = torch.sigmoid(outputs['concept_scores']).cpu().numpy()
                all_concept_preds.append(concept_preds)
                all_concept_labels.append(concept_labels_batch.cpu().numpy())

        all_concept_preds = np.vstack(all_concept_preds)
        all_concept_labels = np.vstack(all_concept_labels)
        concept_pred_binary = (all_concept_preds > 0.7).astype(int)

        concept_f1 = f1_score(all_concept_labels, concept_pred_binary, average='macro', zero_division=0)

        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Val Concept F1: {concept_f1:.4f}")

        if concept_f1 > best_concept_f1:
            best_concept_f1 = concept_f1

            torch.save({
                'model_state_dict': shifamind_model.state_dict(),
                'num_concepts': len(concept_store.concepts),
                'concept_cuis': list(concept_store.concepts.keys()),
                'concept_names': {cui: info['preferred_name'] for cui, info in concept_store.concepts.items()},
                'concept_embeddings': concept_embeddings,
                'concept_f1': best_concept_f1
            }, CHECKPOINT_CONCEPTS)
            print(f"  ✅ Saved checkpoint (F1: {best_concept_f1:.4f})")

    print(f"\n✅ Stage 3 complete. Best Concept F1: {best_concept_f1:.4f}")

torch.cuda.empty_cache()

# ============================================================================
# 17. TRAINING STAGE 4: Joint Fine-Tuning
# ============================================================================

print("\n" + "="*70)
print("STAGE 4: JOINT FINE-TUNING WITH ALIGNMENT")
print("="*70)

if CHECKPOINT_FINAL.exists():
    print(f"\n✅ Found existing checkpoint: {CHECKPOINT_FINAL}")
    print("Skipping Stage 4 (already trained)")
    checkpoint = torch.load(CHECKPOINT_FINAL, map_location=device)
    shifamind_model.load_state_dict(checkpoint['model_state_dict'])
else:
    print("\nStarting Stage 4 training...")

    optimizer = torch.optim.AdamW(shifamind_model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = AlignmentLoss(concept_store, TARGET_CODES)

    num_training_steps = 3 * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_training_steps // 10,
        num_training_steps=num_training_steps
    )

    best_f1 = 0

    for epoch in range(3):
        print(f"\nEpoch {epoch+1}/3")

        shifamind_model.train()
        total_loss = 0
        loss_components = defaultdict(float)

        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            concept_labels_batch = batch['concept_labels'].to(device)

            optimizer.zero_grad()

            outputs = shifamind_model(input_ids, attention_mask, concept_embeddings)

            loss, components = criterion(
                outputs['logits'],
                outputs['concept_scores'],
                labels,
                concept_labels_batch
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(shifamind_model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            for k, v in components.items():
                loss_components[k] += v

        avg_loss = total_loss / len(train_loader)

        print(f"  Loss: {avg_loss:.4f}")
        for k, v in loss_components.items():
            print(f"    {k}: {v/len(train_loader):.4f}")

        # Validation
        shifamind_model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = shifamind_model(input_ids, attention_mask, concept_embeddings)

                preds = torch.sigmoid(outputs['logits']).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())

        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        pred_binary = (all_preds > 0.5).astype(int)

        macro_f1 = f1_score(all_labels, pred_binary, average='macro', zero_division=0)

        print(f"  Val Macro F1: {macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1

            torch.save({
                'model_state_dict': shifamind_model.state_dict(),
                'num_concepts': len(concept_store.concepts),
                'concept_cuis': list(concept_store.concepts.keys()),
                'concept_names': {cui: info['preferred_name'] for cui, info in concept_store.concepts.items()},
                'concept_embeddings': concept_embeddings,
                'target_codes': TARGET_CODES,
                'macro_f1': best_f1
            }, CHECKPOINT_FINAL)
            print(f"  ✅ Saved checkpoint (F1: {best_f1:.4f})")

    print(f"\n✅ Stage 4 complete. Best F1: {best_f1:.4f}")

torch.cuda.empty_cache()

print("\n" + "="*80)
print("✅ ALL TRAINING STAGES COMPLETE!")
print("="*80)
print(f"\n📊 Summary:")
print(f"   Concepts loaded: {len(umls_concepts)}")
print(f"   Whitelist concepts: {len(whitelist_concepts)}")
print(f"   Avg labels/sample: {train_concept_labels.sum(axis=1).mean():.1f}")
print(f"   Final F1: {best_f1:.4f}")
print(f"\n🎯 Targeted loading approach successful!")
print(f"   Next: Run evaluation and create demo")
