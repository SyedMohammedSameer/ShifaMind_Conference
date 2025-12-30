#!/usr/bin/env python3
"""
================================================================================
SHIFAMIND PHASE 1: Concept-Grounded Clinical BERT
================================================================================
Author: Mohammed Sameer Syed
University of Arizona - MS in AI Capstone

Architecture:
- Base: BioClinicalBERT (emilyalsentzer/Bio_ClinicalBERT)
- Gated Cross-Attention at layers [9, 11]
- 40 learned medical concept embeddings
- Pointer Network for evidence extraction
- ForcedCitationHead for explainability

Saves:
- Train/val/test splits (for reproducibility across all phases)
- Phase 1 checkpoint + concept embeddings
- All outputs to 07_ShifaMind/

Expected F1: ~0.76
================================================================================
"""

print("="*80)
print("🚀 SHIFAMIND PHASE 1 - CONCEPT-GROUNDED CLINICAL BERT")
print("="*80)

# ============================================================================
# IMPORTS & SETUP
# ============================================================================

import os
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
from typing import Dict, List, Set
from collections import defaultdict, Counter
import pickle

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Device: {device}")

# ============================================================================
# CONFIGURATION
# ============================================================================

print("\n" + "="*80)
print("⚙️  CONFIGURATION")
print("="*80)

# Paths
BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')
MIMIC_NOTES_PATH = BASE_PATH / '01_Raw_Datasets/Extracted/mimic-iv-note-2.2/note'
UMLS_META_PATH = BASE_PATH / '01_Raw_Datasets/Extracted/umls-2025AA-metathesaurus-full/2025AA/META'
MIMIC_PATH = BASE_PATH / '01_Raw_Datasets/Extracted/mimic-iv-3.1'

# NEW: All outputs to 07_ShifaMind
OUTPUT_BASE = BASE_PATH / '07_ShifaMind'
CHECKPOINT_PATH = OUTPUT_BASE / 'checkpoints/phase1'
RESULTS_PATH = OUTPUT_BASE / 'results/phase1'
SHARED_DATA_PATH = OUTPUT_BASE / 'shared_data'

CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
SHARED_DATA_PATH.mkdir(parents=True, exist_ok=True)

print(f"📁 Checkpoints: {CHECKPOINT_PATH}")
print(f"📁 Results: {RESULTS_PATH}")
print(f"📁 Shared Data: {SHARED_DATA_PATH}")

# Target diagnoses
TARGET_CODES = ['J189', 'I5023', 'A419', 'K8000']
ICD_DESCRIPTIONS = {
    'J189': 'Pneumonia, unspecified organism',
    'I5023': 'Acute on chronic systolic heart failure',
    'A419': 'Sepsis, unspecified organism',
    'K8000': 'Calculus of gallbladder with acute cholecystitis'
}

print(f"\n🎯 Target diagnoses: {len(TARGET_CODES)}")
for code in TARGET_CODES:
    print(f"   {code}: {ICD_DESCRIPTIONS[code]}")

# Required medical terms for UMLS filtering
REQUIRED_MEDICAL_TERMS = {
    'J189': ['Pneumonia', 'Lung infection', 'Respiratory infection', 'Fever', 'Cough',
             'Dyspnea', 'Crackles', 'Hypoxia', 'Pulmonary infiltrate'],
    'I5023': ['Heart failure', 'Cardiac failure', 'Dyspnea', 'Edema', 'Orthopnea',
              'Pulmonary edema', 'Cardiomegaly', 'Elevated BNP'],
    'A419': ['Sepsis', 'Bacteremia', 'Fever', 'Hypotension', 'Shock', 'Confusion',
             'Leukocytosis', 'Lactic acidosis', 'Organ failure'],
    'K8000': ['Cholecystitis', 'Gallstones', 'Abdominal pain', 'Murphy sign',
              'Fever', 'Nausea', 'Gallbladder inflammation']
}

DIAGNOSIS_KEYWORDS = {
    'J189': ['pneumonia', 'lung', 'respiratory', 'infiltrate', 'fever', 'cough'],
    'I5023': ['heart', 'cardiac', 'failure', 'edema', 'dyspnea', 'orthopnea'],
    'A419': ['sepsis', 'bacteremia', 'infection', 'fever', 'hypotension', 'shock'],
    'K8000': ['cholecystitis', 'gallbladder', 'gallstone', 'abdominal', 'murphy']
}

# ============================================================================
# UMLS LOADER
# ============================================================================

print("\n" + "="*80)
print("📚 LOADING UMLS CONCEPTS")
print("="*80)

class TargetedUMLSLoader:
    def __init__(self, umls_path: Path):
        self.mrconso_path = umls_path / 'MRCONSO.RRF'
        self.mrsty_path = umls_path / 'MRSTY.RRF'
        self.mrdef_path = umls_path / 'MRDEF.RRF'

    def load_specific_concepts(self, required_terms: Dict[str, List[str]]):
        print("\n🔍 Searching UMLS...")

        all_terms_flat = []
        for terms_list in required_terms.values():
            all_terms_flat.extend(terms_list)

        search_terms = set([t.lower().strip() for t in all_terms_flat])
        print(f"   Search terms: {len(search_terms)}")

        found_concepts = {}
        term_to_cuis = defaultdict(list)

        print("\n📖 Scanning MRCONSO...")
        with open(self.mrconso_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in tqdm(f, desc="  Searching"):
                fields = line.strip().split('|')
                if len(fields) < 15:
                    continue

                cui, lang, sab, term = fields[0], fields[1], fields[11], fields[14]

                if lang != 'ENG':
                    continue
                if sab not in ['SNOMEDCT_US', 'ICD10CM', 'MSH', 'NCI']:
                    continue

                term_lower = term.lower().strip()

                for search_term in search_terms:
                    if search_term == term_lower or search_term in term_lower:
                        if cui not in found_concepts:
                            found_concepts[cui] = {
                                'cui': cui,
                                'preferred_name': term,
                                'terms': [term],
                                'semantic_types': [],
                                'definition': ''
                            }
                        else:
                            if term not in found_concepts[cui]['terms']:
                                found_concepts[cui]['terms'].append(term)

                        if cui not in term_to_cuis[search_term]:
                            term_to_cuis[search_term].append(cui)
                        break

        print(f"   ✅ Found {len(found_concepts)} concepts")

        # Load semantic types
        print("\n📋 Loading semantic types...")
        with open(self.mrsty_path, 'r', encoding='utf-8') as f:
            for line in f:
                fields = line.strip().split('|')
                if len(fields) >= 2 and fields[0] in found_concepts:
                    found_concepts[fields[0]]['semantic_types'].append(fields[1])

        # Load definitions
        if self.mrdef_path.exists():
            print("\n📖 Loading definitions...")
            with open(self.mrdef_path, 'r', encoding='utf-8') as f:
                for line in f:
                    fields = line.strip().split('|')
                    if len(fields) >= 6 and fields[0] in found_concepts:
                        if not found_concepts[fields[0]]['definition']:
                            found_concepts[fields[0]]['definition'] = fields[5]

        return found_concepts, term_to_cuis

def filter_to_top_concepts(found_concepts, term_to_cuis, required_terms, top_n=10):
    print(f"\n🔍 Filtering to top-{top_n} per diagnosis...")

    diagnosis_scores = {}
    for dx_code, terms_list in required_terms.items():
        concept_scores = Counter()
        for term in terms_list:
            term_lower = term.lower().strip()
            if term_lower in term_to_cuis:
                for cui in term_to_cuis[term_lower]:
                    concept_scores[cui] += 1
        diagnosis_scores[dx_code] = concept_scores

    all_kept_cuis = set()
    for dx_code, scores in diagnosis_scores.items():
        top = [cui for cui, _ in scores.most_common(top_n)]
        all_kept_cuis.update(top)
        print(f"  {dx_code}: {len(top)} concepts")

    filtered = {cui: info for cui, info in found_concepts.items() if cui in all_kept_cuis}
    filtered_terms = {term: [c for c in cuis if c in all_kept_cuis]
                      for term, cuis in term_to_cuis.items()}
    filtered_terms = {k: v for k, v in filtered_terms.items() if v}

    print(f"\n   ✅ Filtered: {len(found_concepts)} → {len(filtered)}")
    return filtered, filtered_terms

# Load UMLS
loader = TargetedUMLSLoader(UMLS_META_PATH)
umls_raw, terms_raw = loader.load_specific_concepts(REQUIRED_MEDICAL_TERMS)
umls_concepts, term_to_cuis = filter_to_top_concepts(umls_raw, terms_raw, REQUIRED_MEDICAL_TERMS, top_n=10)

print(f"\n✅ UMLS loaded: {len(umls_concepts)} concepts")

# ============================================================================
# LOAD MIMIC-IV
# ============================================================================

print("\n" + "="*80)
print("🏥 LOADING MIMIC-IV")
print("="*80)

class MIMICLoader:
    def __init__(self, mimic_path: Path, notes_path: Path):
        self.hosp_path = mimic_path / 'mimic-iv-3.1/hosp'
        self.notes_path = notes_path

    def load_diagnoses(self) -> pd.DataFrame:
        return pd.read_csv(self.hosp_path / 'diagnoses_icd.csv.gz', compression='gzip')

    def load_discharge_notes(self) -> pd.DataFrame:
        return pd.read_csv(self.notes_path / 'discharge.csv.gz', compression='gzip')

mimic_loader = MIMICLoader(MIMIC_PATH, MIMIC_NOTES_PATH)
df_diag = mimic_loader.load_diagnoses()
df_notes = mimic_loader.load_discharge_notes()

print(f"✅ Loaded:")
print(f"   Diagnoses: {len(df_diag):,}")
print(f"   Notes: {len(df_notes):,}")

def prepare_dataset(df_diag, df_notes, target_codes, max_per_code=3000):
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

    # Balance by sampling
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

    print(f"   ✅ Final: {len(df_final)} samples")
    return df_final

df_data = prepare_dataset(df_diag, df_notes, TARGET_CODES, max_per_code=3000)

# ============================================================================
# SPLIT & SAVE (CRITICAL!)
# ============================================================================

print("\n" + "="*80)
print("🔀 CREATING & SAVING SPLITS")
print("="*80)

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

df_train = df_train.drop('primary_dx', axis=1).reset_index(drop=True)
df_val = df_val.drop('primary_dx', axis=1).reset_index(drop=True)
df_test = df_test.drop('primary_dx', axis=1).reset_index(drop=True)

print(f"📊 Splits:")
print(f"   Train: {len(df_train):,}")
print(f"   Val:   {len(df_val):,}")
print(f"   Test:  {len(df_test):,}")

# SAVE SPLITS
print("\n💾 Saving splits...")
with open(SHARED_DATA_PATH / 'train_split.pkl', 'wb') as f:
    pickle.dump(df_train, f)
with open(SHARED_DATA_PATH / 'val_split.pkl', 'wb') as f:
    pickle.dump(df_val, f)
with open(SHARED_DATA_PATH / 'test_split.pkl', 'wb') as f:
    pickle.dump(df_test, f)

# Save split info
split_info = {
    'seed': SEED,
    'train_size': len(df_train),
    'val_size': len(df_val),
    'test_size': len(df_test),
    'target_codes': TARGET_CODES
}
with open(SHARED_DATA_PATH / 'split_info.json', 'w') as f:
    json.dump(split_info, f, indent=2)

print(f"   ✅ Saved to {SHARED_DATA_PATH}")

# ============================================================================
# ARCHITECTURE
# ============================================================================

print("\n" + "="*80)
print("🏗️  BUILDING ARCHITECTURE")
print("="*80)

class GatedCrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=8, dropout=0.1, layer_idx=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, concept_embeddings, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape
        num_concepts = concept_embeddings.shape[0]

        concepts_batch = concept_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        Q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(concepts_batch).view(batch_size, num_concepts, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(concepts_batch).view(batch_size, num_concepts, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        context = self.out_proj(context)

        gate_input = torch.cat([hidden_states, context], dim=-1)
        gate_values = self.gate(gate_input)
        output = hidden_states + gate_values * context
        output = self.layer_norm(output)

        return output, attn_weights.mean(dim=1)

class PointerNetwork(nn.Module):
    def __init__(self, hidden_size, max_spans=5):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_spans = max_spans
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, concept_embeddings, text_hidden, attention_mask=None):
        batch_size, seq_len, hidden = text_hidden.shape

        if concept_embeddings.dim() == 2:
            num_concepts = concept_embeddings.shape[0]
            concept_embeddings = concept_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            num_concepts = concept_embeddings.shape[1]

        queries = self.query_proj(concept_embeddings)
        keys = self.key_proj(text_hidden)

        pointer_scores = torch.bmm(queries, keys.transpose(1, 2)) / (hidden ** 0.5)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).expand(-1, num_concepts, -1)
            pointer_scores = pointer_scores.masked_fill(mask == 0, float('-inf'))

        pointer_probs = F.softmax(pointer_scores, dim=-1)
        top_scores, _ = torch.topk(pointer_probs, k=min(self.max_spans, seq_len), dim=-1)

        return {'span_scores': top_scores.mean(), 'pointer_probs': pointer_probs}

class ForcedCitationHead(nn.Module):
    def __init__(self, hidden_size, num_concepts, max_spans=5, top_k_concepts=5):
        super().__init__()
        self.concept_classifier = nn.Linear(hidden_size, num_concepts)
        self.pointer_network = PointerNetwork(hidden_size, max_spans)
        self.top_k = top_k_concepts

    def forward(self, cls_hidden, text_hidden, concept_embeddings, attention_mask=None):
        concept_logits = self.concept_classifier(cls_hidden)
        concept_probs = torch.sigmoid(concept_logits)
        top_scores, top_indices = torch.topk(concept_probs, k=self.top_k, dim=-1)

        batch_size = cls_hidden.shape[0]
        selected = torch.stack([concept_embeddings[top_indices[b]] for b in range(batch_size)])
        evidence = self.pointer_network(selected, text_hidden, attention_mask)

        return {'concept_logits': concept_logits, 'span_scores': evidence['span_scores']}

class ShifaMindPhase1(nn.Module):
    def __init__(self, base_model, num_concepts, num_classes, fusion_layers=[9, 11]):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = base_model.config.hidden_size
        self.fusion_layers = fusion_layers

        self.fusion_modules = nn.ModuleDict({
            str(layer): GatedCrossAttention(self.hidden_size, layer_idx=layer)
            for layer in fusion_layers
        })

        self.layer_gates = nn.ParameterDict({
            str(layer): nn.Parameter(torch.tensor(0.5), requires_grad=False)
            for layer in fusion_layers
        })

        self.diagnosis_head = nn.Linear(self.hidden_size, num_classes)
        self.concept_head = nn.Linear(self.hidden_size, num_concepts)
        self.diagnosis_concept_interaction = nn.Bilinear(num_classes, num_concepts, num_concepts)
        self.citation_head = ForcedCitationHead(self.hidden_size, num_concepts)
        self.dropout = nn.Dropout(0.1)

        print(f"   Parameters: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, input_ids, attention_mask, concept_embeddings, return_evidence=False):
        outputs = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, return_dict=True
        )

        hidden_states = outputs.hidden_states
        current_hidden = outputs.last_hidden_state

        for layer_idx in self.fusion_layers:
            if str(layer_idx) in self.fusion_modules:
                layer_hidden = hidden_states[layer_idx]
                fused_hidden, _ = self.fusion_modules[str(layer_idx)](
                    layer_hidden, concept_embeddings, attention_mask
                )
                gate = torch.sigmoid(self.layer_gates[str(layer_idx)])
                current_hidden = (1 - gate) * current_hidden + gate * fused_hidden

        cls_hidden = self.dropout(current_hidden[:, 0, :])
        diagnosis_logits = self.diagnosis_head(cls_hidden)
        concept_logits = self.concept_head(cls_hidden)
        refined_concept_logits = self.diagnosis_concept_interaction(
            torch.sigmoid(diagnosis_logits), torch.sigmoid(concept_logits)
        )

        result = {
            'logits': diagnosis_logits,
            'concept_scores': refined_concept_logits,
            'cls_hidden': cls_hidden
        }

        if return_evidence:
            citation_out = self.citation_head(cls_hidden, current_hidden, concept_embeddings, attention_mask)
            result['citations'] = citation_out

        return result

print("✅ Architecture defined")

# ============================================================================
# CONCEPT STORE & EMBEDDINGS
# ============================================================================

print("\n" + "="*80)
print("🧬 CONCEPT STORE")
print("="*80)

class ConceptStore:
    def __init__(self, umls_concepts: Dict):
        self.concepts = umls_concepts
        self.concept_to_idx = {cui: i for i, cui in enumerate(umls_concepts.keys())}
        self.idx_to_concept = {i: cui for cui, i in self.concept_to_idx.items()}
        print(f"   Indexed: {len(self.concepts)} concepts")

    def create_embeddings(self, tokenizer, model, device):
        print("\n🧬 Creating embeddings...")

        concept_texts = []
        for cui, info in self.concepts.items():
            text = f"{info['preferred_name']}."
            if info.get('definition'):
                text += f" {info['definition'][:150]}"
            concept_texts.append(text)

        all_embeddings = []
        model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(concept_texts), 32), desc="  Encoding"):
                batch = concept_texts[i:i+32]
                enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
                outputs = model(**enc)
                all_embeddings.append(outputs.last_hidden_state[:, 0, :].cpu())

        final = torch.cat(all_embeddings, dim=0).to(device)
        print(f"   ✅ Shape: {final.shape}")
        return final

print("🤖 Loading BioClinicalBERT...")
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
base_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)

concept_store = ConceptStore(umls_concepts)
concept_embeddings = concept_store.create_embeddings(tokenizer, base_model, device)

print("\n✅ Concept store ready")

# ============================================================================
# CONCEPT LABELING
# ============================================================================

print("\n" + "="*80)
print("🏷️  GENERATING CONCEPT LABELS")
print("="*80)

class WhitelistLabeler:
    def __init__(self, concept_store, term_to_cuis, required_terms):
        self.concept_store = concept_store
        self.term_to_cuis = term_to_cuis
        self.required_terms = required_terms
        self.whitelist = {}

    def build_whitelist(self):
        for dx_code, terms_list in self.required_terms.items():
            whitelist_cuis = set()
            for term in terms_list:
                term_lower = term.lower().strip()
                if term_lower in self.term_to_cuis:
                    whitelist_cuis.update(self.term_to_cuis[term_lower])
            self.whitelist[dx_code] = list(whitelist_cuis)
            print(f"   {dx_code}: {len(whitelist_cuis)} concepts")
        return self.whitelist

    def generate_labels(self, diagnosis_codes: List[str]) -> List[int]:
        activated = set()
        for dx_code in diagnosis_codes:
            if dx_code in self.whitelist:
                activated.update(self.whitelist[dx_code])

        return [1 if cui in activated else 0 for cui in self.concept_store.concepts.keys()]

    def generate_dataset_labels(self, df):
        print(f"\n🏷️  Labeling {len(df)} samples...")
        all_labels = []
        for row in tqdm(df.itertuples(), total=len(df), desc="  Processing"):
            all_labels.append(self.generate_labels(row.icd_codes))
        return np.array(all_labels)

labeler = WhitelistLabeler(concept_store, term_to_cuis, REQUIRED_MEDICAL_TERMS)
labeler.build_whitelist()

train_concept_labels = labeler.generate_dataset_labels(df_train)
val_concept_labels = labeler.generate_dataset_labels(df_val)
test_concept_labels = labeler.generate_dataset_labels(df_test)

print(f"\n✅ Labels generated (avg: {train_concept_labels.sum(axis=1).mean():.1f} per sample)")

# Save concept labels
print("\n💾 Saving concept labels...")
np.save(SHARED_DATA_PATH / 'train_concept_labels.npy', train_concept_labels)
np.save(SHARED_DATA_PATH / 'val_concept_labels.npy', val_concept_labels)
np.save(SHARED_DATA_PATH / 'test_concept_labels.npy', test_concept_labels)
print("   ✅ Saved")

# ============================================================================
# DATASET
# ============================================================================

class ClinicalDataset(Dataset):
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
            str(self.texts[idx]), padding='max_length', truncation=True,
            max_length=self.max_length, return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(self.labels[idx])
        }

        if self.concept_labels is not None:
            item['concept_labels'] = torch.FloatTensor(self.concept_labels[idx])

        return item

# ============================================================================
# INITIALIZE MODEL
# ============================================================================

print("\n" + "="*80)
print("🚀 INITIALIZING MODEL")
print("="*80)

model = ShifaMindPhase1(
    base_model=base_model,
    num_concepts=len(concept_store.concepts),
    num_classes=len(TARGET_CODES),
    fusion_layers=[9, 11]
).to(device)

print(f"\n✅ Model on {device}")

# ============================================================================
# STAGE 1: DIAGNOSIS HEAD
# ============================================================================

print("\n" + "="*80)
print("🎯 STAGE 1: DIAGNOSIS HEAD")
print("="*80)

stage1_checkpoint = CHECKPOINT_PATH / 'stage1_diagnosis.pt'

train_dataset = ClinicalDataset(df_train['text'].tolist(), df_train['labels'].tolist(), tokenizer)
val_dataset = ClinicalDataset(df_val['text'].tolist(), df_val['labels'].tolist(), tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
criterion = nn.BCEWithLogitsLoss()

num_epochs = 3
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=len(train_loader) // 10,
    num_training_steps=len(train_loader) * num_epochs
)

best_f1 = 0

for epoch in range(num_epochs):
    print(f"\n{'='*60}\nEpoch {epoch+1}/{num_epochs}\n{'='*60}")

    # Train
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc="  Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, concept_embeddings)
        loss = criterion(outputs['logits'], labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    print(f"  Loss: {total_loss/len(train_loader):.4f}")

    # Validate
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="  Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, concept_embeddings)
            preds = torch.sigmoid(outputs['logits']).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    pred_binary = (all_preds > 0.5).astype(int)

    macro_f1 = f1_score(all_labels, pred_binary, average='macro', zero_division=0)
    print(f"  Val F1: {macro_f1:.4f}")

    if macro_f1 > best_f1:
        best_f1 = macro_f1
        torch.save({
            'model_state_dict': model.state_dict(),
            'concept_embeddings': concept_embeddings,
            'num_concepts': len(concept_store.concepts),
            'macro_f1': best_f1,
            'epoch': epoch
        }, stage1_checkpoint)
        print(f"  ✅ Saved (F1: {best_f1:.4f})")

print(f"\n✅ Stage 1 complete! Best F1: {best_f1:.4f}")
torch.cuda.empty_cache()

# ============================================================================
# STAGE 2: CONCEPT HEAD
# ============================================================================

print("\n" + "="*80)
print("🧠 STAGE 2: CONCEPT HEAD")
print("="*80)

stage2_checkpoint = CHECKPOINT_PATH / 'stage2_concepts.pt'

train_dataset = ClinicalDataset(
    df_train['text'].tolist(), df_train['labels'].tolist(), tokenizer,
    concept_labels=train_concept_labels
)
val_dataset = ClinicalDataset(
    df_val['text'].tolist(), df_val['labels'].tolist(), tokenizer,
    concept_labels=val_concept_labels
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
num_epochs = 2
best_concept_f1 = 0

for epoch in range(num_epochs):
    print(f"\n{'='*60}\nEpoch {epoch+1}/{num_epochs}\n{'='*60}")

    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc="  Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        concept_labels_batch = batch['concept_labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, concept_embeddings)

        concept_loss = criterion(outputs['concept_scores'], concept_labels_batch)
        concept_probs = torch.sigmoid(outputs['concept_scores'])
        top_k_probs = torch.topk(concept_probs, k=min(12, concept_probs.size(1)), dim=1)[0]
        confidence_loss = -torch.mean(top_k_probs)

        loss = 0.7 * concept_loss + 0.3 * confidence_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    print(f"  Loss: {total_loss/len(train_loader):.4f}")

    # Validate
    model.eval()
    all_concept_preds, all_concept_labels = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="  Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            concept_labels_batch = batch['concept_labels'].to(device)

            outputs = model(input_ids, attention_mask, concept_embeddings)
            concept_preds = torch.sigmoid(outputs['concept_scores']).cpu().numpy()
            all_concept_preds.append(concept_preds)
            all_concept_labels.append(concept_labels_batch.cpu().numpy())

    all_concept_preds = np.vstack(all_concept_preds)
    all_concept_labels = np.vstack(all_concept_labels)
    concept_pred_binary = (all_concept_preds > 0.7).astype(int)

    concept_f1 = f1_score(all_concept_labels, concept_pred_binary, average='macro', zero_division=0)
    print(f"  Concept F1: {concept_f1:.4f}")

    if concept_f1 > best_concept_f1:
        best_concept_f1 = concept_f1
        torch.save({
            'model_state_dict': model.state_dict(),
            'concept_embeddings': concept_embeddings,
            'num_concepts': len(concept_store.concepts),
            'concept_f1': best_concept_f1,
            'epoch': epoch
        }, stage2_checkpoint)
        print(f"  ✅ Saved (F1: {best_concept_f1:.4f})")

print(f"\n✅ Stage 2 complete! Best Concept F1: {best_concept_f1:.4f}")
torch.cuda.empty_cache()

# ============================================================================
# STAGE 3: JOINT TRAINING
# ============================================================================

print("\n" + "="*80)
print("🔄 STAGE 3: JOINT TRAINING")
print("="*80)

stage3_checkpoint = CHECKPOINT_PATH / 'phase1_final.pt'

class AdaptiveLoss(nn.Module):
    def __init__(self, dx_weight=0.5, align_weight=0.3, cite_weight=0.15, cal_weight=0.05):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.w_dx = dx_weight
        self.w_align = align_weight
        self.w_cite = cite_weight
        self.w_cal = cal_weight

    def forward(self, outputs, dx_labels, concept_labels):
        loss_dx = self.bce(outputs['logits'], dx_labels)
        loss_align = self.bce(outputs['concept_scores'], concept_labels)

        if 'citations' in outputs and outputs['citations'] is not None:
            loss_cite = -outputs['citations']['span_scores'].mean()
        else:
            loss_cite = torch.tensor(0.0, device=outputs['logits'].device)

        # Simple calibration proxy
        probs = torch.sigmoid(outputs['logits'])
        loss_cal = torch.abs(probs - dx_labels).mean()

        total = (self.w_dx * loss_dx + self.w_align * loss_align +
                 self.w_cite * loss_cite + self.w_cal * loss_cal)

        return total, {
            'diagnosis': loss_dx.item(),
            'alignment': loss_align.item(),
            'citation': loss_cite.item() if torch.is_tensor(loss_cite) else 0,
            'calibration': loss_cal.item()
        }

criterion = AdaptiveLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

num_epochs = 3
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=len(train_loader) // 10,
    num_training_steps=len(train_loader) * num_epochs
)

best_f1 = 0

for epoch in range(num_epochs):
    print(f"\n{'='*60}\nEpoch {epoch+1}/{num_epochs}\n{'='*60}")

    model.train()
    total_loss = 0
    loss_components = defaultdict(float)

    for batch in tqdm(train_loader, desc="  Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        concept_labels_batch = batch['concept_labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, concept_embeddings, return_evidence=True)
        loss, components = criterion(outputs, labels, concept_labels_batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        for k, v in components.items():
            loss_components[k] += v

    print(f"\n  Loss: {total_loss/len(train_loader):.4f}")
    for k in ['diagnosis', 'alignment', 'citation', 'calibration']:
        if k in loss_components:
            print(f"    {k}: {loss_components[k]/len(train_loader):.4f}")

    # Validate
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="  Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, concept_embeddings)
            preds = torch.sigmoid(outputs['logits']).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    pred_binary = (all_preds > 0.5).astype(int)

    macro_f1 = f1_score(all_labels, pred_binary, average='macro', zero_division=0)
    print(f"\n  Val F1: {macro_f1:.4f}")

    if macro_f1 > best_f1:
        best_f1 = macro_f1
        torch.save({
            'model_state_dict': model.state_dict(),
            'concept_embeddings': concept_embeddings,
            'num_concepts': len(concept_store.concepts),
            'target_codes': TARGET_CODES,
            'macro_f1': best_f1,
            'epoch': epoch
        }, stage3_checkpoint)
        print(f"  ✅ Saved (F1: {best_f1:.4f})")

print(f"\n✅ Stage 3 complete! Best F1: {best_f1:.4f}")

# ============================================================================
# FINAL EVALUATION
# ============================================================================

print("\n" + "="*80)
print("📊 FINAL EVALUATION")
print("="*80)

checkpoint = torch.load(stage3_checkpoint, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

test_dataset = ClinicalDataset(
    df_test['text'].tolist(), df_test['labels'].tolist(), tokenizer,
    concept_labels=test_concept_labels
)
test_loader = DataLoader(test_dataset, batch_size=16)

model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="  Testing"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask, concept_embeddings)
        preds = torch.sigmoid(outputs['logits']).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

all_preds = np.vstack(all_preds)
all_labels = np.vstack(all_labels)
pred_binary = (all_preds > 0.5).astype(int)

macro_f1 = f1_score(all_labels, pred_binary, average='macro', zero_division=0)
micro_f1 = f1_score(all_labels, pred_binary, average='micro', zero_division=0)
macro_precision = precision_score(all_labels, pred_binary, average='macro', zero_division=0)
macro_recall = recall_score(all_labels, pred_binary, average='macro', zero_division=0)

try:
    macro_auc = roc_auc_score(all_labels, all_preds, average='macro')
except:
    macro_auc = 0.0

per_class_f1 = [f1_score(all_labels[:, i], pred_binary[:, i], zero_division=0)
                for i in range(len(TARGET_CODES))]

print("\n" + "="*80)
print("🎉 PHASE 1 FINAL RESULTS")
print("="*80)

print("\n🎯 Performance:")
print(f"   Macro F1:    {macro_f1:.4f}")
print(f"   Micro F1:    {micro_f1:.4f}")
print(f"   Precision:   {macro_precision:.4f}")
print(f"   Recall:      {macro_recall:.4f}")
print(f"   AUROC:       {macro_auc:.4f}")

print("\n📊 Per-Class F1:")
for i, code in enumerate(TARGET_CODES):
    print(f"   {code}: {per_class_f1[i]:.4f}")

# Save results
results = {
    'phase': 'Phase 1 - Concept-Grounded BERT',
    'test_metrics': {
        'macro_f1': float(macro_f1),
        'micro_f1': float(micro_f1),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_auc': float(macro_auc),
        'per_class_f1': {code: float(f1) for code, f1 in zip(TARGET_CODES, per_class_f1)}
    },
    'config': {
        'fusion_layers': [9, 11],
        'num_concepts': len(concept_store.concepts),
        'target_codes': TARGET_CODES
    }
}

with open(RESULTS_PATH / 'phase1_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save concept embeddings for Phase 2
torch.save(concept_embeddings, SHARED_DATA_PATH / 'concept_embeddings.pt')

print(f"\n💾 Results saved to: {RESULTS_PATH}")
print(f"💾 Checkpoint: {stage3_checkpoint}")
print(f"💾 Shared data: {SHARED_DATA_PATH}")

print("\n" + "="*80)
print("✅ PHASE 1 COMPLETE!")
print("="*80)
print(f"\n📈 Final Macro F1: {macro_f1:.4f}")
print("\n🚀 Ready for Phase 2 (RAG integration)")
print(f"\nAlhamdulillah! 🤲")
