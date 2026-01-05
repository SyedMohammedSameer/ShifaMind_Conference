#!/usr/bin/env python3
"""
================================================================================
SHIFAMIND2 PHASE 1: Concept Bottleneck Model with TOP-50 ICD-10 Labels
================================================================================
Author: Mohammed Sameer Syed
University of Arizona - MS in AI Capstone

CHANGES FROM SHIFAMIND1_P1:
1. ✅ TOP-50 ICD-10 codes computed from MIMIC-IV (icd_version==10)
2. ✅ Fresh artifacts - no reuse of old splits/checkpoints
3. ✅ All outputs to /content/drive/MyDrive/ShifaMind/10_ShifaMind/run_TIMESTAMP/
4. ✅ Builds mimic_dx_data.csv from raw MIMIC-IV files
5. ✅ Fixed concept space (≤120 concepts across all Top-50)

Architecture (unchanged):
1. BioClinicalBERT base encoder
2. Multi-head cross-attention with concepts (MULTIPLICATIVE bottleneck)
3. Concept Head (predicts clinical concepts)
4. Diagnosis Head (predicts TOP-50 ICD-10 codes)

Multi-Objective Loss:
L_total = λ1·L_dx + λ2·L_align + λ3·L_concept

Target Metrics:
- Diagnosis F1: >0.75
- Concept F1: >0.70
- Concept Completeness: >0.80

================================================================================
"""

print("="*80)
print("🚀 SHIFAMIND2 PHASE 1 - TOP-50 ICD-10 LABELS")
print("="*80)

# ============================================================================
# IMPORTS & SETUP
# ============================================================================

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, AutoModel,
    get_linear_schedule_with_warmup
)

import json
import pickle
import gzip
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
import re
from datetime import datetime

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

# Create timestamped run folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')
OUTPUT_BASE = BASE_PATH / '10_ShifaMind' / f'run_{timestamp}'

# Run-specific paths
SHARED_DATA_PATH = OUTPUT_BASE / 'shared_data'
CHECKPOINT_PATH = OUTPUT_BASE / 'checkpoints' / 'phase1'
RESULTS_PATH = OUTPUT_BASE / 'results' / 'phase1'
CONCEPT_STORE_PATH = OUTPUT_BASE / 'concept_store'
LOGS_PATH = OUTPUT_BASE / 'logs'

# Create all directories
for path in [SHARED_DATA_PATH, CHECKPOINT_PATH, RESULTS_PATH, CONCEPT_STORE_PATH, LOGS_PATH]:
    path.mkdir(parents=True, exist_ok=True)

print(f"\n📁 Run Folder: {OUTPUT_BASE}")
print(f"📁 Timestamp: {timestamp}")
print(f"📁 Shared Data: {SHARED_DATA_PATH}")
print(f"📁 Checkpoints: {CHECKPOINT_PATH}")
print(f"📁 Results: {RESULTS_PATH}")
print(f"📁 Concept Store: {CONCEPT_STORE_PATH}")

# Raw MIMIC-IV paths
RAW_MIMIC_PATH = BASE_PATH / '01_Raw_Datasets' / 'Extracted' / 'mimic-iv-3.1' / 'mimic-iv-3.1' / 'hosp'
RAW_MIMIC_NOTE_PATH = BASE_PATH / '01_Raw_Datasets' / 'Extracted' / 'mimic-iv-note-2.2' / 'note'

print(f"\n📂 MIMIC-IV Hosp: {RAW_MIMIC_PATH}")
print(f"📂 MIMIC-IV Note: {RAW_MIMIC_NOTE_PATH}")

# Fixed global concept space (≤120 concepts)
# Common medical concepts applicable across multiple diagnoses
GLOBAL_CONCEPTS = [
    # Symptoms
    'fever', 'cough', 'dyspnea', 'pain', 'nausea', 'vomiting', 'diarrhea', 'fatigue',
    'headache', 'dizziness', 'weakness', 'confusion', 'syncope', 'chest', 'abdominal',
    'dysphagia', 'hemoptysis', 'hematuria', 'hematemesis', 'melena', 'jaundice',
    'edema', 'rash', 'pruritus', 'weight', 'anorexia', 'malaise',
    # Vital signs / Physical findings
    'hypotension', 'hypertension', 'tachycardia', 'bradycardia', 'tachypnea', 'hypoxia',
    'fever', 'hypothermia', 'shock', 'altered', 'lethargic', 'obtunded',
    # Organ systems
    'cardiac', 'pulmonary', 'renal', 'hepatic', 'neurologic', 'gastrointestinal',
    'respiratory', 'cardiovascular', 'genitourinary', 'musculoskeletal', 'endocrine',
    'hematologic', 'dermatologic', 'psychiatric',
    # Common conditions
    'infection', 'sepsis', 'pneumonia', 'uti', 'cellulitis', 'meningitis',
    'failure', 'infarction', 'ischemia', 'hemorrhage', 'thrombosis', 'embolism',
    'obstruction', 'perforation', 'rupture', 'stenosis', 'regurgitation',
    'hypertrophy', 'atrophy', 'neoplasm', 'malignancy', 'metastasis',
    # Lab/diagnostic
    'elevated', 'decreased', 'anemia', 'leukocytosis', 'thrombocytopenia',
    'hyperglycemia', 'hypoglycemia', 'acidosis', 'alkalosis', 'hypoxemia',
    'creatinine', 'bilirubin', 'troponin', 'bnp', 'lactate', 'wbc', 'cultures',
    # Imaging/procedures
    'infiltrate', 'consolidation', 'effusion', 'edema', 'cardiomegaly',
    'ultrasound', 'ct', 'mri', 'xray', 'echo', 'ekg',
    # Treatments
    'antibiotics', 'diuretics', 'vasopressors', 'insulin', 'anticoagulation',
    'oxygen', 'ventilation', 'dialysis', 'transfusion', 'surgery'
]

print(f"\n🧠 Global Concept Space: {len(GLOBAL_CONCEPTS)} concepts")

# Hyperparameters (same as shifamind1)
LAMBDA_DX = 1.0
LAMBDA_ALIGN = 0.5
LAMBDA_CONCEPT = 0.3

print(f"\n⚖️  Loss Weights:")
print(f"   λ1 (Diagnosis): {LAMBDA_DX}")
print(f"   λ2 (Alignment): {LAMBDA_ALIGN}")
print(f"   λ3 (Concept):   {LAMBDA_CONCEPT}")

# ============================================================================
# STEP 1: COMPUTE TOP-50 ICD-10 CODES FROM MIMIC-IV
# ============================================================================

print("\n" + "="*80)
print("📊 COMPUTING TOP-50 ICD-10 CODES FROM MIMIC-IV")
print("="*80)

def normalize_icd10_code(code):
    """Normalize ICD-10 code: uppercase, remove dots"""
    if pd.isna(code):
        return None
    code_str = str(code).upper().replace('.', '').strip()
    return code_str if code_str else None

print("\n1️⃣ Loading diagnoses_icd.csv.gz...")
diagnoses_path = RAW_MIMIC_PATH / 'diagnoses_icd.csv.gz'
df_diag = pd.read_csv(diagnoses_path, compression='gzip')
print(f"   Loaded {len(df_diag):,} diagnosis records")

# Filter ICD-10 only
df_diag_icd10 = df_diag[df_diag['icd_version'] == 10].copy()
print(f"   ICD-10 records: {len(df_diag_icd10):,}")

# Normalize codes
df_diag_icd10['icd_code_normalized'] = df_diag_icd10['icd_code'].apply(normalize_icd10_code)
df_diag_icd10 = df_diag_icd10.dropna(subset=['icd_code_normalized'])
print(f"   After normalization: {len(df_diag_icd10):,}")

print("\n2️⃣ Loading discharge notes...")
discharge_path = RAW_MIMIC_NOTE_PATH / 'discharge.csv.gz'
df_notes = pd.read_csv(discharge_path, compression='gzip')
print(f"   Loaded {len(df_notes):,} discharge notes")

# Keep only non-empty notes
df_notes = df_notes[df_notes['text'].notna() & (df_notes['text'].str.len() > 100)].copy()
print(f"   Non-empty notes: {len(df_notes):,}")

# Get unique hadm_id with notes
valid_hadm_ids = set(df_notes['hadm_id'].unique())
print(f"   Unique hadm_id with discharge notes: {len(valid_hadm_ids):,}")

print("\n3️⃣ Filtering diagnoses to hadm_id with notes...")
df_diag_icd10 = df_diag_icd10[df_diag_icd10['hadm_id'].isin(valid_hadm_ids)].copy()
print(f"   Diagnoses with notes: {len(df_diag_icd10):,}")

print("\n4️⃣ Computing Top-50 ICD-10 codes by admission frequency...")
# Count unique hadm_id per code (not total occurrences)
code_counts = df_diag_icd10.groupby('icd_code_normalized')['hadm_id'].nunique().sort_values(ascending=False)
print(f"   Total unique ICD-10 codes: {len(code_counts):,}")

# Take top 50
TOP_50_CODES = code_counts.head(50).index.tolist()
TOP_50_COUNTS = code_counts.head(50).values.tolist()

print(f"\n✅ TOP-50 ICD-10 CODES:")
print(f"{'Rank':<6} {'Code':<10} {'Admissions':<12}")
print("-" * 30)
for rank, (code, count) in enumerate(zip(TOP_50_CODES, TOP_50_COUNTS), 1):
    print(f"{rank:<6} {code:<10} {count:<12,}")

# Save Top-50 info
top50_info = {
    'timestamp': timestamp,
    'top_50_codes': TOP_50_CODES,
    'top_50_counts': {code: int(count) for code, count in zip(TOP_50_CODES, TOP_50_COUNTS)},
    'total_unique_codes': len(code_counts),
    'total_icd10_records': len(df_diag_icd10),
    'valid_admissions': len(valid_hadm_ids)
}

with open(SHARED_DATA_PATH / 'top50_icd10_info.json', 'w') as f:
    json.dump(top50_info, f, indent=2)

print(f"\n💾 Saved Top-50 info to: {SHARED_DATA_PATH / 'top50_icd10_info.json'}")

# ============================================================================
# STEP 2: BUILD MIMIC_DX_DATA.CSV WITH TOP-50 LABELS
# ============================================================================

print("\n" + "="*80)
print("📊 BUILDING mimic_dx_data.csv WITH TOP-50 LABELS")
print("="*80)

print("\n1️⃣ Creating multi-label matrix...")
# For each hadm_id, create binary vector for Top-50 codes
hadm_labels = defaultdict(lambda: [0] * len(TOP_50_CODES))
code_to_idx = {code: idx for idx, code in enumerate(TOP_50_CODES)}

for _, row in tqdm(df_diag_icd10.iterrows(), total=len(df_diag_icd10), desc="Processing diagnoses"):
    hadm_id = row['hadm_id']
    code = row['icd_code_normalized']
    if code in code_to_idx:
        hadm_labels[hadm_id][code_to_idx[code]] = 1

print(f"   Labeled {len(hadm_labels):,} admissions")

print("\n2️⃣ Merging with discharge notes...")
# Merge notes with labels
df_notes_with_labels = df_notes.copy()
df_notes_with_labels['labels'] = df_notes_with_labels['hadm_id'].map(
    lambda x: hadm_labels.get(x, [0] * len(TOP_50_CODES))
)

# Keep only admissions that have at least one Top-50 label
df_notes_with_labels['has_top50'] = df_notes_with_labels['labels'].apply(lambda x: sum(x) > 0)
df_final = df_notes_with_labels[df_notes_with_labels['has_top50']].copy()

print(f"   Admissions with Top-50 labels: {len(df_final):,}")

# Add individual code columns for easier analysis
for idx, code in enumerate(TOP_50_CODES):
    df_final[code] = df_final['labels'].apply(lambda x: x[idx])

print("\n3️⃣ Label distribution:")
label_counts = [df_final[code].sum() for code in TOP_50_CODES]
print(f"   Mean labels per admission: {np.mean([sum(x) for x in df_final['labels']]):.2f}")
print(f"   Median labels per admission: {np.median([sum(x) for x in df_final['labels']]):.0f}")
print(f"\n   Top-10 most frequent codes in dataset:")
top_10_in_dataset = sorted(zip(TOP_50_CODES, label_counts), key=lambda x: x[1], reverse=True)[:10]
for code, count in top_10_in_dataset:
    print(f"      {code}: {count:,} ({count/len(df_final)*100:.1f}%)")

# Save to CSV
mimic_dx_path = OUTPUT_BASE / 'mimic_dx_data_top50.csv'
df_final[['subject_id', 'hadm_id', 'text'] + TOP_50_CODES].to_csv(mimic_dx_path, index=False)
print(f"\n💾 Saved dataset to: {mimic_dx_path}")
print(f"   Rows: {len(df_final):,}")
print(f"   Columns: {len(df_final.columns)}")

# ============================================================================
# STEP 3: CREATE TRAIN/VAL/TEST SPLITS (FRESH)
# ============================================================================

print("\n" + "="*80)
print("📊 CREATING TRAIN/VAL/TEST SPLITS (FRESH)")
print("="*80)

# Prepare for splitting
df = df_final[['text', 'labels'] + TOP_50_CODES].copy()
df = df.dropna(subset=['text'])

print(f"\n📊 Dataset size: {len(df):,} samples")

# Random split (stratification not feasible with Top-50 multilabel)
# Split: 70% train, 15% val, 15% test
train_idx, temp_idx = train_test_split(
    range(len(df)),
    test_size=0.3,
    random_state=SEED
)
val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.5,
    random_state=SEED
)

df_train = df.iloc[train_idx].reset_index(drop=True)
df_val = df.iloc[val_idx].reset_index(drop=True)
df_test = df.iloc[test_idx].reset_index(drop=True)

print(f"\n✅ Splits created:")
print(f"   Train: {len(df_train):,} ({len(df_train)/len(df)*100:.1f}%)")
print(f"   Val:   {len(df_val):,} ({len(df_val)/len(df)*100:.1f}%)")
print(f"   Test:  {len(df_test):,} ({len(df_test)/len(df)*100:.1f}%)")

# Label distribution per split
print(f"\n📊 Label distribution per split:")
for split_name, split_df in [('Train', df_train), ('Val', df_val), ('Test', df_test)]:
    avg_labels = np.mean([sum(x) for x in split_df['labels']])
    total_positives = sum([sum(x) for x in split_df['labels']])
    print(f"   {split_name}: avg={avg_labels:.2f} labels/sample, total={total_positives:,} positive labels")

# Save splits
with open(SHARED_DATA_PATH / 'train_split.pkl', 'wb') as f:
    pickle.dump(df_train, f)
with open(SHARED_DATA_PATH / 'val_split.pkl', 'wb') as f:
    pickle.dump(df_val, f)
with open(SHARED_DATA_PATH / 'test_split.pkl', 'wb') as f:
    pickle.dump(df_test, f)

# Save split info
split_info = {
    'timestamp': timestamp,
    'total_samples': len(df),
    'train_samples': len(df_train),
    'val_samples': len(df_val),
    'test_samples': len(df_test),
    'num_labels': len(TOP_50_CODES),
    'label_codes': TOP_50_CODES,
    'random_seed': SEED
}

with open(SHARED_DATA_PATH / 'split_info.json', 'w') as f:
    json.dump(split_info, f, indent=2)

print(f"\n💾 Saved splits to: {SHARED_DATA_PATH}")

# ============================================================================
# STEP 4: GENERATE CONCEPT LABELS (KEYWORD-BASED)
# ============================================================================

print("\n" + "="*80)
print("🧠 GENERATING CONCEPT LABELS (KEYWORD-BASED)")
print("="*80)

def generate_concept_labels(texts, concepts):
    """Generate binary concept labels based on keyword presence"""
    labels = []
    for text in tqdm(texts, desc="Labeling"):
        text_lower = str(text).lower()
        concept_label = [1 if concept in text_lower else 0 for concept in concepts]
        labels.append(concept_label)
    return np.array(labels)

print(f"\n🔍 Using {len(GLOBAL_CONCEPTS)} global concepts")

train_concept_labels = generate_concept_labels(df_train['text'], GLOBAL_CONCEPTS)
val_concept_labels = generate_concept_labels(df_val['text'], GLOBAL_CONCEPTS)
test_concept_labels = generate_concept_labels(df_test['text'], GLOBAL_CONCEPTS)

print(f"\n✅ Concept labels generated:")
print(f"   Shape: {train_concept_labels.shape}")
print(f"   Concepts per sample (train): {train_concept_labels.sum(axis=1).mean():.2f}")

# Save concept labels
np.save(SHARED_DATA_PATH / 'train_concept_labels.npy', train_concept_labels)
np.save(SHARED_DATA_PATH / 'val_concept_labels.npy', val_concept_labels)
np.save(SHARED_DATA_PATH / 'test_concept_labels.npy', test_concept_labels)

# Save concept list
with open(SHARED_DATA_PATH / 'concept_list.json', 'w') as f:
    json.dump(GLOBAL_CONCEPTS, f, indent=2)

print(f"💾 Saved concept labels to: {SHARED_DATA_PATH}")

# ============================================================================
# ARCHITECTURE: CONCEPT BOTTLENECK (SAME AS SHIFAMIND1)
# ============================================================================

print("\n" + "="*80)
print("🏗️  ARCHITECTURE: CONCEPT BOTTLENECK")
print("="*80)

class ConceptBottleneckCrossAttention(nn.Module):
    """Multiplicative concept bottleneck with cross-attention"""
    def __init__(self, hidden_size, num_heads=8, dropout=0.1, layer_idx=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.layer_idx = layer_idx

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

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

        pooled_text = hidden_states.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)
        pooled_context = context.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)
        gate_input = torch.cat([pooled_text, pooled_context], dim=-1)
        gate = self.gate_net(gate_input)

        output = gate * context
        output = self.layer_norm(output)

        return output, attn_weights.mean(dim=1), gate.mean()


class ShifaMind2Phase1(nn.Module):
    """ShifaMind2 Phase 1: Concept Bottleneck with Top-50 ICD-10"""
    def __init__(self, base_model, num_concepts, num_classes, fusion_layers=[9, 11]):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = base_model.config.hidden_size
        self.num_concepts = num_concepts
        self.fusion_layers = fusion_layers

        self.concept_embeddings = nn.Parameter(
            torch.randn(num_concepts, self.hidden_size) * 0.02
        )

        self.fusion_modules = nn.ModuleDict({
            str(layer): ConceptBottleneckCrossAttention(self.hidden_size, layer_idx=layer)
            for layer in fusion_layers
        })

        self.concept_head = nn.Linear(self.hidden_size, num_concepts)
        self.diagnosis_head = nn.Linear(self.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, return_attention=False):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        hidden_states = outputs.hidden_states
        current_hidden = outputs.last_hidden_state

        attention_maps = {}
        gate_values = []

        for layer_idx in self.fusion_layers:
            if str(layer_idx) in self.fusion_modules:
                layer_hidden = hidden_states[layer_idx]
                fused_hidden, attn, gate = self.fusion_modules[str(layer_idx)](
                    layer_hidden, self.concept_embeddings, attention_mask
                )
                current_hidden = fused_hidden
                gate_values.append(gate.item())

                if return_attention:
                    attention_maps[f'layer_{layer_idx}'] = attn

        cls_hidden = self.dropout(current_hidden[:, 0, :])
        concept_scores = torch.sigmoid(self.concept_head(cls_hidden))
        diagnosis_logits = self.diagnosis_head(cls_hidden)

        result = {
            'logits': diagnosis_logits,
            'concept_scores': concept_scores,
            'hidden_states': current_hidden,
            'cls_hidden': cls_hidden,
            'avg_gate': np.mean(gate_values) if gate_values else 0.0
        }

        if return_attention:
            result['attention_maps'] = attention_maps

        return result


class MultiObjectiveLoss(nn.Module):
    """Multi-objective loss: L_dx + L_align + L_concept"""
    def __init__(self, lambda_dx=1.0, lambda_align=0.5, lambda_concept=0.3):
        super().__init__()
        self.lambda_dx = lambda_dx
        self.lambda_align = lambda_align
        self.lambda_concept = lambda_concept
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs, dx_labels, concept_labels):
        loss_dx = self.bce(outputs['logits'], dx_labels)

        dx_probs = torch.sigmoid(outputs['logits'])
        concept_scores = outputs['concept_scores']
        loss_align = torch.abs(
            dx_probs.unsqueeze(-1) - concept_scores.unsqueeze(1)
        ).mean()

        concept_logits = torch.logit(concept_scores.clamp(1e-7, 1-1e-7))
        loss_concept = self.bce(concept_logits, concept_labels)

        total_loss = (
            self.lambda_dx * loss_dx +
            self.lambda_align * loss_align +
            self.lambda_concept * loss_concept
        )

        components = {
            'total': total_loss.item(),
            'dx': loss_dx.item(),
            'align': loss_align.item(),
            'concept': loss_concept.item()
        }

        return total_loss, components


print("✅ Architecture defined")

# ============================================================================
# DATASET
# ============================================================================

class ConceptDataset(Dataset):
    def __init__(self, texts, labels, concept_labels, tokenizer, max_length=384):
        self.texts = texts
        self.labels = labels
        self.concept_labels = concept_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

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

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(self.labels[idx]),
            'concept_labels': torch.FloatTensor(self.concept_labels[idx])
        }

# ============================================================================
# TRAINING
# ============================================================================

print("\n" + "="*80)
print("🏋️  TRAINING PHASE 1")
print("="*80)

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
base_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)

model = ShifaMind2Phase1(
    base_model,
    num_concepts=len(GLOBAL_CONCEPTS),
    num_classes=len(TOP_50_CODES),
    fusion_layers=[9, 11]
).to(device)

print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"   Num concepts: {len(GLOBAL_CONCEPTS)}")
print(f"   Num diagnoses: {len(TOP_50_CODES)}")

# Create datasets
train_dataset = ConceptDataset(
    df_train['text'].tolist(),
    df_train['labels'].tolist(),
    train_concept_labels,
    tokenizer
)
val_dataset = ConceptDataset(
    df_val['text'].tolist(),
    df_val['labels'].tolist(),
    val_concept_labels,
    tokenizer
)
test_dataset = ConceptDataset(
    df_test['text'].tolist(),
    df_test['labels'].tolist(),
    test_concept_labels,
    tokenizer
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

print(f"✅ Datasets ready")

# Training setup
criterion = MultiObjectiveLoss(
    lambda_dx=LAMBDA_DX,
    lambda_align=LAMBDA_ALIGN,
    lambda_concept=LAMBDA_CONCEPT
)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

num_epochs = 5
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=len(train_loader) // 2,
    num_training_steps=len(train_loader) * num_epochs
)

best_f1 = 0.0
history = {'train_loss': [], 'val_f1': [], 'concept_f1': []}

# Training loop
for epoch in range(num_epochs):
    print(f"\n{'='*70}\nEpoch {epoch+1}/{num_epochs}\n{'='*70}")

    model.train()
    epoch_losses = defaultdict(list)

    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        dx_labels = batch['labels'].to(device)
        concept_labels = batch['concept_labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss, components = criterion(outputs, dx_labels, concept_labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        for k, v in components.items():
            epoch_losses[k].append(v)

    print(f"\n📊 Epoch {epoch+1} Losses:")
    print(f"   Total:     {np.mean(epoch_losses['total']):.4f}")
    print(f"   Diagnosis: {np.mean(epoch_losses['dx']):.4f}")
    print(f"   Alignment: {np.mean(epoch_losses['align']):.4f}")
    print(f"   Concept:   {np.mean(epoch_losses['concept']):.4f}")

    # Validation
    model.eval()
    all_dx_preds, all_dx_labels = [], []
    all_concept_preds, all_concept_labels = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            dx_labels = batch['labels'].to(device)
            concept_labels = batch['concept_labels'].to(device)

            outputs = model(input_ids, attention_mask)

            all_dx_preds.append(torch.sigmoid(outputs['logits']).cpu())
            all_dx_labels.append(dx_labels.cpu())
            all_concept_preds.append(outputs['concept_scores'].cpu())
            all_concept_labels.append(concept_labels.cpu())

    all_dx_preds = torch.cat(all_dx_preds, dim=0).numpy()
    all_dx_labels = torch.cat(all_dx_labels, dim=0).numpy()
    all_concept_preds = torch.cat(all_concept_preds, dim=0).numpy()
    all_concept_labels = torch.cat(all_concept_labels, dim=0).numpy()

    dx_pred_binary = (all_dx_preds > 0.5).astype(int)
    concept_pred_binary = (all_concept_preds > 0.5).astype(int)

    dx_f1 = f1_score(all_dx_labels, dx_pred_binary, average='macro', zero_division=0)
    concept_f1 = f1_score(all_concept_labels, concept_pred_binary, average='macro', zero_division=0)

    print(f"\n📈 Validation:")
    print(f"   Diagnosis F1: {dx_f1:.4f}")
    print(f"   Concept F1:   {concept_f1:.4f}")

    history['train_loss'].append(np.mean(epoch_losses['total']))
    history['val_f1'].append(dx_f1)
    history['concept_f1'].append(concept_f1)

    if dx_f1 > best_f1:
        best_f1 = dx_f1
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'macro_f1': best_f1,
            'concept_f1': concept_f1,
            'concept_embeddings': model.concept_embeddings.data.cpu(),
            'num_concepts': model.num_concepts,
            'config': {
                'num_concepts': len(GLOBAL_CONCEPTS),
                'num_classes': len(TOP_50_CODES),
                'fusion_layers': [9, 11],
                'lambda_dx': LAMBDA_DX,
                'lambda_align': LAMBDA_ALIGN,
                'lambda_concept': LAMBDA_CONCEPT,
                'top_50_codes': TOP_50_CODES,
                'timestamp': timestamp
            }
        }
        torch.save(checkpoint, CHECKPOINT_PATH / 'phase1_best.pt')
        print(f"   ✅ Saved best model (F1: {best_f1:.4f})")

print(f"\n✅ Training complete! Best Diagnosis F1: {best_f1:.4f}")

# ============================================================================
# FINAL EVALUATION
# ============================================================================

print("\n" + "="*80)
print("📊 FINAL TEST EVALUATION")
print("="*80)

checkpoint = torch.load(CHECKPOINT_PATH / 'phase1_best.pt', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

all_dx_preds, all_dx_labels = [], []
all_concept_preds, all_concept_labels = [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        dx_labels = batch['labels'].to(device)
        concept_labels = batch['concept_labels'].to(device)

        outputs = model(input_ids, attention_mask)

        all_dx_preds.append(torch.sigmoid(outputs['logits']).cpu())
        all_dx_labels.append(dx_labels.cpu())
        all_concept_preds.append(outputs['concept_scores'].cpu())
        all_concept_labels.append(concept_labels.cpu())

all_dx_preds = torch.cat(all_dx_preds, dim=0).numpy()
all_dx_labels = torch.cat(all_dx_labels, dim=0).numpy()
all_concept_preds = torch.cat(all_concept_preds, dim=0).numpy()
all_concept_labels = torch.cat(all_concept_labels, dim=0).numpy()

dx_pred_binary = (all_dx_preds > 0.5).astype(int)
concept_pred_binary = (all_concept_preds > 0.5).astype(int)

macro_f1 = f1_score(all_dx_labels, dx_pred_binary, average='macro', zero_division=0)
micro_f1 = f1_score(all_dx_labels, dx_pred_binary, average='micro', zero_division=0)
macro_precision = precision_score(all_dx_labels, dx_pred_binary, average='macro', zero_division=0)
macro_recall = recall_score(all_dx_labels, dx_pred_binary, average='macro', zero_division=0)

per_class_f1 = [
    f1_score(all_dx_labels[:, i], dx_pred_binary[:, i], zero_division=0)
    for i in range(len(TOP_50_CODES))
]

concept_f1 = f1_score(all_concept_labels, concept_pred_binary, average='macro', zero_division=0)

print("\n" + "="*80)
print("🎉 SHIFAMIND2 PHASE 1 - FINAL RESULTS")
print("="*80)

print("\n🎯 Diagnosis Performance (Top-50 ICD-10):")
print(f"   Macro F1:    {macro_f1:.4f}")
print(f"   Micro F1:    {micro_f1:.4f}")
print(f"   Precision:   {macro_precision:.4f}")
print(f"   Recall:      {macro_recall:.4f}")

print(f"\n🧠 Concept Performance:")
print(f"   Concept F1:  {concept_f1:.4f}")

print(f"\n📊 Top-10 Best Performing Diagnoses:")
top_10_best = sorted(zip(TOP_50_CODES, per_class_f1), key=lambda x: x[1], reverse=True)[:10]
for rank, (code, f1) in enumerate(top_10_best, 1):
    count = top50_info['top_50_counts'].get(code, 0)
    print(f"   {rank}. {code}: F1={f1:.4f} (n={count:,})")

print(f"\n📊 Top-10 Worst Performing Diagnoses:")
top_10_worst = sorted(zip(TOP_50_CODES, per_class_f1), key=lambda x: x[1])[:10]
for rank, (code, f1) in enumerate(top_10_worst, 1):
    count = top50_info['top_50_counts'].get(code, 0)
    print(f"   {rank}. {code}: F1={f1:.4f} (n={count:,})")

# Save results
results = {
    'phase': 'ShifaMind2 Phase 1 - Top-50 ICD-10',
    'timestamp': timestamp,
    'run_folder': str(OUTPUT_BASE),
    'diagnosis_metrics': {
        'macro_f1': float(macro_f1),
        'micro_f1': float(micro_f1),
        'precision': float(macro_precision),
        'recall': float(macro_recall),
        'per_class_f1': {code: float(f1) for code, f1 in zip(TOP_50_CODES, per_class_f1)}
    },
    'concept_metrics': {
        'concept_f1': float(concept_f1),
        'num_concepts': len(GLOBAL_CONCEPTS)
    },
    'dataset_info': {
        'num_labels': len(TOP_50_CODES),
        'train_samples': len(df_train),
        'val_samples': len(df_val),
        'test_samples': len(df_test)
    },
    'loss_weights': {
        'lambda_dx': LAMBDA_DX,
        'lambda_align': LAMBDA_ALIGN,
        'lambda_concept': LAMBDA_CONCEPT
    },
    'training_history': history
}

with open(RESULTS_PATH / 'results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save per-label F1 scores as CSV
per_label_df = pd.DataFrame({
    'icd_code': TOP_50_CODES,
    'f1_score': per_class_f1,
    'train_count': [top50_info['top_50_counts'].get(code, 0) for code in TOP_50_CODES]
})
per_label_df = per_label_df.sort_values('f1_score', ascending=False)
per_label_df.to_csv(RESULTS_PATH / 'per_label_f1.csv', index=False)

print(f"\n💾 Results saved to: {RESULTS_PATH / 'results.json'}")
print(f"💾 Per-label F1 saved to: {RESULTS_PATH / 'per_label_f1.csv'}")
print(f"💾 Best model saved to: {CHECKPOINT_PATH / 'phase1_best.pt'}")

print("\n" + "="*80)
print("✅ SHIFAMIND2 PHASE 1 COMPLETE!")
print("="*80)
print("\n📍 Summary:")
print(f"   ✅ Top-50 ICD-10 codes computed from MIMIC-IV")
print(f"   ✅ Fresh dataset built: {len(df):,} samples")
print(f"   ✅ Fresh train/val/test splits created")
print(f"   ✅ Concept bottleneck model trained")
print(f"   ✅ Macro F1: {macro_f1:.4f} | Micro F1: {micro_f1:.4f}")
print(f"\n📁 All artifacts saved to: {OUTPUT_BASE}")
print(f"\nNext: Run shifamind2_p2.py (GraphSAGE) with this run folder")
print("\nAlhamdulillah! 🤲")
