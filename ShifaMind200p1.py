#!/usr/bin/env python3
"""
================================================================================
SHIFAMIND PHASE 1: Concept Bottleneck Model for 50 ICD-10 Codes
================================================================================
Author: Mohammed Sameer Syed
University of Arizona - MS in AI Capstone

PROPER IMPLEMENTATION:
- 50 ICD-10 diagnosis codes (top 50 from MIMIC-IV)
- 75 clinical concepts
- Multiplicative concept bottleneck (forces information through concepts)
- Multi-objective loss: diagnosis + alignment + concept

Architecture:
1. BioClinicalBERT encoder
2. Cross-attention with concept embeddings (MULTIPLICATIVE bottleneck)
3. Concept prediction head
4. Diagnosis prediction head

Loss:
L_total = λ_dx·L_dx + λ_align·L_align + λ_concept·L_concept

Target Metrics:
- Diagnosis F1: >0.70 (harder with 50 codes)
- Concept F1: >0.65
- Concept Completeness: >0.75

================================================================================
"""

print("="*80)
print("🚀 SHIFAMIND PHASE 1 - CONCEPT BOTTLENECK FOR 50 ICD CODES")
print("="*80)

# ============================================================================
# IMPORTS
# ============================================================================

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

import json
import pickle
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from collections import defaultdict
import re

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
OUTPUT_BASE = BASE_PATH / '09_ShifaMind'

# Phase 1 specific paths
CHECKPOINT_PATH = OUTPUT_BASE / 'checkpoints/phase1'
RESULTS_PATH = OUTPUT_BASE / 'results/phase1'
SHARED_DATA_PATH = OUTPUT_BASE / 'shared_data'

# Create directories
for path in [CHECKPOINT_PATH, RESULTS_PATH, SHARED_DATA_PATH]:
    path.mkdir(parents=True, exist_ok=True)

print(f"📁 Base Path: {BASE_PATH}")
print(f"📁 Checkpoints: {CHECKPOINT_PATH}")
print(f"📁 Shared Data: {SHARED_DATA_PATH}")
print(f"📁 Results: {RESULTS_PATH}")

# ============================================================================
# LOAD DATA FIRST TO AUTO-DETECT ICD CODES
# ============================================================================

print("\n" + "="*80)
print("📊 LOADING MIMIC-IV DATA")
print("="*80)

MIMIC_DATA_PATH = BASE_PATH / 'mimic_dx_data.csv'

print(f"\n📥 Loading data from: {MIMIC_DATA_PATH}")

if not MIMIC_DATA_PATH.exists():
    raise FileNotFoundError(
        f"❌ MIMIC data not found at: {MIMIC_DATA_PATH}\n\n"
        f"Please run prepare_mimic_data_50codes.py first to create this file.\n\n"
        f"Steps:\n"
        f"  1. Run: python prepare_mimic_data_50codes.py\n"
        f"  2. Then run this script"
    )

# Load CSV
df = pd.read_csv(MIMIC_DATA_PATH)

print(f"✅ Loaded {len(df):,} total clinical notes")
print(f"📋 Columns found: {len(df.columns)}")

# Auto-detect ICD code columns (all columns except metadata)
metadata_cols = ['text', 'subject_id', 'hadm_id', 'note_id', 'charttime', 'storetime']
TARGET_CODES = [col for col in df.columns if col not in metadata_cols]

print(f"\n🎯 Detected {len(TARGET_CODES)} ICD-10 diagnosis codes")
print(f"   First 10: {TARGET_CODES[:10]}")
print(f"   Last 10:  {TARGET_CODES[-10:]}")

# 75 Clinical Concepts
COMMON_CLINICAL_CONCEPTS = [
    # Vital signs & measurements
    'hypertension', 'hypotension', 'tachycardia', 'bradycardia', 'fever', 'hypothermia',
    'tachypnea', 'hypoxia', 'hypercapnia', 'acidosis', 'alkalosis',
    # Symptoms
    'dyspnea', 'chest pain', 'edema', 'fatigue', 'confusion', 'syncope',
    'nausea', 'vomiting', 'diarrhea', 'abdominal pain', 'headache',
    # Cardiac
    'cardiac arrest', 'heart failure', 'myocardial infarction', 'arrhythmia',
    'cardiogenic shock', 'pericardial effusion', 'valvular disease',
    # Respiratory
    'respiratory failure', 'pneumonia', 'copd exacerbation', 'asthma',
    'pulmonary edema', 'pleural effusion', 'pneumothorax',
    # Renal
    'acute kidney injury', 'chronic kidney disease', 'dialysis', 'oliguria',
    'anuria', 'proteinuria', 'hematuria',
    # Infection/Sepsis
    'sepsis', 'septic shock', 'bacteremia', 'uti', 'cellulitis',
    # Metabolic
    'diabetes', 'hyperglycemia', 'hypoglycemia', 'electrolyte imbalance',
    'hyponatremia', 'hypernatremia', 'hypokalemia', 'hyperkalemia',
    # Neurological
    'altered mental status', 'seizure', 'stroke', 'delirium', 'coma',
    # GI/Hepatic
    'gi bleed', 'hepatic encephalopathy', 'ascites', 'liver failure',
    # Hematologic
    'anemia', 'thrombocytopenia', 'coagulopathy', 'dic',
    # Other
    'mechanical ventilation', 'intubation', 'vasopressor support'
]

NUM_DIAGNOSES = len(TARGET_CODES)
NUM_CONCEPTS = len(COMMON_CLINICAL_CONCEPTS)

print(f"🧠 Concepts: {NUM_CONCEPTS} clinical concepts")

# Hyperparameters
LAMBDA_DX = 2.0      # Diagnosis loss weight (increased for 50 codes)
LAMBDA_ALIGN = 0.7   # Alignment loss weight (forces concept bottleneck)
LAMBDA_CONCEPT = 0.4 # Concept loss weight

BATCH_SIZE = 8
EPOCHS = 5
LR = 2e-5
MAX_LENGTH = 512

print(f"\n⚖️  Loss Weights:")
print(f"   λ_dx (Diagnosis): {LAMBDA_DX}")
print(f"   λ_align (Alignment): {LAMBDA_ALIGN}")
print(f"   λ_concept (Concept): {LAMBDA_CONCEPT}")

# ============================================================================
# FILTER DATA
# ============================================================================

print("\n" + "="*80)
print("🔍 FILTERING DATA")
print("="*80)

# Verify text column exists
if 'text' not in df.columns:
    raise ValueError("❌ 'text' column not found in data")

# Filter to samples that have at least one of the target codes
mask = df[TARGET_CODES].sum(axis=1) > 0
df_filtered = df[mask].copy()

print(f"✅ Filtered to {len(df_filtered)} samples with target diagnoses")
print(f"   (Removed {len(df) - len(df_filtered)} samples with no target codes)")

# Verify text column exists
if 'text' not in df_filtered.columns:
    raise ValueError("❌ 'text' column not found in data")

# Create labels array
labels = df_filtered[TARGET_CODES].values.astype(np.float32)

print(f"\n📊 Label Statistics:")
print(f"   Shape: {labels.shape}")
print(f"   Total positive labels: {labels.sum():.0f}")
print(f"   Avg labels per sample: {labels.sum(axis=1).mean():.2f}")
print(f"   Label frequency (top 10):")
for i in range(min(10, NUM_DIAGNOSES)):
    count = labels[:, i].sum()
    pct = (count / len(labels)) * 100
    print(f"      {TARGET_CODES[i]:8s}: {count:5.0f} ({pct:5.2f}%)")

# ============================================================================
# TRAIN/VAL/TEST SPLIT
# ============================================================================

print("\n" + "="*80)
print("🔪 CREATING TRAIN/VAL/TEST SPLITS")
print("="*80)

# Split: 70% train, 15% val, 15% test
texts = df_filtered['text'].tolist()

# First split: 70% train, 30% temp
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    texts, labels, test_size=0.3, random_state=SEED, stratify=labels[:, 0]
)

# Second split: 15% val, 15% test
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, random_state=SEED, stratify=temp_labels[:, 0]
)

print(f"✅ Train: {len(train_texts)} samples")
print(f"✅ Val:   {len(val_texts)} samples")
print(f"✅ Test:  {len(test_texts)} samples")

# ============================================================================
# CONCEPT LABELING
# ============================================================================

print("\n" + "="*80)
print("🏷️  GENERATING CONCEPT LABELS")
print("="*80)

def label_concepts(texts: List[str], concepts: List[str]) -> np.ndarray:
    """
    Fast keyword-based concept labeling
    Returns binary matrix: (n_samples, n_concepts)
    """
    print(f"   Processing {len(texts):,} texts for {len(concepts)} concepts...")

    # Pre-lowercase concepts once
    concepts_lower = [c.lower() for c in concepts]

    # Vectorized approach - much faster
    concept_labels = np.zeros((len(texts), len(concepts)), dtype=np.float32)

    # Process in chunks for progress bar
    chunk_size = 10000
    for i in tqdm(range(0, len(texts), chunk_size), desc="Labeling concepts"):
        chunk_end = min(i + chunk_size, len(texts))
        chunk_texts = texts[i:chunk_end]

        for j, concept in enumerate(concepts_lower):
            # Vectorized string search
            for k, text in enumerate(chunk_texts):
                if concept in text.lower():
                    concept_labels[i + k, j] = 1.0

    return concept_labels

train_concept_labels = label_concepts(train_texts, COMMON_CLINICAL_CONCEPTS)
val_concept_labels = label_concepts(val_texts, COMMON_CLINICAL_CONCEPTS)
test_concept_labels = label_concepts(test_texts, COMMON_CLINICAL_CONCEPTS)

print(f"\n✅ Concept labels generated:")
print(f"   Train: {train_concept_labels.shape}")
print(f"   Val:   {val_concept_labels.shape}")
print(f"   Test:  {test_concept_labels.shape}")
print(f"   Avg concepts per sample: {train_concept_labels.sum(axis=1).mean():.2f}")

# Save splits
print("\n💾 Saving splits and concept labels...")

with open(SHARED_DATA_PATH / 'train_split.pkl', 'wb') as f:
    pickle.dump(pd.DataFrame({'text': train_texts, 'labels': list(train_labels)}), f)
with open(SHARED_DATA_PATH / 'val_split.pkl', 'wb') as f:
    pickle.dump(pd.DataFrame({'text': val_texts, 'labels': list(val_labels)}), f)
with open(SHARED_DATA_PATH / 'test_split.pkl', 'wb') as f:
    pickle.dump(pd.DataFrame({'text': test_texts, 'labels': list(test_labels)}), f)

np.save(SHARED_DATA_PATH / 'train_concept_labels.npy', train_concept_labels)
np.save(SHARED_DATA_PATH / 'val_concept_labels.npy', val_concept_labels)
np.save(SHARED_DATA_PATH / 'test_concept_labels.npy', test_concept_labels)

# Save concept list
with open(SHARED_DATA_PATH / 'concept_list.json', 'w') as f:
    json.dump(COMMON_CLINICAL_CONCEPTS, f, indent=2)

# Save target codes
with open(SHARED_DATA_PATH / 'target_codes.json', 'w') as f:
    json.dump(TARGET_CODES, f, indent=2)

print(f"✅ Saved to: {SHARED_DATA_PATH}")

# ============================================================================
# DATASET
# ============================================================================

tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

class MIMICDataset(Dataset):
    def __init__(self, texts, dx_labels, concept_labels, tokenizer, max_length=512):
        self.texts = texts
        self.dx_labels = dx_labels
        self.concept_labels = concept_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'dx_labels': torch.tensor(self.dx_labels[idx], dtype=torch.float32),
            'concept_labels': torch.tensor(self.concept_labels[idx], dtype=torch.float32)
        }

train_dataset = MIMICDataset(train_texts, train_labels, train_concept_labels, tokenizer, MAX_LENGTH)
val_dataset = MIMICDataset(val_texts, val_labels, val_concept_labels, tokenizer, MAX_LENGTH)
test_dataset = MIMICDataset(test_texts, test_labels, test_concept_labels, tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"\n✅ Datasets ready:")
print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches:   {len(val_loader)}")
print(f"   Test batches:  {len(test_loader)}")

# ============================================================================
# MODEL: CONCEPT BOTTLENECK
# ============================================================================

print("\n" + "="*80)
print("🏗️  BUILDING CONCEPT BOTTLENECK MODEL")
print("="*80)

class ConceptBottleneckModel(nn.Module):
    """
    Multiplicative Concept Bottleneck Model
    Forces all diagnosis information through concept representations
    """
    def __init__(self, encoder_name, num_concepts, num_diagnoses, hidden_dim=768):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.hidden_dim = hidden_dim
        self.num_concepts = num_concepts
        self.num_diagnoses = num_diagnoses

        # Concept embeddings (learnable)
        self.concept_embeddings = nn.Parameter(
            torch.randn(num_concepts, hidden_dim) * 0.02
        )

        # Cross-attention: concepts attend to encoder outputs
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Concept prediction head
        self.concept_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Diagnosis prediction head (from concepts only!)
        self.diagnosis_head = nn.Sequential(
            nn.Linear(num_concepts, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_diagnoses)
        )

        # Layer norm
        self.concept_norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.size(0)

        # 1. Encode text with BioClinicalBERT
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = encoder_outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

        # 2. Cross-attention: concepts attend to text
        concept_emb = self.concept_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        concept_emb = self.concept_norm(concept_emb)

        attended_concepts, attn_weights = self.cross_attention(
            query=concept_emb,           # (batch, num_concepts, hidden_dim)
            key=hidden_states,           # (batch, seq_len, hidden_dim)
            value=hidden_states,
            key_padding_mask=~attention_mask.bool()
        )  # Output: (batch, num_concepts, hidden_dim)

        # 3. Predict concepts (binary for each concept)
        concept_logits = self.concept_head(attended_concepts).squeeze(-1)  # (batch, num_concepts)

        # 4. BOTTLENECK: Use concept predictions for diagnosis
        # Apply sigmoid to get concept activations, then pass to diagnosis head
        concept_activations = torch.sigmoid(concept_logits)  # (batch, num_concepts)

        # 5. Predict diagnoses from concepts
        diagnosis_logits = self.diagnosis_head(concept_activations)  # (batch, num_diagnoses)

        return {
            'logits': diagnosis_logits,
            'concept_logits': concept_logits,
            'concept_activations': concept_activations,
            'attention_weights': attn_weights
        }

model = ConceptBottleneckModel(
    encoder_name='emilyalsentzer/Bio_ClinicalBERT',
    num_concepts=NUM_CONCEPTS,
    num_diagnoses=NUM_DIAGNOSES
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"✅ Model loaded: {total_params:,} parameters")

# ============================================================================
# LOSS FUNCTION
# ============================================================================

class MultiObjectiveLoss(nn.Module):
    """
    L_total = λ_dx·L_dx + λ_align·L_align + λ_concept·L_concept
    """
    def __init__(self, lambda_dx=2.0, lambda_align=0.7, lambda_concept=0.4):
        super().__init__()
        self.lambda_dx = lambda_dx
        self.lambda_align = lambda_align
        self.lambda_concept = lambda_concept
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs, dx_labels, concept_labels):
        # 1. Diagnosis loss
        loss_dx = self.bce(outputs['logits'], dx_labels)

        # 2. Concept prediction loss
        loss_concept = self.bce(outputs['concept_logits'], concept_labels)

        # 3. Alignment loss (concepts should correlate with diagnoses)
        # When diagnosis is present, relevant concepts should activate
        concept_activations = outputs['concept_activations']

        # Simple alignment: encourage high concept activation when diagnosis present
        # For each diagnosis, we want some concepts to be active
        concept_contrib = concept_activations.mean(dim=1)  # (batch,)
        dx_present = (dx_labels.sum(dim=1) > 0).float()    # (batch,)

        # When diagnosis present, concepts should be active
        loss_align = F.mse_loss(concept_contrib, dx_present)

        # Total loss
        total_loss = (
            self.lambda_dx * loss_dx +
            self.lambda_concept * loss_concept +
            self.lambda_align * loss_align
        )

        return total_loss, {
            'loss_dx': loss_dx.item(),
            'loss_concept': loss_concept.item(),
            'loss_align': loss_align.item(),
            'total': total_loss.item()
        }

criterion = MultiObjectiveLoss(
    lambda_dx=LAMBDA_DX,
    lambda_align=LAMBDA_ALIGN,
    lambda_concept=LAMBDA_CONCEPT
)

# ============================================================================
# TRAINING
# ============================================================================

print("\n" + "="*80)
print("🏋️  TRAINING")
print("="*80)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=len(train_loader) * 2,
    num_training_steps=len(train_loader) * EPOCHS
)

best_val_f1 = 0.0
history = {'train_loss': [], 'val_loss': [], 'val_dx_f1': [], 'val_concept_f1': []}

for epoch in range(EPOCHS):
    print(f"\n{'='*80}")
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"{'='*80}")

    # TRAIN
    model.train()
    train_loss = 0.0
    train_components = defaultdict(float)

    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        dx_labels = batch['dx_labels'].to(device)
        concept_labels = batch['concept_labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss, components = criterion(outputs, dx_labels, concept_labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        for k, v in components.items():
            train_components[k] += v

    avg_train_loss = train_loss / len(train_loader)
    print(f"\n📉 Train Loss: {avg_train_loss:.4f}")
    print(f"   • Diagnosis:  {train_components['loss_dx']/len(train_loader):.4f}")
    print(f"   • Concept:    {train_components['loss_concept']/len(train_loader):.4f}")
    print(f"   • Alignment:  {train_components['loss_align']/len(train_loader):.4f}")

    # VALIDATION
    model.eval()
    val_loss = 0.0
    all_dx_preds, all_dx_labels = [], []
    all_concept_preds, all_concept_labels = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            dx_labels = batch['dx_labels'].to(device)
            concept_labels = batch['concept_labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss, _ = criterion(outputs, dx_labels, concept_labels)
            val_loss += loss.item()

            # Predictions
            dx_preds = torch.sigmoid(outputs['logits']).cpu().numpy() > 0.5
            concept_preds = torch.sigmoid(outputs['concept_logits']).cpu().numpy() > 0.5

            all_dx_preds.append(dx_preds)
            all_dx_labels.append(dx_labels.cpu().numpy())
            all_concept_preds.append(concept_preds)
            all_concept_labels.append(concept_labels.cpu().numpy())

    # Metrics
    all_dx_preds = np.vstack(all_dx_preds)
    all_dx_labels = np.vstack(all_dx_labels)
    all_concept_preds = np.vstack(all_concept_preds)
    all_concept_labels = np.vstack(all_concept_labels)

    val_dx_f1 = f1_score(all_dx_labels, all_dx_preds, average='macro', zero_division=0)
    val_concept_f1 = f1_score(all_concept_labels, all_concept_preds, average='macro', zero_division=0)

    avg_val_loss = val_loss / len(val_loader)

    print(f"\n📊 Validation Results:")
    print(f"   Loss:        {avg_val_loss:.4f}")
    print(f"   Diagnosis F1:  {val_dx_f1:.4f}")
    print(f"   Concept F1:    {val_concept_f1:.4f}")

    # Save best model
    if val_dx_f1 > best_val_f1:
        best_val_f1 = val_dx_f1
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_dx_f1': val_dx_f1,
            'val_concept_f1': val_concept_f1,
            'concept_embeddings': model.concept_embeddings.data.cpu()
        }
        torch.save(checkpoint, CHECKPOINT_PATH / 'best_model.pt')
        print(f"   ✅ Best model saved! (F1: {val_dx_f1:.4f})")

    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['val_dx_f1'].append(val_dx_f1)
    history['val_concept_f1'].append(val_concept_f1)

# ============================================================================
# EVALUATION ON TEST SET
# ============================================================================

print("\n" + "="*80)
print("📊 FINAL EVALUATION ON TEST SET")
print("="*80)

# Load best model
checkpoint = torch.load(CHECKPOINT_PATH / 'best_model.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

all_dx_preds, all_dx_labels = [], []
all_concept_preds, all_concept_labels = [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        dx_labels = batch['dx_labels'].to(device)
        concept_labels = batch['concept_labels'].to(device)

        outputs = model(input_ids, attention_mask)

        dx_preds = torch.sigmoid(outputs['logits']).cpu().numpy() > 0.5
        concept_preds = torch.sigmoid(outputs['concept_logits']).cpu().numpy() > 0.5

        all_dx_preds.append(dx_preds)
        all_dx_labels.append(dx_labels.cpu().numpy())
        all_concept_preds.append(concept_preds)
        all_concept_labels.append(concept_labels.cpu().numpy())

all_dx_preds = np.vstack(all_dx_preds)
all_dx_labels = np.vstack(all_dx_labels)
all_concept_preds = np.vstack(all_concept_preds)
all_concept_labels = np.vstack(all_concept_labels)

# Calculate metrics
test_dx_f1_macro = f1_score(all_dx_labels, all_dx_preds, average='macro', zero_division=0)
test_dx_f1_micro = f1_score(all_dx_labels, all_dx_preds, average='micro', zero_division=0)
test_concept_f1 = f1_score(all_concept_labels, all_concept_preds, average='macro', zero_division=0)

print(f"\n{'='*80}")
print("✅ FINAL TEST RESULTS")
print(f"{'='*80}")
print(f"\n📊 Diagnosis Performance:")
print(f"   Macro F1:  {test_dx_f1_macro:.4f}")
print(f"   Micro F1:  {test_dx_f1_micro:.4f}")
print(f"\n🧠 Concept Performance:")
print(f"   Macro F1:  {test_concept_f1:.4f}")

# Per-code performance
print(f"\n📋 Per-Code F1 (Top 10):")
per_code_f1 = []
for i in range(NUM_DIAGNOSES):
    f1 = f1_score(all_dx_labels[:, i], all_dx_preds[:, i], zero_division=0)
    per_code_f1.append(f1)

sorted_indices = np.argsort(per_code_f1)[::-1]
for idx in sorted_indices[:10]:
    code = TARGET_CODES[idx]
    f1 = per_code_f1[idx]
    support = all_dx_labels[:, idx].sum()
    print(f"   {code:8s}: F1={f1:.4f} (n={support:.0f})")

# Save results
results = {
    'test_dx_f1_macro': float(test_dx_f1_macro),
    'test_dx_f1_micro': float(test_dx_f1_micro),
    'test_concept_f1': float(test_concept_f1),
    'per_code_f1': {TARGET_CODES[i]: float(per_code_f1[i]) for i in range(NUM_DIAGNOSES)},
    'history': history,
    'config': {
        'num_diagnoses': NUM_DIAGNOSES,
        'num_concepts': NUM_CONCEPTS,
        'lambda_dx': LAMBDA_DX,
        'lambda_align': LAMBDA_ALIGN,
        'lambda_concept': LAMBDA_CONCEPT,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'lr': LR
    }
}

with open(RESULTS_PATH / 'phase1_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to: {RESULTS_PATH / 'phase1_results.json'}")
print(f"✅ Model saved to: {CHECKPOINT_PATH / 'best_model.pt'}")
print(f"✅ Splits saved to: {SHARED_DATA_PATH}")

print("\n" + "="*80)
print("🎉 PHASE 1 COMPLETE!")
print("="*80)
print(f"\n💾 Outputs:")
print(f"   • Model checkpoint: {CHECKPOINT_PATH / 'best_model.pt'}")
print(f"   • Train/val/test splits: {SHARED_DATA_PATH}")
print(f"   • Concept labels: {SHARED_DATA_PATH}")
print(f"   • Results: {RESULTS_PATH / 'phase1_results.json'}")
print("\n✅ Ready for Phase 2 (GraphSAGE)")
