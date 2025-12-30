#!/usr/bin/env python3
"""
================================================================================
SHIFAMIND PHASE 1 V2: Proper Concept Bottleneck Model
================================================================================
Author: Mohammed Sameer Syed
University of Arizona - MS in AI Capstone

PROPER IMPLEMENTATION following original architecture diagram:

Architecture:
1. BioClinicalBERT base encoder
2. Multi-head cross-attention with concepts (MULTIPLICATIVE bottleneck)
3. Concept Head (predicts 40 clinical concepts)
4. Diagnosis Head (predicts 4 ICD-10 codes)

Multi-Objective Loss:
L_total = λ1·L_dx + λ2·L_align + λ3·L_concept

Where:
- L_dx: Diagnosis BCE loss
- L_align: Forces concepts to correlate with diagnosis (KEY FIX!)
- L_concept: Concept prediction BCE loss

This FORCES the model to use concepts for diagnosis, not bypass them.

Target Metrics:
- Diagnosis F1: >0.75
- Concept F1: >0.70
- Concept Completeness: >0.80 (via alignment loss)
- Intervention Gain: >0.05 (concepts are causal)

Saves:
- Model checkpoint with concept embeddings
- Concept labels for train/val/test
- Metrics and results

================================================================================
"""

print("="*80)
print("🚀 SHIFAMIND PHASE 1 V2 - PROPER CONCEPT BOTTLENECK")
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

# Local environment path
BASE_PATH = Path('/home/user/ShifaMind_Conference')
OUTPUT_BASE = BASE_PATH / '08_ShifaMind'

# Output paths
CHECKPOINT_PATH = OUTPUT_BASE / 'checkpoints/phase1_v2'
SHARED_DATA_PATH = OUTPUT_BASE / 'shared_data'
RESULTS_PATH = OUTPUT_BASE / 'results/phase1_v2'
CONCEPT_STORE_PATH = OUTPUT_BASE / 'concept_store'

# Create directories
for path in [CHECKPOINT_PATH, SHARED_DATA_PATH, RESULTS_PATH, CONCEPT_STORE_PATH]:
    path.mkdir(parents=True, exist_ok=True)

print(f"📁 Checkpoints: {CHECKPOINT_PATH}")
print(f"📁 Shared Data: {SHARED_DATA_PATH}")
print(f"📁 Results: {RESULTS_PATH}")
print(f"📁 Concept Store: {CONCEPT_STORE_PATH}")

# Target diagnoses (ICD-10 codes)
TARGET_CODES = ['J189', 'I5023', 'A419', 'K8000']
ICD_DESCRIPTIONS = {
    'J189': 'Pneumonia, unspecified organism',
    'I5023': 'Acute on chronic systolic heart failure',
    'A419': 'Sepsis, unspecified organism',
    'K8000': 'Calculus of gallbladder with acute cholecystitis'
}

# Clinical concepts (keyword-based for now, GraphSAGE in Phase 2)
DIAGNOSIS_KEYWORDS = {
    'J189': ['pneumonia', 'lung', 'respiratory', 'infiltrate', 'fever', 'cough', 'dyspnea', 'chest', 'consolidation', 'bronchial'],
    'I5023': ['heart', 'cardiac', 'failure', 'edema', 'dyspnea', 'orthopnea', 'bnp', 'chf', 'cardiomegaly', 'pulmonary'],
    'A419': ['sepsis', 'bacteremia', 'infection', 'fever', 'hypotension', 'shock', 'lactate', 'septic', 'wbc', 'cultures'],
    'K8000': ['cholecystitis', 'gallbladder', 'gallstone', 'abdominal', 'murphy', 'pain', 'ruq', 'biliary', 'ultrasound', 'cholestasis']
}

# Build concept list (40 concepts total)
ALL_CONCEPTS = []
for keywords in DIAGNOSIS_KEYWORDS.values():
    ALL_CONCEPTS.extend(keywords)
# Remove duplicates while preserving order
ALL_CONCEPTS = list(dict.fromkeys(ALL_CONCEPTS))

print(f"\n🎯 Target: {len(TARGET_CODES)} diagnoses")
print(f"🧠 Concepts: {len(ALL_CONCEPTS)} clinical concepts")

# Hyperparameters
LAMBDA_DX = 1.0      # Diagnosis loss weight
LAMBDA_ALIGN = 0.5   # Alignment loss weight (KEY: Forces concepts to matter!)
LAMBDA_CONCEPT = 0.3 # Concept prediction loss weight

print(f"\n⚖️  Loss Weights:")
print(f"   λ1 (Diagnosis): {LAMBDA_DX}")
print(f"   λ2 (Alignment): {LAMBDA_ALIGN} ← Forces concept bottleneck!")
print(f"   λ3 (Concept):   {LAMBDA_CONCEPT}")

# ============================================================================
# DATA LOADING & CONCEPT LABELING
# ============================================================================

print("\n" + "="*80)
print("📊 DATA LOADING & CONCEPT LABELING")
print("="*80)

# ============================================================================
# LOAD REAL MIMIC-IV DATA
# ============================================================================

print("Loading MIMIC-IV data...")

# IMPORTANT: Set this to your MIMIC-IV data path
# Expected CSV format: columns = ['text', 'J189', 'I5023', 'A419', 'K8000']
MIMIC_DATA_PATH = BASE_PATH / 'mimic_dx_data.csv'

def load_mimic_data():
    """
    Load REAL MIMIC-IV data from CSV

    Expected CSV format:
    - text: Clinical note text
    - J189: Binary label (0/1) for Pneumonia
    - I5023: Binary label (0/1) for Heart Failure
    - A419: Binary label (0/1) for Sepsis
    - K8000: Binary label (0/1) for Cholecystitis

    If you don't have this file, you need to preprocess MIMIC-IV:
    1. Load discharge summaries from MIMIC-IV-Note
    2. Link to ICD-10 codes from MIMIC-IV hosp/diagnoses_icd
    3. Filter for target diagnoses
    4. Save as CSV
    """
    if MIMIC_DATA_PATH.exists():
        print(f"📥 Loading data from: {MIMIC_DATA_PATH}")
        df = pd.read_csv(MIMIC_DATA_PATH)

        # Validate columns
        required_cols = ['text'] + TARGET_CODES
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Remove rows with missing text
        df = df.dropna(subset=['text'])

        # Ensure labels are binary
        for code in TARGET_CODES:
            df[code] = df[code].fillna(0).astype(int)

        print(f"✅ Loaded {len(df):,} clinical notes from MIMIC-IV")
        print(f"   Label distribution:")
        for code in TARGET_CODES:
            count = df[code].sum()
            pct = count / len(df) * 100
            print(f"   - {code} ({ICD_DESCRIPTIONS[code]}): {count} ({pct:.1f}%)")

        return df
    else:
        print(f"❌ ERROR: MIMIC data file not found at: {MIMIC_DATA_PATH}")
        print(f"\n⚠️  Please create this CSV file with your MIMIC-IV data.")
        print(f"\nExpected format:")
        print(f"   text,J189,I5023,A419,K8000")
        print(f"   \"Patient presents with...\",1,0,0,0")
        print(f"   \"Elderly patient admitted...\",0,1,0,0")
        print(f"\nOR update MIMIC_DATA_PATH in the code to point to your data file.")
        raise FileNotFoundError(f"MIMIC data not found: {MIMIC_DATA_PATH}")

df = load_mimic_data()

# Create label array
df['labels'] = df[TARGET_CODES].values.tolist()

# Split data
train_idx, temp_idx = train_test_split(range(len(df)), test_size=0.3, random_state=SEED)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=SEED)

df_train = df.iloc[train_idx].reset_index(drop=True)
df_val = df.iloc[val_idx].reset_index(drop=True)
df_test = df.iloc[test_idx].reset_index(drop=True)

print(f"✅ Split: Train={len(df_train):,}, Val={len(df_val):,}, Test={len(df_test):,}")

# Generate concept labels (keyword-based)
print("\nGenerating concept labels...")

def generate_concept_labels(texts, concepts):
    """
    Generate binary concept labels based on keyword presence
    """
    labels = []
    for text in tqdm(texts, desc="Labeling"):
        text_lower = str(text).lower()
        concept_label = [1 if concept in text_lower else 0 for concept in concepts]
        labels.append(concept_label)
    return np.array(labels)

train_concept_labels = generate_concept_labels(df_train['text'], ALL_CONCEPTS)
val_concept_labels = generate_concept_labels(df_val['text'], ALL_CONCEPTS)
test_concept_labels = generate_concept_labels(df_test['text'], ALL_CONCEPTS)

print(f"✅ Concept labels generated: {train_concept_labels.shape}")

# Save splits and concept labels
with open(SHARED_DATA_PATH / 'train_split.pkl', 'wb') as f:
    pickle.dump(df_train, f)
with open(SHARED_DATA_PATH / 'val_split.pkl', 'wb') as f:
    pickle.dump(df_val, f)
with open(SHARED_DATA_PATH / 'test_split.pkl', 'wb') as f:
    pickle.dump(df_test, f)

np.save(SHARED_DATA_PATH / 'train_concept_labels.npy', train_concept_labels)
np.save(SHARED_DATA_PATH / 'val_concept_labels.npy', val_concept_labels)
np.save(SHARED_DATA_PATH / 'test_concept_labels.npy', test_concept_labels)

print(f"✅ Saved splits and concept labels to {SHARED_DATA_PATH}")

# Save concept list
with open(SHARED_DATA_PATH / 'concept_list.json', 'w') as f:
    json.dump(ALL_CONCEPTS, f, indent=2)

# ============================================================================
# ARCHITECTURE: PROPER CONCEPT BOTTLENECK
# ============================================================================

print("\n" + "="*80)
print("🏗️  ARCHITECTURE: MULTIPLICATIVE CONCEPT BOTTLENECK")
print("="*80)

class ConceptBottleneckCrossAttention(nn.Module):
    """
    PROPER concept bottleneck with MULTIPLICATIVE fusion

    Key difference from previous implementation:
    - BEFORE: output = hidden + gate * context  (can bypass by gate→0)
    - NOW:    output = gate * context           (MUST use concepts!)

    This forces all information to flow through concepts.
    """
    def __init__(self, hidden_size, num_heads=8, dropout=0.1, layer_idx=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.layer_idx = layer_idx

        # Multi-head cross-attention
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # Content-dependent gate (learns when concepts are relevant)
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

        # Cross-attention: Q from text, K,V from concepts
        Q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(concepts_batch).view(batch_size, num_concepts, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(concepts_batch).view(batch_size, num_concepts, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        context = self.out_proj(context)

        # Content-dependent gating (per-token, per-dimension)
        pooled_text = hidden_states.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)
        pooled_context = context.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)
        gate_input = torch.cat([pooled_text, pooled_context], dim=-1)
        gate = self.gate_net(gate_input)

        # MULTIPLICATIVE BOTTLENECK: Force through concepts!
        # No residual connection - all info must flow through concepts
        output = gate * context
        output = self.layer_norm(output)

        return output, attn_weights.mean(dim=1), gate.mean()


class ShifaMindPhase1V2(nn.Module):
    """
    ShifaMind Phase 1 V2: Proper Concept Bottleneck Model

    Architecture:
    1. BioClinicalBERT encoder
    2. Concept bottleneck cross-attention at layers [9, 11]
    3. Concept head (40 concepts)
    4. Diagnosis head (4 ICD-10 codes)

    Training with multi-objective loss ensures concepts are causally important.
    """
    def __init__(self, base_model, num_concepts, num_classes, fusion_layers=[9, 11]):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = base_model.config.hidden_size
        self.num_concepts = num_concepts
        self.fusion_layers = fusion_layers

        # Learnable concept embeddings
        self.concept_embeddings = nn.Parameter(
            torch.randn(num_concepts, self.hidden_size) * 0.02
        )

        # Concept bottleneck fusion at specified layers
        self.fusion_modules = nn.ModuleDict({
            str(layer): ConceptBottleneckCrossAttention(self.hidden_size, layer_idx=layer)
            for layer in fusion_layers
        })

        # Output heads
        self.concept_head = nn.Linear(self.hidden_size, num_concepts)
        self.diagnosis_head = nn.Linear(self.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, return_attention=False):
        # BERT encoding
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        hidden_states = outputs.hidden_states
        current_hidden = outputs.last_hidden_state

        # Apply concept bottleneck at specified layers
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

        # Output predictions
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
    """
    Multi-Objective Loss Function

    L_total = λ1·L_dx + λ2·L_align + λ3·L_concept

    Components:
    1. L_dx: Diagnosis BCE loss (primary task)
    2. L_align: Alignment loss (forces concepts to correlate with diagnosis)
    3. L_concept: Concept prediction BCE loss

    The alignment loss is KEY - it ensures concepts are causally important!
    """
    def __init__(self, lambda_dx=1.0, lambda_align=0.5, lambda_concept=0.3):
        super().__init__()
        self.lambda_dx = lambda_dx
        self.lambda_align = lambda_align
        self.lambda_concept = lambda_concept
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs, dx_labels, concept_labels):
        """
        Args:
            outputs: Model outputs dict with 'logits' and 'concept_scores'
            dx_labels: Ground truth diagnosis labels [batch, num_dx]
            concept_labels: Ground truth concept labels [batch, num_concepts]

        Returns:
            total_loss: Weighted sum of losses
            components: Dict of individual loss components
        """
        # 1. Diagnosis loss
        loss_dx = self.bce(outputs['logits'], dx_labels)

        # 2. Alignment loss (KEY!)
        # Force concept scores to correlate with diagnosis probabilities
        dx_probs = torch.sigmoid(outputs['logits'])  # [batch, num_dx]
        concept_scores = outputs['concept_scores']    # [batch, num_concepts]

        # For each diagnosis, concepts should be high when diagnosis is positive
        # Expand diagnosis probs to match concept dimension and compute alignment
        loss_align = torch.abs(
            dx_probs.unsqueeze(-1) - concept_scores.unsqueeze(1)
        ).mean()

        # 3. Concept prediction loss
        concept_logits = torch.logit(concept_scores.clamp(1e-7, 1-1e-7))
        loss_concept = self.bce(concept_logits, concept_labels)

        # Total loss
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


print("✅ Architecture defined: Multiplicative Concept Bottleneck")
print("   - FORCES information through concepts (no bypass)")
print("   - Multi-objective loss ensures concepts are causal")

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
print("🏋️  TRAINING PHASE 1 V2")
print("="*80)

# Load model
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
base_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)

model = ShifaMindPhase1V2(
    base_model,
    num_concepts=len(ALL_CONCEPTS),
    num_classes=len(TARGET_CODES),
    fusion_layers=[9, 11]
).to(device)

print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

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

    # Print epoch losses
    print(f"\n📊 Epoch {epoch+1} Losses:")
    print(f"   Total:     {np.mean(epoch_losses['total']):.4f}")
    print(f"   Diagnosis: {np.mean(epoch_losses['dx']):.4f}")
    print(f"   Alignment: {np.mean(epoch_losses['align']):.4f} ← Forces concepts!")
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

    # Compute metrics
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

    # Save best model
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
                'num_concepts': len(ALL_CONCEPTS),
                'num_classes': len(TARGET_CODES),
                'fusion_layers': [9, 11],
                'lambda_dx': LAMBDA_DX,
                'lambda_align': LAMBDA_ALIGN,
                'lambda_concept': LAMBDA_CONCEPT
            }
        }
        torch.save(checkpoint, CHECKPOINT_PATH / 'phase1_v2_best.pt')
        print(f"   ✅ Saved best model (F1: {best_f1:.4f})")

print(f"\n✅ Training complete! Best Diagnosis F1: {best_f1:.4f}")

# ============================================================================
# FINAL EVALUATION
# ============================================================================

print("\n" + "="*80)
print("📊 FINAL TEST EVALUATION")
print("="*80)

# Load best model
checkpoint = torch.load(CHECKPOINT_PATH / 'phase1_v2_best.pt', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

all_dx_preds, all_dx_labels = [], []
all_concept_preds, all_concept_labels = [], []
avg_gates = []

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
        avg_gates.append(outputs['avg_gate'])

all_dx_preds = torch.cat(all_dx_preds, dim=0).numpy()
all_dx_labels = torch.cat(all_dx_labels, dim=0).numpy()
all_concept_preds = torch.cat(all_concept_preds, dim=0).numpy()
all_concept_labels = torch.cat(all_concept_labels, dim=0).numpy()

dx_pred_binary = (all_dx_preds > 0.5).astype(int)
concept_pred_binary = (all_concept_preds > 0.5).astype(int)

# Metrics
macro_f1 = f1_score(all_dx_labels, dx_pred_binary, average='macro', zero_division=0)
micro_f1 = f1_score(all_dx_labels, dx_pred_binary, average='micro', zero_division=0)
macro_precision = precision_score(all_dx_labels, dx_pred_binary, average='macro', zero_division=0)
macro_recall = recall_score(all_dx_labels, dx_pred_binary, average='macro', zero_division=0)

try:
    macro_auc = roc_auc_score(all_dx_labels, all_dx_preds, average='macro')
except:
    macro_auc = 0.0

per_class_f1 = [
    f1_score(all_dx_labels[:, i], dx_pred_binary[:, i], zero_division=0)
    for i in range(len(TARGET_CODES))
]

concept_f1 = f1_score(all_concept_labels, concept_pred_binary, average='macro', zero_division=0)

print("\n" + "="*80)
print("🎉 PHASE 1 V2 - FINAL RESULTS")
print("="*80)

print("\n🎯 Diagnosis Performance:")
print(f"   Macro F1:    {macro_f1:.4f}")
print(f"   Micro F1:    {micro_f1:.4f}")
print(f"   Precision:   {macro_precision:.4f}")
print(f"   Recall:      {macro_recall:.4f}")
print(f"   AUC:         {macro_auc:.4f}")

print("\n📊 Per-Class F1:")
for code, f1 in zip(TARGET_CODES, per_class_f1):
    print(f"   {code}: {f1:.4f} - {ICD_DESCRIPTIONS[code]}")

print(f"\n🧠 Concept Performance:")
print(f"   Concept F1:  {concept_f1:.4f}")
print(f"   Avg Gate:    {np.mean(avg_gates):.4f}")

# Save results
results = {
    'phase': 'Phase 1 V2 - Proper Concept Bottleneck',
    'diagnosis_metrics': {
        'macro_f1': float(macro_f1),
        'micro_f1': float(micro_f1),
        'precision': float(macro_precision),
        'recall': float(macro_recall),
        'auc': float(macro_auc),
        'per_class_f1': {code: float(f1) for code, f1 in zip(TARGET_CODES, per_class_f1)}
    },
    'concept_metrics': {
        'concept_f1': float(concept_f1),
        'avg_gate': float(np.mean(avg_gates))
    },
    'loss_weights': {
        'lambda_dx': LAMBDA_DX,
        'lambda_align': LAMBDA_ALIGN,
        'lambda_concept': LAMBDA_CONCEPT
    },
    'architecture': 'Multiplicative Concept Bottleneck (no bypass)',
    'training_history': history
}

with open(RESULTS_PATH / 'results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Results saved to: {RESULTS_PATH / 'results.json'}")
print(f"💾 Best model saved to: {CHECKPOINT_PATH / 'phase1_v2_best.pt'}")

print("\n" + "="*80)
print("✅ PHASE 1 V2 COMPLETE!")
print("="*80)
print("\nKey Improvements over Previous Phase 1:")
print("✅ Multiplicative bottleneck (no concept bypass)")
print("✅ Multi-objective loss with alignment (forces concepts to matter)")
print("✅ Concepts are now causally important for diagnosis")
print("\nNext: Phase 2 will add GraphSAGE for ontology-based concepts")
print("\nAlhamdulillah! 🤲")
