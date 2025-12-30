#!/usr/bin/env python3
"""
================================================================================
SHIFAMIND PHASE 4: Ablation Studies
================================================================================
Author: Mohammed Sameer Syed
University of Arizona - MS in AI Capstone

SELF-CONTAINED COLAB SCRIPT - Copy-paste ready!

Ablations (5):
1. Full Model (Phase 2 Fixed + RAG)
2. No RAG (Phase 1 Fixed only)
3. Middle Layers [5, 7]
4. Early Layers [2, 3, 4]
5. BERT + Concept Head (no fusion)

Loads:
- Train/val/test splits from Phase 1
- Phase 1 Fixed & Phase 2 Fixed checkpoints

Saves:
- Ablation results to 07_ShifaMind/results/phase4_fixed/

TARGET: Show that our architecture choices matter
================================================================================
"""

print("="*80)
print("🚀 SHIFAMIND PHASE 4 - ABLATION STUDIES")
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
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from transformers import (
    AutoTokenizer, AutoModel,
    get_linear_schedule_with_warmup
)

import json
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, List
import pickle
import time

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

# ⚠️ IMPORTANT: Update this path to match YOUR Google Drive structure
# The script expects these checkpoint files to exist from running Phase 1 & 2:
#   - checkpoints/phase1_fixed/phase1_fixed_best.pt (or phase1_fixed_final.pt)
#   - checkpoints/phase2_fixed/phase2_fixed_best.pt (or phase2_fixed_final.pt)
BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')
OUTPUT_BASE = BASE_PATH / '07_ShifaMind'

# Input paths
SHARED_DATA_PATH = OUTPUT_BASE / 'shared_data'

# Try both possible Phase 1 checkpoint names
PHASE1_FIXED_CHECKPOINT_BEST = OUTPUT_BASE / 'checkpoints/phase1_fixed/phase1_fixed_best.pt'
PHASE1_FIXED_CHECKPOINT_FINAL = OUTPUT_BASE / 'checkpoints/phase1_fixed/phase1_fixed_final.pt'

# Try both possible Phase 2 checkpoint names
PHASE2_FIXED_CHECKPOINT_BEST = OUTPUT_BASE / 'checkpoints/phase2_fixed/phase2_fixed_best.pt'
PHASE2_FIXED_CHECKPOINT_FINAL = OUTPUT_BASE / 'checkpoints/phase2_fixed/phase2_fixed_final.pt'

# Check which checkpoints exist
if PHASE1_FIXED_CHECKPOINT_BEST.exists():
    PHASE1_FIXED_CHECKPOINT = PHASE1_FIXED_CHECKPOINT_BEST
elif PHASE1_FIXED_CHECKPOINT_FINAL.exists():
    PHASE1_FIXED_CHECKPOINT = PHASE1_FIXED_CHECKPOINT_FINAL
else:
    PHASE1_FIXED_CHECKPOINT = PHASE1_FIXED_CHECKPOINT_BEST  # Default, will error later with helpful message

if PHASE2_FIXED_CHECKPOINT_BEST.exists():
    PHASE2_FIXED_CHECKPOINT = PHASE2_FIXED_CHECKPOINT_BEST
elif PHASE2_FIXED_CHECKPOINT_FINAL.exists():
    PHASE2_FIXED_CHECKPOINT = PHASE2_FIXED_CHECKPOINT_FINAL
else:
    PHASE2_FIXED_CHECKPOINT = PHASE2_FIXED_CHECKPOINT_BEST  # Default, will error later with helpful message

# Output paths
CHECKPOINT_PATH = OUTPUT_BASE / 'checkpoints/phase4_fixed'
RESULTS_PATH = OUTPUT_BASE / 'results/phase4_fixed'

CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

print(f"📁 Checkpoints: {CHECKPOINT_PATH}")
print(f"📁 Results: {RESULTS_PATH}")

TARGET_CODES = ['J189', 'I5023', 'A419', 'K8000']
ICD_DESCRIPTIONS = {
    'J189': 'Pneumonia, unspecified organism',
    'I5023': 'Acute on chronic systolic heart failure',
    'A419': 'Sepsis, unspecified organism',
    'K8000': 'Calculus of gallbladder with acute cholecystitis'
}

print(f"\n🎯 Target: {len(TARGET_CODES)} diagnoses")

# ============================================================================
# ARCHITECTURE (ALL INLINE - NO IMPORTS)
# ============================================================================

print("\n" + "="*80)
print("🏗️  ARCHITECTURE COMPONENTS")
print("="*80)

class AdaptiveGatedCrossAttention(nn.Module):
    """Fixed cross-attention with learnable gates"""
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

        # Adaptive gate (4 layers, input = hidden*2 + 1 for scalar relevance)
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size * 2 + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
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

        # Compute relevance
        text_pooled = hidden_states.mean(dim=1)
        concept_pooled = concepts_batch.mean(dim=1)
        relevance = F.cosine_similarity(text_pooled, concept_pooled, dim=-1)
        relevance = relevance.unsqueeze(-1).unsqueeze(-1).expand(-1, seq_len, -1)

        # Learnable gate
        gate_input = torch.cat([hidden_states, context, relevance], dim=-1)
        gate_values = self.gate_net(gate_input)

        output = hidden_states + gate_values * context
        output = self.layer_norm(output)

        return output, attn_weights.mean(dim=1)


class FlexibleShifaMind(nn.Module):
    """Flexible architecture for ablations"""
    def __init__(self, base_model, num_concepts, num_classes, fusion_layers=None, use_fusion=True):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = base_model.config.hidden_size
        self.fusion_layers = fusion_layers if fusion_layers else []
        self.use_fusion = use_fusion

        if use_fusion and len(self.fusion_layers) > 0:
            self.fusion_modules = nn.ModuleDict({
                str(layer): AdaptiveGatedCrossAttention(self.hidden_size, layer_idx=layer)
                for layer in self.fusion_layers
            })

        self.diagnosis_head = nn.Linear(self.hidden_size, num_classes)
        self.concept_head = nn.Linear(self.hidden_size, num_concepts) if use_fusion else None
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, concept_embeddings):
        outputs = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, return_dict=True
        )

        hidden_states = outputs.hidden_states
        current_hidden = outputs.last_hidden_state

        if self.use_fusion and len(self.fusion_layers) > 0:
            for layer_idx in self.fusion_layers:
                if str(layer_idx) in self.fusion_modules:
                    layer_hidden = hidden_states[layer_idx]
                    fused_hidden, _ = self.fusion_modules[str(layer_idx)](
                        layer_hidden, concept_embeddings, attention_mask
                    )
                    current_hidden = fused_hidden

        cls_hidden = self.dropout(current_hidden[:, 0, :])
        diagnosis_logits = self.diagnosis_head(cls_hidden)

        return {'logits': diagnosis_logits}

print("✅ Architecture components defined")

# ============================================================================
# LOAD SAVED SPLITS
# ============================================================================

print("\n" + "="*80)
print("📥 LOADING SAVED SPLITS")
print("="*80)

with open(SHARED_DATA_PATH / 'train_split.pkl', 'rb') as f:
    df_train = pickle.load(f)
with open(SHARED_DATA_PATH / 'val_split.pkl', 'rb') as f:
    df_val = pickle.load(f)
with open(SHARED_DATA_PATH / 'test_split.pkl', 'rb') as f:
    df_test = pickle.load(f)

print(f"✅ Loaded:")
print(f"   Train: {len(df_train):,}")
print(f"   Val:   {len(df_val):,}")
print(f"   Test:  {len(df_test):,}")

# ============================================================================
# DATASET
# ============================================================================

class SimpleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=384):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]), padding='max_length', truncation=True,
            max_length=self.max_length, return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(self.labels[idx])
        }

# ============================================================================
# TRAINING HELPER
# ============================================================================

def train_and_evaluate(model, train_loader, val_loader, test_loader, model_name, num_epochs=3, lr=2e-5):
    """Train and evaluate a model"""
    print(f"\n{'='*70}")
    print(f"Training: {model_name}")
    print(f"{'='*70}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=len(train_loader) // 10,
        num_training_steps=len(train_loader) * num_epochs
    )

    best_f1 = 0
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Train
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc="  Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
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

                outputs = model(input_ids, attention_mask)
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

    training_time = time.time() - start_time

    # Test
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="  Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
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

    print(f"\n✅ Test Results:")
    print(f"   Macro F1: {macro_f1:.4f}")
    print(f"   Micro F1: {micro_f1:.4f}")
    print(f"   Training Time: {training_time:.1f}s")

    return {
        'macro_f1': float(macro_f1),
        'micro_f1': float(micro_f1),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_auc': float(macro_auc),
        'per_class_f1': {code: float(f1) for code, f1 in zip(TARGET_CODES, per_class_f1)},
        'training_time': float(training_time)
    }

# ============================================================================
# LOAD PHASE 1 & 2 FIXED FOR COMPARISON
# ============================================================================

print("\n" + "="*80)
print("📥 LOADING PHASE 1 & 2 FIXED RESULTS")
print("="*80)

# Check if checkpoint files exist
if not PHASE1_FIXED_CHECKPOINT.exists():
    print(f"\n❌ ERROR: Phase 1 checkpoint not found!")
    print(f"   Expected: {PHASE1_FIXED_CHECKPOINT}")
    print(f"   Also tried: {PHASE1_FIXED_CHECKPOINT_FINAL if PHASE1_FIXED_CHECKPOINT == PHASE1_FIXED_CHECKPOINT_BEST else PHASE1_FIXED_CHECKPOINT_BEST}")
    print(f"\n📋 Required files from Phase 1 & 2:")
    print(f"   1. Run phase1_fixed.py to create: checkpoints/phase1_fixed/phase1_fixed_best.pt (or phase1_fixed_final.pt)")
    print(f"   2. Run phase2_fixed.py to create: checkpoints/phase2_fixed/phase2_fixed_best.pt (or phase2_fixed_final.pt)")
    print(f"\n⚠️  Make sure these files exist in your Google Drive at:")
    print(f"   {OUTPUT_BASE / 'checkpoints'}")
    raise FileNotFoundError(f"Phase 1 checkpoint not found: {PHASE1_FIXED_CHECKPOINT}")

if not PHASE2_FIXED_CHECKPOINT.exists():
    print(f"\n❌ ERROR: Phase 2 checkpoint not found!")
    print(f"   Expected: {PHASE2_FIXED_CHECKPOINT}")
    print(f"   Also tried: {PHASE2_FIXED_CHECKPOINT_FINAL if PHASE2_FIXED_CHECKPOINT == PHASE2_FIXED_CHECKPOINT_BEST else PHASE2_FIXED_CHECKPOINT_BEST}")
    print(f"\n📋 Required files from Phase 1 & 2:")
    print(f"   1. Run phase1_fixed.py to create: checkpoints/phase1_fixed/phase1_fixed_best.pt (or phase1_fixed_final.pt)")
    print(f"   2. Run phase2_fixed.py to create: checkpoints/phase2_fixed/phase2_fixed_best.pt (or phase2_fixed_final.pt)")
    print(f"\n⚠️  Make sure these files exist in your Google Drive at:")
    print(f"   {OUTPUT_BASE / 'checkpoints'}")
    raise FileNotFoundError(f"Phase 2 checkpoint not found: {PHASE2_FIXED_CHECKPOINT}")

print(f"✅ Found Phase 1: {PHASE1_FIXED_CHECKPOINT.name}")
print(f"✅ Found Phase 2: {PHASE2_FIXED_CHECKPOINT.name}")

phase1_fixed_ckpt = torch.load(PHASE1_FIXED_CHECKPOINT, map_location=device, weights_only=False)
phase2_fixed_ckpt = torch.load(PHASE2_FIXED_CHECKPOINT, map_location=device, weights_only=False)
concept_embeddings = phase1_fixed_ckpt['concept_embeddings'].to(device)
num_concepts = concept_embeddings.shape[0]

phase1_fixed_f1 = phase1_fixed_ckpt.get('macro_f1', 0.0)
phase2_fixed_f1 = phase2_fixed_ckpt.get('macro_f1', 0.0)

print(f"✅ Phase 1 Fixed F1: {phase1_fixed_f1:.4f}")
print(f"✅ Phase 2 Fixed F1: {phase2_fixed_f1:.4f}")

# ============================================================================
# ABLATIONS
# ============================================================================

print("\n" + "="*80)
print("🔬 ABLATION STUDIES")
print("="*80)

ablation_results = {}

# Prepare dataloaders
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

train_dataset = SimpleDataset(df_train['text'].tolist(), df_train['labels'].tolist(), tokenizer)
val_dataset = SimpleDataset(df_val['text'].tolist(), df_val['labels'].tolist(), tokenizer)
test_dataset = SimpleDataset(df_test['text'].tolist(), df_test['labels'].tolist(), tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# Quick wrapper to add concept_embeddings
class ModelWrapper:
    def __init__(self, model, concept_embeddings):
        self.model = model
        self.concept_embeddings = concept_embeddings

    def __call__(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask, self.concept_embeddings)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def parameters(self):
        return self.model.parameters()

# ABLATION 1: Full Model (Phase 2 Fixed) - Already done
print("\n1️⃣  Full Model (Phase 2 Fixed + RAG)")
ablation_results['full_model'] = {
    'macro_f1': float(phase2_fixed_f1),
    'description': 'Phase 2 Fixed with diagnosis-aware RAG',
    'fusion_layers': [9, 11],
    'uses_rag': True
}
print(f"   F1: {phase2_fixed_f1:.4f} (from checkpoint)")

# ABLATION 2: No RAG (Phase 1 Fixed only)
print("\n2️⃣  No RAG (Phase 1 Fixed only)")
ablation_results['no_rag'] = {
    'macro_f1': float(phase1_fixed_f1),
    'description': 'Phase 1 Fixed without RAG',
    'fusion_layers': [9, 11],
    'uses_rag': False
}
print(f"   F1: {phase1_fixed_f1:.4f} (from checkpoint)")

# ABLATION 3: Middle Layers
print("\n3️⃣  Middle Layers [5, 7]")
base_model_mid = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)
model_mid = FlexibleShifaMind(
    base_model_mid, num_concepts, len(TARGET_CODES),
    fusion_layers=[5, 7], use_fusion=True
).to(device)

model_mid_wrapper = ModelWrapper(model_mid, concept_embeddings)
ablation_results['middle_layers'] = train_and_evaluate(
    model_mid_wrapper, train_loader, val_loader, test_loader,
    "Middle Layers [5, 7]", num_epochs=2
)
ablation_results['middle_layers']['description'] = 'Fusion at middle layers [5, 7]'
ablation_results['middle_layers']['fusion_layers'] = [5, 7]

del model_mid, model_mid_wrapper, base_model_mid
torch.cuda.empty_cache()

# ABLATION 4: Early Layers
print("\n4️⃣  Early Layers [2, 3, 4]")
base_model_early = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)
model_early = FlexibleShifaMind(
    base_model_early, num_concepts, len(TARGET_CODES),
    fusion_layers=[2, 3, 4], use_fusion=True
).to(device)

model_early_wrapper = ModelWrapper(model_early, concept_embeddings)
ablation_results['early_layers'] = train_and_evaluate(
    model_early_wrapper, train_loader, val_loader, test_loader,
    "Early Layers [2, 3, 4]", num_epochs=2
)
ablation_results['early_layers']['description'] = 'Fusion at early layers [2, 3, 4]'
ablation_results['early_layers']['fusion_layers'] = [2, 3, 4]

del model_early, model_early_wrapper, base_model_early
torch.cuda.empty_cache()

# ABLATION 5: BERT + Concept Head (no fusion)
print("\n5️⃣  BERT + Concept Head (no fusion)")
base_model_no_fusion = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)
model_no_fusion = FlexibleShifaMind(
    base_model_no_fusion, num_concepts, len(TARGET_CODES),
    fusion_layers=[], use_fusion=False
).to(device)

model_no_fusion_wrapper = ModelWrapper(model_no_fusion, concept_embeddings)
ablation_results['no_fusion'] = train_and_evaluate(
    model_no_fusion_wrapper, train_loader, val_loader, test_loader,
    "BERT + Concept Head (no fusion)", num_epochs=2
)
ablation_results['no_fusion']['description'] = 'BioClinicalBERT with concept head, no fusion'
ablation_results['no_fusion']['fusion_layers'] = []

del model_no_fusion, model_no_fusion_wrapper, base_model_no_fusion
torch.cuda.empty_cache()


# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("💾 SAVING RESULTS")
print("="*80)

ablation_results_summary = {
    'phase': 'Phase 4 - Ablation Studies',
    'ablations': ablation_results,
    'summary': {
        'full_model_f1': ablation_results['full_model']['macro_f1'],
        'no_rag_f1': ablation_results['no_rag']['macro_f1'],
        'best_ablation_f1': max([v['macro_f1'] for v in ablation_results.values()])
    }
}

with open(RESULTS_PATH / 'ablation_results.json', 'w') as f:
    json.dump(ablation_results_summary, f, indent=2)

print(f"✅ Saved: {RESULTS_PATH / 'ablation_results.json'}")

# Create ablation comparison table
ablation_data = []
for name, results in ablation_results.items():
    ablation_data.append({
        'Model': name.replace('_', ' ').title(),
        'Macro F1': f"{results['macro_f1']:.4f}",
        'Description': results.get('description', ''),
        'Fusion Layers': str(results.get('fusion_layers', 'N/A'))
    })

ablation_df = pd.DataFrame(ablation_data)
ablation_df = ablation_df.sort_values('Macro F1', ascending=False)
ablation_df.to_csv(RESULTS_PATH / 'ablation_table.csv', index=False)

print(f"✅ Saved: {RESULTS_PATH / 'ablation_table.csv'}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("🎉 PHASE 4 COMPLETE - ABLATION STUDIES")
print("="*80)

print("\n📊 Ablation Results:")
for name, results in ablation_results.items():
    print(f"   {name.replace('_', ' ').title():30s}: F1 = {results['macro_f1']:.4f}")

print(f"\n🏆 Best Configuration: Full Model (Phase 2 Fixed + RAG)")
print(f"   F1: {ablation_results['full_model']['macro_f1']:.4f}")

print(f"\n📈 RAG Contribution:")
rag_contribution = ablation_results['full_model']['macro_f1'] - ablation_results['no_rag']['macro_f1']
print(f"   Phase 2 Fixed vs Phase 1 Fixed: {rag_contribution:+.4f}")

print(f"\n💾 Results: {RESULTS_PATH}")
print("\n🚀 Ready for Phase 5 (Baseline Comparisons)")
print(f"\nAlhamdulillah! 🤲")
