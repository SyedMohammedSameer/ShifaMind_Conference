#!/usr/bin/env python3
"""
================================================================================
SHIFAMIND PHASE 5: Baseline Comparisons
================================================================================
Author: Mohammed Sameer Syed
University of Arizona - MS in AI Capstone

SELF-CONTAINED COLAB SCRIPT - Copy-paste ready!

Baselines (3):
1. BioClinicalBERT (vanilla fine-tuned)
2. PubMedBERT
3. BlueBERT

Loads:
- Train/val/test splits from Phase 1
- Phase 2 Fixed results for comparison

Saves:
- Baseline results to 07_ShifaMind/results/phase5_fixed/

TARGET: Prove Phase 2 Fixed beats all baselines
================================================================================
"""

print("="*80)
print("🚀 SHIFAMIND PHASE 5 - BASELINE COMPARISONS")
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

BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')
OUTPUT_BASE = BASE_PATH / '07_ShifaMind'

# Input paths
SHARED_DATA_PATH = OUTPUT_BASE / 'shared_data'
PHASE2_FIXED_CHECKPOINT = OUTPUT_BASE / 'checkpoints/phase2_fixed/phase2_fixed_best.pt'

# Output paths
RESULTS_PATH = OUTPUT_BASE / 'results/phase5_fixed'
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

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
    """Train and evaluate a baseline model"""
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
# LOAD PHASE 2 FIXED RESULT
# ============================================================================

print("\n" + "="*80)
print("📥 LOADING PHASE 2 FIXED RESULT")
print("="*80)

phase2_fixed_ckpt = torch.load(PHASE2_FIXED_CHECKPOINT, map_location=device, weights_only=False)
phase2_fixed_f1 = phase2_fixed_ckpt.get('macro_f1', 0.0)

print(f"✅ Phase 2 Fixed F1: {phase2_fixed_f1:.4f}")

# ============================================================================
# BASELINES
# ============================================================================

print("\n" + "="*80)
print("📊 BASELINE COMPARISONS")
print("="*80)

baseline_results = {}

class VanillaClassifier(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(base_model.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = self.dropout(outputs.last_hidden_state[:, 0, :])
        logits = self.classifier(cls_hidden)
        return {'logits': logits}

# BASELINE 1: BioClinicalBERT (vanilla)
print("\n1️⃣  BioClinicalBERT (vanilla fine-tuned)")
tokenizer_bio = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
base_model_bio = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)
model_vanilla = VanillaClassifier(base_model_bio, len(TARGET_CODES)).to(device)

train_dataset_bio = SimpleDataset(df_train['text'].tolist(), df_train['labels'].tolist(), tokenizer_bio)
val_dataset_bio = SimpleDataset(df_val['text'].tolist(), df_val['labels'].tolist(), tokenizer_bio)
test_dataset_bio = SimpleDataset(df_test['text'].tolist(), df_test['labels'].tolist(), tokenizer_bio)

train_loader_bio = DataLoader(train_dataset_bio, batch_size=8, shuffle=True)
val_loader_bio = DataLoader(val_dataset_bio, batch_size=16)
test_loader_bio = DataLoader(test_dataset_bio, batch_size=16)

baseline_results['bio_clinical_bert'] = train_and_evaluate(
    model_vanilla, train_loader_bio, val_loader_bio, test_loader_bio,
    "BioClinicalBERT (vanilla)", num_epochs=3
)
baseline_results['bio_clinical_bert']['description'] = 'Vanilla BioClinicalBERT fine-tuned'

del model_vanilla, base_model_bio, tokenizer_bio
del train_dataset_bio, val_dataset_bio, test_dataset_bio
del train_loader_bio, val_loader_bio, test_loader_bio
torch.cuda.empty_cache()

# BASELINE 2: PubMedBERT
print("\n2️⃣  PubMedBERT")
try:
    tokenizer_pubmed = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
    base_model_pubmed = AutoModel.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext").to(device)
    model_pubmed = VanillaClassifier(base_model_pubmed, len(TARGET_CODES)).to(device)

    train_dataset_pubmed = SimpleDataset(df_train['text'].tolist(), df_train['labels'].tolist(), tokenizer_pubmed)
    val_dataset_pubmed = SimpleDataset(df_val['text'].tolist(), df_val['labels'].tolist(), tokenizer_pubmed)
    test_dataset_pubmed = SimpleDataset(df_test['text'].tolist(), df_test['labels'].tolist(), tokenizer_pubmed)

    train_loader_pubmed = DataLoader(train_dataset_pubmed, batch_size=8, shuffle=True)
    val_loader_pubmed = DataLoader(val_dataset_pubmed, batch_size=16)
    test_loader_pubmed = DataLoader(test_dataset_pubmed, batch_size=16)

    baseline_results['pubmed_bert'] = train_and_evaluate(
        model_pubmed, train_loader_pubmed, val_loader_pubmed, test_loader_pubmed,
        "PubMedBERT", num_epochs=3
    )
    baseline_results['pubmed_bert']['description'] = 'PubMedBERT fine-tuned'

    del model_pubmed, base_model_pubmed, tokenizer_pubmed
    del train_dataset_pubmed, val_dataset_pubmed, test_dataset_pubmed
    del train_loader_pubmed, val_loader_pubmed, test_loader_pubmed
    torch.cuda.empty_cache()
except Exception as e:
    print(f"   ⚠️  Skipping PubMedBERT: {str(e)}")
    baseline_results['pubmed_bert'] = {'macro_f1': 0.0, 'description': 'Failed to load', 'error': str(e)}

# BASELINE 3: BlueBERT
print("\n3️⃣  BlueBERT")
try:
    tokenizer_blue = AutoTokenizer.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12")
    base_model_blue = AutoModel.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12").to(device)
    model_blue = VanillaClassifier(base_model_blue, len(TARGET_CODES)).to(device)

    train_dataset_blue = SimpleDataset(df_train['text'].tolist(), df_train['labels'].tolist(), tokenizer_blue)
    val_dataset_blue = SimpleDataset(df_val['text'].tolist(), df_val['labels'].tolist(), tokenizer_blue)
    test_dataset_blue = SimpleDataset(df_test['text'].tolist(), df_test['labels'].tolist(), tokenizer_blue)

    train_loader_blue = DataLoader(train_dataset_blue, batch_size=8, shuffle=True)
    val_loader_blue = DataLoader(val_dataset_blue, batch_size=16)
    test_loader_blue = DataLoader(test_dataset_blue, batch_size=16)

    baseline_results['blue_bert'] = train_and_evaluate(
        model_blue, train_loader_blue, val_loader_blue, test_loader_blue,
        "BlueBERT", num_epochs=3
    )
    baseline_results['blue_bert']['description'] = 'BlueBERT (PubMed+MIMIC) fine-tuned'

    del model_blue, base_model_blue, tokenizer_blue
    del train_dataset_blue, val_dataset_blue, test_dataset_blue
    del train_loader_blue, val_loader_blue, test_loader_blue
    torch.cuda.empty_cache()
except Exception as e:
    print(f"   ⚠️  Skipping BlueBERT: {str(e)}")
    baseline_results['blue_bert'] = {'macro_f1': 0.0, 'description': 'Failed to load', 'error': str(e)}

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("💾 SAVING RESULTS")
print("="*80)

all_results = {
    'phase': 'Phase 5 - Baseline Comparisons',
    'phase2_fixed_f1': float(phase2_fixed_f1),
    'baselines': baseline_results,
    'summary': {
        'phase2_fixed_f1': float(phase2_fixed_f1),
        'best_baseline_f1': max([v.get('macro_f1', 0) for v in baseline_results.values()]),
        'improvement': float(phase2_fixed_f1) - max([v.get('macro_f1', 0) for v in baseline_results.values()])
    }
}

with open(RESULTS_PATH / 'baseline_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"✅ Saved: {RESULTS_PATH / 'baseline_results.json'}")

# Create baseline comparison table
baseline_data = []
baseline_data.append({
    'Model': 'Phase 2 Fixed (Ours)',
    'Macro F1': f"{phase2_fixed_f1:.4f}",
    'Description': 'Our model with RAG'
})

for name, results in baseline_results.items():
    baseline_data.append({
        'Model': name.replace('_', ' ').title(),
        'Macro F1': f"{results.get('macro_f1', 0):.4f}",
        'Description': results.get('description', '')
    })

baseline_df = pd.DataFrame(baseline_data)
baseline_df = baseline_df.sort_values('Macro F1', ascending=False)
baseline_df.to_csv(RESULTS_PATH / 'baseline_table.csv', index=False)

print(f"✅ Saved: {RESULTS_PATH / 'baseline_table.csv'}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("🎉 PHASE 5 COMPLETE - BASELINE COMPARISONS")
print("="*80)

print(f"\n🏆 Our Model (Phase 2 Fixed):")
print(f"   F1: {phase2_fixed_f1:.4f}")

print("\n📊 Baseline Results:")
for name, results in baseline_results.items():
    print(f"   {name.replace('_', ' ').title():30s}: F1 = {results.get('macro_f1', 0):.4f}")

best_baseline_f1 = max([v.get('macro_f1', 0) for v in baseline_results.values()])
if best_baseline_f1 > 0:
    improvement = phase2_fixed_f1 - best_baseline_f1
    print(f"\n📈 Improvement Over Best Baseline:")
    print(f"   Δ: {improvement:+.4f} ({improvement/best_baseline_f1*100:+.1f}%)")

print(f"\n💾 Results: {RESULTS_PATH}")
print("\n✅ ALL PHASES COMPLETE!")
print(f"\nAlhamdulillah! 🤲")
