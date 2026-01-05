#!/usr/bin/env python3
"""
================================================================================
SHIFAMIND2 PHASE 5: FINAL COMPARISON (NO TRAINING)
================================================================================

Generates final comparison table by:
1. Loading ShifaMind ablation results
2. Loading existing baseline checkpoints (CAML, DR-CAML, MultiResCNN, LAAT)
3. Quick evaluation (inference only)
4. Complete comparison table

NO TRAINING - EVALUATION ONLY!
================================================================================
"""

print("="*80)
print("🚀 PHASE 5 - FINAL COMPARISON (EVALUATION ONLY)")
print("="*80)

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from transformers import AutoTokenizer
from tqdm.auto import tqdm

import json
import pickle
from pathlib import Path
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Device: {device}")

# ============================================================================
# CONFIG
# ============================================================================

print("\n" + "="*80)
print("⚙️  CONFIGURATION")
print("="*80)

BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')
SHIFAMIND2_BASE = BASE_PATH / '10_ShifaMind'

run_folders = sorted([d for d in SHIFAMIND2_BASE.glob('run_*') if d.is_dir()], reverse=True)
OUTPUT_BASE = run_folders[0]
print(f"📁 Run folder: {OUTPUT_BASE.name}")

PHASE1_CHECKPOINT_PATH = OUTPUT_BASE / 'checkpoints' / 'phase1' / 'phase1_best.pt'
checkpoint = torch.load(PHASE1_CHECKPOINT_PATH, map_location='cpu', weights_only=False)
TOP_50_CODES = checkpoint['config']['top_50_codes']

SHARED_DATA_PATH = OUTPUT_BASE / 'shared_data'
RESULTS_PATH = OUTPUT_BASE / 'results' / 'phase5_complete'
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

with open(SHARED_DATA_PATH / 'concept_list.json', 'r') as f:
    ALL_CONCEPTS = json.load(f)

print(f"✅ {len(TOP_50_CODES)} diagnoses, {len(ALL_CONCEPTS)} concepts")

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n" + "="*80)
print("📊 LOADING DATA")
print("="*80)

with open(SHARED_DATA_PATH / 'val_split.pkl', 'rb') as f:
    df_val = pickle.load(f)

with open(SHARED_DATA_PATH / 'test_split.pkl', 'rb') as f:
    df_test = pickle.load(f)

print(f"✅ Val: {len(df_val)}, Test: {len(df_test)}")
TOP_K = 5

# ============================================================================
# LOAD SHIFAMIND RESULTS
# ============================================================================

print("\n" + "="*80)
print("📍 LOADING SHIFAMIND ABLATION RESULTS")
print("="*80)

fair_results_path = OUTPUT_BASE / 'results' / 'phase5_fair' / 'fair_evaluation_results.json'

with open(fair_results_path, 'r') as f:
    fair_data = json.load(f)
    all_results = fair_data['models'].copy()

print(f"✅ Loaded ShifaMind ablations:")
for model_name in all_results:
    print(f"   - {model_name}: {all_results[model_name]['test']['tuned']['macro_f1']:.4f}")

# Load Phase 2
phase2_path = OUTPUT_BASE / 'results' / 'phase2' / 'results.json'
if phase2_path.exists():
    with open(phase2_path, 'r') as f:
        p2 = json.load(f)
    all_results['ShifaMind w/ GraphSAGE w/o RAG (Phase 2)'] = {
        'validation': {'fixed_05': {'macro_f1': 0.0}, 'tuned': {'macro_f1': 0.0}, 'topk': {'macro_f1': 0.0}},
        'test': {
            'fixed_05': {'macro_f1': 0.0},
            'tuned': {'macro_f1': p2['diagnosis_metrics']['macro_f1'], 'micro_f1': p2['diagnosis_metrics']['micro_f1']},
            'topk': {'macro_f1': 0.0}
        },
        'tuned_threshold': p2.get('threshold', 0.5)
    }
    print(f"   - Phase 2: {p2['diagnosis_metrics']['macro_f1']:.4f}")

# ============================================================================
# BASELINE ARCHITECTURES
# ============================================================================

print("\n" + "="*80)
print("🏗️  BASELINE ARCHITECTURES")
print("="*80)

class CAML(nn.Module):
    def __init__(self, vocab_size=30522, embed_dim=100, num_filters=50, num_labels=50):
        super().__init__()
        self.num_labels = num_labels
        self.num_filters = num_filters
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, num_filters, kernel_size=4, padding=2)
        self.U = nn.Linear(num_filters, num_labels, bias=False)
        self.final_weight = nn.Parameter(torch.randn(num_labels, num_filters))
        self.final_bias = nn.Parameter(torch.zeros(num_labels))

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids).transpose(1, 2)
        H = torch.tanh(self.conv(x)).transpose(1, 2)
        alpha = torch.softmax(self.U(H), dim=1)
        m = torch.bmm(alpha.transpose(1, 2), H)
        return torch.sum(m * self.final_weight.unsqueeze(0), dim=2) + self.final_bias

class DR_CAML(nn.Module):
    def __init__(self, vocab_size=30522, embed_dim=100, num_filters=50, num_labels=50,
                 descriptions=None, tokenizer=None, lambda_desc=0.1):
        super().__init__()
        self.num_labels = num_labels
        self.num_filters = num_filters
        self.lambda_desc = lambda_desc
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, num_filters, kernel_size=4, padding=2)
        self.U = nn.Linear(num_filters, num_labels, bias=False)
        self.final_weight = nn.Parameter(torch.randn(num_labels, num_filters))
        self.final_bias = nn.Parameter(torch.zeros(num_labels))
        if descriptions and tokenizer:
            self.register_buffer('desc_embeddings', self._encode_descriptions(descriptions, tokenizer))
        else:
            self.register_buffer('desc_embeddings', torch.zeros(num_labels, num_filters))

    def _encode_descriptions(self, descriptions, tokenizer):
        desc_vecs = []
        for desc in descriptions:
            tokens = tokenizer(desc, truncation=True, max_length=128, padding='max_length', return_tensors='pt')
            with torch.no_grad():
                x = self.embedding(tokens['input_ids']).transpose(1, 2)
                h = torch.tanh(self.conv(x))
                desc_vecs.append(F.max_pool1d(h, kernel_size=h.size(2)).squeeze())
        return torch.stack(desc_vecs)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids).transpose(1, 2)
        H = torch.tanh(self.conv(x)).transpose(1, 2)
        alpha = torch.softmax(self.U(H), dim=1)
        m = torch.bmm(alpha.transpose(1, 2), H)
        return torch.sum(m * self.final_weight.unsqueeze(0), dim=2) + self.final_bias

class MultiResCNN(nn.Module):
    def __init__(self, vocab_size=30522, embed_dim=100, num_labels=50):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv3 = nn.Conv1d(embed_dim, 100, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(embed_dim, 100, kernel_size=5, padding=2)
        self.conv9 = nn.Conv1d(embed_dim, 100, kernel_size=9, padding=4)
        self.U = nn.Linear(300, num_labels, bias=False)
        self.final_weight = nn.Parameter(torch.randn(num_labels, 300))
        self.final_bias = nn.Parameter(torch.zeros(num_labels))

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids).transpose(1, 2)
        C = torch.cat([torch.relu(self.conv3(x)), torch.relu(self.conv5(x)), torch.relu(self.conv9(x))], dim=1).transpose(1, 2)
        alpha = torch.softmax(self.U(C), dim=1)
        m = torch.bmm(alpha.transpose(1, 2), C)
        return torch.sum(m * self.final_weight.unsqueeze(0), dim=2) + self.final_bias

class LAAT(nn.Module):
    def __init__(self, vocab_size=30522, embed_dim=100, hidden_dim=256, num_labels=50):
        super().__init__()
        self.hidden_dim = hidden_dim * 2
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.label_queries = nn.Parameter(torch.randn(num_labels, self.hidden_dim))
        self.W_attn = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.output_weight = nn.Parameter(torch.randn(num_labels, self.hidden_dim))
        self.output_bias = nn.Parameter(torch.zeros(num_labels))

    def forward(self, input_ids, attention_mask=None):
        H, _ = self.lstm(self.embedding(input_ids))
        scores = torch.einsum('bth,lh->blt', self.W_attn(H), self.label_queries)
        m = torch.bmm(torch.softmax(scores, dim=2), H)
        return torch.sum(m * self.output_weight.unsqueeze(0), dim=2) + self.output_bias

print("✅ Architectures loaded")

# ============================================================================
# EVALUATION
# ============================================================================

class ICDDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = df['text'].tolist()
        self.labels = df['labels'].tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(str(self.texts[idx]), truncation=True, max_length=512, padding='max_length', return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }

def tune_threshold(probs, y):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.61, 0.01):
        f1 = f1_score(y, (probs > t).astype(int), average='micro', zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t

def eval_threshold(probs, y, t):
    preds = (probs > t).astype(int)
    return {'macro_f1': float(f1_score(y, preds, average='macro', zero_division=0)),
            'micro_f1': float(f1_score(y, preds, average='micro', zero_division=0))}

def eval_topk(probs, y, k):
    preds = np.zeros_like(probs)
    for i in range(len(probs)):
        preds[i, np.argsort(probs[i])[-k:]] = 1
    return {'macro_f1': float(f1_score(y, preds, average='macro', zero_division=0)),
            'micro_f1': float(f1_score(y, preds, average='micro', zero_division=0))}

def get_probs(model, loader):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            probs.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(batch['labels'].numpy())
    return np.vstack(probs), np.vstack(labels)

def evaluate_complete(model, val_loader, test_loader, name):
    print(f"\n📊 {name}...")
    pv, yv = get_probs(model, val_loader)
    pt, yt = get_probs(model, test_loader)
    t = tune_threshold(pv, yv)

    results = {
        'validation': {'fixed_05': eval_threshold(pv, yv, 0.5), 'tuned': eval_threshold(pv, yv, t), 'topk': eval_topk(pv, yv, TOP_K)},
        'test': {'fixed_05': eval_threshold(pt, yt, 0.5), 'tuned': eval_threshold(pt, yt, t), 'topk': eval_topk(pt, yt, TOP_K)},
        'tuned_threshold': t
    }

    print(f"   Test Macro-F1: Fixed={results['test']['fixed_05']['macro_f1']:.4f}, "
          f"Tuned@{t:.2f}={results['test']['tuned']['macro_f1']:.4f}, Top-5={results['test']['topk']['macro_f1']:.4f}")
    return results

# ============================================================================
# LOAD AND EVALUATE BASELINES
# ============================================================================

print("\n" + "="*80)
print("📍 LOADING & EVALUATING BASELINE CHECKPOINTS")
print("="*80)

tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
descriptions = [f"ICD-10 code {code}" for code in TOP_50_CODES]

val_loader = DataLoader(ICDDataset(df_val, tokenizer), batch_size=32, shuffle=False)
test_loader = DataLoader(ICDDataset(df_test, tokenizer), batch_size=32, shuffle=False)

baseline_checkpoints = {
    'CAML': (CAML(num_labels=len(TOP_50_CODES)), 'caml_best.pt'),
    'DR-CAML': (DR_CAML(num_labels=len(TOP_50_CODES), descriptions=descriptions, tokenizer=tokenizer), 'dr-caml_best.pt'),
    'MultiResCNN': (MultiResCNN(num_labels=len(TOP_50_CODES)), 'multirescnn_best.pt'),
    'LAAT': (LAAT(num_labels=len(TOP_50_CODES)), 'laat_best.pt')
}

for name, (model, ckpt_name) in baseline_checkpoints.items():
    ckpt_path = OUTPUT_BASE / 'checkpoints' / 'baselines' / ckpt_name
    if ckpt_path.exists():
        model.to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        all_results[name] = evaluate_complete(model, val_loader, test_loader, name)
        del model
        torch.cuda.empty_cache()
    else:
        print(f"\n⚠️  {name}: checkpoint not found, skipping")

# ============================================================================
# FINAL TABLE
# ============================================================================

print("\n" + "="*80)
print("📊 FINAL COMPARISON TABLE")
print("="*80)

sorted_models = sorted(all_results.items(), key=lambda x: x[1]['test']['tuned']['macro_f1'], reverse=True)

print("\n" + "="*135)
print(f"{'Model':<50} {'Test Macro@0.5':<17} {'Test Macro@Tuned':<19} {'Test Macro@Top-5':<18} {'Tuned Threshold':<15} {'Category'}")
print("="*135)

for model_name, results in sorted_models:
    category = 'Ablation' if 'ShifaMind' in model_name or 'Phase' in model_name else 'Baseline'
    print(f"{model_name:<50} {results['test']['fixed_05']['macro_f1']:<17.4f} "
          f"{results['test']['tuned']['macro_f1']:<19.4f} {results['test']['topk']['macro_f1']:<18.4f} "
          f"{results['tuned_threshold']:<15.2f} {category}")

print("="*135)

# Save
with open(RESULTS_PATH / 'final_comparison.json', 'w') as f:
    json.dump(all_results, f, indent=2)

table_data = [{
    'Model': name,
    'Test_Macro_Fixed_0.5': r['test']['fixed_05']['macro_f1'],
    'Test_Macro_Tuned': r['test']['tuned']['macro_f1'],
    'Test_Macro_Top_5': r['test']['topk']['macro_f1'],
    'Tuned_Threshold': r['tuned_threshold'],
    'Category': 'Ablation' if 'ShifaMind' in name else 'Baseline'
} for name, r in sorted_models]

pd.DataFrame(table_data).to_csv(RESULTS_PATH / 'final_comparison.csv', index=False)

print(f"\n✅ Results saved to: {RESULTS_PATH}")
print(f"\nBEST MODEL: {sorted_models[0][0]}")
print(f"Test Macro-F1 @ Tuned Threshold: {sorted_models[0][1]['test']['tuned']['macro_f1']:.4f}")
print(f"\nTotal models compared: {len(all_results)}")
print(f"  - ShifaMind Ablations: {sum(1 for n, _ in sorted_models if 'ShifaMind' in n or 'Phase' in n)}")
print(f"  - SOTA Baselines: {sum(1 for n, _ in sorted_models if 'ShifaMind' not in n and 'Phase' not in n)}")
print("\nAlhamdulillah! 🤲")
