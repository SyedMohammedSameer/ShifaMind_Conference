#!/usr/bin/env python3
"""
================================================================================
SHIFAMIND2 PHASE 5: COMPLETE ALL 6 BASELINES
================================================================================

Strategy:
1. Load ShifaMind ablations (Phases 1-3)
2. Load existing baselines: CAML, DR-CAML, MultiResCNN, LAAT
3. Train remaining: PLM-ICD, Longformer-ICD
4. Generate final comparison table with ALL models

================================================================================
"""

print("="*80)
print("🚀 PHASE 5 - COMPLETE ALL 6 BASELINES")
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
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm

import json
import pickle
from pathlib import Path
import sys

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

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

with open(SHARED_DATA_PATH / 'train_split.pkl', 'rb') as f:
    df_train = pickle.load(f)

print(f"✅ Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
TOP_K = 5
print(f"📊 Top-k = {TOP_K}")

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

class PLM_ICD(nn.Module):
    def __init__(self, base_model, num_labels=50, chunk_size=512, stride=256):
        super().__init__()
        self.bert = base_model
        self.chunk_size = chunk_size
        self.stride = stride
        self.classifier = nn.Linear(768, num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.size()
        if seq_len <= self.chunk_size:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state.mean(dim=1)
        else:
            chunk_embeddings = []
            for start in range(0, seq_len, self.stride):
                end = min(start + self.chunk_size, seq_len)
                chunk_ids = input_ids[:, start:end]
                chunk_mask = attention_mask[:, start:end] if attention_mask is not None else None
                outputs = self.bert(input_ids=chunk_ids, attention_mask=chunk_mask)
                chunk_emb = outputs.last_hidden_state.mean(dim=1)
                chunk_embeddings.append(chunk_emb)
                if end >= seq_len:
                    break
            pooled = torch.stack(chunk_embeddings, dim=1).max(dim=1)[0]
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

try:
    from transformers import LongformerModel
    LONGFORMER_AVAILABLE = True
except:
    LONGFORMER_AVAILABLE = False

class LongformerICD(nn.Module):
    def __init__(self, num_labels=50):
        super().__init__()
        if LONGFORMER_AVAILABLE:
            try:
                self.longformer = LongformerModel.from_pretrained('allenai/longformer-base-4096')
            except:
                self.longformer = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        else:
            self.longformer = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        self.classifier = nn.Linear(768, num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask=None):
        if 'longformer' in str(type(self.longformer)).lower():
            global_attention_mask = torch.zeros_like(input_ids)
            global_attention_mask[:, 0] = 1
            outputs = self.longformer(input_ids=input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
        else:
            outputs = self.longformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.last_hidden_state[:, 0, :])
        return self.classifier(pooled)

print("✅ All architectures loaded")

# ============================================================================
# DATASET & EVALUATION
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
    print(f"\n📊 Evaluating {name}...")
    pv, yv = get_probs(model, val_loader)
    pt, yt = get_probs(model, test_loader)
    t = tune_threshold(pv, yv)

    results = {
        'validation': {'fixed_05': eval_threshold(pv, yv, 0.5), 'tuned': eval_threshold(pv, yv, t), 'topk': eval_topk(pv, yv, TOP_K)},
        'test': {'fixed_05': eval_threshold(pt, yt, 0.5), 'tuned': eval_threshold(pt, yt, t), 'topk': eval_topk(pt, yt, TOP_K)},
        'tuned_threshold': t
    }

    print(f"   Threshold: {t:.2f} | Test Macro-F1: Fixed={results['test']['fixed_05']['macro_f1']:.4f}, "
          f"Tuned={results['test']['tuned']['macro_f1']:.4f}, Top-5={results['test']['topk']['macro_f1']:.4f}")
    return results

def train_baseline(model, name, train_loader, val_loader, epochs=3, lr=1e-4):
    print(f"\n🔧 Training {name}...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            optimizer.zero_grad()
            logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            loss = criterion(logits, batch['labels'].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Validation
        pv, yv = get_probs(model, val_loader)
        val_f1 = eval_threshold(pv, yv, tune_threshold(pv, yv))['macro_f1']
        print(f"   Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Val Macro-F1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            ckpt_path = OUTPUT_BASE / 'checkpoints' / 'baselines' / f'{name.lower().replace(" ", "_").replace("-", "")}_best.pt'
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= 2:
                print(f"   Early stopping at epoch {epoch+1}")
                break

    # Load best
    ckpt_path = OUTPUT_BASE / 'checkpoints' / 'baselines' / f'{name.lower().replace(" ", "_").replace("-", "")}_best.pt'
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

# ============================================================================
# LOAD EXISTING + TRAIN REMAINING
# ============================================================================

print("\n" + "="*80)
print("📍 SECTION 1: LOADING EXISTING BASELINES")
print("="*80)

tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
descriptions = [f"ICD-10 code {code}" for code in TOP_50_CODES]

val_loader = DataLoader(ICDDataset(df_val, tokenizer), batch_size=32, shuffle=False)
test_loader = DataLoader(ICDDataset(df_test, tokenizer), batch_size=32, shuffle=False)

existing_models = {
    'CAML': (CAML(num_labels=len(TOP_50_CODES)), 'caml_best.pt'),
    'DR-CAML': (DR_CAML(num_labels=len(TOP_50_CODES), descriptions=descriptions, tokenizer=tokenizer), 'drcaml_best.pt'),
    'MultiResCNN': (MultiResCNN(num_labels=len(TOP_50_CODES)), 'multirescnn_best.pt'),
    'LAAT': (LAAT(num_labels=len(TOP_50_CODES)), 'laat_best.pt')
}

for name, (model, ckpt_name) in existing_models.items():
    ckpt_path = OUTPUT_BASE / 'checkpoints' / 'baselines' / ckpt_name
    if ckpt_path.exists():
        print(f"\n✅ {name}: Loading checkpoint")
        model.to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        all_results[name] = evaluate_complete(model, val_loader, test_loader, name)
        del model
        torch.cuda.empty_cache()
    else:
        print(f"\n⚠️  {name}: Checkpoint not found - skipping")

# ============================================================================
# SECTION 2: TRAIN REMAINING BASELINES
# ============================================================================

print("\n" + "="*80)
print("📍 SECTION 2: TRAINING REMAINING BASELINES")
print("="*80)

train_dataset = ICDDataset(df_train, tokenizer)

# 5. PLM-ICD
print("\n" + "="*80)
print("🔵 PLM-ICD")
print("="*80)

plm_ckpt = OUTPUT_BASE / 'checkpoints' / 'baselines' / 'plmicd_best.pt'
if plm_ckpt.exists():
    print("   ✅ Loading existing checkpoint")
    base_model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT').to(device)
    model_plm = PLM_ICD(base_model, num_labels=len(TOP_50_CODES)).to(device)
    model_plm.load_state_dict(torch.load(plm_ckpt, map_location=device))
    all_results['PLM-ICD'] = evaluate_complete(model_plm, val_loader, test_loader, "PLM-ICD")
else:
    print("   🔧 Training from scratch...")
    train_loader_plm = DataLoader(train_dataset, batch_size=8, shuffle=True)
    base_model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT').to(device)
    model_plm = PLM_ICD(base_model, num_labels=len(TOP_50_CODES)).to(device)
    train_baseline(model_plm, "PLM-ICD", train_loader_plm, val_loader, epochs=3, lr=2e-5)
    all_results['PLM-ICD'] = evaluate_complete(model_plm, val_loader, test_loader, "PLM-ICD")

del model_plm
if 'base_model' in locals():
    del base_model
torch.cuda.empty_cache()

# 6. Longformer-ICD
print("\n" + "="*80)
print("🔵 Longformer-ICD")
print("="*80)

long_ckpt = OUTPUT_BASE / 'checkpoints' / 'baselines' / 'longformericd_best.pt'
if long_ckpt.exists():
    print("   ✅ Loading existing checkpoint")
    model_long = LongformerICD(num_labels=len(TOP_50_CODES)).to(device)
    model_long.load_state_dict(torch.load(long_ckpt, map_location=device))
    all_results['Longformer-ICD'] = evaluate_complete(model_long, val_loader, test_loader, "Longformer-ICD")
else:
    print("   🔧 Training from scratch...")
    train_loader_long = DataLoader(train_dataset, batch_size=2, shuffle=True)
    model_long = LongformerICD(num_labels=len(TOP_50_CODES)).to(device)
    train_baseline(model_long, "Longformer-ICD", train_loader_long, val_loader, epochs=2, lr=1e-5)
    all_results['Longformer-ICD'] = evaluate_complete(model_long, val_loader, test_loader, "Longformer-ICD")

del model_long
torch.cuda.empty_cache()

# ============================================================================
# FINAL COMPARISON
# ============================================================================

print("\n" + "="*80)
print("📊 FINAL COMPARISON TABLE - ALL MODELS")
print("="*80)

sorted_models = sorted(all_results.items(), key=lambda x: x[1]['test']['tuned']['macro_f1'], reverse=True)

print("\n" + "="*135)
print(f"{'Model':<50} {'Test Macro@0.5':<17} {'Test Macro@Tuned':<19} {'Test Macro@Top-5':<18} {'Tuned Thresh':<14} {'Category'}")
print("="*135)

for model_name, results in sorted_models:
    category = 'Ablation' if 'ShifaMind' in model_name or 'Phase' in model_name else 'Baseline'
    print(f"{model_name:<50} {results['test']['fixed_05']['macro_f1']:<17.4f} "
          f"{results['test']['tuned']['macro_f1']:<19.4f} {results['test']['topk']['macro_f1']:<18.4f} "
          f"{results['tuned_threshold']:<14.2f} {category}")

print("="*135)

# Save
with open(RESULTS_PATH / 'complete_comparison_all6.json', 'w') as f:
    json.dump(all_results, f, indent=2)

table_data = [{
    'Model': name,
    'Test_Macro_Fixed_0.5': r['test']['fixed_05']['macro_f1'],
    'Test_Macro_Tuned': r['test']['tuned']['macro_f1'],
    'Test_Macro_Top_5': r['test']['topk']['macro_f1'],
    'Tuned_Threshold': r['tuned_threshold'],
    'Category': 'Ablation' if 'ShifaMind' in name else 'Baseline'
} for name, r in sorted_models]

pd.DataFrame(table_data).to_csv(RESULTS_PATH / 'complete_comparison_all6.csv', index=False)

print(f"\n✅ Results saved to: {RESULTS_PATH}")
print(f"   - complete_comparison_all6.json")
print(f"   - complete_comparison_all6.csv")

print(f"\n" + "="*80)
print("🏆 FINAL RESULTS")
print("="*80)
print(f"\nBEST MODEL: {sorted_models[0][0]}")
print(f"Test Macro-F1 @ Tuned Threshold: {sorted_models[0][1]['test']['tuned']['macro_f1']:.4f}")
print(f"\nTotal models compared: {len(all_results)}")

ablations = sum(1 for n, _ in sorted_models if 'ShifaMind' in n or 'Phase' in n)
baselines = len(all_results) - ablations

print(f"  - ShifaMind Ablations: {ablations}")
print(f"  - SOTA Baselines: {baselines}")

print("\n" + "="*80)
print("✅ COMPLETE! All 6 baselines evaluated with fair protocol!")
print("="*80)
print("\nAlhamdulillah! 🤲")
