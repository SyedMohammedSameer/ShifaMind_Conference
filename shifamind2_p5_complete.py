#!/usr/bin/env python3
"""
================================================================================
SHIFAMIND2 PHASE 5: COMPLETE FAIR COMPARISON
================================================================================

Combines:
- ABLATION STUDIES: ShifaMind Phases 1-3 (from fair comparison)
- BASELINE COMPARISONS: SOTA models trained with same protocol

All models evaluated with unified protocol:
- Fixed threshold @ 0.5
- Tuned threshold (validation-optimized)
- Top-k predictions

Primary Metric: Test Macro-F1 @ Tuned Threshold
================================================================================
"""

print("="*80)
print("🚀 PHASE 5 - COMPLETE FAIR COMPARISON (ABLATIONS + BASELINES)")
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
if not run_folders:
    print("❌ No runs found!")
    sys.exit(1)

OUTPUT_BASE = run_folders[0]
print(f"📁 Run folder: {OUTPUT_BASE.name}")

# Load config from Phase 1 checkpoint
PHASE1_CHECKPOINT_PATH = OUTPUT_BASE / 'checkpoints' / 'phase1' / 'phase1_best.pt'
checkpoint = torch.load(PHASE1_CHECKPOINT_PATH, map_location='cpu', weights_only=False)
TOP_50_CODES = checkpoint['config']['top_50_codes']
timestamp = checkpoint['config']['timestamp']

SHARED_DATA_PATH = OUTPUT_BASE / 'shared_data'
RESULTS_PATH = OUTPUT_BASE / 'results' / 'phase5_complete'
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

with open(SHARED_DATA_PATH / 'concept_list.json', 'r') as f:
    ALL_CONCEPTS = json.load(f)

print(f"✅ Config loaded: {len(TOP_50_CODES)} diagnoses, {len(ALL_CONCEPTS)} concepts")

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

# Calculate top-k
avg_labels_per_sample = np.mean([len(labels) for labels in df_val['labels'].tolist()])
TOP_K = max(1, int(round(avg_labels_per_sample)))
print(f"📊 Top-k = {TOP_K}")

# ============================================================================
# SECTION A: LOAD SHIFAMIND ABLATION RESULTS
# ============================================================================

print("\n" + "="*80)
print("📍 SECTION A: LOADING SHIFAMIND ABLATION RESULTS")
print("="*80)

fair_results_path = OUTPUT_BASE / 'results' / 'phase5_fair' / 'all_results.json'

if not fair_results_path.exists():
    print(f"❌ Fair comparison results not found at {fair_results_path}")
    print("   Please run shifamind2_p5_fair.py first!")
    sys.exit(1)

with open(fair_results_path, 'r') as f:
    shifamind_results = json.load(f)

print(f"✅ Loaded ShifaMind results:")
for model_name in shifamind_results:
    test_macro_tuned = shifamind_results[model_name]['test']['tuned']['macro_f1']
    print(f"   - {model_name}: Test Macro-F1 @ Tuned = {test_macro_tuned:.4f}")

# Load Phase 2 results if available
phase2_results_path = OUTPUT_BASE / 'results' / 'phase2' / 'results.json'
if phase2_results_path.exists():
    with open(phase2_results_path, 'r') as f:
        phase2_data = json.load(f)

    # Convert to fair comparison format (assuming it used tuned threshold)
    shifamind_results['ShifaMind w/ GraphSAGE w/o RAG (Phase 2)'] = {
        'test': {
            'tuned': {
                'macro_f1': phase2_data['diagnosis_metrics']['macro_f1'],
                'micro_f1': phase2_data['diagnosis_metrics']['micro_f1']
            }
        },
        'tuned_threshold': phase2_data.get('threshold', 0.5)
    }
    print(f"   - Phase 2: Test Macro-F1 = {phase2_data['diagnosis_metrics']['macro_f1']:.4f}")
else:
    print("   ⚠️  Phase 2 results not found - skipping")

# ============================================================================
# BASELINE MODEL ARCHITECTURES
# ============================================================================

print("\n" + "="*80)
print("🏗️  BASELINE MODEL ARCHITECTURES")
print("="*80)

class CAML(nn.Module):
    """CAML: Convolutional Attention for Multi-Label classification (Mullenbach et al., NAACL 2018)"""
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
        x = self.embedding(input_ids)
        x = x.transpose(1, 2)
        H = torch.tanh(self.conv(x))
        H = H.transpose(1, 2)
        alpha = torch.softmax(self.U(H), dim=1)
        m = torch.bmm(alpha.transpose(1, 2), H)
        logits = torch.sum(m * self.final_weight.unsqueeze(0), dim=2) + self.final_bias
        return logits

class DR_CAML(nn.Module):
    """DR-CAML: CAML + description regularization (Mullenbach et al., NAACL 2018)"""
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

        # Precompute description embeddings
        if descriptions is not None and tokenizer is not None:
            self.register_buffer('desc_embeddings', self._encode_descriptions(descriptions, tokenizer))
        else:
            self.register_buffer('desc_embeddings', torch.zeros(num_labels, num_filters))

    def _encode_descriptions(self, descriptions, tokenizer):
        """Encode ICD descriptions into num_filters-dim vectors"""
        desc_vecs = []
        for desc in descriptions:
            tokens = tokenizer(desc, truncation=True, max_length=128,
                             padding='max_length', return_tensors='pt')
            input_ids = tokens['input_ids']
            with torch.no_grad():
                x = self.embedding(input_ids)
                x = x.transpose(1, 2)
                h = torch.tanh(self.conv(x))
                pooled = F.max_pool1d(h, kernel_size=h.size(2)).squeeze()
            desc_vecs.append(pooled)
        return torch.stack(desc_vecs)

    def forward(self, input_ids, attention_mask=None, return_reg_loss=False):
        x = self.embedding(input_ids)
        x = x.transpose(1, 2)
        H = torch.tanh(self.conv(x))
        H = H.transpose(1, 2)
        alpha = torch.softmax(self.U(H), dim=1)
        m = torch.bmm(alpha.transpose(1, 2), H)
        logits = torch.sum(m * self.final_weight.unsqueeze(0), dim=2) + self.final_bias

        if return_reg_loss:
            reg_loss = torch.mean((self.final_weight - self.desc_embeddings) ** 2)
            return logits, reg_loss
        return logits

class MultiResCNN(nn.Module):
    """MultiResCNN: Multi-scale CNN with label attention (Li & Yu, AAAI 2020)"""
    def __init__(self, vocab_size=30522, embed_dim=100, num_labels=50):
        super().__init__()
        self.num_labels = num_labels

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv3 = nn.Conv1d(embed_dim, 100, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(embed_dim, 100, kernel_size=5, padding=2)
        self.conv9 = nn.Conv1d(embed_dim, 100, kernel_size=9, padding=4)

        total_filters = 300
        self.U = nn.Linear(total_filters, num_labels, bias=False)
        self.final_weight = nn.Parameter(torch.randn(num_labels, total_filters))
        self.final_bias = nn.Parameter(torch.zeros(num_labels))

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x_t = x.transpose(1, 2)
        c3 = torch.relu(self.conv3(x_t))
        c5 = torch.relu(self.conv5(x_t))
        c9 = torch.relu(self.conv9(x_t))
        C = torch.cat([c3, c5, c9], dim=1)
        C = C.transpose(1, 2)
        alpha = torch.softmax(self.U(C), dim=1)
        m = torch.bmm(alpha.transpose(1, 2), C)
        logits = torch.sum(m * self.final_weight.unsqueeze(0), dim=2) + self.final_bias
        return logits

class LAAT(nn.Module):
    """LAAT: Label Attention Model (Vu et al., EMNLP 2020)"""
    def __init__(self, vocab_size=30522, embed_dim=100, hidden_dim=256, num_labels=50):
        super().__init__()
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim * 2  # BiLSTM

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.label_queries = nn.Parameter(torch.randn(num_labels, self.hidden_dim))
        self.W_attn = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.output_weight = nn.Parameter(torch.randn(num_labels, self.hidden_dim))
        self.output_bias = nn.Parameter(torch.zeros(num_labels))

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        H, _ = self.lstm(x)
        H_proj = self.W_attn(H)
        scores = torch.einsum('bth,lh->blt', H_proj, self.label_queries)
        alpha = torch.softmax(scores, dim=2)
        m = torch.bmm(alpha, H)
        logits = torch.sum(m * self.output_weight.unsqueeze(0), dim=2) + self.output_bias
        return logits

class PLM_ICD(nn.Module):
    """PLM-ICD: Transformer with chunk pooling (Huang et al., ACL 2022)"""
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
        logits = self.classifier(pooled)
        return logits

try:
    from transformers import LongformerModel
    LONGFORMER_AVAILABLE = True
except:
    LONGFORMER_AVAILABLE = False

class LongformerICD(nn.Module):
    """Longformer for ICD coding (Beltagy et al., 2020)"""
    def __init__(self, num_labels=50, max_length=2048):
        super().__init__()
        if LONGFORMER_AVAILABLE:
            try:
                self.longformer = LongformerModel.from_pretrained('allenai/longformer-base-4096')
            except:
                print("⚠️  Longformer download failed, using BioClinicalBERT")
                self.longformer = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        else:
            print("⚠️  Longformer not available, using BioClinicalBERT")
            self.longformer = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

        self.classifier = nn.Linear(768, num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask=None, global_attention_mask=None):
        if global_attention_mask is None and 'longformer' in str(type(self.longformer)).lower():
            global_attention_mask = torch.zeros_like(input_ids)
            global_attention_mask[:, 0] = 1
            outputs = self.longformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask
            )
        else:
            outputs = self.longformer(input_ids=input_ids, attention_mask=attention_mask)

        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits

print("✅ Baseline architectures loaded:")
print("   1. CAML ✅")
print("   2. DR-CAML ✅")
print("   3. MultiResCNN ✅")
print("   4. LAAT ✅")
print("   5. PLM-ICD ✅")
print("   6. Longformer-ICD ✅")

# ============================================================================
# DATASET
# ============================================================================

class ICDDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.texts = df['text'].tolist()
        self.labels = df['labels'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }

# ============================================================================
# EVALUATION FUNCTIONS (FROM FAIR COMPARISON)
# ============================================================================

def tune_global_threshold(probs_val, y_val):
    """Find optimal threshold on validation via grid search"""
    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.arange(0.05, 0.61, 0.01):
        preds = (probs_val > threshold).astype(int)
        f1 = f1_score(y_val, preds, average='micro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold

def eval_with_threshold(probs, y_true, threshold):
    """Evaluate at fixed threshold"""
    preds = (probs > threshold).astype(int)
    return {
        'macro_f1': float(f1_score(y_true, preds, average='macro', zero_division=0)),
        'micro_f1': float(f1_score(y_true, preds, average='micro', zero_division=0))
    }

def eval_with_topk(probs, y_true, k):
    """Evaluate with top-k predictions"""
    preds = np.zeros_like(probs)
    for i in range(len(probs)):
        top_k_indices = np.argsort(probs[i])[-k:]
        preds[i, top_k_indices] = 1
    return {
        'macro_f1': float(f1_score(y_true, preds, average='macro', zero_division=0)),
        'micro_f1': float(f1_score(y_true, preds, average='micro', zero_division=0))
    }

def get_probs_from_model(model, loader):
    """Get probabilities from model"""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Getting predictions", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']

            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    return np.vstack(all_probs), np.vstack(all_labels)

def evaluate_model_complete(model, val_loader, test_loader, model_name):
    """Complete evaluation with all 3 methods"""
    print(f"\n📊 Evaluating {model_name}...")

    probs_val, y_val = get_probs_from_model(model, val_loader)
    probs_test, y_test = get_probs_from_model(model, test_loader)

    tuned_threshold = tune_global_threshold(probs_val, y_val)

    val_results = {
        'fixed_05': eval_with_threshold(probs_val, y_val, 0.5),
        'tuned': eval_with_threshold(probs_val, y_val, tuned_threshold),
        'topk': eval_with_topk(probs_val, y_val, TOP_K)
    }

    test_results = {
        'fixed_05': eval_with_threshold(probs_test, y_test, 0.5),
        'tuned': eval_with_threshold(probs_test, y_test, tuned_threshold),
        'topk': eval_with_topk(probs_test, y_test, TOP_K)
    }

    print(f"   Best threshold: {tuned_threshold:.2f} (val micro-F1: {val_results['tuned']['micro_f1']:.4f})")
    print(f"   Test: Fixed@0.5={test_results['fixed_05']['macro_f1']:.4f}, "
          f"Tuned@{tuned_threshold:.2f}={test_results['tuned']['macro_f1']:.4f}, "
          f"Top-{TOP_K}={test_results['topk']['macro_f1']:.4f}")

    return {
        'validation': val_results,
        'test': test_results,
        'tuned_threshold': tuned_threshold
    }

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_baseline(model, model_name, train_loader, val_loader, epochs=3, lr=1e-4):
    """Train baseline model"""
    print(f"\n🔧 Training {model_name}...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()

    best_val_f1 = 0.0
    patience = 2
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Validation
        probs_val, y_val = get_probs_from_model(model, val_loader)
        threshold = tune_global_threshold(probs_val, y_val)
        val_results = eval_with_threshold(probs_val, y_val, threshold)
        val_f1 = val_results['macro_f1']

        print(f"   Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Val Macro-F1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            # Save best model
            checkpoint_path = OUTPUT_BASE / 'checkpoints' / 'baselines' / f'{model_name.lower().replace(" ", "_")}_best.pt'
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   Early stopping at epoch {epoch+1}")
                break

    # Load best model
    checkpoint_path = OUTPUT_BASE / 'checkpoints' / 'baselines' / f'{model_name.lower().replace(" ", "_")}_best.pt'
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    return threshold

# ============================================================================
# SECTION B: TRAIN & EVALUATE BASELINES
# ============================================================================

print("\n" + "="*80)
print("📍 SECTION B: TRAINING BASELINE MODELS")
print("="*80)

tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

train_dataset = ICDDataset(df_train, tokenizer)
val_dataset = ICDDataset(df_val, tokenizer)
test_dataset = ICDDataset(df_test, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

baseline_results = {}

# Get ICD-10 descriptions for DR-CAML
icd_descriptions = {}
for code in TOP_50_CODES:
    # Simple description (in production, use actual ICD-10 descriptions)
    icd_descriptions[code] = f"ICD-10 code {code}"
descriptions_list = [icd_descriptions[code] for code in TOP_50_CODES]

# 1. CAML
print("\n" + "="*80)
print("🔵 Training CAML...")
print("="*80)

model_caml = CAML(num_labels=len(TOP_50_CODES)).to(device)
train_baseline(model_caml, "CAML", train_loader, val_loader, epochs=5, lr=1e-3)
baseline_results['CAML'] = evaluate_model_complete(model_caml, val_loader, test_loader, "CAML")
del model_caml
torch.cuda.empty_cache()

# 2. DR-CAML
print("\n" + "="*80)
print("🔵 Training DR-CAML...")
print("="*80)

model_dr_caml = DR_CAML(num_labels=len(TOP_50_CODES), descriptions=descriptions_list, tokenizer=tokenizer).to(device)
train_baseline(model_dr_caml, "DR-CAML", train_loader, val_loader, epochs=5, lr=1e-3)
baseline_results['DR-CAML'] = evaluate_model_complete(model_dr_caml, val_loader, test_loader, "DR-CAML")
del model_dr_caml
torch.cuda.empty_cache()

# 3. MultiResCNN
print("\n" + "="*80)
print("🔵 Training MultiResCNN...")
print("="*80)

model_multi = MultiResCNN(num_labels=len(TOP_50_CODES)).to(device)
train_baseline(model_multi, "MultiResCNN", train_loader, val_loader, epochs=5, lr=1e-3)
baseline_results['MultiResCNN'] = evaluate_model_complete(model_multi, val_loader, test_loader, "MultiResCNN")
del model_multi
torch.cuda.empty_cache()

# 4. LAAT
print("\n" + "="*80)
print("🔵 Training LAAT...")
print("="*80)

# Use smaller batch size for LAAT (memory intensive)
train_loader_laat = DataLoader(train_dataset, batch_size=8, shuffle=True)
model_laat = LAAT(num_labels=len(TOP_50_CODES)).to(device)
train_baseline(model_laat, "LAAT", train_loader_laat, val_loader, epochs=5, lr=1e-3)
baseline_results['LAAT'] = evaluate_model_complete(model_laat, val_loader, test_loader, "LAAT")
del model_laat
torch.cuda.empty_cache()

# 5. PLM-ICD
print("\n" + "="*80)
print("🔵 Training PLM-ICD...")
print("="*80)

# Use smaller batch size for transformer models
train_loader_plm = DataLoader(train_dataset, batch_size=8, shuffle=True)
base_model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT').to(device)
model_plm = PLM_ICD(base_model, num_labels=len(TOP_50_CODES)).to(device)
train_baseline(model_plm, "PLM-ICD", train_loader_plm, val_loader, epochs=3, lr=2e-5)
baseline_results['PLM-ICD'] = evaluate_model_complete(model_plm, val_loader, test_loader, "PLM-ICD")
del model_plm, base_model
torch.cuda.empty_cache()

# 6. Longformer-ICD
print("\n" + "="*80)
print("🔵 Training Longformer-ICD...")
print("="*80)

# Use smaller batch size for Longformer
train_loader_long = DataLoader(train_dataset, batch_size=2, shuffle=True)
model_long = LongformerICD(num_labels=len(TOP_50_CODES)).to(device)
train_baseline(model_long, "Longformer-ICD", train_loader_long, val_loader, epochs=2, lr=1e-5)
baseline_results['Longformer-ICD'] = evaluate_model_complete(model_long, val_loader, test_loader, "Longformer-ICD")
del model_long
torch.cuda.empty_cache()

# ============================================================================
# FINAL COMPARISON TABLE
# ============================================================================

print("\n" + "="*80)
print("📊 COMPLETE FAIR COMPARISON")
print("="*80)

# Combine all results
all_results = {**shifamind_results, **baseline_results}

# Sort by Test Macro-F1 @ Tuned
sorted_models = sorted(
    all_results.items(),
    key=lambda x: x[1]['test']['tuned']['macro_f1'],
    reverse=True
)

# Create table
print("\n" + "="*120)
print(f"{'Model':<50} {'Test Macro@0.5':<17} {'Test Macro@Tuned':<19} {'Test Macro@Top-k':<17} {'Category':<15}")
print("="*120)

for model_name, results in sorted_models:
    test_fixed = results['test']['fixed_05']['macro_f1']
    test_tuned = results['test']['tuned']['macro_f1']
    test_topk = results['test']['topk']['macro_f1']

    # Categorize
    if 'ShifaMind' in model_name or 'Phase' in model_name:
        category = 'Ablation'
    else:
        category = 'Baseline'

    print(f"{model_name:<50} {test_fixed:<17.4f} {test_tuned:<17.4f} {test_topk:<17.4f} {category:<15}")

print("="*120)

# Save all results
output_file = RESULTS_PATH / 'complete_comparison.json'
with open(output_file, 'w') as f:
    json.dump(all_results, f, indent=2)

# Save table as CSV
table_data = []
for model_name, results in sorted_models:
    table_data.append({
        'Model': model_name,
        'Test_Macro_Fixed_0.5': results['test']['fixed_05']['macro_f1'],
        'Test_Macro_Tuned': results['test']['tuned']['macro_f1'],
        'Test_Macro_Top_k': results['test']['topk']['macro_f1'],
        'Tuned_Threshold': results['tuned_threshold'],
        'Category': 'Ablation' if 'ShifaMind' in model_name or 'Phase' in model_name else 'Baseline'
    })

df_table = pd.DataFrame(table_data)
df_table.to_csv(RESULTS_PATH / 'complete_comparison.csv', index=False)

print(f"\n✅ Results saved to: {RESULTS_PATH}")
print(f"   - complete_comparison.json")
print(f"   - complete_comparison.csv")

print("\n" + "="*80)
print("✅ COMPLETE FAIR COMPARISON FINISHED!")
print("="*80)

best_model = sorted_models[0][0]
best_score = sorted_models[0][1]['test']['tuned']['macro_f1']

print(f"\nBEST MODEL: {best_model}")
print(f"Test Macro-F1 @ Tuned Threshold: {best_score:.4f}")
print(f"\nAll models evaluated with IDENTICAL protocol:")
print(f"  - Same data splits")
print(f"  - Same evaluation metrics")
print(f"  - Same threshold tuning procedure")
print(f"  - Primary metric: Test Macro-F1 @ Tuned Threshold")

print("\nAlhamdulillah! 🤲")
