#!/usr/bin/env python3
"""
================================================================================
SHIFAMIND2 PHASE 5: Ablations + SOTA Baselines (Top-50 ICD-10) - FIXED
================================================================================
Author: Mohammed Sameer Syed
University of Arizona - MS in AI Capstone

MAJOR FIXES APPLIED:
1. ✅ Threshold tuning on validation set (per-label + global)
2. ✅ Top-k prediction (k=5, avg labels per sample)
3. ✅ Positive class weighting (BCEWithLogitsLoss pos_weight)
4. ✅ Correct CAML: per-label weights (not shared classifier)
5. ✅ Correct DR-CAML: description regularization with ICD-10 descriptions
6. ✅ Correct MultiResCNN: label-wise attention added
7. ✅ Correct LAAT: label-specific attention mechanism
8. ✅ Correct PLM-ICD: chunk pooling implemented
9. ✅ Real Longformer: allenai/longformer-base-4096
10. ✅ Model-specific training configs (5 epochs CNN, 3 transformers, etc.)
11. ✅ Early stopping on validation micro-F1
12. ✅ Diagnostics: mean prediction tracking, all-zero warnings

================================================================================
"""

print("="*80)
print("🚀 SHIFAMIND2 PHASE 5 - ABLATIONS + SOTA BASELINES (TOP-50) [FIXED]")
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
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, LongformerTokenizer, LongformerModel

import json
import pickle
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import sys
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
# CONFIGURATION: LOAD FROM PREVIOUS PHASES
# ============================================================================

print("\n" + "="*80)
print("⚙️  CONFIGURATION: LOADING FROM PHASE 1-4")
print("="*80)

BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')
SHIFAMIND2_BASE = BASE_PATH / '10_ShifaMind'

run_folders = sorted([d for d in SHIFAMIND2_BASE.glob('run_*') if d.is_dir()], reverse=True)

if not run_folders:
    print("❌ No previous runs found!")
    sys.exit(1)

OUTPUT_BASE = run_folders[0]
print(f"📁 Using run folder: {OUTPUT_BASE.name}")

# Load config from Phase 1
PHASE1_CHECKPOINT = OUTPUT_BASE / 'checkpoints' / 'phase1' / 'phase1_best.pt'
checkpoint = torch.load(PHASE1_CHECKPOINT, map_location='cpu', weights_only=False)
TOP_50_CODES = checkpoint['config']['top_50_codes']
timestamp = checkpoint['config']['timestamp']

print(f"✅ Loaded config:")
print(f"   Timestamp: {timestamp}")
print(f"   Top-50 codes: {len(TOP_50_CODES)}")

SHARED_DATA_PATH = OUTPUT_BASE / 'shared_data'
RESULTS_PATH = OUTPUT_BASE / 'results' / 'phase5'
BASELINES_CHECKPOINT_PATH = OUTPUT_BASE / 'checkpoints' / 'baselines'

RESULTS_PATH.mkdir(parents=True, exist_ok=True)
BASELINES_CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)

with open(SHARED_DATA_PATH / 'concept_list.json', 'r') as f:
    ALL_CONCEPTS = json.load(f)

print(f"\n🎯 Target: {len(TOP_50_CODES)} diagnoses")
print(f"🧠 Concepts: {len(ALL_CONCEPTS)} clinical concepts")

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n" + "="*80)
print("📊 LOADING DATA (SAME SPLITS FOR ALL MODELS)")
print("="*80)

with open(SHARED_DATA_PATH / 'train_split.pkl', 'rb') as f:
    df_train = pickle.load(f)
with open(SHARED_DATA_PATH / 'val_split.pkl', 'rb') as f:
    df_val = pickle.load(f)
with open(SHARED_DATA_PATH / 'test_split.pkl', 'rb') as f:
    df_test = pickle.load(f)

print(f"✅ Train set: {len(df_train)} samples")
print(f"✅ Val set:   {len(df_val)} samples")
print(f"✅ Test set:  {len(df_test)} samples")

# Compute positive class weights for imbalance
train_labels = np.array(df_train['labels'].tolist())
pos_counts = train_labels.sum(axis=0)  # [num_labels]
neg_counts = len(train_labels) - pos_counts
pos_weight = np.clip(neg_counts / np.maximum(pos_counts, 1), 1.0, 50.0)
pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32)

print(f"\n⚖️  Positive class weights computed (median: {np.median(pos_weight):.2f}, max: {np.max(pos_weight):.2f})")

# Average labels per sample (for top-k)
avg_labels_per_sample = train_labels.sum(axis=1).mean()
TOP_K = int(round(avg_labels_per_sample))
print(f"📊 Average labels per sample: {avg_labels_per_sample:.2f} → Top-k = {TOP_K}")

# ============================================================================
# LOAD ICD-10 DESCRIPTIONS FOR DR-CAML
# ============================================================================

print("\n" + "="*80)
print("📚 LOADING ICD-10 DESCRIPTIONS")
print("="*80)

ICD_DESC_PATH = BASE_PATH / '01_Raw_Datasets' / 'Extracted' / 'mimic-iv-3.1' / 'mimic-iv-3.1' / 'hosp' / 'd_icd_diagnoses.csv.gz'

if ICD_DESC_PATH.exists():
    icd_df = pd.read_csv(ICD_DESC_PATH, compression='gzip')
    icd_df = icd_df[icd_df['icd_version'] == 10].copy()

    # Normalize codes
    def normalize_icd(code):
        if pd.isna(code):
            return None
        code = str(code).strip().upper().replace('.', '')
        return code

    icd_df['icd_code_norm'] = icd_df['icd_code'].apply(normalize_icd)
    icd_desc_map = dict(zip(icd_df['icd_code_norm'], icd_df['long_title'].fillna('')))

    # Map to TOP_50_CODES
    ICD_DESCRIPTIONS = {}
    for code in TOP_50_CODES:
        ICD_DESCRIPTIONS[code] = icd_desc_map.get(code, f"ICD-10 code {code}")

    print(f"✅ Loaded {len(ICD_DESCRIPTIONS)} ICD-10 descriptions")
    print(f"   Example: {TOP_50_CODES[0]} → {ICD_DESCRIPTIONS[TOP_50_CODES[0]][:60]}...")
else:
    print("⚠️  ICD descriptions not found - DR-CAML will use placeholders")
    ICD_DESCRIPTIONS = {code: f"ICD-10 diagnosis code {code}" for code in TOP_50_CODES}

# ============================================================================
# BASELINE MODEL ARCHITECTURES (CORRECTED)
# ============================================================================

print("\n" + "="*80)
print("🏗️  BASELINE MODEL ARCHITECTURES (FIXED IMPLEMENTATIONS)")
print("="*80)

# ----------------------------------------------------------------------------
# 1. CAML (FIXED: per-label weights)
# ----------------------------------------------------------------------------

class CAML(nn.Module):
    """
    CAML: Convolutional Attention for Multi-Label classification
    Mullenbach et al., NAACL 2018

    FIXED: Per-label weight vectors (not shared classifier)
    """
    def __init__(self, vocab_size=30522, embed_dim=100, num_filters=50, num_labels=50):
        super().__init__()
        self.num_labels = num_labels
        self.num_filters = num_filters

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, num_filters, kernel_size=4, padding=2)

        # Per-label attention
        self.U = nn.Linear(num_filters, num_labels, bias=False)  # Attention weights

        # Per-label classification weights
        self.final_weight = nn.Parameter(torch.randn(num_labels, num_filters))
        self.final_bias = nn.Parameter(torch.zeros(num_labels))

    def forward(self, input_ids, attention_mask=None):
        batch_size = input_ids.size(0)

        # Embed + Conv
        x = self.embedding(input_ids)  # [B, T, E]
        x = x.transpose(1, 2)  # [B, E, T]
        H = torch.tanh(self.conv(x))  # [B, F, T]
        H = H.transpose(1, 2)  # [B, T, F]

        # Per-label attention
        alpha = torch.softmax(self.U(H), dim=1)  # [B, T, L]

        # Weighted sum per label
        m = torch.bmm(alpha.transpose(1, 2), H)  # [B, L, F]

        # Per-label logits
        logits = torch.sum(m * self.final_weight.unsqueeze(0), dim=2) + self.final_bias

        return logits

# ----------------------------------------------------------------------------
# 2. DR-CAML (FIXED: description regularization)
# ----------------------------------------------------------------------------

class DR_CAML(nn.Module):
    """
    DR-CAML: CAML + description regularization
    Mullenbach et al., NAACL 2018

    FIXED: Actual description regularization with ICD-10 text
    """
    def __init__(self, vocab_size=30522, embed_dim=100, num_filters=50, num_labels=50,
                 descriptions=None, tokenizer=None, lambda_desc=0.1):
        super().__init__()
        self.num_labels = num_labels
        self.num_filters = num_filters
        self.lambda_desc = lambda_desc

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, num_filters, kernel_size=4, padding=2)

        # Per-label attention
        self.U = nn.Linear(num_filters, num_labels, bias=False)

        # Per-label classification weights
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
            # Tokenize
            tokens = tokenizer(desc, truncation=True, max_length=128,
                             padding='max_length', return_tensors='pt')
            input_ids = tokens['input_ids']  # [1, 128]

            # Embed + Conv + Maxpool
            with torch.no_grad():
                x = self.embedding(input_ids)  # [1, 128, embed_dim]
                x = x.transpose(1, 2)  # [1, embed_dim, 128]
                h = torch.tanh(self.conv(x))  # [1, num_filters, 128]
                pooled = F.max_pool1d(h, kernel_size=h.size(2)).squeeze()  # [num_filters]

            desc_vecs.append(pooled)

        return torch.stack(desc_vecs)  # [num_labels, num_filters]

    def forward(self, input_ids, attention_mask=None, return_reg_loss=False):
        batch_size = input_ids.size(0)

        # Same as CAML
        x = self.embedding(input_ids)
        x = x.transpose(1, 2)
        H = torch.tanh(self.conv(x))
        H = H.transpose(1, 2)

        alpha = torch.softmax(self.U(H), dim=1)
        m = torch.bmm(alpha.transpose(1, 2), H)

        logits = torch.sum(m * self.final_weight.unsqueeze(0), dim=2) + self.final_bias

        if return_reg_loss:
            # Description regularization: ||W - D||^2
            reg_loss = torch.mean((self.final_weight - self.desc_embeddings) ** 2)
            return logits, reg_loss

        return logits

# ----------------------------------------------------------------------------
# 3. MultiResCNN (FIXED: label-wise attention)
# ----------------------------------------------------------------------------

class MultiResCNN(nn.Module):
    """
    MultiResCNN: Multi-scale CNN with label attention
    Li & Yu, AAAI 2020

    FIXED: Added label-wise attention (was missing)
    """
    def __init__(self, vocab_size=30522, embed_dim=100, num_labels=50):
        super().__init__()
        self.num_labels = num_labels

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Multi-scale convolutions
        self.conv3 = nn.Conv1d(embed_dim, 100, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(embed_dim, 100, kernel_size=5, padding=2)
        self.conv9 = nn.Conv1d(embed_dim, 100, kernel_size=9, padding=4)

        total_filters = 300

        # Label-wise attention (like CAML)
        self.U = nn.Linear(total_filters, num_labels, bias=False)

        # Per-label weights
        self.final_weight = nn.Parameter(torch.randn(num_labels, total_filters))
        self.final_bias = nn.Parameter(torch.zeros(num_labels))

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)  # [B, T, E]
        x_t = x.transpose(1, 2)  # [B, E, T]

        # Multi-scale convolutions
        c3 = torch.relu(self.conv3(x_t))  # [B, 100, T]
        c5 = torch.relu(self.conv5(x_t))  # [B, 100, T]
        c9 = torch.relu(self.conv9(x_t))  # [B, 100, T]

        # Concatenate
        C = torch.cat([c3, c5, c9], dim=1)  # [B, 300, T]
        C = C.transpose(1, 2)  # [B, T, 300]

        # Label-wise attention
        alpha = torch.softmax(self.U(C), dim=1)  # [B, T, L]
        m = torch.bmm(alpha.transpose(1, 2), C)  # [B, L, 300]

        # Per-label logits
        logits = torch.sum(m * self.final_weight.unsqueeze(0), dim=2) + self.final_bias

        return logits

# ----------------------------------------------------------------------------
# 4. LAAT (FIXED: label-specific attention)
# ----------------------------------------------------------------------------

class LAAT(nn.Module):
    """
    LAAT: Label Attention Model
    Vu et al., EMNLP 2020

    FIXED: Label-specific attention (was global)
    """
    def __init__(self, vocab_size=30522, embed_dim=100, hidden_dim=256, num_labels=50):
        super().__init__()
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim * 2  # BiLSTM

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)

        # Label query vectors
        self.label_queries = nn.Parameter(torch.randn(num_labels, self.hidden_dim))

        # Attention projection
        self.W_attn = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        # Output per label
        self.output_weight = nn.Parameter(torch.randn(num_labels, self.hidden_dim))
        self.output_bias = nn.Parameter(torch.zeros(num_labels))

    def forward(self, input_ids, attention_mask=None):
        batch_size = input_ids.size(0)

        x = self.embedding(input_ids)
        H, _ = self.lstm(x)  # [B, T, hidden_dim*2]

        # Label-specific attention
        H_proj = self.W_attn(H)  # [B, T, H]

        # Compute scores for each label
        # scores[b,l,t] = H_proj[b,t,:] · label_queries[l,:]
        scores = torch.einsum('bth,lh->blt', H_proj, self.label_queries)  # [B, L, T]
        alpha = torch.softmax(scores, dim=2)  # [B, L, T]

        # Weighted context per label
        m = torch.bmm(alpha, H)  # [B, L, H]

        # Per-label logits
        logits = torch.sum(m * self.output_weight.unsqueeze(0), dim=2) + self.output_bias

        return logits

# ----------------------------------------------------------------------------
# 5. PLM-ICD (FIXED: chunk pooling)
# ----------------------------------------------------------------------------

class PLM_ICD(nn.Module):
    """
    PLM-ICD: Transformer with chunk pooling
    Huang et al., ACL 2022

    FIXED: Actual chunk-based processing
    """
    def __init__(self, base_model, num_labels=50, chunk_size=512, stride=256):
        super().__init__()
        self.bert = base_model
        self.chunk_size = chunk_size
        self.stride = stride
        self.classifier = nn.Linear(768, num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.size()

        # If short enough, process normally
        if seq_len <= self.chunk_size:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state.mean(dim=1)
        else:
            # Chunk processing
            chunk_embeddings = []

            for start in range(0, seq_len, self.stride):
                end = min(start + self.chunk_size, seq_len)

                chunk_ids = input_ids[:, start:end]
                chunk_mask = attention_mask[:, start:end] if attention_mask is not None else None

                outputs = self.bert(input_ids=chunk_ids, attention_mask=chunk_mask)
                chunk_emb = outputs.last_hidden_state.mean(dim=1)  # [B, 768]
                chunk_embeddings.append(chunk_emb)

                if end >= seq_len:
                    break

            # Max pooling across chunks
            pooled = torch.stack(chunk_embeddings, dim=1).max(dim=1)[0]  # [B, 768]

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        return logits

# ----------------------------------------------------------------------------
# 6. Longformer-ICD (FIXED: real Longformer)
# ----------------------------------------------------------------------------

class LongformerICD(nn.Module):
    """
    Longformer for ICD coding
    Beltagy et al., 2020

    FIXED: Uses actual Longformer (was BioClinicalBERT)
    """
    def __init__(self, num_labels=50, max_length=2048):
        super().__init__()
        try:
            self.longformer = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        except:
            print("⚠️  Longformer not available, using BioClinicalBERT as fallback")
            self.longformer = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

        self.classifier = nn.Linear(768, num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask=None, global_attention_mask=None):
        # Set global attention on CLS token
        if global_attention_mask is None and hasattr(self.longformer, 'config') and 'longformer' in str(type(self.longformer)).lower():
            global_attention_mask = torch.zeros_like(input_ids)
            global_attention_mask[:, 0] = 1

            outputs = self.longformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask
            )
        else:
            outputs = self.longformer(input_ids=input_ids, attention_mask=attention_mask)

        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        return logits

print("✅ Architectures defined with corrections:")
print("   1. CAML - Per-label weights ✅")
print("   2. DR-CAML - Description regularization ✅")
print("   3. MultiResCNN - Label-wise attention ✅")
print("   4. LAAT - Label-specific attention ✅")
print("   5. PLM-ICD - Chunk pooling ✅")
print("   6. Longformer-ICD - Real Longformer ✅")

# ============================================================================
# DATASET
# ============================================================================

class SimpleDataset(Dataset):
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
# THRESHOLD TUNING
# ============================================================================

def tune_thresholds(model, val_loader, model_name="Model", search_global=True):
    """
    Tune decision thresholds on validation set

    Returns:
        - best_threshold (float or np.array): global or per-label thresholds
        - best_f1 (float): validation micro-F1 at best threshold
    """
    print(f"\n🔧 Tuning thresholds for {model_name} on validation set...")

    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Collecting predictions"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    if search_global:
        # Global threshold search
        best_threshold = 0.5
        best_f1 = 0.0

        for threshold in np.arange(0.05, 0.61, 0.05):
            preds = (all_probs > threshold).astype(int)
            f1 = f1_score(all_labels, preds, average='micro', zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        print(f"   Best global threshold: {best_threshold:.2f} (val micro-F1: {best_f1:.4f})")
        return best_threshold, best_f1

    else:
        # Per-label threshold search (more expensive)
        per_label_thresholds = np.full(all_labels.shape[1], 0.5)

        for label_idx in range(all_labels.shape[1]):
            best_t = 0.5
            best_f1_label = 0.0

            for t in np.arange(0.05, 0.61, 0.05):
                preds_label = (all_probs[:, label_idx] > t).astype(int)
                f1_label = f1_score(all_labels[:, label_idx], preds_label, zero_division=0)

                if f1_label > best_f1_label:
                    best_f1_label = f1_label
                    best_t = t

            per_label_thresholds[label_idx] = best_t

        # Compute overall micro-F1 with per-label thresholds
        preds = np.zeros_like(all_probs)
        for label_idx in range(all_labels.shape[1]):
            preds[:, label_idx] = (all_probs[:, label_idx] > per_label_thresholds[label_idx]).astype(int)

        best_f1 = f1_score(all_labels, preds, average='micro', zero_division=0)

        print(f"   Per-label thresholds: median={np.median(per_label_thresholds):.2f}, val micro-F1: {best_f1:.4f}")
        return per_label_thresholds, best_f1

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, test_loader, model_name="Model", threshold=0.5, top_k=5):
    """
    Comprehensive evaluation with:
    - Tuned threshold predictions
    - Top-k predictions
    - Fixed 0.5 threshold (for reference)
    """
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating {model_name}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()

            # Shape checks
            assert probs.shape[1] == 50, f"Expected 50 labels, got {probs.shape[1]}"

            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    # Diagnostics
    mean_prob = all_probs.mean()
    max_prob = all_probs.max()
    frac_above_05 = (all_probs > 0.5).mean()

    print(f"\n   Diagnostics: mean_prob={mean_prob:.4f}, max_prob={max_prob:.4f}, frac>0.5={frac_above_05:.4f}")

    # 1. Tuned threshold predictions
    if isinstance(threshold, np.ndarray):
        # Per-label thresholds
        preds_tuned = np.zeros_like(all_probs)
        for label_idx in range(all_labels.shape[1]):
            preds_tuned[:, label_idx] = (all_probs[:, label_idx] > threshold[label_idx]).astype(int)
    else:
        # Global threshold
        preds_tuned = (all_probs > threshold).astype(int)

    # 2. Top-k predictions
    preds_topk = np.zeros_like(all_probs)
    for i in range(len(all_probs)):
        top_k_indices = np.argsort(all_probs[i])[-top_k:]
        preds_topk[i, top_k_indices] = 1

    # 3. Fixed 0.5 threshold
    preds_05 = (all_probs > 0.5).astype(int)

    # Check for all-zero predictions
    if preds_tuned.sum() == 0:
        print(f"   ⚠️  WARNING: All predictions are zero with tuned threshold!")
    if preds_topk.sum() == 0:
        print(f"   ⚠️  WARNING: All top-k predictions are zero!")

    # Compute metrics
    results = {
        'tuned': {
            'macro_f1': float(f1_score(all_labels, preds_tuned, average='macro', zero_division=0)),
            'micro_f1': float(f1_score(all_labels, preds_tuned, average='micro', zero_division=0)),
            'threshold': threshold if isinstance(threshold, float) else 'per-label'
        },
        'topk': {
            'macro_f1': float(f1_score(all_labels, preds_topk, average='macro', zero_division=0)),
            'micro_f1': float(f1_score(all_labels, preds_topk, average='micro', zero_division=0)),
            'k': top_k
        },
        'fixed_05': {
            'macro_f1': float(f1_score(all_labels, preds_05, average='macro', zero_division=0)),
            'micro_f1': float(f1_score(all_labels, preds_05, average='micro', zero_division=0))
        }
    }

    return results

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_baseline(model, model_name, train_loader, val_loader, config):
    """
    Train a baseline model with config-specific settings

    config: {
        'epochs': int,
        'lr': float,
        'patience': int,
        'batch_size': int,
        'criterion': str ('bce' or 'bce_weighted'),
        'grad_clip': float
    }
    """
    print(f"\n🏋️  Training {model_name}...")
    print(f"   Config: {config}")

    # Setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config.get('weight_decay', 0.0))

    if config.get('criterion') == 'bce_weighted':
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor.to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()

    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0.0
        epoch_probs = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            # Handle DR-CAML special case
            if isinstance(model, DR_CAML):
                logits, reg_loss = model(input_ids, attention_mask, return_reg_loss=True)
                loss = criterion(logits, labels) + model.lambda_desc * reg_loss
            else:
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

            loss.backward()

            if config.get('grad_clip'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])

            optimizer.step()

            epoch_loss += loss.item()
            epoch_probs.append(torch.sigmoid(logits).detach().cpu().numpy())

            pbar.set_postfix({'loss': loss.item()})

        avg_loss = epoch_loss / len(train_loader)
        mean_train_prob = np.vstack(epoch_probs).mean()

        print(f"   Epoch {epoch+1}: loss={avg_loss:.4f}, mean_prob={mean_train_prob:.4f}")

        # Validation
        val_threshold, val_f1 = tune_thresholds(model, val_loader, model_name, search_global=True)

        print(f"   Val F1: {val_f1:.4f} (threshold: {val_threshold:.2f})")

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            # Save checkpoint
            checkpoint_path = BASELINES_CHECKPOINT_PATH / f'{model_name.lower().replace("-", "_")}_baseline.pt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_f1': val_f1,
                'threshold': val_threshold,
                'epoch': epoch
            }, checkpoint_path)
            print(f"   ✅ Saved best model (F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"   Early stopping at epoch {epoch+1}")
                break

    # Load best model
    checkpoint_path = BASELINES_CHECKPOINT_PATH / f'{model_name.lower().replace("-", "_")}_baseline.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    return checkpoint['threshold']

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

MODEL_CONFIGS = {
    'CAML': {
        'epochs': 5,
        'lr': 1e-3,
        'patience': 2,
        'batch_size': 32,
        'criterion': 'bce_weighted',
        'grad_clip': 1.0,
        'weight_decay': 1e-4
    },
    'DR-CAML': {
        'epochs': 5,
        'lr': 1e-3,
        'patience': 2,
        'batch_size': 32,
        'criterion': 'bce_weighted',
        'grad_clip': 1.0,
        'weight_decay': 1e-4
    },
    'MultiResCNN': {
        'epochs': 5,
        'lr': 1e-3,
        'patience': 2,
        'batch_size': 32,
        'criterion': 'bce_weighted',
        'grad_clip': 1.0,
        'weight_decay': 1e-4
    },
    'LAAT': {
        'epochs': 5,
        'lr': 1e-3,
        'patience': 2,
        'batch_size': 16,
        'criterion': 'bce_weighted',
        'grad_clip': 1.0,
        'weight_decay': 1e-4
    },
    'PLM-ICD': {
        'epochs': 3,
        'lr': 2e-5,
        'patience': 1,
        'batch_size': 8,
        'criterion': 'bce_weighted',
        'grad_clip': 1.0,
        'weight_decay': 0.01
    },
    'Longformer-ICD': {
        'epochs': 2,
        'lr': 1e-5,
        'patience': 1,
        'batch_size': 2,
        'criterion': 'bce_weighted',
        'grad_clip': 1.0,
        'weight_decay': 0.01
    }
}

# ============================================================================
# SECTION A: LOAD SHIFAMIND RESULTS
# ============================================================================

print("\n" + "="*80)
print("📍 SECTION A: LOADING SHIFAMIND RESULTS")
print("="*80)

ablation_results = {}

# Load Phase 1-3 results
phase1_results_path = OUTPUT_BASE / 'results' / 'phase1' / 'results.json'
if phase1_results_path.exists():
    with open(phase1_results_path, 'r') as f:
        phase1_results = json.load(f)
    ablation_results['without_graphsage'] = {
        'macro_f1': phase1_results['diagnosis_metrics']['macro_f1'],
        'micro_f1': phase1_results['diagnosis_metrics']['micro_f1']
    }
    print(f"✅ Phase 1 (w/o GraphSAGE): Macro-F1 = {ablation_results['without_graphsage']['macro_f1']:.4f}")

phase2_results_path = OUTPUT_BASE / 'results' / 'phase2' / 'results.json'
if phase2_results_path.exists():
    with open(phase2_results_path, 'r') as f:
        phase2_results = json.load(f)
    ablation_results['without_rag'] = {
        'macro_f1': phase2_results['diagnosis_metrics']['macro_f1'],
        'micro_f1': phase2_results['diagnosis_metrics']['micro_f1']
    }
    print(f"✅ Phase 2 (w/o RAG): Macro-F1 = {ablation_results['without_rag']['macro_f1']:.4f}")

phase3_results_path = OUTPUT_BASE / 'results' / 'phase3' / 'results.json'
if phase3_results_path.exists():
    with open(phase3_results_path, 'r') as f:
        phase3_results = json.load(f)
    ablation_results['full_model'] = {
        'macro_f1': phase3_results['diagnosis_metrics']['macro_f1'],
        'micro_f1': phase3_results['diagnosis_metrics']['micro_f1']
    }
    print(f"✅ Phase 3 (Full ShifaMind): Macro-F1 = {ablation_results['full_model']['macro_f1']:.4f}")

# ============================================================================
# SECTION B: TRAIN & EVALUATE BASELINES
# ============================================================================

print("\n" + "="*80)
print("📍 SECTION B: TRAINING SOTA BASELINES")
print("="*80)

tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

# Prepare description list for DR-CAML
desc_list = [ICD_DESCRIPTIONS[code] for code in TOP_50_CODES]

sota_results = {}

# Baseline models
baseline_models = {
    'CAML': CAML(num_labels=len(TOP_50_CODES)),
    'DR-CAML': DR_CAML(num_labels=len(TOP_50_CODES), descriptions=desc_list,
                       tokenizer=tokenizer, lambda_desc=0.1),
    'MultiResCNN': MultiResCNN(num_labels=len(TOP_50_CODES)),
    'LAAT': LAAT(num_labels=len(TOP_50_CODES)),
}

# Add transformer models
base_model_plm = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT').to(device)
baseline_models['PLM-ICD'] = PLM_ICD(base_model_plm, num_labels=len(TOP_50_CODES))

baseline_models['Longformer-ICD'] = LongformerICD(num_labels=len(TOP_50_CODES))

print(f"\n🏋️  Training {len(baseline_models)} baselines with proper configs...")

for model_name, model in baseline_models.items():
    print(f"\n{'='*70}")
    print(f"🏆 {model_name}")
    print(f"{'='*70}")

    model = model.to(device)
    config = MODEL_CONFIGS[model_name]

    checkpoint_path = BASELINES_CHECKPOINT_PATH / f'{model_name.lower().replace("-", "_")}_baseline.pt'

    # Prepare dataloaders with model-specific batch size
    train_dataset = SimpleDataset(df_train, tokenizer, max_length=512)
    val_dataset = SimpleDataset(df_val, tokenizer, max_length=512)
    test_dataset = SimpleDataset(df_test, tokenizer, max_length=512)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Train or load
    if checkpoint_path.exists():
        print(f"📥 Loading existing checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        tuned_threshold = checkpoint.get('threshold', 0.5)
    else:
        tuned_threshold = train_baseline(model, model_name, train_loader, val_loader, config)

    # Evaluate on test set
    results = evaluate_model(model, test_loader, model_name, threshold=tuned_threshold, top_k=TOP_K)
    sota_results[model_name] = results

    print(f"\n📊 Test Results for {model_name}:")
    print(f"   Tuned threshold: Macro-F1={results['tuned']['macro_f1']:.4f}, Micro-F1={results['tuned']['micro_f1']:.4f}")
    print(f"   Top-{TOP_K}:         Macro-F1={results['topk']['macro_f1']:.4f}, Micro-F1={results['topk']['micro_f1']:.4f}")
    print(f"   Fixed 0.5:       Macro-F1={results['fixed_05']['macro_f1']:.4f}, Micro-F1={results['fixed_05']['micro_f1']:.4f}")

    # Save individual results
    with open(RESULTS_PATH / f'baseline_{model_name.lower().replace("-", "_")}.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Cleanup
    del model
    torch.cuda.empty_cache()

# ============================================================================
# COMPREHENSIVE COMPARISON
# ============================================================================

print("\n" + "="*80)
print("📊 COMPREHENSIVE COMPARISON TABLE")
print("="*80)

comparison_data = []

# ShifaMind ablations
if 'full_model' in ablation_results:
    comparison_data.append({
        'Model': 'ShifaMind (Full)',
        'Macro-F1': ablation_results['full_model']['macro_f1'],
        'Micro-F1': ablation_results['full_model']['micro_f1'],
        'Interpretable': 'Yes',
        'Type': 'Ours'
    })

if 'without_rag' in ablation_results:
    comparison_data.append({
        'Model': 'ShifaMind w/o RAG',
        'Macro-F1': ablation_results['without_rag']['macro_f1'],
        'Micro-F1': ablation_results['without_rag']['micro_f1'],
        'Interpretable': 'Yes',
        'Type': 'Ablation'
    })

if 'without_graphsage' in ablation_results:
    comparison_data.append({
        'Model': 'ShifaMind w/o GraphSAGE',
        'Macro-F1': ablation_results['without_graphsage']['macro_f1'],
        'Micro-F1': ablation_results['without_graphsage']['micro_f1'],
        'Interpretable': 'Yes',
        'Type': 'Ablation'
    })

# SOTA baselines (use tuned threshold results)
for model_name, results in sota_results.items():
    comparison_data.append({
        'Model': model_name,
        'Macro-F1': results['tuned']['macro_f1'],
        'Micro-F1': results['tuned']['micro_f1'],
        'Interpretable': 'No',
        'Type': 'SOTA Baseline'
    })

# Sort by macro-F1
comparison_df = pd.DataFrame(comparison_data).sort_values('Macro-F1', ascending=False)

print("\n" + "="*95)
print(f"{'Model':<30} {'Macro-F1':<12} {'Micro-F1':<12} {'Interpretable':<15} {'Type':<15}")
print("="*95)
for _, row in comparison_df.iterrows():
    print(f"{row['Model']:<30} {row['Macro-F1']:<12.4f} {row['Micro-F1']:<12.4f} {row['Interpretable']:<15} {row['Type']:<15}")
print("="*95)

# Save
comparison_df.to_csv(RESULTS_PATH / 'comparison_table.csv', index=False)

final_results = {
    'timestamp': timestamp,
    'run_folder': str(OUTPUT_BASE),
    'ablation_studies': ablation_results,
    'sota_comparison': sota_results,
    'comparison_table': comparison_data,
    'top_k': TOP_K
}

with open(RESULTS_PATH / 'ablation_sota_results.json', 'w') as f:
    json.dump(final_results, f, indent=2)

print(f"\n✅ Results saved to: {RESULTS_PATH}")

print("\n" + "="*80)
print("✅ SHIFAMIND2 PHASE 5 COMPLETE!")
print("="*80)
print("\nAlhamdulillah! 🤲")
