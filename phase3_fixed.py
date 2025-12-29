#!/usr/bin/env python3
"""
================================================================================
SHIFAMIND PHASE 3 FIXED: XAI Metrics Evaluation
================================================================================
Author: Mohammed Sameer Syed
University of Arizona - MS in AI Capstone

SELF-CONTAINED COLAB SCRIPT - Copy-paste ready!

XAI Metrics for Phase 2 Fixed:
1. ERASER Faithfulness (Comprehensiveness, Sufficiency)
2. Concept-Diagnosis Association (Discriminability, Consistency)
3. Keyword Attention Ratio
4. Clinical Validity (Concept F1, FP Rate)

Loads:
- Train/val/test splits from Phase 1
- Phase 2 Fixed checkpoint + concept embeddings

Saves:
- XAI metrics to 07_ShifaMind/results/phase3_fixed/

TARGET XAI METRICS (vs broken original):
- Keyword Attention: >0.10 (was 0.025)
- Concept F1: >0.50 (was 0.254)
- Comprehensiveness: >0.15 (was 0.082)
- Sufficiency: >0.70 (was 0.623)
================================================================================
"""

print("="*80)
print("🚀 SHIFAMIND PHASE 3 FIXED - XAI METRICS EVALUATION")
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
from transformers import AutoTokenizer, AutoModel

import json
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from collections import defaultdict
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

BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')
OUTPUT_BASE = BASE_PATH / '07_ShifaMind'

# Input paths
SHARED_DATA_PATH = OUTPUT_BASE / 'shared_data'
PHASE2_FIXED_CHECKPOINT = OUTPUT_BASE / 'checkpoints/phase2_fixed/phase2_fixed_best.pt'

# Output paths
RESULTS_PATH = OUTPUT_BASE / 'results/phase3_fixed'
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

print(f"📁 Results: {RESULTS_PATH}")

TARGET_CODES = ['J189', 'I5023', 'A419', 'K8000']
ICD_DESCRIPTIONS = {
    'J189': 'Pneumonia, unspecified organism',
    'I5023': 'Acute on chronic systolic heart failure',
    'A419': 'Sepsis, unspecified organism',
    'K8000': 'Calculus of gallbladder with acute cholecystitis'
}

# Clinical keywords for attention analysis
DIAGNOSIS_KEYWORDS = {
    'J189': ['pneumonia', 'lung', 'respiratory', 'infiltrate', 'fever', 'cough', 'dyspnea'],
    'I5023': ['heart', 'cardiac', 'failure', 'edema', 'dyspnea', 'orthopnea', 'bnp'],
    'A419': ['sepsis', 'bacteremia', 'infection', 'fever', 'hypotension', 'shock', 'lactate'],
    'K8000': ['cholecystitis', 'gallbladder', 'gallstone', 'abdominal', 'murphy', 'pain']
}

print(f"\n🎯 Target: {len(TARGET_CODES)} diagnoses")

# ============================================================================
# ARCHITECTURE (ALL INLINE - NO IMPORTS)
# ============================================================================

print("\n" + "="*80)
print("🏗️  ARCHITECTURE COMPONENTS")
print("="*80)

class AdaptiveGatedCrossAttention(nn.Module):
    """
    Fixed cross-attention with learnable content-dependent gates
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

        # Adaptive gate network (content-dependent)
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, concept_embeddings, attention_mask=None):
        """
        Args:
            hidden_states: [batch, seq_len, hidden]
            concept_embeddings: [num_concepts, hidden]
            attention_mask: [batch, seq_len]
        Returns:
            output: [batch, seq_len, hidden]
            attn_weights: [batch, seq_len, num_concepts] - for XAI analysis
        """
        batch_size, seq_len, _ = hidden_states.shape
        num_concepts = concept_embeddings.shape[0]

        # Expand concepts to batch
        concepts_batch = concept_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        # Multi-head cross-attention
        Q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(concepts_batch).view(batch_size, num_concepts, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(concepts_batch).view(batch_size, num_concepts, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        context = self.out_proj(context)

        # Compute relevance score (cosine similarity)
        text_pooled = hidden_states.mean(dim=1)  # [batch, hidden]
        concept_pooled = concepts_batch.mean(dim=1)  # [batch, hidden]
        relevance = F.cosine_similarity(text_pooled, concept_pooled, dim=-1)  # [batch]
        relevance = relevance.unsqueeze(-1).unsqueeze(-1)  # [batch, 1, 1]
        relevance = relevance.expand(-1, seq_len, -1)  # [batch, seq_len, 1]
        relevance_features = relevance.expand(-1, -1, self.hidden_size)  # [batch, seq_len, hidden]

        # Learnable content-dependent gate
        gate_input = torch.cat([hidden_states, context, relevance_features], dim=-1)  # [batch, seq_len, hidden*3]
        gate_values = self.gate_net(gate_input)  # [batch, seq_len, 1]

        # Apply gating
        output = hidden_states + gate_values * context
        output = self.layer_norm(output)

        # Return attention weights averaged over heads for XAI analysis
        attn_weights_avg = attn_weights.mean(dim=1)  # [batch, seq_len, num_concepts]

        return output, attn_weights_avg


class ShifaMindPhase1Fixed(nn.Module):
    """
    Fixed Phase 1 with learnable cross-attention (matches Phase 2 architecture)
    """
    def __init__(self, base_model, concept_embeddings_init, num_classes, fusion_layers=[9, 11]):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = base_model.config.hidden_size
        self.concept_embeddings = nn.Parameter(concept_embeddings_init.clone())
        self.num_concepts = concept_embeddings_init.shape[0]

        self.fusion_modules = nn.ModuleDict({
            str(layer): AdaptiveGatedCrossAttention(self.hidden_size, layer_idx=layer)
            for layer in fusion_layers
        })

        self.diagnosis_head = nn.Linear(self.hidden_size, num_classes)
        self.concept_head = nn.Linear(self.hidden_size, self.num_concepts)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, return_attention=False):
        # Base BERT encoding
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        hidden_states = outputs.hidden_states
        current_hidden = outputs.last_hidden_state

        # Apply fusion at specified layers
        fusion_attentions = {}
        for layer_idx in [9, 11]:
            if str(layer_idx) in self.fusion_modules:
                layer_hidden = hidden_states[layer_idx]
                fused_hidden, attn_weights = self.fusion_modules[str(layer_idx)](
                    layer_hidden, self.concept_embeddings, attention_mask
                )
                current_hidden = fused_hidden
                if return_attention:
                    fusion_attentions[f'layer_{layer_idx}'] = attn_weights

        # Classification
        cls_hidden = self.dropout(current_hidden[:, 0, :])
        diagnosis_logits = self.diagnosis_head(cls_hidden)
        concept_logits = self.concept_head(cls_hidden)

        result = {
            'logits': diagnosis_logits,
            'concept_scores': concept_logits,
            'cls_hidden': cls_hidden,
            'hidden_states': current_hidden
        }

        if return_attention:
            result['fusion_attentions'] = fusion_attentions

        return result


class AdaptiveRAGFusion(nn.Module):
    """
    Adaptive RAG fusion with relevance-based gating
    """
    def __init__(self, hidden_size=768):
        super().__init__()
        self.rag_proj = nn.Linear(hidden_size, hidden_size)

        # Adaptive gate based on text quality, RAG quality, and relevance
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, text_cls, rag_cls, relevance_score):
        """
        Args:
            text_cls: [batch, hidden] - text CLS
            rag_cls: [batch, hidden] - RAG CLS
            relevance_score: [batch, 1] - RAG relevance score
        """
        rag_projected = self.rag_proj(rag_cls)

        # Content-dependent gate
        gate_input = torch.cat([text_cls, rag_projected, relevance_score], dim=-1)
        gate = self.gate_net(gate_input)

        # Adaptive fusion
        fused = (1 - gate) * text_cls + gate * rag_projected
        fused = self.layer_norm(fused)

        return fused, gate


class ShifaMindPhase2Fixed(nn.Module):
    """
    Phase 2 Fixed with diagnosis-aware RAG
    """
    def __init__(self, phase1_model):
        super().__init__()
        self.phase1_model = phase1_model
        self.rag_fusion = AdaptiveRAGFusion(hidden_size=phase1_model.hidden_size)

    def forward(self, input_ids, attention_mask, concept_embeddings,
                rag_input_ids=None, rag_attention_mask=None, relevance_score=None,
                return_attention=False):
        """
        Forward pass with optional RAG
        For XAI evaluation, we primarily analyze Phase 1 attention patterns
        """
        # Get Phase 1 outputs
        phase1_out = self.phase1_model(
            input_ids, attention_mask, concept_embeddings,
            return_attention=return_attention
        )

        # If RAG available, fuse
        if rag_input_ids is not None and relevance_score is not None:
            rag_outputs = self.phase1_model.base_model(
                input_ids=rag_input_ids,
                attention_mask=rag_attention_mask
            )
            rag_cls = rag_outputs.last_hidden_state[:, 0, :]

            fused_cls, gate = self.rag_fusion(
                phase1_out['cls_hidden'], rag_cls, relevance_score
            )

            # Re-classify with fused representation
            diagnosis_logits = self.phase1_model.diagnosis_head(fused_cls)
            phase1_out['logits'] = diagnosis_logits
            phase1_out['rag_gate'] = gate

        return phase1_out

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

# Load concept labels
train_concept_labels = np.load(SHARED_DATA_PATH / 'train_concept_labels.npy')
val_concept_labels = np.load(SHARED_DATA_PATH / 'val_concept_labels.npy')
test_concept_labels = np.load(SHARED_DATA_PATH / 'test_concept_labels.npy')

print(f"✅ Concept labels loaded")

# ============================================================================
# LOAD PHASE 2 FIXED MODEL
# ============================================================================

print("\n" + "="*80)
print("📥 LOADING PHASE 2 FIXED MODEL")
print("="*80)

print("Loading BioClinicalBERT tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
print("✅ Tokenizer")

print("\nLoading Phase 2 Fixed checkpoint...")
checkpoint = torch.load(PHASE2_FIXED_CHECKPOINT, map_location=device, weights_only=False)
print(f"✅ Checkpoint loaded (F1: {checkpoint.get('macro_f1', 0):.4f})")

# Load concept embeddings
concept_embeddings = checkpoint['concept_embeddings'].to(device)
num_concepts = concept_embeddings.shape[0]

print(f"Concept embeddings: {concept_embeddings.shape}")

# Reconstruct model
base_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)
phase1_model = ShifaMindPhase1Fixed(
    base_model=base_model,
    concept_embeddings_init=concept_embeddings,
    num_classes=len(TARGET_CODES),
    fusion_layers=[9, 11]
).to(device)

model = ShifaMindPhase2Fixed(phase1_model).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✅ Phase 2 Fixed model loaded and ready")

# ============================================================================
# DATASET
# ============================================================================

class XAIDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, concept_labels=None, max_length=384):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.concept_labels = concept_labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text, padding='max_length', truncation=True,
            max_length=self.max_length, return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(self.labels[idx]),
            'text': text
        }

        if self.concept_labels is not None:
            item['concept_labels'] = torch.FloatTensor(self.concept_labels[idx])

        return item

test_dataset = XAIDataset(
    df_test['text'].tolist(), df_test['labels'].tolist(), tokenizer,
    concept_labels=test_concept_labels
)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"\n✅ XAI dataset ready: {len(test_dataset)} samples")

# ============================================================================
# XAI METRIC 1: KEYWORD ATTENTION RATIO
# ============================================================================

print("\n" + "="*80)
print("📐 METRIC 1: KEYWORD ATTENTION RATIO")
print("="*80)

def compute_keyword_attention(model, test_loader, concept_embeddings, tokenizer, keywords_dict, device):
    """
    Keyword Attention Ratio: % of attention on clinical keywords
    TARGET: >0.10 (was 0.025 in broken version)
    """
    print("\n🔬 Computing Keyword Attention Ratio...")

    keyword_attention_ratios = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="  Keyword Attention"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()
            texts = batch['text']

            outputs = model(input_ids, attention_mask, return_attention=True)

            if 'fusion_attentions' in outputs:
                # Use fusion attention from layer 11 (most semantic)
                fusion_attn = outputs['fusion_attentions'].get('layer_11')

                if fusion_attn is not None:
                    # fusion_attn: [batch, seq_len, num_concepts]
                    for i in range(len(texts)):
                        text = texts[i].lower()
                        tokens = tokenizer.tokenize(text)

                        # Find keyword tokens for this diagnosis
                        for j, code in enumerate(TARGET_CODES):
                            if labels[i][j] == 1:
                                keywords = keywords_dict.get(code, [])
                                keyword_mask = torch.zeros(input_ids.shape[1], device=device)

                                for k, token in enumerate(tokens[:input_ids.shape[1]-2]):  # Account for [CLS] and [SEP]
                                    actual_idx = k + 1  # Skip [CLS]
                                    if any(kw in token for kw in keywords):
                                        keyword_mask[actual_idx] = 1

                                if keyword_mask.sum() > 0:
                                    # Average attention over concepts dimension
                                    avg_attn = fusion_attn[i].mean(dim=1)  # [seq_len]
                                    attn_on_keywords = (avg_attn * keyword_mask).sum() / keyword_mask.sum()
                                    keyword_attention_ratios.append(attn_on_keywords.item())

    keyword_attention_ratio = np.mean(keyword_attention_ratios) if keyword_attention_ratios else 0

    print(f"\n  ✅ Keyword Attention Ratio: {keyword_attention_ratio:.4f}")
    print(f"     (Target: >0.10, Original: 0.025)")

    return {
        'keyword_attention_ratio': float(keyword_attention_ratio),
        'num_samples': len(keyword_attention_ratios)
    }

keyword_metrics = compute_keyword_attention(
    model, test_loader, concept_embeddings, tokenizer, DIAGNOSIS_KEYWORDS, device
)

# ============================================================================
# XAI METRIC 2: CLINICAL VALIDITY (CONCEPT F1)
# ============================================================================

print("\n" + "="*80)
print("📐 METRIC 2: CLINICAL VALIDITY")
print("="*80)

def compute_clinical_validity(model, test_loader, concept_embeddings, device):
    """
    Clinical Validity: Concept prediction accuracy
    TARGET: Concept F1 >0.50 (was 0.254 in broken version)
    """
    print("\n🔬 Computing Clinical Validity...")

    all_concept_preds = []
    all_concept_labels = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="  Clinical Validity"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            concept_labels = batch['concept_labels'].cpu().numpy()

            outputs = model(input_ids, attention_mask)
            concept_preds = (torch.sigmoid(outputs['concept_scores']) > 0.5).cpu().numpy().astype(int)

            all_concept_preds.append(concept_preds)
            all_concept_labels.append(concept_labels)

    all_concept_preds = np.vstack(all_concept_preds)
    all_concept_labels = np.vstack(all_concept_labels)

    # Compute metrics
    tp_rate = precision_score(all_concept_labels, all_concept_preds, average='macro', zero_division=0)
    recall = recall_score(all_concept_labels, all_concept_preds, average='macro', zero_division=0)
    f1 = f1_score(all_concept_labels, all_concept_preds, average='macro', zero_division=0)

    # False positive rate
    fp_rate = ((all_concept_preds == 1) & (all_concept_labels == 0)).sum() / max((all_concept_labels == 0).sum(), 1)

    print(f"\n  ✅ Concept F1:           {f1:.4f}")
    print(f"     (Target: >0.50, Original: 0.254)")
    print(f"  ✅ Precision:            {tp_rate:.4f}")
    print(f"  ✅ Recall:               {recall:.4f}")
    print(f"  ✅ False Positive Rate:  {fp_rate:.4f}")

    return {
        'true_positive_rate': float(tp_rate),
        'recall': float(recall),
        'f1': float(f1),
        'false_positive_rate': float(fp_rate)
    }

clinical_validity_metrics = compute_clinical_validity(model, test_loader, concept_embeddings, device)

# ============================================================================
# XAI METRIC 3: ERASER FAITHFULNESS
# ============================================================================

print("\n" + "="*80)
print("📐 METRIC 3: ERASER FAITHFULNESS")
print("="*80)

def compute_comprehensiveness(model, test_loader, concept_embeddings, device, top_k_tokens=20):
    """
    Comprehensiveness: F1 drop when high-attention tokens are removed
    Higher = better (evidence matters)
    TARGET: >0.15 (was 0.082 in broken version)
    """
    print("\n🔬 Computing Comprehensiveness...")

    all_preds_full = []
    all_preds_removed = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="  Comprehensiveness"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Full prediction with attention
            outputs_full = model(input_ids, attention_mask, return_attention=True)
            preds_full = torch.sigmoid(outputs_full['logits']).cpu().numpy()

            # Get top-k attended tokens
            if 'fusion_attentions' in outputs_full:
                fusion_attn = outputs_full['fusion_attentions'].get('layer_11')

                if fusion_attn is not None:
                    # Average over concepts: [batch, seq_len]
                    avg_attn = fusion_attn.mean(dim=2)

                    # Find top-k tokens per sample
                    topk_values, topk_indices = torch.topk(avg_attn, k=min(top_k_tokens, avg_attn.shape[1]), dim=1)

                    # Create modified mask (remove top-k tokens)
                    modified_mask = attention_mask.clone()
                    for b in range(input_ids.shape[0]):
                        modified_mask[b, topk_indices[b]] = 0

                    # Predict with evidence removed
                    outputs_removed = model(input_ids, modified_mask)
                    preds_removed = torch.sigmoid(outputs_removed['logits']).cpu().numpy()
                else:
                    preds_removed = preds_full
            else:
                preds_removed = preds_full

            all_preds_full.append(preds_full)
            all_preds_removed.append(preds_removed)
            all_labels.append(labels.cpu().numpy())

    all_preds_full = np.vstack(all_preds_full)
    all_preds_removed = np.vstack(all_preds_removed)
    all_labels = np.vstack(all_labels)

    # Compute F1 drop
    f1_full = f1_score(all_labels, (all_preds_full > 0.5).astype(int), average='macro', zero_division=0)
    f1_removed = f1_score(all_labels, (all_preds_removed > 0.5).astype(int), average='macro', zero_division=0)

    comprehensiveness = f1_full - f1_removed

    print(f"\n  ✅ Comprehensiveness: {comprehensiveness:.4f}")
    print(f"     (Target: >0.15, Original: 0.082)")
    print(f"     F1 (full):        {f1_full:.4f}")
    print(f"     F1 (removed):     {f1_removed:.4f}")

    return {
        'comprehensiveness': float(comprehensiveness),
        'f1_full': float(f1_full),
        'f1_removed': float(f1_removed)
    }

def compute_sufficiency(model, test_loader, concept_embeddings, device, top_k_tokens=20):
    """
    Sufficiency: F1 with ONLY high-attention tokens
    Higher = better (evidence is sufficient)
    TARGET: >0.70 (was 0.623 in broken version)
    """
    print("\n🔬 Computing Sufficiency...")

    all_preds_full = []
    all_preds_only = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="  Sufficiency"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Full prediction
            outputs_full = model(input_ids, attention_mask, return_attention=True)
            preds_full = torch.sigmoid(outputs_full['logits']).cpu().numpy()

            # Keep only top-k attended tokens
            if 'fusion_attentions' in outputs_full:
                fusion_attn = outputs_full['fusion_attentions'].get('layer_11')

                if fusion_attn is not None:
                    avg_attn = fusion_attn.mean(dim=2)
                    topk_values, topk_indices = torch.topk(avg_attn, k=min(top_k_tokens, avg_attn.shape[1]), dim=1)

                    # Create modified mask (keep only top-k)
                    modified_mask = torch.zeros_like(attention_mask)
                    for b in range(input_ids.shape[0]):
                        modified_mask[b, topk_indices[b]] = 1
                        modified_mask[b, 0] = 1  # Keep [CLS]

                    # Predict with only evidence
                    outputs_only = model(input_ids, modified_mask)
                    preds_only = torch.sigmoid(outputs_only['logits']).cpu().numpy()
                else:
                    preds_only = preds_full
            else:
                preds_only = preds_full

            all_preds_full.append(preds_full)
            all_preds_only.append(preds_only)
            all_labels.append(labels.cpu().numpy())

    all_preds_full = np.vstack(all_preds_full)
    all_preds_only = np.vstack(all_preds_only)
    all_labels = np.vstack(all_labels)

    # Compute sufficiency ratio
    f1_full = f1_score(all_labels, (all_preds_full > 0.5).astype(int), average='macro', zero_division=0)
    f1_only = f1_score(all_labels, (all_preds_only > 0.5).astype(int), average='macro', zero_division=0)

    sufficiency = f1_only / f1_full if f1_full > 0 else 0

    print(f"\n  ✅ Sufficiency: {sufficiency:.4f}")
    print(f"     (Target: >0.70, Original: 0.623)")
    print(f"     F1 (full): {f1_full:.4f}")
    print(f"     F1 (only): {f1_only:.4f}")

    return {
        'sufficiency': float(sufficiency),
        'f1_full': float(f1_full),
        'f1_only': float(f1_only)
    }

comprehensiveness_metrics = compute_comprehensiveness(model, test_loader, concept_embeddings, device)
sufficiency_metrics = compute_sufficiency(model, test_loader, concept_embeddings, device)

eraser_metrics = {
    'comprehensiveness': comprehensiveness_metrics,
    'sufficiency': sufficiency_metrics
}

# ============================================================================
# XAI METRIC 4: CONCEPT-DIAGNOSIS ASSOCIATION
# ============================================================================

print("\n" + "="*80)
print("📐 METRIC 4: CONCEPT-DIAGNOSIS ASSOCIATION")
print("="*80)

def compute_concept_association(model, test_loader, concept_embeddings, device):
    """
    Discriminability: Can concepts distinguish diagnoses?
    Consistency: Are concept activations stable within diagnosis?
    """
    print("\n🔬 Computing Concept-Diagnosis Association...")

    diagnosis_concepts = {code: [] for code in TARGET_CODES}

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="  Association"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()

            outputs = model(input_ids, attention_mask)
            concept_probs = torch.sigmoid(outputs['concept_scores']).cpu().numpy()

            # Group by diagnosis
            for i in range(len(labels)):
                for j, code in enumerate(TARGET_CODES):
                    if labels[i][j] == 1:
                        diagnosis_concepts[code].append(concept_probs[i])

    # Discriminability: KL divergence between diagnosis concept distributions
    discriminability_scores = []
    for code1 in TARGET_CODES:
        for code2 in TARGET_CODES:
            if code1 != code2 and len(diagnosis_concepts[code1]) > 0 and len(diagnosis_concepts[code2]) > 0:
                dist1 = np.mean(diagnosis_concepts[code1], axis=0) + 1e-10
                dist2 = np.mean(diagnosis_concepts[code2], axis=0) + 1e-10
                kl_div = np.sum(dist1 * np.log(dist1 / dist2))
                discriminability_scores.append(kl_div)

    discriminability = np.mean(discriminability_scores) if discriminability_scores else 0

    # Consistency: Lower std = higher consistency
    consistency_scores = []
    for code in TARGET_CODES:
        if len(diagnosis_concepts[code]) > 1:
            std = np.std(diagnosis_concepts[code], axis=0).mean()
            consistency_scores.append(1.0 / (1.0 + std))

    consistency = np.mean(consistency_scores) if consistency_scores else 0

    print(f"\n  ✅ Discriminability: {discriminability:.4f}")
    print(f"  ✅ Consistency:      {consistency:.4f}")

    return {
        'discriminability': float(discriminability),
        'consistency': float(consistency)
    }

concept_association_metrics = compute_concept_association(model, test_loader, concept_embeddings, device)

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("💾 SAVING RESULTS")
print("="*80)

xai_metrics = {
    'phase': 'Phase 3 Fixed - XAI Metrics',
    'keyword_attention': keyword_metrics,
    'clinical_validity': clinical_validity_metrics,
    'eraser_faithfulness': eraser_metrics,
    'concept_diagnosis_association': concept_association_metrics,
    'summary': {
        'keyword_attention_ratio': keyword_metrics['keyword_attention_ratio'],
        'concept_f1': clinical_validity_metrics['f1'],
        'comprehensiveness': eraser_metrics['comprehensiveness']['comprehensiveness'],
        'sufficiency': eraser_metrics['sufficiency']['sufficiency'],
        'discriminability': concept_association_metrics['discriminability'],
        'consistency': concept_association_metrics['consistency'],
        'false_positive_rate': clinical_validity_metrics['false_positive_rate']
    },
    'targets': {
        'keyword_attention_ratio': '>0.10',
        'concept_f1': '>0.50',
        'comprehensiveness': '>0.15',
        'sufficiency': '>0.70'
    },
    'original_broken': {
        'keyword_attention_ratio': 0.025,
        'concept_f1': 0.254,
        'comprehensiveness': 0.082,
        'sufficiency': 0.623
    }
}

with open(RESULTS_PATH / 'xai_metrics.json', 'w') as f:
    json.dump(xai_metrics, f, indent=2)

print(f"✅ Saved: {RESULTS_PATH / 'xai_metrics.json'}")

# Create summary table
summary_df = pd.DataFrame({
    'Metric': [
        'Keyword Attention Ratio',
        'Concept F1',
        'Comprehensiveness',
        'Sufficiency',
        'Discriminability',
        'Consistency',
        'False Positive Rate'
    ],
    'Value': [
        f"{xai_metrics['summary']['keyword_attention_ratio']:.4f}",
        f"{xai_metrics['summary']['concept_f1']:.4f}",
        f"{xai_metrics['summary']['comprehensiveness']:.4f}",
        f"{xai_metrics['summary']['sufficiency']:.4f}",
        f"{xai_metrics['summary']['discriminability']:.4f}",
        f"{xai_metrics['summary']['consistency']:.4f}",
        f"{xai_metrics['summary']['false_positive_rate']:.4f}"
    ],
    'Target': [
        '>0.10',
        '>0.50',
        '>0.15',
        '>0.70',
        'Higher',
        'Higher',
        'Lower'
    ],
    'Original': [
        '0.025',
        '0.254',
        '0.082',
        '0.623',
        '-',
        '-',
        '-'
    ]
})

summary_df.to_csv(RESULTS_PATH / 'xai_summary.csv', index=False)
print(f"✅ Saved: {RESULTS_PATH / 'xai_summary.csv'}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("🎉 PHASE 3 FIXED COMPLETE - XAI METRICS")
print("="*80)

print("\n📊 XAI Metrics Summary:")
print(f"   Keyword Attention Ratio:  {xai_metrics['summary']['keyword_attention_ratio']:.4f} (Target: >0.10, was 0.025)")
print(f"   Concept F1:               {xai_metrics['summary']['concept_f1']:.4f} (Target: >0.50, was 0.254)")
print(f"   Comprehensiveness:        {xai_metrics['summary']['comprehensiveness']:.4f} (Target: >0.15, was 0.082)")
print(f"   Sufficiency:              {xai_metrics['summary']['sufficiency']:.4f} (Target: >0.70, was 0.623)")
print(f"   Discriminability:         {xai_metrics['summary']['discriminability']:.4f}")
print(f"   Consistency:              {xai_metrics['summary']['consistency']:.4f}")
print(f"   False Positive Rate:      {xai_metrics['summary']['false_positive_rate']:.4f}")

# Check if targets met
targets_met = {
    'keyword_attention': xai_metrics['summary']['keyword_attention_ratio'] > 0.10,
    'concept_f1': xai_metrics['summary']['concept_f1'] > 0.50,
    'comprehensiveness': xai_metrics['summary']['comprehensiveness'] > 0.15,
    'sufficiency': xai_metrics['summary']['sufficiency'] > 0.70
}

print("\n✅ Targets Met:")
for metric, met in targets_met.items():
    status = "✓" if met else "✗"
    print(f"   {status} {metric.replace('_', ' ').title()}")

print(f"\n💾 Results: {RESULTS_PATH}")
print("\n🚀 Ready for Phase 4 (Ablations & Baselines)")
print(f"\nAlhamdulillah! 🤲")
