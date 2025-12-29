#!/usr/bin/env python3
"""
================================================================================
SHIFAMIND PHASE 3: XAI Metrics Evaluation
================================================================================
Author: Mohammed Sameer Syed
University of Arizona - MS in AI Capstone

XAI Metrics:
1. ERASER Faithfulness:
   - Comprehensiveness: Model performance drop when evidence is removed
   - Sufficiency: Model performance with only evidence

2. Concept-Diagnosis Association:
   - Discriminability: Concept activation differences between diagnoses
   - Consistency: Concept activation stability within diagnosis

3. Text-Based Clinical Relevance:
   - Keyword Attention Ratio: Attention on clinical keywords

4. Clinical Validity:
   - True positive concept activation rate
   - False positive concept activation rate

Loads:
- Train/val/test splits from Phase 1
- Phase 2 checkpoint + concept embeddings

Saves:
- XAI metrics to 07_ShifaMind/results/phase3/
================================================================================
"""

print("="*80)
print("🚀 SHIFAMIND PHASE 3 - XAI METRICS EVALUATION")
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
PHASE2_CHECKPOINT = OUTPUT_BASE / 'checkpoints/phase2/phase2_final.pt'

# Output paths
RESULTS_PATH = OUTPUT_BASE / 'results/phase3'
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
# LOAD SAVED SPLITS
# ============================================================================

print("\n" + "="*80)
print("📥 LOADING SAVED SPLITS")
print("="*80)

print("Loading splits...")
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

print(f"\n✅ Concept labels loaded")

# ============================================================================
# LOAD PHASE 2 MODEL
# ============================================================================

print("\n" + "="*80)
print("📥 LOADING PHASE 2 MODEL")
print("="*80)

# Architecture definitions (must match Phase 2)
class GatedCrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=8, dropout=0.1, layer_idx=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.Sigmoid())
        self.out_proj = nn.Linear(hidden_size, hidden_size)
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

        gate_input = torch.cat([hidden_states, context], dim=-1)
        gate_values = self.gate(gate_input)
        output = hidden_states + gate_values * context
        output = self.layer_norm(output)

        return output, attn_weights.mean(dim=1)

class PointerNetwork(nn.Module):
    def __init__(self, hidden_size, max_spans=5):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_spans = max_spans
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, concept_embeddings, text_hidden, attention_mask=None):
        batch_size, seq_len, hidden = text_hidden.shape
        if concept_embeddings.dim() == 2:
            num_concepts = concept_embeddings.shape[0]
            concept_embeddings = concept_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            num_concepts = concept_embeddings.shape[1]
        queries = self.query_proj(concept_embeddings)
        keys = self.key_proj(text_hidden)
        pointer_scores = torch.bmm(queries, keys.transpose(1, 2)) / (hidden ** 0.5)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).expand(-1, num_concepts, -1)
            pointer_scores = pointer_scores.masked_fill(mask == 0, float('-inf'))
        pointer_probs = F.softmax(pointer_scores, dim=-1)
        top_scores, top_indices = torch.topk(pointer_probs, k=min(self.max_spans, seq_len), dim=-1)
        return {
            'span_scores': top_scores.mean(),
            'pointer_probs': pointer_probs,
            'top_indices': top_indices
        }

class ForcedCitationHead(nn.Module):
    def __init__(self, hidden_size, num_concepts, max_spans=5, top_k_concepts=5):
        super().__init__()
        self.concept_classifier = nn.Linear(hidden_size, num_concepts)
        self.pointer_network = PointerNetwork(hidden_size, max_spans)
        self.top_k = top_k_concepts

    def forward(self, cls_hidden, text_hidden, concept_embeddings, attention_mask=None):
        concept_logits = self.concept_classifier(cls_hidden)
        concept_probs = torch.sigmoid(concept_logits)
        top_scores, top_indices = torch.topk(concept_probs, k=self.top_k, dim=-1)
        batch_size = cls_hidden.shape[0]
        selected = torch.stack([concept_embeddings[top_indices[b]] for b in range(batch_size)])
        evidence = self.pointer_network(selected, text_hidden, attention_mask)
        return {
            'concept_logits': concept_logits,
            'concept_probs': concept_probs,
            'top_concept_indices': top_indices,
            'span_scores': evidence['span_scores'],
            'pointer_probs': evidence['pointer_probs'],
            'top_span_indices': evidence.get('top_indices')
        }

class ShifaMindPhase1(nn.Module):
    def __init__(self, base_model, num_concepts, num_classes, fusion_layers=[9, 11]):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = base_model.config.hidden_size
        self.fusion_layers = fusion_layers

        self.fusion_modules = nn.ModuleDict({
            str(layer): GatedCrossAttention(self.hidden_size, layer_idx=layer)
            for layer in fusion_layers
        })

        self.layer_gates = nn.ParameterDict({
            str(layer): nn.Parameter(torch.tensor(0.5), requires_grad=False)
            for layer in fusion_layers
        })

        self.diagnosis_head = nn.Linear(self.hidden_size, num_classes)
        self.concept_head = nn.Linear(self.hidden_size, num_concepts)
        self.diagnosis_concept_interaction = nn.Bilinear(num_classes, num_concepts, num_concepts)
        self.citation_head = ForcedCitationHead(self.hidden_size, num_concepts)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, concept_embeddings, return_evidence=False, return_attention=False):
        outputs = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, output_attentions=return_attention,
            return_dict=True
        )

        hidden_states = outputs.hidden_states
        current_hidden = outputs.last_hidden_state

        fusion_attentions = {}
        for layer_idx in self.fusion_layers:
            if str(layer_idx) in self.fusion_modules:
                layer_hidden = hidden_states[layer_idx]
                fused_hidden, attn = self.fusion_modules[str(layer_idx)](
                    layer_hidden, concept_embeddings, attention_mask
                )
                gate = torch.sigmoid(self.layer_gates[str(layer_idx)])
                current_hidden = (1 - gate) * current_hidden + gate * fused_hidden
                if return_attention:
                    fusion_attentions[f'layer_{layer_idx}'] = attn

        cls_hidden = self.dropout(current_hidden[:, 0, :])
        diagnosis_logits = self.diagnosis_head(cls_hidden)
        concept_logits = self.concept_head(cls_hidden)
        refined_concept_logits = self.diagnosis_concept_interaction(
            torch.sigmoid(diagnosis_logits), torch.sigmoid(concept_logits)
        )

        result = {
            'logits': diagnosis_logits,
            'concept_scores': refined_concept_logits,
            'cls_hidden': cls_hidden,
            'hidden_states': current_hidden
        }

        if return_evidence:
            citation_out = self.citation_head(cls_hidden, current_hidden, concept_embeddings, attention_mask)
            result['citations'] = citation_out

        if return_attention:
            result['fusion_attentions'] = fusion_attentions
            result['base_attentions'] = outputs.attentions

        return result

class SimpleRAGFusion(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.rag_gate = nn.Parameter(torch.tensor(0.2))
        self.rag_proj = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, text_cls, rag_cls):
        rag_projected = self.rag_proj(rag_cls)
        gate = torch.sigmoid(self.rag_gate).clamp(0.0, 0.4)
        fused = (1 - gate) * text_cls + gate * rag_projected
        fused = self.layer_norm(fused)
        return fused, gate

class ShifaMindPhase2(nn.Module):
    def __init__(self, phase1_model, tokenizer):
        super().__init__()
        self.phase1_model = phase1_model
        self.tokenizer = tokenizer
        self.rag_fusion = SimpleRAGFusion(hidden_size=phase1_model.hidden_size)

    def forward(self, input_ids, attention_mask, concept_embeddings, return_evidence=False, return_attention=False):
        # For XAI, we use Phase 1 base model (without RAG for clearer analysis)
        return self.phase1_model(
            input_ids, attention_mask, concept_embeddings,
            return_evidence=return_evidence, return_attention=return_attention
        )

print("Loading BioClinicalBERT tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
print("✅ Tokenizer")

print("\nLoading Phase 2 checkpoint...")
checkpoint = torch.load(PHASE2_CHECKPOINT, map_location=device)
print(f"✅ Checkpoint loaded (F1: {checkpoint.get('macro_f1', 0):.4f})")

# Load Phase 1 checkpoint for num_concepts
phase1_checkpoint_path = checkpoint.get('phase1_checkpoint')
if phase1_checkpoint_path:
    phase1_ckpt = torch.load(phase1_checkpoint_path, map_location=device)
    num_concepts = phase1_ckpt['num_concepts']
    concept_embeddings = phase1_ckpt['concept_embeddings'].to(device)
else:
    # Fallback
    concept_embeddings = torch.load(SHARED_DATA_PATH / 'concept_embeddings.pt', map_location=device)
    num_concepts = concept_embeddings.shape[0]

base_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)
phase1_model = ShifaMindPhase1(
    base_model=base_model, num_concepts=num_concepts,
    num_classes=len(TARGET_CODES), fusion_layers=[9, 11]
).to(device)

model = ShifaMindPhase2(phase1_model, tokenizer).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✅ Phase 2 model loaded")

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
# XAI METRIC 1: ERASER FAITHFULNESS
# ============================================================================

print("\n" + "="*80)
print("📐 METRIC 1: ERASER FAITHFULNESS")
print("="*80)

def compute_comprehensiveness(model, test_loader, concept_embeddings, device, top_k=5):
    """
    Comprehensiveness: How much does performance drop when evidence is removed?
    Higher = better (evidence is important)
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

            # Full prediction
            outputs_full = model(input_ids, attention_mask, concept_embeddings, return_evidence=True)
            preds_full = torch.sigmoid(outputs_full['logits']).cpu().numpy()

            # Get evidence spans
            if 'citations' in outputs_full and outputs_full['citations'] is not None:
                top_spans = outputs_full['citations'].get('top_span_indices')

                # Remove evidence (mask out top attention tokens)
                if top_spans is not None:
                    modified_mask = attention_mask.clone()
                    for b in range(input_ids.shape[0]):
                        if top_spans.shape[0] > b:
                            span_indices = top_spans[b, :, :top_k].flatten()
                            modified_mask[b, span_indices] = 0

                    outputs_removed = model(input_ids, modified_mask, concept_embeddings)
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
    print(f"     F1 (full):    {f1_full:.4f}")
    print(f"     F1 (removed): {f1_removed:.4f}")

    return {
        'comprehensiveness': float(comprehensiveness),
        'f1_full': float(f1_full),
        'f1_removed': float(f1_removed)
    }

def compute_sufficiency(model, test_loader, concept_embeddings, device, top_k=5):
    """
    Sufficiency: How well does model perform with ONLY evidence?
    Higher = better (evidence is sufficient)
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
            outputs_full = model(input_ids, attention_mask, concept_embeddings, return_evidence=True)
            preds_full = torch.sigmoid(outputs_full['logits']).cpu().numpy()

            # Keep only evidence
            if 'citations' in outputs_full and outputs_full['citations'] is not None:
                top_spans = outputs_full['citations'].get('top_span_indices')

                if top_spans is not None:
                    modified_mask = torch.zeros_like(attention_mask)
                    for b in range(input_ids.shape[0]):
                        if top_spans.shape[0] > b:
                            span_indices = top_spans[b, :, :top_k].flatten()
                            modified_mask[b, span_indices] = 1

                    outputs_only = model(input_ids, modified_mask, concept_embeddings)
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

    # Compute F1 drop
    f1_full = f1_score(all_labels, (all_preds_full > 0.5).astype(int), average='macro', zero_division=0)
    f1_only = f1_score(all_labels, (all_preds_only > 0.5).astype(int), average='macro', zero_division=0)

    sufficiency = f1_only / f1_full if f1_full > 0 else 0

    print(f"\n  ✅ Sufficiency: {sufficiency:.4f}")
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
# XAI METRIC 2: CONCEPT-DIAGNOSIS ASSOCIATION
# ============================================================================

print("\n" + "="*80)
print("📐 METRIC 2: CONCEPT-DIAGNOSIS ASSOCIATION")
print("="*80)

def compute_concept_association(model, test_loader, concept_embeddings, device):
    """
    Discriminability: Can concepts distinguish between diagnoses?
    Consistency: Are concept activations consistent within diagnosis?
    """
    print("\n🔬 Computing Concept-Diagnosis Association...")

    diagnosis_concepts = {code: [] for code in TARGET_CODES}

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="  Association"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()

            outputs = model(input_ids, attention_mask, concept_embeddings)
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

    # Consistency: Average std within each diagnosis
    consistency_scores = []
    for code in TARGET_CODES:
        if len(diagnosis_concepts[code]) > 1:
            std = np.std(diagnosis_concepts[code], axis=0).mean()
            consistency_scores.append(1.0 / (1.0 + std))  # Lower std = higher consistency

    consistency = np.mean(consistency_scores) if consistency_scores else 0

    print(f"\n  ✅ Discriminability: {discriminability:.4f}")
    print(f"  ✅ Consistency:      {consistency:.4f}")

    return {
        'discriminability': float(discriminability),
        'consistency': float(consistency)
    }

concept_association_metrics = compute_concept_association(model, test_loader, concept_embeddings, device)

# ============================================================================
# XAI METRIC 3: KEYWORD ATTENTION RATIO
# ============================================================================

print("\n" + "="*80)
print("📐 METRIC 3: KEYWORD ATTENTION RATIO")
print("="*80)

def compute_keyword_attention(model, test_loader, concept_embeddings, tokenizer, keywords_dict, device):
    """
    Keyword Attention Ratio: % of attention on clinical keywords
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

            outputs = model(input_ids, attention_mask, concept_embeddings, return_attention=True)

            if 'fusion_attentions' in outputs:
                # Use fusion attention from layer 11
                fusion_attn = outputs['fusion_attentions'].get('layer_11')

                if fusion_attn is not None:
                    # fusion_attn: (batch, num_concepts, seq_len)
                    for i in range(len(texts)):
                        text = texts[i].lower()
                        tokens = tokenizer.tokenize(text)

                        # Find keyword tokens
                        for j, code in enumerate(TARGET_CODES):
                            if labels[i][j] == 1:
                                keywords = keywords_dict.get(code, [])
                                keyword_mask = torch.zeros(input_ids.shape[1], device=device)

                                for k, token in enumerate(tokens[:input_ids.shape[1]]):
                                    if any(kw in token for kw in keywords):
                                        keyword_mask[k] = 1

                                if keyword_mask.sum() > 0:
                                    # Average attention over concepts first, then compute keyword attention
                                    # fusion_attn[i] shape: (seq_len, num_concepts)
                                    avg_attn = fusion_attn[i].mean(dim=1)  # Average over concepts -> (seq_len,)
                                    attn_on_keywords = (avg_attn * keyword_mask).sum() / keyword_mask.sum()
                                    keyword_attention_ratios.append(attn_on_keywords.item())

    keyword_attention_ratio = np.mean(keyword_attention_ratios) if keyword_attention_ratios else 0

    print(f"\n  ✅ Keyword Attention Ratio: {keyword_attention_ratio:.4f}")

    return {
        'keyword_attention_ratio': float(keyword_attention_ratio),
        'num_samples': len(keyword_attention_ratios)
    }

keyword_metrics = compute_keyword_attention(
    model, test_loader, concept_embeddings, tokenizer, DIAGNOSIS_KEYWORDS, device
)

# ============================================================================
# XAI METRIC 4: CLINICAL VALIDITY
# ============================================================================

print("\n" + "="*80)
print("📐 METRIC 4: CLINICAL VALIDITY")
print("="*80)

def compute_clinical_validity(model, test_loader, concept_embeddings, device):
    """
    Clinical Validity:
    - True Positive Rate: Correct concept activations when diagnosis is present
    - False Positive Rate: Incorrect concept activations when diagnosis is absent
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

            outputs = model(input_ids, attention_mask, concept_embeddings)
            concept_preds = (torch.sigmoid(outputs['concept_scores']) > 0.7).cpu().numpy().astype(int)

            all_concept_preds.append(concept_preds)
            all_concept_labels.append(concept_labels)

    all_concept_preds = np.vstack(all_concept_preds)
    all_concept_labels = np.vstack(all_concept_labels)

    # Compute metrics
    tp_rate = precision_score(all_concept_labels, all_concept_preds, average='macro', zero_division=0)
    recall = recall_score(all_concept_labels, all_concept_preds, average='macro', zero_division=0)
    f1 = f1_score(all_concept_labels, all_concept_preds, average='macro', zero_division=0)

    # False positive rate
    fp_rate = ((all_concept_preds == 1) & (all_concept_labels == 0)).sum() / (all_concept_labels == 0).sum()

    print(f"\n  ✅ True Positive Rate:  {tp_rate:.4f}")
    print(f"  ✅ Recall:              {recall:.4f}")
    print(f"  ✅ F1:                  {f1:.4f}")
    print(f"  ✅ False Positive Rate: {fp_rate:.4f}")

    return {
        'true_positive_rate': float(tp_rate),
        'recall': float(recall),
        'f1': float(f1),
        'false_positive_rate': float(fp_rate)
    }

clinical_validity_metrics = compute_clinical_validity(model, test_loader, concept_embeddings, device)

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("💾 SAVING RESULTS")
print("="*80)

xai_metrics = {
    'phase': 'Phase 3 - XAI Metrics',
    'eraser_faithfulness': eraser_metrics,
    'concept_diagnosis_association': concept_association_metrics,
    'keyword_attention': keyword_metrics,
    'clinical_validity': clinical_validity_metrics,
    'summary': {
        'comprehensiveness': eraser_metrics['comprehensiveness']['comprehensiveness'],
        'sufficiency': eraser_metrics['sufficiency']['sufficiency'],
        'discriminability': concept_association_metrics['discriminability'],
        'consistency': concept_association_metrics['consistency'],
        'keyword_attention_ratio': keyword_metrics['keyword_attention_ratio'],
        'concept_f1': clinical_validity_metrics['f1'],
        'false_positive_rate': clinical_validity_metrics['false_positive_rate']
    }
}

with open(RESULTS_PATH / 'xai_metrics.json', 'w') as f:
    json.dump(xai_metrics, f, indent=2)

print(f"✅ Saved: {RESULTS_PATH / 'xai_metrics.json'}")

# Create summary table
summary_df = pd.DataFrame({
    'Metric': [
        'Comprehensiveness',
        'Sufficiency',
        'Discriminability',
        'Consistency',
        'Keyword Attention Ratio',
        'Concept F1',
        'False Positive Rate'
    ],
    'Value': [
        f"{xai_metrics['summary']['comprehensiveness']:.4f}",
        f"{xai_metrics['summary']['sufficiency']:.4f}",
        f"{xai_metrics['summary']['discriminability']:.4f}",
        f"{xai_metrics['summary']['consistency']:.4f}",
        f"{xai_metrics['summary']['keyword_attention_ratio']:.4f}",
        f"{xai_metrics['summary']['concept_f1']:.4f}",
        f"{xai_metrics['summary']['false_positive_rate']:.4f}"
    ]
})

summary_df.to_csv(RESULTS_PATH / 'xai_summary.csv', index=False)
print(f"✅ Saved: {RESULTS_PATH / 'xai_summary.csv'}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("🎉 PHASE 3 COMPLETE - XAI METRICS")
print("="*80)

print("\n📊 XAI Metrics Summary:")
print(f"   Comprehensiveness:        {xai_metrics['summary']['comprehensiveness']:.4f}")
print(f"   Sufficiency:              {xai_metrics['summary']['sufficiency']:.4f}")
print(f"   Discriminability:         {xai_metrics['summary']['discriminability']:.4f}")
print(f"   Consistency:              {xai_metrics['summary']['consistency']:.4f}")
print(f"   Keyword Attention Ratio:  {xai_metrics['summary']['keyword_attention_ratio']:.4f}")
print(f"   Concept F1:               {xai_metrics['summary']['concept_f1']:.4f}")
print(f"   False Positive Rate:      {xai_metrics['summary']['false_positive_rate']:.4f}")

print(f"\n💾 Results: {RESULTS_PATH}")
print("\n🚀 Ready for Phase 4 (Ablations & Baselines)")
print(f"\nAlhamdulillah! 🤲")
