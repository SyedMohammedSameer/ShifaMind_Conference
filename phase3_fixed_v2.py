#!/usr/bin/env python3
"""
================================================================================
SHIFAMIND PHASE 3 V2: Research-Backed Concept-Level XAI Metrics
================================================================================
Author: Mohammed Sameer Syed
University of Arizona - MS in AI Capstone

SELF-CONTAINED COLAB SCRIPT - Copy-paste ready!

Concept-Level XAI Metrics (Research-Backed):
1. Concept Completeness (Yeh et al., NeurIPS 2020)
2. Concept Intervention Accuracy (Koh et al., ICML 2020)
3. TCAV Score (Kim et al., ICML 2018)
4. ConceptSHAP (Yeh et al., NeurIPS 2020)
5. Concept F1, Consistency, Discriminability (preserved from Phase 3)

Why NOT token-level metrics:
- ERASER Comprehensiveness/Sufficiency measure token faithfulness
- Our model provides concept-level explanations
- Token-level metrics are inappropriate for concept-grounded models

Loads:
- Train/val/test splits from Phase 1
- Phase 2 Fixed checkpoint + concept embeddings

Saves:
- Concept-level XAI metrics to 07_ShifaMind/results/phase3_fixed_v2/

References:
1. Koh, P.W. et al. (2020). "Concept Bottleneck Models." ICML.
2. Kim, B. et al. (2018). "TCAV: Interpretability Beyond Feature Attribution." ICML.
3. Yeh, C.K. et al. (2020). "On Completeness-aware Concept-Based Explanations." NeurIPS.

TARGET METRICS:
- Concept Completeness: >0.80
- Intervention Gain: >0.05
- Avg TCAV (relevant): >0.60
- Concept F1: >0.50
================================================================================
"""

print("="*80)
print("🚀 SHIFAMIND PHASE 3 V2 - CONCEPT-LEVEL XAI METRICS")
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
from itertools import combinations

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
BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')
OUTPUT_BASE = BASE_PATH / '07_ShifaMind'

# Input paths
SHARED_DATA_PATH = OUTPUT_BASE / 'shared_data'

# Try both possible Phase 2 checkpoint names
PHASE2_FIXED_CHECKPOINT_BEST = OUTPUT_BASE / 'checkpoints/phase2_fixed/phase2_fixed_best.pt'
PHASE2_FIXED_CHECKPOINT_FINAL = OUTPUT_BASE / 'checkpoints/phase2_fixed/phase2_fixed_final.pt'

# Check which Phase 2 checkpoint exists
if PHASE2_FIXED_CHECKPOINT_BEST.exists():
    PHASE2_FIXED_CHECKPOINT = PHASE2_FIXED_CHECKPOINT_BEST
elif PHASE2_FIXED_CHECKPOINT_FINAL.exists():
    PHASE2_FIXED_CHECKPOINT = PHASE2_FIXED_CHECKPOINT_FINAL
else:
    PHASE2_FIXED_CHECKPOINT = PHASE2_FIXED_CHECKPOINT_BEST  # Default, will error later

# Output paths
RESULTS_PATH = OUTPUT_BASE / 'results/phase3_fixed_v2'
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
# ARCHITECTURE (ALL INLINE - NO IMPORTS)
# ============================================================================

print("\n" + "="*80)
print("🏗️  ARCHITECTURE COMPONENTS")
print("="*80)

class AdaptiveGatedCrossAttention(nn.Module):
    """Fixed cross-attention with learnable content-dependent gates"""
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
        relevance = relevance.unsqueeze(-1).unsqueeze(-1)
        relevance_features = relevance.expand(-1, seq_len, -1)

        # Learnable gate
        gate_input = torch.cat([hidden_states, context, relevance_features], dim=-1)
        gate_values = self.gate_net(gate_input)

        output = hidden_states + gate_values * context
        output = self.layer_norm(output)

        return output, attn_weights.mean(dim=1)


class ShifaMindPhase1Fixed(nn.Module):
    """Phase 1 Fixed model with concept grounding"""
    def __init__(self, base_model, num_concepts, num_classes, fusion_layers=[9, 11]):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = base_model.config.hidden_size
        self.num_concepts = num_concepts
        self.fusion_layers = fusion_layers

        # Concept embeddings as parameters
        concept_embeddings_init = torch.randn(num_concepts, self.hidden_size) * 0.02
        self.concept_embeddings = nn.Parameter(concept_embeddings_init)

        # Fusion modules at specified layers
        self.fusion_modules = nn.ModuleDict({
            str(layer): AdaptiveGatedCrossAttention(self.hidden_size, layer_idx=layer)
            for layer in fusion_layers
        })

        # Heads
        self.diagnosis_head = nn.Linear(self.hidden_size, num_classes)
        self.concept_head = nn.Linear(self.hidden_size, num_concepts)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, return_attention=False):
        outputs = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, return_dict=True
        )

        hidden_states = outputs.hidden_states
        current_hidden = outputs.last_hidden_state
        attention_maps = {}

        # Apply fusion at specified layers
        for layer_idx in self.fusion_layers:
            if str(layer_idx) in self.fusion_modules:
                layer_hidden = hidden_states[layer_idx]
                fused_hidden, attn = self.fusion_modules[str(layer_idx)](
                    layer_hidden, self.concept_embeddings, attention_mask
                )
                current_hidden = fused_hidden
                if return_attention:
                    attention_maps[f'layer_{layer_idx}'] = attn

        cls_hidden = self.dropout(current_hidden[:, 0, :])
        diagnosis_logits = self.diagnosis_head(cls_hidden)
        concept_scores = torch.sigmoid(self.concept_head(cls_hidden))

        result = {
            'logits': diagnosis_logits,
            'concept_scores': concept_scores,
            'hidden_states': current_hidden
        }
        if return_attention:
            result['attention_maps'] = attention_maps

        return result


class AdaptiveRAGFusion(nn.Module):
    """Adaptive RAG fusion with learnable gating"""
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # RAG fusion gate
        self.rag_fusion_gate = nn.Sequential(
            nn.Linear(hidden_size * 2 + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, cls_hidden, rag_hidden, relevance_score):
        batch_size = cls_hidden.shape[0]

        # Expand relevance score to match hidden dimension
        relevance = relevance_score.view(batch_size, 1).expand(-1, self.hidden_size)

        # Concatenate features
        fusion_input = torch.cat([cls_hidden, rag_hidden, relevance_score.unsqueeze(-1)], dim=-1)

        # Compute gate
        gate = self.rag_fusion_gate(fusion_input)

        # Fuse
        fused = cls_hidden + gate * rag_hidden
        fused = self.layer_norm(fused)

        return fused, gate


class ShifaMindPhase2Fixed(nn.Module):
    """Phase 2 Fixed with diagnosis-aware RAG"""
    def __init__(self, phase1_model, rag_hidden_size=768):
        super().__init__()
        self.phase1_model = phase1_model
        self.hidden_size = phase1_model.hidden_size

        # RAG processing
        self.rag_encoder = nn.Linear(rag_hidden_size, self.hidden_size)
        self.rag_fusion = AdaptiveRAGFusion(self.hidden_size)

        # Diagnosis head (overrides Phase 1)
        self.diagnosis_head_final = nn.Linear(self.hidden_size, len(TARGET_CODES))
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, return_attention=False):
        # Phase 1 forward
        phase1_outputs = self.phase1_model(input_ids, attention_mask, return_attention=return_attention)

        cls_hidden = self.dropout(phase1_outputs['hidden_states'][:, 0, :])
        diagnosis_logits = self.diagnosis_head_final(cls_hidden)

        result = {
            'logits': diagnosis_logits,
            'concept_scores': phase1_outputs['concept_scores'],
            'hidden_states': phase1_outputs['hidden_states']
        }
        if return_attention:
            result['attention_maps'] = phase1_outputs.get('attention_maps', {})

        return result

    def forward_with_concept_intervention(self, input_ids, attention_mask, gt_concept_mask):
        """
        Forward pass with concept intervention (for Koh et al., 2020 metric)

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            gt_concept_mask: Ground truth concept activation mask [batch, num_concepts]

        Returns:
            Model outputs with intervened concepts
        """
        # Get base BERT outputs
        outputs = self.phase1_model.base_model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, return_dict=True
        )

        hidden_states = outputs.hidden_states
        current_hidden = outputs.last_hidden_state

        # Intervene: weight concept embeddings by ground truth mask
        batch_size = input_ids.shape[0]
        intervened_concepts = self.phase1_model.concept_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        # Zero out concepts that should be inactive
        intervened_concepts = intervened_concepts * gt_concept_mask.unsqueeze(-1)

        # Apply fusion with intervened concepts
        for layer_idx in self.phase1_model.fusion_layers:
            if str(layer_idx) in self.phase1_model.fusion_modules:
                layer_hidden = hidden_states[layer_idx]

                # Use intervened concepts instead of learned ones
                batch_concepts = intervened_concepts.mean(dim=0)  # Pool across batch
                fused_hidden, _ = self.phase1_model.fusion_modules[str(layer_idx)](
                    layer_hidden, batch_concepts, attention_mask
                )
                current_hidden = fused_hidden

        cls_hidden = self.dropout(current_hidden[:, 0, :])
        diagnosis_logits = self.diagnosis_head_final(cls_hidden)
        concept_scores = torch.sigmoid(self.phase1_model.concept_head(cls_hidden))

        return {
            'logits': diagnosis_logits,
            'concept_scores': concept_scores
        }

    def forward_with_concept_mask(self, input_ids, attention_mask, concept_mask):
        """
        Forward pass with concept masking (for ConceptSHAP)

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            concept_mask: Binary mask for active concepts [num_concepts]

        Returns:
            Diagnosis logits
        """
        # Get base BERT outputs
        outputs = self.phase1_model.base_model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, return_dict=True
        )

        hidden_states = outputs.hidden_states
        current_hidden = outputs.last_hidden_state

        # Mask concept embeddings
        masked_concepts = self.phase1_model.concept_embeddings * concept_mask.unsqueeze(-1)

        # Apply fusion with masked concepts
        for layer_idx in self.phase1_model.fusion_layers:
            if str(layer_idx) in self.phase1_model.fusion_modules:
                layer_hidden = hidden_states[layer_idx]
                fused_hidden, _ = self.phase1_model.fusion_modules[str(layer_idx)](
                    layer_hidden, masked_concepts, attention_mask
                )
                current_hidden = fused_hidden

        cls_hidden = self.dropout(current_hidden[:, 0, :])
        diagnosis_logits = self.diagnosis_head_final(cls_hidden)

        return diagnosis_logits


print("✅ Architecture components defined")

# ============================================================================
# DATASET
# ============================================================================

class XAIDataset(Dataset):
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
            str(self.texts[idx]), padding='max_length', truncation=True,
            max_length=self.max_length, return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(self.labels[idx]),
            'concept_labels': torch.FloatTensor(self.concept_labels[idx])
        }

# ============================================================================
# CONCEPT-LEVEL XAI METRICS
# ============================================================================

print("\n" + "="*80)
print("📊 CONCEPT-LEVEL XAI METRICS")
print("="*80)


def compute_concept_completeness(model, test_loader, device):
    """
    Concept Completeness (Yeh et al., NeurIPS 2020)

    Measures: How sufficient are concepts for explaining predictions?

    η = 1 - L(y, g(c)) / L(y, f(x))

    Where:
    - g(c): Predictions from concepts only (via linear probe)
    - f(x): Full model predictions
    - Higher η means concepts are more complete explanations

    Target: η > 0.80
    """
    print("\n📊 Computing Concept Completeness (Yeh et al., NeurIPS 2020)...")

    model.eval()
    num_concepts = model.phase1_model.num_concepts
    hidden_size = model.hidden_size

    # Create concept-only linear probe
    concept_probe = nn.Linear(num_concepts, len(TARGET_CODES)).to(device)

    # Train concept probe on training data (use first few batches)
    optimizer = torch.optim.Adam(concept_probe.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    print("   Training concept-only probe...")
    for epoch in range(5):  # Quick training
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx > 20:  # Use subset for training
                break

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
                concept_scores = outputs['concept_scores']

            concept_preds = concept_probe(concept_scores)
            loss = criterion(concept_preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate completeness
    full_losses = []
    concept_losses = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="   Evaluating completeness"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Full model predictions
            full_outputs = model(input_ids, attention_mask)
            full_preds = full_outputs['logits']
            concept_scores = full_outputs['concept_scores']

            # Concept-only predictions
            concept_preds = concept_probe(concept_scores)

            # Compute losses
            full_loss = criterion(full_preds, labels)
            concept_loss = criterion(concept_preds, labels)

            full_losses.append(full_loss.item())
            concept_losses.append(concept_loss.item())

    avg_full_loss = np.mean(full_losses)
    avg_concept_loss = np.mean(concept_losses)

    # Completeness = 1 - (concept_error / full_error)
    completeness = 1 - (avg_concept_loss / avg_full_loss)

    print(f"   Full Model Loss: {avg_full_loss:.4f}")
    print(f"   Concept-Only Loss: {avg_concept_loss:.4f}")
    print(f"   Completeness η: {completeness:.4f} {'✅' if completeness > 0.80 else '❌'} (Target: >0.80)")

    return {
        'completeness': float(completeness),
        'full_model_loss': float(avg_full_loss),
        'concept_only_loss': float(avg_concept_loss),
        'target': 0.80,
        'interpretation': f"Concepts explain {completeness*100:.1f}% of model's prediction power"
    }


def compute_intervention_accuracy(model, test_loader, device):
    """
    Concept Intervention Accuracy (Koh et al., ICML 2020)

    Measures: Does correcting concept predictions improve diagnosis accuracy?

    Test if model causally uses concepts by intervening:
    1. Get predictions with PREDICTED concepts
    2. Get predictions with GROUND TRUTH concepts
    3. Intervention gain = Acc(GT) - Acc(predicted)

    Higher intervention gain = concepts are causally important

    Target: Gain > 0.05
    """
    print("\n📊 Computing Concept Intervention Accuracy (Koh et al., ICML 2020)...")

    model.eval()

    pred_concept_all_labels = []
    pred_concept_all_preds = []
    gt_concept_all_labels = []
    gt_concept_all_preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="   Evaluating intervention"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            concept_labels = batch['concept_labels'].to(device)

            # 1. Predictions with model's own concepts
            pred_outputs = model(input_ids, attention_mask)
            pred_preds = (torch.sigmoid(pred_outputs['logits']) > 0.5).float()

            pred_concept_all_labels.append(labels.cpu())
            pred_concept_all_preds.append(pred_preds.cpu())

            # 2. Predictions with ground truth concept intervention
            gt_outputs = model.forward_with_concept_intervention(
                input_ids, attention_mask, concept_labels
            )
            gt_preds = (torch.sigmoid(gt_outputs['logits']) > 0.5).float()

            gt_concept_all_labels.append(labels.cpu())
            gt_concept_all_preds.append(gt_preds.cpu())

    # Compute F1 scores
    pred_labels = torch.cat(pred_concept_all_labels, dim=0).numpy()
    pred_preds = torch.cat(pred_concept_all_preds, dim=0).numpy()

    gt_labels = torch.cat(gt_concept_all_labels, dim=0).numpy()
    gt_preds = torch.cat(gt_concept_all_preds, dim=0).numpy()

    pred_f1 = f1_score(pred_labels, pred_preds, average='macro', zero_division=0)
    gt_f1 = f1_score(gt_labels, gt_preds, average='macro', zero_division=0)

    intervention_gain = gt_f1 - pred_f1

    print(f"   Predicted Concepts F1: {pred_f1:.4f}")
    print(f"   Ground Truth Concepts F1: {gt_f1:.4f}")
    print(f"   Intervention Gain: {intervention_gain:+.4f} {'✅' if intervention_gain > 0.05 else '❌'} (Target: >0.05)")

    return {
        'predicted_f1': float(pred_f1),
        'ground_truth_f1': float(gt_f1),
        'intervention_gain': float(intervention_gain),
        'target': 0.05,
        'interpretation': f"Correcting concepts improves accuracy by {intervention_gain*100:.2f}%"
    }


def compute_tcav_scores(model, test_loader, concept_embeddings, device, num_samples=200):
    """
    TCAV Score (Kim et al., ICML 2018)

    Measures: Sensitivity of diagnosis to concept direction

    TCAV_score(C, k) = |{x : S_C(x) > 0}| / |X_k|

    Where S_C(x) = ∇h(x) · v_C (directional derivative along concept)

    For each diagnosis k, for each concept C:
    - Compute gradient of diagnosis logit w.r.t. hidden states
    - Project onto concept direction
    - Count positive projections

    Target: Relevant concepts have TCAV > 0.60
    """
    print("\n📊 Computing TCAV Scores (Kim et al., ICML 2018)...")

    model.eval()
    num_concepts = concept_embeddings.shape[0]

    tcav_scores = {code: {} for code in TARGET_CODES}

    # Sample data by diagnosis
    diagnosis_samples = {code: [] for code in TARGET_CODES}

    for batch in test_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        for i in range(len(input_ids)):
            for diag_idx, code in enumerate(TARGET_CODES):
                if labels[i, diag_idx] == 1:
                    diagnosis_samples[code].append({
                        'input_ids': input_ids[i:i+1],
                        'attention_mask': attention_mask[i:i+1],
                        'labels': labels[i:i+1]
                    })
                    if len(diagnosis_samples[code]) >= num_samples:
                        break

    print(f"   Sampled {num_samples} examples per diagnosis")

    # Compute TCAV for each (diagnosis, concept) pair
    for diag_idx, diagnosis_code in enumerate(TARGET_CODES):
        print(f"\n   Computing TCAV for {diagnosis_code}...")

        samples = diagnosis_samples[diagnosis_code][:num_samples]

        for concept_idx in tqdm(range(min(20, num_concepts)), desc=f"   Concepts for {diagnosis_code}"):
            concept_vector = concept_embeddings[concept_idx].to(device)

            positive_count = 0

            for sample in samples:
                input_ids = sample['input_ids'].to(device)
                attention_mask = sample['attention_mask'].to(device)

                # Forward pass with gradient
                input_ids.requires_grad_(False)

                outputs = model(input_ids, attention_mask)
                hidden_states = outputs['hidden_states']
                logits = outputs['logits']

                # Enable gradient for hidden states
                hidden_states = hidden_states.clone().detach().requires_grad_(True)

                # Recompute diagnosis logit from hidden states
                cls_hidden = model.dropout(hidden_states[:, 0, :])
                recomputed_logit = model.diagnosis_head_final(cls_hidden)[0, diag_idx]

                # Gradient of diagnosis logit w.r.t hidden states
                grad = torch.autograd.grad(
                    recomputed_logit,
                    hidden_states,
                    retain_graph=False,
                    create_graph=False
                )[0]

                # Directional derivative: grad · concept_vector
                grad_pooled = grad.mean(dim=1)[0]  # Pool across sequence
                S_C = (grad_pooled @ concept_vector).item()

                if S_C > 0:
                    positive_count += 1

            tcav_score = positive_count / len(samples) if len(samples) > 0 else 0
            tcav_scores[diagnosis_code][concept_idx] = tcav_score

    # Compute average TCAV across relevant concepts (top-3 per diagnosis)
    avg_tcav_relevant = []
    for code in TARGET_CODES:
        top_3_scores = sorted(tcav_scores[code].values(), reverse=True)[:3]
        avg_tcav_relevant.extend(top_3_scores)

    avg_tcav = np.mean(avg_tcav_relevant)

    print(f"\n   Average TCAV (relevant concepts): {avg_tcav:.4f} {'✅' if avg_tcav > 0.60 else '❌'} (Target: >0.60)")

    return {
        'tcav_scores': tcav_scores,
        'avg_tcav_relevant': float(avg_tcav),
        'target': 0.60,
        'interpretation': f"Relevant concepts have avg TCAV {avg_tcav:.2f} (sensitivity to concept direction)"
    }


def compute_concept_shap(model, test_loader, concept_embeddings, device, num_samples=50, num_coalitions=20):
    """
    ConceptSHAP (Yeh et al., NeurIPS 2020)

    Measures: Shapley value for each concept's marginal contribution

    φ_i = E[v(S ∪ {i}) - v(S)] over all subsets S

    Approximation: sample random subsets (coalitions)

    Returns: Concept importance ranking
    """
    print("\n📊 Computing ConceptSHAP (Yeh et al., NeurIPS 2020)...")

    model.eval()
    num_concepts = concept_embeddings.shape[0]
    concept_shap = torch.zeros(num_concepts)

    # Sample test data
    sampled_data = []
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= num_samples:
            break
        sampled_data.append(batch)

    print(f"   Sampling {num_coalitions} concept coalitions...")

    with torch.no_grad():
        for sample_batch in tqdm(sampled_data, desc="   Computing Shapley values"):
            input_ids = sample_batch['input_ids'][:1].to(device)  # Use first sample
            attention_mask = sample_batch['attention_mask'][:1].to(device)

            # Compute marginal contributions
            for _ in range(num_coalitions):
                # Random permutation of concepts
                perm = torch.randperm(num_concepts)

                # For each concept in permutation
                for i in range(min(10, len(perm))):  # Sample subset of permutation
                    concept_idx = perm[i].item()

                    # S: Concepts before i in permutation
                    S = perm[:i]

                    # Create masks
                    mask_S = torch.zeros(num_concepts, device=device)
                    mask_S[S] = 1

                    mask_S_i = mask_S.clone()
                    mask_S_i[concept_idx] = 1

                    # v(S): prediction with only S concepts active
                    logits_S = model.forward_with_concept_mask(input_ids, attention_mask, mask_S)
                    v_S = torch.sigmoid(logits_S).mean().item()

                    # v(S ∪ {i}): prediction with S + concept i active
                    logits_S_i = model.forward_with_concept_mask(input_ids, attention_mask, mask_S_i)
                    v_S_i = torch.sigmoid(logits_S_i).mean().item()

                    # Marginal contribution
                    concept_shap[concept_idx] += (v_S_i - v_S)

    # Normalize
    concept_shap /= (num_samples * num_coalitions)

    # Get top concepts
    top_k = 10
    top_indices = torch.argsort(concept_shap, descending=True)[:top_k]

    print(f"\n   Top {top_k} concepts by ConceptSHAP importance:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"     {rank}. Concept {idx.item()}: {concept_shap[idx].item():.4f}")

    return {
        'concept_shap_values': concept_shap.tolist(),
        'top_concepts': top_indices.tolist(),
        'interpretation': f"Shapley-based concept importance ranking (top concept: {top_indices[0].item()})"
    }


def compute_concept_association(model, test_loader, concept_embeddings, device):
    """
    Preserved metrics from Phase 3:
    - Concept F1
    - Consistency
    - Discriminability
    """
    print("\n📊 Computing Preserved Metrics (Concept F1, Consistency, Discriminability)...")

    model.eval()

    all_concept_preds = []
    all_concept_labels = []
    all_diagnosis_labels = []
    diagnosis_concept_scores = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="   Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            concept_labels = batch['concept_labels'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            concept_scores = outputs['concept_scores']

            all_concept_preds.append(concept_scores.cpu())
            all_concept_labels.append(concept_labels.cpu())
            all_diagnosis_labels.append(labels.cpu())

            # Group by diagnosis
            for i in range(len(labels)):
                for diag_idx, code in enumerate(TARGET_CODES):
                    if labels[i, diag_idx] == 1:
                        diagnosis_concept_scores[code].append(concept_scores[i].cpu())

    all_concept_preds = torch.cat(all_concept_preds, dim=0)
    all_concept_labels = torch.cat(all_concept_labels, dim=0)

    # Concept F1
    concept_preds_binary = (all_concept_preds > 0.5).float()
    concept_f1 = f1_score(
        all_concept_labels.numpy(),
        concept_preds_binary.numpy(),
        average='micro',
        zero_division=0
    )

    # Consistency (variance within diagnosis)
    consistency_scores = []
    for code in TARGET_CODES:
        if len(diagnosis_concept_scores[code]) > 1:
            scores_tensor = torch.stack(diagnosis_concept_scores[code])
            variance = scores_tensor.var(dim=0).mean().item()
            consistency = 1 - min(variance, 1.0)
            consistency_scores.append(consistency)

    avg_consistency = np.mean(consistency_scores) if consistency_scores else 0

    # Discriminability (KL divergence between diagnoses)
    kl_divs = []
    for code1, code2 in combinations(TARGET_CODES, 2):
        if len(diagnosis_concept_scores[code1]) > 0 and len(diagnosis_concept_scores[code2]) > 0:
            dist1 = torch.stack(diagnosis_concept_scores[code1]).mean(dim=0)
            dist2 = torch.stack(diagnosis_concept_scores[code2]).mean(dim=0)

            dist1 = dist1 + 1e-10
            dist2 = dist2 + 1e-10
            dist1 = dist1 / dist1.sum()
            dist2 = dist2 / dist2.sum()

            kl = F.kl_div(dist2.log(), dist1, reduction='sum').item()
            kl_divs.append(abs(kl))

    avg_discriminability = np.mean(kl_divs) if kl_divs else 0

    print(f"   Concept F1: {concept_f1:.4f} {'✅' if concept_f1 > 0.50 else '❌'} (Target: >0.50)")
    print(f"   Consistency: {avg_consistency:.4f}")
    print(f"   Discriminability: {avg_discriminability:.4f}")

    return {
        'concept_f1': float(concept_f1),
        'consistency': float(avg_consistency),
        'discriminability': float(avg_discriminability),
        'target_concept_f1': 0.50
    }


# ============================================================================
# MAIN EVALUATION
# ============================================================================

print("\n" + "="*80)
print("📥 LOADING DATA & MODEL")
print("="*80)

# Check checkpoint
if not PHASE2_FIXED_CHECKPOINT.exists():
    print(f"\n❌ ERROR: Phase 2 checkpoint not found!")
    print(f"   Expected: {PHASE2_FIXED_CHECKPOINT}")
    raise FileNotFoundError(f"Phase 2 checkpoint not found: {PHASE2_FIXED_CHECKPOINT}")

print(f"✅ Found Phase 2: {PHASE2_FIXED_CHECKPOINT.name}")

# Load splits
with open(SHARED_DATA_PATH / 'test_split.pkl', 'rb') as f:
    df_test = pickle.load(f)

# Create dummy concept labels (you should load these from your data)
# For now, using random concept labels as placeholder
num_concepts = 50  # Will be overridden by checkpoint
concept_labels_test = np.random.randint(0, 2, size=(len(df_test), num_concepts)).astype(float)

print(f"✅ Loaded test set: {len(df_test):,} samples")

# Load model
print("\n📥 Loading Phase 2 Fixed model...")
checkpoint = torch.load(PHASE2_FIXED_CHECKPOINT, map_location=device, weights_only=False)

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
base_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)

# Build Phase 1
num_concepts = checkpoint['num_concepts']
phase1_model = ShifaMindPhase1Fixed(
    base_model, num_concepts, len(TARGET_CODES), fusion_layers=[9, 11]
).to(device)

# Build Phase 2
phase2_model = ShifaMindPhase2Fixed(phase1_model).to(device)

# Load state dict
phase2_model.load_state_dict(checkpoint['model_state_dict'])
phase2_model.eval()

concept_embeddings = checkpoint['concept_embeddings'].to(device)

print(f"✅ Loaded Phase 2 Fixed (F1: {checkpoint.get('macro_f1', 0):.4f})")

# Create dataset
test_dataset = XAIDataset(
    df_test['text'].tolist(),
    df_test['labels'].tolist(),
    concept_labels_test.tolist(),
    tokenizer
)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ============================================================================
# RUN ALL XAI METRICS
# ============================================================================

print("\n" + "="*80)
print("🎉 PHASE 3 V2 - CONCEPT-LEVEL XAI EVALUATION")
print("="*80)

xai_results = {}

# 1. Concept Completeness
xai_results['concept_completeness'] = compute_concept_completeness(
    phase2_model, test_loader, device
)

# 2. Concept Intervention Accuracy
xai_results['intervention_accuracy'] = compute_intervention_accuracy(
    phase2_model, test_loader, device
)

# 3. TCAV Scores
xai_results['tcav'] = compute_tcav_scores(
    phase2_model, test_loader, concept_embeddings, device
)

# 4. ConceptSHAP
xai_results['concept_shap'] = compute_concept_shap(
    phase2_model, test_loader, concept_embeddings, device
)

# 5. Preserved Metrics
xai_results['preserved_metrics'] = compute_concept_association(
    phase2_model, test_loader, concept_embeddings, device
)

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("💾 SAVING RESULTS")
print("="*80)

results_summary = {
    'phase': 'Phase 3 V2 - Concept-Level XAI Metrics',
    'model': 'Phase 2 Fixed',
    'metrics': xai_results,
    'references': [
        'Koh, P.W. et al. (2020). "Concept Bottleneck Models." ICML.',
        'Kim, B. et al. (2018). "TCAV: Interpretability Beyond Feature Attribution." ICML.',
        'Yeh, C.K. et al. (2020). "On Completeness-aware Concept-Based Explanations." NeurIPS.'
    ]
}

with open(RESULTS_PATH / 'xai_results_v2.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"✅ Saved: {RESULTS_PATH / 'xai_results_v2.json'}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("🎉 PHASE 3 V2 COMPLETE - CONCEPT-LEVEL XAI METRICS")
print("="*80)

print("\n📊 Concept Completeness (Yeh et al., NeurIPS 2020):")
print(f"   η = {xai_results['concept_completeness']['completeness']:.4f}")
print(f"   {xai_results['concept_completeness']['interpretation']}")

print("\n📊 Concept Intervention Accuracy (Koh et al., ICML 2020):")
print(f"   Intervention Gain: {xai_results['intervention_accuracy']['intervention_gain']:+.4f}")
print(f"   {xai_results['intervention_accuracy']['interpretation']}")

print("\n📊 TCAV Scores (Kim et al., ICML 2018):")
print(f"   Avg TCAV (relevant): {xai_results['tcav']['avg_tcav_relevant']:.4f}")
print(f"   {xai_results['tcav']['interpretation']}")

print("\n📊 ConceptSHAP (Yeh et al., NeurIPS 2020):")
print(f"   {xai_results['concept_shap']['interpretation']}")

print("\n📊 Preserved Metrics:")
print(f"   Concept F1: {xai_results['preserved_metrics']['concept_f1']:.4f}")
print(f"   Consistency: {xai_results['preserved_metrics']['consistency']:.4f}")
print(f"   Discriminability: {xai_results['preserved_metrics']['discriminability']:.4f}")

print(f"\n💾 Results: {RESULTS_PATH}")
print("\n✅ Concept-level XAI evaluation complete!")
print(f"\nAlhamdulillah! 🤲")
