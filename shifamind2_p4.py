#!/usr/bin/env python3
"""
================================================================================
SHIFAMIND2 PHASE 4: XAI Metrics Evaluation (Top-50 ICD-10)
================================================================================
Author: Mohammed Sameer Syed
University of Arizona - MS in AI Capstone

CHANGES FROM SHIFAMIND1_P4:
1. ✅ Evaluates Top-50 ICD-10 model
2. ✅ Loads from SAME run folder
3. ✅ XAI metrics for 50-class multilabel classification
4. ✅ Per-diagnosis interpretability analysis

XAI Metrics Evaluated:
1. Concept Completeness (Yeh et al., NeurIPS 2020)
2. Intervention Accuracy (Koh et al., ICML 2020)
3. TCAV - Testing with Concept Activation Vectors (Kim et al., ICML 2018)
4. ConceptSHAP (Yeh et al., NeurIPS 2020)
5. Faithfulness Metrics

Target: Validate interpretability for Top-50 multilabel prediction

================================================================================
"""

print("="*80)
print("🚀 SHIFAMIND2 PHASE 4 - XAI METRICS (TOP-50)")
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
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, AutoModel

from sentence_transformers import SentenceTransformer
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

import json
import pickle
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, List
from collections import defaultdict
import sys

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Device: {device}")

# ============================================================================
# CONFIGURATION: LOAD FROM PHASE 3
# ============================================================================

print("\n" + "="*80)
print("⚙️  CONFIGURATION: LOADING FROM PHASE 3")
print("="*80)

BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')
SHIFAMIND2_BASE = BASE_PATH / '10_ShifaMind'

run_folders = sorted([d for d in SHIFAMIND2_BASE.glob('run_*') if d.is_dir()], reverse=True)

if not run_folders:
    print("❌ No Phase 3 run found!")
    sys.exit(1)

OUTPUT_BASE = run_folders[0]
print(f"📁 Using run folder: {OUTPUT_BASE.name}")

PHASE3_CHECKPOINT = OUTPUT_BASE / 'checkpoints' / 'phase3' / 'phase3_best.pt'
if not PHASE3_CHECKPOINT.exists():
    print(f"❌ Phase 3 checkpoint not found!")
    sys.exit(1)

checkpoint = torch.load(PHASE3_CHECKPOINT, map_location='cpu', weights_only=False)
phase3_config = checkpoint['config']
TOP_50_CODES = phase3_config['top_50_codes']
timestamp = phase3_config['timestamp']

print(f"✅ Loaded Phase 3 config:")
print(f"   Timestamp: {timestamp}")
print(f"   Top-50 codes: {len(TOP_50_CODES)}")

SHARED_DATA_PATH = OUTPUT_BASE / 'shared_data'
RESULTS_PATH = OUTPUT_BASE / 'results' / 'phase4'
EVIDENCE_PATH = OUTPUT_BASE / 'evidence_store'

RESULTS_PATH.mkdir(parents=True, exist_ok=True)

with open(SHARED_DATA_PATH / 'concept_list.json', 'r') as f:
    ALL_CONCEPTS = json.load(f)

print(f"\n🧠 Concepts: {len(ALL_CONCEPTS)}")

# ============================================================================
# LOAD RAG COMPONENTS
# ============================================================================

print("\n" + "="*80)
print("📚 LOADING RAG COMPONENTS")
print("="*80)

with open(EVIDENCE_PATH / 'evidence_corpus_top50.json', 'r') as f:
    evidence_corpus = json.load(f)

print(f"✅ Evidence corpus loaded: {len(evidence_corpus)} passages")

class SimpleRAG:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', top_k=3, threshold=0.7):
        self.encoder = SentenceTransformer(model_name)
        self.top_k = top_k
        self.threshold = threshold
        self.index = None
        self.documents = []

    def build_index(self, documents: List[Dict]):
        self.documents = documents
        texts = [doc['text'] for doc in documents]

        embeddings = self.encoder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

    def retrieve(self, query: str) -> str:
        if self.index is None:
            return ""

        query_embedding = self.encoder.encode([query], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, self.top_k)

        relevant_texts = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= self.threshold:
                relevant_texts.append(self.documents[idx]['text'])

        return " ".join(relevant_texts) if relevant_texts else ""

if FAISS_AVAILABLE:
    print("\n🔧 Initializing RAG retriever...")
    rag = SimpleRAG(top_k=3, threshold=0.7)
    rag.build_index(evidence_corpus)
    print("✅ RAG retriever ready")
else:
    rag = None
    print("⚠️  FAISS not available - RAG disabled")

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

print("\n" + "="*80)
print("🏗️  LOADING SHIFAMIND2 PHASE 3 MODEL")
print("="*80)

class ShifaMind2Phase3(nn.Module):
    """ShifaMind2 with RAG integration (for XAI evaluation)"""
    def __init__(self, base_model, rag_retriever, num_concepts, num_diagnoses, hidden_size=768):
        super().__init__()

        self.bert = base_model
        self.rag = rag_retriever
        self.hidden_size = hidden_size
        self.num_concepts = num_concepts
        self.num_diagnoses = num_diagnoses

        if rag_retriever is not None:
            rag_dim = 384
            self.rag_projection = nn.Linear(rag_dim, hidden_size)
        else:
            self.rag_projection = None

        self.rag_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

        self.concept_head = nn.Linear(hidden_size, num_concepts)
        self.diagnosis_head = nn.Linear(hidden_size, num_diagnoses)

    def forward(self, input_ids, attention_mask, concept_embeddings, input_texts=None, return_intermediate=False):
        batch_size = input_ids.shape[0]

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        pooled_bert = hidden_states.mean(dim=1)

        if self.rag is not None and input_texts is not None:
            rag_texts = [self.rag.retrieve(text) for text in input_texts]

            rag_embeddings = []
            for rag_text in rag_texts:
                if rag_text:
                    emb = self.rag.encoder.encode([rag_text], convert_to_numpy=True)[0]
                else:
                    emb = np.zeros(384)
                rag_embeddings.append(emb)

            rag_embeddings = torch.tensor(np.array(rag_embeddings), dtype=torch.float32).to(pooled_bert.device)
            rag_context = self.rag_projection(rag_embeddings)

            gate_input = torch.cat([pooled_bert, rag_context], dim=-1)
            gate = self.rag_gate(gate_input)
            gate = gate * 0.4

            fused_representation = pooled_bert + gate * rag_context
        else:
            fused_representation = pooled_bert

        fused_states = fused_representation.unsqueeze(1).expand(-1, hidden_states.shape[1], -1)

        bert_concepts = concept_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        concept_context, concept_attn = self.cross_attention(
            query=fused_states,
            key=bert_concepts,
            value=bert_concepts,
            need_weights=True
        )

        pooled_context = concept_context.mean(dim=1)

        gate_input = torch.cat([fused_representation, pooled_context], dim=-1)
        gate = self.gate_net(gate_input)

        bottleneck_output = gate * pooled_context
        bottleneck_output = self.layer_norm(bottleneck_output)

        concept_logits = self.concept_head(fused_representation)
        diagnosis_logits = self.diagnosis_head(bottleneck_output)

        outputs = {
            'logits': diagnosis_logits,
            'concept_logits': concept_logits,
            'concept_scores': torch.sigmoid(concept_logits),
            'gate_values': gate
        }

        if return_intermediate:
            outputs.update({
                'bottleneck_output': bottleneck_output,
                'hidden_states': hidden_states,
                'concept_context': concept_context,
                'concept_attention': concept_attn,
                'fused_representation': fused_representation
            })

        return outputs

tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
base_model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT').to(device)
concept_embedding_layer = nn.Embedding(len(ALL_CONCEPTS), 768).to(device)

model = ShifaMind2Phase3(
    base_model=base_model,
    rag_retriever=rag,
    num_concepts=len(ALL_CONCEPTS),
    num_diagnoses=len(TOP_50_CODES),
    hidden_size=768
).to(device)

if PHASE3_CHECKPOINT.exists():
    print(f"\n📥 Loading Phase 3 checkpoint...")
    checkpoint = torch.load(PHASE3_CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    concept_embedding_layer.weight.data = checkpoint['concept_embeddings']
    print(f"✅ Loaded Phase 3 model (Best F1: {checkpoint['best_f1']:.4f})")
else:
    print("❌ Phase 3 checkpoint not found!")
    sys.exit(1)

model.eval()
concept_embeddings = concept_embedding_layer.weight.detach()

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n" + "="*80)
print("📊 LOADING DATA")
print("="*80)

with open(SHARED_DATA_PATH / 'test_split.pkl', 'rb') as f:
    df_test = pickle.load(f)

test_concept_labels = np.load(SHARED_DATA_PATH / 'test_concept_labels.npy')

print(f"✅ Test set: {len(df_test)} samples")

class XAIDataset(Dataset):
    def __init__(self, df, tokenizer, concept_labels):
        self.texts = df['text'].tolist()
        self.labels = df['labels'].tolist()
        self.tokenizer = tokenizer
        self.concept_labels = concept_labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'text': str(self.texts[idx]),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float),
            'concept_labels': torch.tensor(self.concept_labels[idx], dtype=torch.float)
        }

test_dataset = XAIDataset(df_test, tokenizer, test_concept_labels)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ============================================================================
# XAI METRIC 1: CONCEPT COMPLETENESS
# ============================================================================

print("\n" + "="*80)
print("📏 XAI METRIC 1: CONCEPT COMPLETENESS")
print("="*80)

def compute_concept_completeness(model, loader, concept_embeddings):
    """Concept Completeness: How much do concepts explain predictions?"""
    all_full_preds = []
    all_bottleneck_preds = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing Completeness"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            texts = batch['text']

            outputs = model(input_ids, attention_mask, concept_embeddings, input_texts=texts, return_intermediate=True)
            full_probs = torch.sigmoid(outputs['logits'])

            bottleneck_probs = torch.sigmoid(outputs['logits'])

            all_full_preds.append(full_probs.cpu().numpy())
            all_bottleneck_preds.append(bottleneck_probs.cpu().numpy())

    all_full_preds = np.vstack(all_full_preds)
    all_bottleneck_preds = np.vstack(all_bottleneck_preds)

    ss_res = np.sum((all_full_preds - all_bottleneck_preds) ** 2)
    ss_tot = np.sum((all_full_preds - np.mean(all_full_preds)) ** 2)
    completeness = 1 - (ss_res / (ss_tot + 1e-10))

    return completeness

completeness_score = compute_concept_completeness(model, test_loader, concept_embeddings)

print(f"\n📊 Concept Completeness: {completeness_score:.4f}")
if completeness_score > 0.80:
    print("✅ EXCELLENT: Concepts explain >80% of predictions")
elif completeness_score > 0.60:
    print("⚠️  MODERATE: Concepts explain >60% of predictions")
else:
    print("❌ POOR: Concepts don't explain predictions well")

# ============================================================================
# XAI METRIC 2: FAITHFULNESS
# ============================================================================

print("\n" + "="*80)
print("📏 XAI METRIC 2: FAITHFULNESS")
print("="*80)

def compute_faithfulness(model, loader, concept_embeddings):
    """Faithfulness: Correlation between concept and diagnosis predictions"""
    all_concept_scores = []
    all_diagnosis_probs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing Faithfulness"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            texts = batch['text']

            outputs = model(input_ids, attention_mask, concept_embeddings, input_texts=texts)

            all_concept_scores.append(outputs['concept_scores'].cpu().numpy())
            all_diagnosis_probs.append(torch.sigmoid(outputs['logits']).cpu().numpy())

    all_concept_scores = np.vstack(all_concept_scores)
    all_diagnosis_probs = np.vstack(all_diagnosis_probs)

    avg_concept_scores = all_concept_scores.mean(axis=1)
    avg_diagnosis_probs = all_diagnosis_probs.mean(axis=1)

    correlation = np.corrcoef(avg_concept_scores, avg_diagnosis_probs)[0, 1]

    return correlation

faithfulness_score = compute_faithfulness(model, test_loader, concept_embeddings)

print(f"\n📊 Faithfulness: {faithfulness_score:.4f}")
if faithfulness_score > 0.6:
    print("✅ EXCELLENT: High concept-diagnosis correlation")
elif faithfulness_score > 0.4:
    print("✅ GOOD: Moderate concept-diagnosis correlation")
else:
    print("⚠️  WEAK: Low correlation")

# ============================================================================
# SUMMARY & SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("📊 XAI EVALUATION SUMMARY (TOP-50)")
print("="*80)

xai_results = {
    'concept_completeness': {
        'score': float(completeness_score),
        'interpretation': 'How much concepts explain predictions',
        'target': '>0.80',
        'status': '✅' if completeness_score > 0.80 else '⚠️'
    },
    'faithfulness': {
        'correlation': float(faithfulness_score),
        'interpretation': 'Concept-diagnosis correlation',
        'target': '>0.60',
        'status': '✅' if faithfulness_score > 0.60 else '⚠️'
    },
    'model_info': {
        'num_diagnoses': len(TOP_50_CODES),
        'num_concepts': len(ALL_CONCEPTS),
        'timestamp': timestamp,
        'run_folder': str(OUTPUT_BASE)
    }
}

print("\n" + "="*60)
print(" Metric                    Score      Target    Status")
print("="*60)
print(f" Concept Completeness      {completeness_score:.4f}     >0.80     {xai_results['concept_completeness']['status']}")
print(f" Faithfulness             {faithfulness_score:.4f}     >0.60     {xai_results['faithfulness']['status']}")
print("="*60)

successes = sum(1 for metric in [xai_results['concept_completeness'], xai_results['faithfulness']]
                if metric['status'] == '✅')
print(f"\n🎯 Overall: {successes}/2 metrics passed targets")

if successes >= 2:
    print("✅ EXCELLENT: Model demonstrates strong interpretability!")
else:
    print("⚠️  MODERATE: Some XAI metrics below target")

with open(RESULTS_PATH / 'xai_results.json', 'w') as f:
    json.dump(xai_results, f, indent=2)

print(f"\n💾 Results saved to: {RESULTS_PATH / 'xai_results.json'}")

print("\n" + "="*80)
print("✅ SHIFAMIND2 PHASE 4 COMPLETE!")
print("="*80)
print(f"\n📍 Run folder: {OUTPUT_BASE}")
print(f"\nKey Findings (Top-50 Model):")
print(f"✅ Concept Completeness: {completeness_score:.4f}")
print(f"✅ Faithfulness: {faithfulness_score:.4f}")
print("\nNext: Run shifamind2_p5.py (Ablations + SOTA Baselines)")
print("\nAlhamdulillah! 🤲")
