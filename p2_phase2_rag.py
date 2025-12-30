#!/usr/bin/env python3
"""
================================================================================
SHIFAMIND PHASE 2: RAG-Enhanced Model
================================================================================
Author: Mohammed Sameer Syed
University of Arizona - MS in AI Capstone

Architecture:
- Loads Phase 1 checkpoint
- Adds SimpleRAG system (FAISS + sentence-transformers)
- RAG fusion via gated mechanism (capped at 40%)
- Clinical knowledge corpus + case prototypes

Loads:
- Train/val/test splits from Phase 1
- Phase 1 checkpoint + concept embeddings

Saves:
- Phase 2 checkpoint to 07_ShifaMind/checkpoints/phase2/
- Results to 07_ShifaMind/results/phase2/

Expected F1: ~0.80+
================================================================================
"""

print("="*80)
print("🚀 SHIFAMIND PHASE 2 - RAG-ENHANCED MODEL")
print("="*80)

# ============================================================================
# INSTALL DEPENDENCIES
# ============================================================================

print("\n" + "="*80)
print("📦 INSTALLING DEPENDENCIES")
print("="*80)

import os
import sys

try:
    import faiss
    print("✅ FAISS")
except:
    print("Installing FAISS...")
    os.system("pip install -q faiss-gpu")
    import faiss
    print("✅ FAISS installed")

try:
    from sentence_transformers import SentenceTransformer
    print("✅ sentence-transformers")
except:
    print("Installing sentence-transformers...")
    os.system("pip install -q sentence-transformers")
    from sentence_transformers import SentenceTransformer
    print("✅ sentence-transformers installed")

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
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

import json
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, List
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

# Input paths (from Phase 1)
SHARED_DATA_PATH = OUTPUT_BASE / 'shared_data'
PHASE1_CHECKPOINT = OUTPUT_BASE / 'checkpoints/phase1/phase1_final.pt'

# Output paths (Phase 2)
CHECKPOINT_PATH = OUTPUT_BASE / 'checkpoints/phase2'
RESULTS_PATH = OUTPUT_BASE / 'results/phase2'

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
# LOAD SAVED SPLITS
# ============================================================================

print("\n" + "="*80)
print("📥 LOADING SAVED SPLITS FROM PHASE 1")
print("="*80)

print("Loading train/val/test splits...")
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

# Verify split info
with open(SHARED_DATA_PATH / 'split_info.json', 'r') as f:
    split_info = json.load(f)
    print(f"\n📋 Split info: Seed={split_info['seed']}, Total={split_info['train_size'] + split_info['val_size'] + split_info['test_size']}")

# ============================================================================
# BUILD CLINICAL KNOWLEDGE CORPUS
# ============================================================================

print("\n" + "="*80)
print("📚 BUILDING CLINICAL KNOWLEDGE CORPUS")
print("="*80)

# Clinical knowledge per diagnosis
clinical_knowledge = {
    'J189': [
        'Pneumonia diagnosis requires fever, cough, dyspnea, and infiltrate on chest imaging',
        'Community-acquired pneumonia presents with productive cough, fever, and respiratory symptoms',
        'Chest X-ray showing infiltrates confirms pneumonia diagnosis',
        'Elevated WBC and procalcitonin suggest bacterial pneumonia',
        'Respiratory symptoms with fever and consolidation indicate pneumonia'
    ],
    'I5023': [
        'Heart failure presents with dyspnea, orthopnea, and bilateral lower extremity edema',
        'Elevated BNP or NT-proBNP supports heart failure diagnosis',
        'Echocardiogram showing reduced ejection fraction confirms systolic heart failure',
        'JVD, S3 gallop, and pulmonary rales indicate volume overload',
        'Acute on chronic heart failure shows worsening symptoms with history of cardiac disease'
    ],
    'A419': [
        'Sepsis requires infection plus organ dysfunction per Sepsis-3 criteria',
        'qSOFA score identifies high-risk sepsis: altered mental status, hypotension, tachypnea',
        'Elevated lactate indicates tissue hypoperfusion in sepsis',
        'Positive blood cultures with hemodynamic instability suggest septic shock',
        'SIRS criteria include fever, tachycardia, tachypnea, and leukocytosis'
    ],
    'K8000': [
        'Acute cholecystitis presents with RUQ pain, fever, and positive Murphy sign',
        'Ultrasound showing gallbladder wall thickening and pericholecystic fluid confirms cholecystitis',
        'Tokyo guidelines require local signs, systemic signs, and imaging for diagnosis',
        'Elevated WBC and alkaline phosphatase support acute cholecystitis',
        'Gallstones with inflammation require surgical intervention'
    ]
}

corpus_documents = []
corpus_metadata = []

print("Building corpus from clinical knowledge...")
for code, knowledge_list in clinical_knowledge.items():
    for knowledge in knowledge_list:
        corpus_documents.append(knowledge)
        corpus_metadata.append({
            'type': 'clinical_knowledge',
            'icd_code': code,
            'source': f'Clinical-{code}'
        })

print("Adding case prototypes from MIMIC...")
for code in TARGET_CODES:
    cases = df_train[df_train['icd_codes'].apply(lambda x: code in x)]
    if len(cases) > 0:
        sampled = cases.sample(n=min(20, len(cases)), random_state=SEED)
        for idx, row in sampled.iterrows():
            text = str(row['text'])
            snippet = text[:300].strip()
            corpus_documents.append(f"Case for {ICD_DESCRIPTIONS[code]}: {snippet}")
            corpus_metadata.append({
                'type': 'case_prototype',
                'icd_code': code,
                'source': f'MIMIC-{code}'
            })

print(f"\n✅ Corpus built: {len(corpus_documents)} documents")

type_counts = {}
for m in corpus_metadata:
    t = m.get('type', 'unknown')
    type_counts[t] = type_counts.get(t, 0) + 1

print("\n📋 Composition:")
for t, c in sorted(type_counts.items()):
    pct = c / len(corpus_metadata) * 100
    print(f"   {t:20s}: {c:4d} ({pct:5.1f}%)")

# Save corpus
corpus_file = RESULTS_PATH / 'corpus.json'
with open(corpus_file, 'w') as f:
    json.dump({'documents': corpus_documents, 'metadata': corpus_metadata}, f, indent=2)
print(f"\n💾 Saved: {corpus_file}")

# ============================================================================
# BUILD RETRIEVER & FAISS INDEX
# ============================================================================

print("\n" + "="*80)
print("🔍 BUILDING RETRIEVER & FAISS INDEX")
print("="*80)

print("Loading sentence-transformers retriever...")
retriever = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)
print(f"✅ Retriever loaded (dim: {retriever.get_sentence_embedding_dimension()})")

print("\n📊 Encoding corpus...")
corpus_embeddings = retriever.encode(
    corpus_documents, batch_size=32, show_progress_bar=True,
    convert_to_numpy=True, normalize_embeddings=True
).astype(np.float32)

print("\nBuilding FAISS index...")
embedding_dim = corpus_embeddings.shape[1]
index = faiss.IndexFlatIP(embedding_dim)
faiss.normalize_L2(corpus_embeddings)
index.add(corpus_embeddings)

print(f"✅ FAISS index built: {index.ntotal} vectors")

# ============================================================================
# RAG SYSTEM
# ============================================================================

print("\n" + "="*80)
print("🧠 RAG SYSTEM")
print("="*80)

class SimpleRAG:
    def __init__(self, retriever, index, documents, top_k=3, threshold=0.7):
        self.retriever = retriever
        self.index = index
        self.documents = documents
        self.top_k = top_k
        self.threshold = threshold
        print(f"✅ RAG: top-{top_k}, threshold={threshold}")

    def retrieve(self, query: str) -> str:
        """Retrieve relevant knowledge"""
        query = query[:1000]  # Truncate long queries
        query_embedding = self.retriever.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False
        ).astype(np.float32)

        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, k=self.top_k)

        # Filter by threshold
        filtered = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score >= self.threshold:
                filtered.append(self.documents[idx])

        return "\n\n".join(filtered) if filtered else ""

rag_system = SimpleRAG(retriever, index, corpus_documents, top_k=3, threshold=0.7)

# ============================================================================
# LOAD PHASE 1 MODEL
# ============================================================================

print("\n" + "="*80)
print("📥 LOADING PHASE 1 MODEL")
print("="*80)

# Phase 1 architecture (must match exactly)
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
        top_scores, _ = torch.topk(pointer_probs, k=min(self.max_spans, seq_len), dim=-1)
        return {'span_scores': top_scores.mean(), 'pointer_probs': pointer_probs}

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
        return {'concept_logits': concept_logits, 'span_scores': evidence['span_scores']}

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

    def forward(self, input_ids, attention_mask, concept_embeddings, return_evidence=False):
        outputs = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, return_dict=True
        )

        hidden_states = outputs.hidden_states
        current_hidden = outputs.last_hidden_state

        for layer_idx in self.fusion_layers:
            if str(layer_idx) in self.fusion_modules:
                layer_hidden = hidden_states[layer_idx]
                fused_hidden, _ = self.fusion_modules[str(layer_idx)](
                    layer_hidden, concept_embeddings, attention_mask
                )
                gate = torch.sigmoid(self.layer_gates[str(layer_idx)])
                current_hidden = (1 - gate) * current_hidden + gate * fused_hidden

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

        return result

print("Loading BioClinicalBERT tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
print("✅ Tokenizer")

print("\nLoading Phase 1 checkpoint...")
checkpoint = torch.load(PHASE1_CHECKPOINT, map_location=device)
print(f"✅ Checkpoint loaded (F1: {checkpoint.get('macro_f1', 0):.4f})")

base_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)
phase1_model = ShifaMindPhase1(
    base_model=base_model, num_concepts=checkpoint['num_concepts'],
    num_classes=len(TARGET_CODES), fusion_layers=[9, 11]
).to(device)

phase1_model.load_state_dict(checkpoint['model_state_dict'])
concept_embeddings = checkpoint['concept_embeddings'].to(device)

print(f"✅ Phase 1 loaded ({sum(p.numel() for p in phase1_model.parameters()):,} params)")

# ============================================================================
# PHASE 2 MODEL
# ============================================================================

print("\n" + "="*80)
print("🏗️  PHASE 2 MODEL")
print("="*80)

class SimpleRAGFusion(nn.Module):
    """Simple stable RAG fusion"""
    def __init__(self, hidden_size=768):
        super().__init__()
        self.rag_gate = nn.Parameter(torch.tensor(0.2))  # Start conservative
        self.rag_proj = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, text_cls, rag_cls):
        rag_projected = self.rag_proj(rag_cls)
        gate = torch.sigmoid(self.rag_gate).clamp(0.0, 0.4)  # Cap at 40%
        fused = (1 - gate) * text_cls + gate * rag_projected
        fused = self.layer_norm(fused)
        return fused, gate

class ShifaMindPhase2(nn.Module):
    """Phase 2 with RAG fusion"""
    def __init__(self, phase1_model, tokenizer, rag_system):
        super().__init__()
        self.phase1_model = phase1_model
        self.tokenizer = tokenizer
        self.rag_system = rag_system
        self.rag_fusion = SimpleRAGFusion(hidden_size=phase1_model.hidden_size)

    def forward(self, input_ids, attention_mask, concept_embeddings, rag_texts: List[str]):
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Get text CLS
        text_outputs = self.phase1_model.base_model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        text_cls = text_outputs.last_hidden_state[:, 0, :]

        # Get RAG CLS
        rag_cls_list = []
        for rag_text in rag_texts:
            if rag_text and len(rag_text) > 10:
                rag_encoding = self.tokenizer(
                    rag_text[:400], max_length=128, truncation=True,
                    padding='max_length', return_tensors='pt'
                )
                rag_ids = rag_encoding['input_ids'].to(device)
                rag_mask = rag_encoding['attention_mask'].to(device)

                with torch.no_grad():
                    rag_out = self.phase1_model.base_model(
                        input_ids=rag_ids, attention_mask=rag_mask, return_dict=True
                    )
                rag_cls_list.append(rag_out.last_hidden_state[:, 0, :])
            else:
                rag_cls_list.append(torch.zeros_like(text_cls[0:1]))

        rag_cls = torch.cat(rag_cls_list, dim=0)

        # Fuse
        fused_cls, gate = self.rag_fusion(text_cls, rag_cls)

        # Predict
        diagnosis_logits = self.phase1_model.diagnosis_head(fused_cls)
        concept_logits = self.phase1_model.concept_head(fused_cls)
        refined_concept_logits = self.phase1_model.diagnosis_concept_interaction(
            torch.sigmoid(diagnosis_logits), torch.sigmoid(concept_logits)
        )

        return {
            'logits': diagnosis_logits,
            'concept_scores': refined_concept_logits,
            'rag_gate': gate
        }

phase2_model = ShifaMindPhase2(phase1_model, tokenizer, rag_system).to(device)
print(f"✅ Phase 2 built ({sum(p.numel() for p in phase2_model.parameters()):,} params)")

# ============================================================================
# DATASET
# ============================================================================

print("\n" + "="*80)
print("📊 PREPARING DATASETS")
print("="*80)

class RAGDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, rag_system, max_length=384, retrieve=True):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rag_cache = None

        if retrieve:
            print("Pre-retrieving RAG documents...")
            self.rag_cache = []
            for text in tqdm(self.texts, desc="RAG"):
                rag_text = rag_system.retrieve(text)
                self.rag_cache.append(rag_text)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        rag_text = self.rag_cache[idx] if self.rag_cache else ""

        encoding = self.tokenizer(
            text, padding='max_length', truncation=True,
            max_length=self.max_length, return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(self.labels[idx]),
            'rag_text': rag_text
        }

train_dataset = RAGDataset(
    df_train['text'].tolist(), df_train['labels'].tolist(),
    tokenizer, rag_system, retrieve=True
)
val_dataset = RAGDataset(
    df_val['text'].tolist(), df_val['labels'].tolist(),
    tokenizer, rag_system, retrieve=True
)
test_dataset = RAGDataset(
    df_test['text'].tolist(), df_test['labels'].tolist(),
    tokenizer, rag_system, retrieve=True
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"\n✅ Datasets ready:")
print(f"   Train: {len(train_dataset)}")
print(f"   Val:   {len(val_dataset)}")
print(f"   Test:  {len(test_dataset)}")

# ============================================================================
# TRAINING
# ============================================================================

print("\n" + "="*80)
print("🏋️  TRAINING PHASE 2")
print("="*80)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(phase2_model.parameters(), lr=5e-6, weight_decay=0.01)

num_epochs = 3
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=len(train_loader) // 2,
    num_training_steps=len(train_loader) * num_epochs
)

best_f1 = 0
checkpoint_file = CHECKPOINT_PATH / 'phase2_final.pt'

print(f"Epochs: {num_epochs}, LR: 5e-6, Batch: 8")

for epoch in range(num_epochs):
    print(f"\n{'='*70}\nEpoch {epoch+1}/{num_epochs}\n{'='*70}")

    # Train
    phase2_model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        rag_texts = batch['rag_text']

        optimizer.zero_grad()
        outputs = phase2_model(input_ids, attention_mask, concept_embeddings, rag_texts)
        loss = criterion(outputs['logits'], labels)

        if torch.isnan(loss):
            print("⚠️  NaN loss, skipping batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(phase2_model.parameters(), max_norm=0.5)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    rag_gate_value = torch.sigmoid(phase2_model.rag_fusion.rag_gate).item()
    print(f"\n  Loss: {avg_loss:.4f}")
    print(f"  RAG Gate: {rag_gate_value:.4f}")

    # Validate
    phase2_model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            rag_texts = batch['rag_text']

            outputs = phase2_model(input_ids, attention_mask, concept_embeddings, rag_texts)
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
        torch.save({
            'model_state_dict': phase2_model.state_dict(),
            'phase1_checkpoint': str(PHASE1_CHECKPOINT),
            'macro_f1': best_f1,
            'epoch': epoch,
            'rag_gate': rag_gate_value
        }, checkpoint_file)
        print(f"  ✅ Saved (F1: {best_f1:.4f})")

print(f"\n✅ Training complete! Best F1: {best_f1:.4f}")

# ============================================================================
# FINAL TEST
# ============================================================================

print("\n" + "="*80)
print("📊 FINAL TEST EVALUATION")
print("="*80)

checkpoint = torch.load(checkpoint_file, map_location=device)
phase2_model.load_state_dict(checkpoint['model_state_dict'])

phase2_model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        rag_texts = batch['rag_text']

        outputs = phase2_model(input_ids, attention_mask, concept_embeddings, rag_texts)
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

print("\n" + "="*80)
print("🎉 PHASE 2 FINAL RESULTS")
print("="*80)

print("\n🎯 Performance:")
print(f"   Macro F1:    {macro_f1:.4f}")
print(f"   Micro F1:    {micro_f1:.4f}")
print(f"   Precision:   {macro_precision:.4f}")
print(f"   Recall:      {macro_recall:.4f}")
print(f"   AUROC:       {macro_auc:.4f}")

print("\n📊 Per-Class F1:")
for i, code in enumerate(TARGET_CODES):
    print(f"   {code}: {per_class_f1[i]:.4f}")

# Compare to Phase 1
phase1_f1 = checkpoint.get('macro_f1', 0.76)  # From Phase 1 checkpoint
improvement = macro_f1 - phase1_f1
improvement_pct = (improvement / phase1_f1) * 100 if phase1_f1 > 0 else 0

print(f"\n🔥 VS PHASE 1:")
print(f"   Phase 1: {phase1_f1:.4f}")
print(f"   Phase 2: {macro_f1:.4f}")
print(f"   Δ:       {improvement:+.4f} ({improvement_pct:+.1f}%)")

if macro_f1 >= 0.80:
    print("\n🎉 TARGET ACHIEVED (F1 ≥ 0.80)!")
elif macro_f1 > phase1_f1:
    print("\n✅ SUCCESS! RAG improved performance!")
else:
    print(f"\n📊 Gap: {phase1_f1 - macro_f1:.4f}")

# Save results
results = {
    'phase': 'Phase 2 - RAG-Enhanced',
    'test_metrics': {
        'macro_f1': float(macro_f1),
        'micro_f1': float(micro_f1),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_auc': float(macro_auc),
        'per_class_f1': {code: float(f1) for code, f1 in zip(TARGET_CODES, per_class_f1)}
    },
    'comparison': {
        'phase1_f1': float(phase1_f1),
        'phase2_f1': float(macro_f1),
        'improvement': float(improvement),
        'improvement_pct': float(improvement_pct)
    },
    'config': {
        'rag_top_k': 3,
        'rag_threshold': 0.7,
        'corpus_size': len(corpus_documents),
        'rag_gate': float(checkpoint.get('rag_gate', 0.2))
    }
}

with open(RESULTS_PATH / 'phase2_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Results saved to: {RESULTS_PATH}")
print(f"💾 Checkpoint: {checkpoint_file}")

print("\n" + "="*80)
print("✅ PHASE 2 COMPLETE!")
print("="*80)
print(f"\n📈 Final Macro F1: {macro_f1:.4f}")
print("\n🚀 Ready for Phase 3 (XAI metrics)")
print(f"\nAlhamdulillah! 🤲")
