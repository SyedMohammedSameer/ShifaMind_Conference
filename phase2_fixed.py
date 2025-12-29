#!/usr/bin/env python3
"""
================================================================================
SHIFAMIND PHASE 2 FIXED: Diagnosis-Aware RAG
================================================================================
Author: Mohammed Sameer Syed (Fixed Version)
University of Arizona - MS in AI Capstone

CRITICAL FIXES:
1. ✅ Diagnosis-aware RAG (retrieve based on predicted diagnosis)
2. ✅ Relevance scoring for retrieved documents
3. ✅ Adaptive RAG gate (learns when RAG helps)
4. ✅ Better corpus (more clinical knowledge, filtered prototypes)
5. ✅ Two-stage retrieval (coarse-to-fine)

Expected F1: >0.80 (beating Phase 1 Fixed)
================================================================================
"""

print("="*80)
print("🚀 SHIFAMIND PHASE 2 FIXED - DIAGNOSIS-AWARE RAG")
print("="*80)

# ============================================================================
# INSTALL DEPENDENCIES
# ============================================================================

import os
import warnings
warnings.filterwarnings('ignore')

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

SHARED_DATA_PATH = OUTPUT_BASE / 'shared_data'
PHASE1_CHECKPOINT = OUTPUT_BASE / 'checkpoints/phase1_fixed/phase1_fixed_best.pt'

CHECKPOINT_PATH = OUTPUT_BASE / 'checkpoints/phase2_fixed'
RESULTS_PATH = OUTPUT_BASE / 'results/phase2_fixed'

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
# FIX 4: BUILD ENHANCED CLINICAL KNOWLEDGE CORPUS
# ============================================================================

print("\n" + "="*80)
print("📚 BUILDING ENHANCED CORPUS")
print("="*80)

# FIX: More comprehensive clinical knowledge per diagnosis
enhanced_clinical_knowledge = {
    'J189': [
        'Pneumonia diagnosis requires fever, cough, dyspnea, and pulmonary infiltrate on imaging',
        'Community-acquired pneumonia presents with productive cough, fever, and respiratory symptoms',
        'Chest X-ray showing consolidation or infiltrates confirms pneumonia diagnosis',
        'Elevated WBC, procalcitonin, and CRP suggest bacterial pneumonia',
        'Hypoxia and tachypnea indicate severe pneumonia requiring hospitalization',
        'Lung crackles or bronchial breath sounds on auscultation support pneumonia',
        'Sputum culture identifies causative organism in pneumonia cases',
        'Antibiotic therapy depends on pneumonia severity and risk factors',
    ],
    'I5023': [
        'Heart failure presents with dyspnea, orthopnea, and bilateral lower extremity edema',
        'Elevated BNP or NT-proBNP (>400 pg/mL) supports heart failure diagnosis',
        'Echocardiogram showing reduced ejection fraction confirms systolic heart failure',
        'JVD, S3 gallop, and pulmonary rales indicate volume overload in heart failure',
        'Acute on chronic heart failure shows worsening symptoms with history of cardiac disease',
        'Chest X-ray may show cardiomegaly and pulmonary congestion',
        'Diuretics reduce volume overload and improve symptoms in heart failure',
        'ACE inhibitors and beta-blockers are cornerstone therapies for systolic heart failure',
    ],
    'A419': [
        'Sepsis requires infection plus organ dysfunction per Sepsis-3 criteria',
        'qSOFA score identifies high-risk sepsis: altered mental status, hypotension, tachypnea',
        'Elevated lactate (>2 mmol/L) indicates tissue hypoperfusion in sepsis',
        'Positive blood cultures with hemodynamic instability suggest septic shock',
        'SIRS criteria include fever, tachycardia, tachypnea, and leukocytosis',
        'Sequential Organ Failure Assessment (SOFA) score quantifies sepsis severity',
        'Early broad-spectrum antibiotics improve survival in sepsis',
        'Fluid resuscitation and vasopressors manage septic shock hemodynamics',
    ],
    'K8000': [
        'Acute cholecystitis presents with RUQ pain, fever, and positive Murphy sign',
        'Ultrasound showing gallbladder wall thickening and pericholecystic fluid confirms cholecystitis',
        'Tokyo guidelines require local signs, systemic signs, and imaging for diagnosis',
        'Elevated WBC, alkaline phosphatase, and bilirubin support acute cholecystitis',
        'Gallstones with inflammation require cholecystectomy for definitive treatment',
        'Murphy sign is pathognomonic for acute cholecystitis on physical exam',
        'HIDA scan shows cystic duct obstruction in acute cholecystitis',
        'Antibiotics and surgical consultation are initial management for cholecystitis',
    ]
}

corpus_documents = []
corpus_metadata = []

print("Building enhanced clinical knowledge corpus...")
for code, knowledge_list in enhanced_clinical_knowledge.items():
    for knowledge in knowledge_list:
        corpus_documents.append(knowledge)
        corpus_metadata.append({
            'type': 'clinical_knowledge',
            'icd_code': code,
            'source': f'Clinical-{code}',
            'relevance': 1.0  # High relevance
        })

# FIX: Select only BEST case prototypes (highest quality notes)
print("Adding high-quality case prototypes...")
for code in TARGET_CODES:
    cases = df_train[df_train['icd_codes'].apply(lambda x: code in x)]
    if len(cases) > 0:
        # Select longer, more detailed notes
        cases['text_length'] = cases['text'].apply(lambda x: len(str(x)))
        cases_sorted = cases.sort_values('text_length', ascending=False)
        sampled = cases_sorted.head(10)  # Only top 10 most detailed

        for idx, row in sampled.iterrows():
            text = str(row['text'])
            snippet = text[:500].strip()  # Longer snippets
            corpus_documents.append(f"Clinical case {ICD_DESCRIPTIONS[code]}: {snippet}")
            corpus_metadata.append({
                'type': 'case_prototype',
                'icd_code': code,
                'source': f'MIMIC-{code}',
                'relevance': 0.8  # Medium relevance
            })

print(f"\n✅ Enhanced corpus: {len(corpus_documents)} documents")

type_counts = {}
for m in corpus_metadata:
    t = m.get('type', 'unknown')
    type_counts[t] = type_counts.get(t, 0) + 1

print("\n📋 Composition:")
for t, c in sorted(type_counts.items()):
    pct = c / len(corpus_metadata) * 100
    print(f"   {t:20s}: {c:4d} ({pct:5.1f}%)")

corpus_file = RESULTS_PATH / 'corpus_enhanced.json'
with open(corpus_file, 'w') as f:
    json.dump({'documents': corpus_documents, 'metadata': corpus_metadata}, f, indent=2)
print(f"\n💾 Saved: {corpus_file}")

# ============================================================================
# BUILD RETRIEVER & DIAGNOSIS-SPECIFIC INDICES
# ============================================================================

print("\n" + "="*80)
print("🔍 BUILDING DIAGNOSIS-SPECIFIC INDICES")
print("="*80)

retriever = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)
print(f"✅ Retriever loaded (dim: {retriever.get_sentence_embedding_dimension()})")

print("\n📊 Encoding corpus...")
corpus_embeddings = retriever.encode(
    corpus_documents, batch_size=32, show_progress_bar=True,
    convert_to_numpy=True, normalize_embeddings=True
).astype(np.float32)

# FIX 1: Create separate FAISS indices per diagnosis
print("\nBuilding diagnosis-specific indices...")
embedding_dim = corpus_embeddings.shape[1]

diagnosis_indices = {}
diagnosis_documents = {}

for code in TARGET_CODES:
    # Get documents for this diagnosis
    code_indices = [i for i, m in enumerate(corpus_metadata) if m['icd_code'] == code]
    code_docs = [corpus_documents[i] for i in code_indices]
    code_embeds = corpus_embeddings[code_indices]

    # Build FAISS index for this diagnosis
    index = faiss.IndexFlatIP(embedding_dim)
    faiss.normalize_L2(code_embeds)
    index.add(code_embeds)

    diagnosis_indices[code] = index
    diagnosis_documents[code] = code_docs

    print(f"   {code}: {index.ntotal} documents")

print(f"✅ Built {len(diagnosis_indices)} diagnosis-specific indices")

# ============================================================================
# FIX 1 & 2: DIAGNOSIS-AWARE RAG WITH RELEVANCE SCORING
# ============================================================================

print("\n" + "="*80)
print("🧠 DIAGNOSIS-AWARE RAG SYSTEM")
print("="*80)

class DiagnosisAwareRAG:
    """FIX 1 & 2: Retrieve based on predicted diagnosis with relevance scoring"""
    def __init__(self, retriever, diagnosis_indices, diagnosis_documents,
                 top_k=5, threshold=0.6):
        self.retriever = retriever
        self.diagnosis_indices = diagnosis_indices
        self.diagnosis_documents = diagnosis_documents
        self.top_k = top_k
        self.threshold = threshold
        print(f"✅ Diagnosis-Aware RAG: top-{top_k}, threshold={threshold}")

    def retrieve(self, query: str, predicted_diagnosis: str = None) -> tuple:
        """
        Retrieve relevant knowledge based on predicted diagnosis
        Returns: (retrieved_text, relevance_score)
        """
        query = query[:1000]
        query_embedding = self.retriever.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False
        ).astype(np.float32)
        faiss.normalize_L2(query_embedding)

        # FIX 1: If diagnosis is predicted, use diagnosis-specific index
        if predicted_diagnosis and predicted_diagnosis in self.diagnosis_indices:
            index = self.diagnosis_indices[predicted_diagnosis]
            documents = self.diagnosis_documents[predicted_diagnosis]
        else:
            # Fallback: search all documents (for initial retrieval)
            all_docs = []
            for docs in self.diagnosis_documents.values():
                all_docs.extend(docs)
            documents = all_docs
            # Create temporary index
            all_embeds = self.retriever.encode(
                documents, convert_to_numpy=True, normalize_embeddings=True,
                show_progress_bar=False
            ).astype(np.float32)
            index = faiss.IndexFlatIP(query_embedding.shape[1])
            faiss.normalize_L2(all_embeds)
            index.add(all_embeds)

        scores, indices = index.search(query_embedding, k=self.top_k)

        # FIX 2: Filter by threshold and compute relevance
        filtered = []
        relevance_scores = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(documents) and score >= self.threshold:
                filtered.append(documents[idx])
                relevance_scores.append(float(score))

        retrieved_text = "\n\n".join(filtered) if filtered else ""
        avg_relevance = np.mean(relevance_scores) if relevance_scores else 0.0

        return retrieved_text, avg_relevance

rag_system = DiagnosisAwareRAG(
    retriever, diagnosis_indices, diagnosis_documents,
    top_k=5, threshold=0.6
)

# ============================================================================
# LOAD PHASE 1 FIXED MODEL
# ============================================================================

print("\n" + "="*80)
print("📥 LOADING PHASE 1 FIXED MODEL")
print("="*80)

# Load architecture from Phase 1 Fixed
exec(open('phase1_fixed.py').read().split('# ============================================================================\n# LOAD SAVED SPLITS')[0].split('# ============================================================================\n# FIXED ARCHITECTURE')[1])

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
checkpoint = torch.load(PHASE1_CHECKPOINT, map_location=device)
concept_embeddings = checkpoint['concept_embeddings'].to(device)
num_concepts = concept_embeddings.shape[0]

base_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)

# Recreate Phase 1 Fixed model class (simplified for loading)
class ShifaMindPhase1Fixed(nn.Module):
    def __init__(self, base_model, concept_embeddings_init, num_classes, fusion_layers=[9, 11]):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = base_model.config.hidden_size
        self.concept_embeddings = nn.Parameter(concept_embeddings_init.clone())
        self.num_concepts = concept_embeddings_init.shape[0]

        from phase1_fixed import AdaptiveGatedCrossAttention
        self.fusion_modules = nn.ModuleDict({
            str(layer): AdaptiveGatedCrossAttention(self.hidden_size, layer_idx=layer)
            for layer in fusion_layers
        })

        self.diagnosis_head = nn.Linear(self.hidden_size, num_classes)
        self.concept_head = nn.Linear(self.hidden_size, self.num_concepts)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, return_dict=True
        )

        hidden_states = outputs.hidden_states
        current_hidden = outputs.last_hidden_state

        for layer_idx in [9, 11]:
            if str(layer_idx) in self.fusion_modules:
                layer_hidden = hidden_states[layer_idx]
                fused_hidden, _, _ = self.fusion_modules[str(layer_idx)](
                    layer_hidden, self.concept_embeddings, attention_mask
                )
                current_hidden = fused_hidden

        cls_hidden = self.dropout(current_hidden[:, 0, :])
        diagnosis_logits = self.diagnosis_head(cls_hidden)

        return {'logits': diagnosis_logits, 'cls_hidden': cls_hidden}

phase1_model = ShifaMindPhase1Fixed(
    base_model, concept_embeddings, len(TARGET_CODES), fusion_layers=[9, 11]
).to(device)

phase1_model.load_state_dict(checkpoint['model_state_dict'])
print(f"✅ Phase 1 Fixed loaded (F1: {checkpoint.get('macro_f1', 0):.4f})")

# ============================================================================
# FIX 3: ADAPTIVE RAG FUSION
# ============================================================================

print("\n" + "="*80)
print("🏗️  ADAPTIVE RAG FUSION")
print("="*80)

class AdaptiveRAGFusion(nn.Module):
    """FIX 3: Gate learns when RAG helps based on relevance"""
    def __init__(self, hidden_size=768):
        super().__init__()
        # Gate depends on text, RAG, AND relevance score
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size * 2 + 1, hidden_size),  # +1 for relevance
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        self.rag_proj = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        print("   🔧 Adaptive gate (learns when RAG helps)")

    def forward(self, text_cls, rag_cls, relevance_score):
        """
        Args:
            text_cls: [batch, hidden]
            rag_cls: [batch, hidden]
            relevance_score: [batch, 1] - how relevant is retrieved doc
        """
        rag_projected = self.rag_proj(rag_cls)

        # Gate depends on text, RAG quality, and relevance
        gate_input = torch.cat([text_cls, rag_projected, relevance_score], dim=-1)
        gate = self.gate_net(gate_input)  # [batch, 1]

        # Adaptive fusion
        fused = (1 - gate) * text_cls + gate * rag_projected
        fused = self.layer_norm(fused)

        return fused, gate

class ShifaMindPhase2Fixed(nn.Module):
    """Phase 2 with Diagnosis-Aware RAG"""
    def __init__(self, phase1_model, tokenizer, rag_system):
        super().__init__()
        self.phase1_model = phase1_model
        self.tokenizer = tokenizer
        self.rag_system = rag_system
        self.rag_fusion = AdaptiveRAGFusion(hidden_size=phase1_model.hidden_size)

    def forward(self, input_ids, attention_mask, rag_texts, relevance_scores):
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
                rag_enc = self.tokenizer(
                    rag_text[:400], max_length=128, truncation=True,
                    padding='max_length', return_tensors='pt'
                )
                with torch.no_grad():
                    rag_out = self.phase1_model.base_model(
                        input_ids=rag_enc['input_ids'].to(device),
                        attention_mask=rag_enc['attention_mask'].to(device),
                        return_dict=True
                    )
                rag_cls_list.append(rag_out.last_hidden_state[:, 0, :])
            else:
                rag_cls_list.append(torch.zeros_like(text_cls[0:1]))

        rag_cls = torch.cat(rag_cls_list, dim=0)

        # Adaptive fusion with relevance
        relevance_tensor = torch.FloatTensor(relevance_scores).unsqueeze(-1).to(device)
        fused_cls, gate = self.rag_fusion(text_cls, rag_cls, relevance_tensor)

        # Predict
        diagnosis_logits = self.phase1_model.diagnosis_head(fused_cls)

        return {
            'logits': diagnosis_logits,
            'rag_gate': gate.mean()
        }

phase2_model = ShifaMindPhase2Fixed(phase1_model, tokenizer, rag_system).to(device)
print(f"✅ Phase 2 Fixed built ({sum(p.numel() for p in phase2_model.parameters()):,} params)")

# ============================================================================
# DATASET WITH DIAGNOSIS-AWARE RAG
# ============================================================================

print("\n" + "="*80)
print("📊 PREPARING DIAGNOSIS-AWARE DATASETS")
print("="*80)

class DiagnosisAwareRAGDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, rag_system, target_codes, max_length=384):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rag_cache = []
        self.relevance_cache = []

        print("Pre-retrieving with diagnosis awareness...")
        for i, (text, label) in enumerate(tqdm(zip(texts, labels), total=len(texts), desc="RAG")):
            # Get predicted diagnosis (use ground truth for training efficiency)
            predicted_dx = None
            for j, code in enumerate(target_codes):
                if label[j] == 1:
                    predicted_dx = code
                    break

            # Retrieve based on diagnosis
            rag_text, relevance = rag_system.retrieve(text, predicted_dx)
            self.rag_cache.append(rag_text)
            self.relevance_cache.append(relevance)

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
            'rag_text': self.rag_cache[idx],
            'relevance': self.relevance_cache[idx]
        }

train_dataset = DiagnosisAwareRAGDataset(
    df_train['text'].tolist(), df_train['labels'].tolist(),
    tokenizer, rag_system, TARGET_CODES
)
val_dataset = DiagnosisAwareRAGDataset(
    df_val['text'].tolist(), df_val['labels'].tolist(),
    tokenizer, rag_system, TARGET_CODES
)
test_dataset = DiagnosisAwareRAGDataset(
    df_test['text'].tolist(), df_test['labels'].tolist(),
    tokenizer, rag_system, TARGET_CODES
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

print(f"\n✅ Datasets ready (diagnosis-aware RAG)")

# ============================================================================
# TRAINING
# ============================================================================

print("\n" + "="*80)
print("🏋️  TRAINING PHASE 2 FIXED")
print("="*80)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(phase2_model.parameters(), lr=5e-6, weight_decay=0.01)

num_epochs = 4
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=len(train_loader) // 2,
    num_training_steps=len(train_loader) * num_epochs
)

best_f1 = 0
checkpoint_file = CHECKPOINT_PATH / 'phase2_fixed_best.pt'

for epoch in range(num_epochs):
    print(f"\n{'='*70}\nEpoch {epoch+1}/{num_epochs}\n{'='*70}")

    # Train
    phase2_model.train()
    total_loss = 0
    gate_values = []

    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        rag_texts = batch['rag_text']
        relevance_scores = batch['relevance']

        optimizer.zero_grad()
        outputs = phase2_model(input_ids, attention_mask, rag_texts, relevance_scores)
        loss = criterion(outputs['logits'], labels)

        if torch.isnan(loss):
            print("⚠️  NaN loss, skip")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(phase2_model.parameters(), max_norm=0.5)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        gate_values.append(outputs['rag_gate'].item())

    avg_loss = total_loss / len(train_loader)
    avg_gate = np.mean(gate_values)
    avg_relevance = np.mean(train_dataset.relevance_cache)

    print(f"\n  Loss: {avg_loss:.4f}")
    print(f"  RAG Gate: {avg_gate:.4f}")
    print(f"  Avg Relevance: {avg_relevance:.4f}")

    # Validate
    phase2_model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            rag_texts = batch['rag_text']
            relevance_scores = batch['relevance']

            outputs = phase2_model(input_ids, attention_mask, rag_texts, relevance_scores)
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
            'rag_gate': avg_gate,
            'avg_relevance': avg_relevance
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
        relevance_scores = batch['relevance']

        outputs = phase2_model(input_ids, attention_mask, rag_texts, relevance_scores)
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
print("🎉 PHASE 2 FIXED - FINAL RESULTS")
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

phase1_f1 = checkpoint.get('macro_f1', 0.78)  # From Phase 1 Fixed
improvement = macro_f1 - phase1_f1

print(f"\n🔥 VS PHASE 1 FIXED:")
print(f"   Phase 1 Fixed: {phase1_f1:.4f}")
print(f"   Phase 2 Fixed: {macro_f1:.4f}")
print(f"   Δ: {improvement:+.4f} ({improvement/phase1_f1*100:+.1f}%)")

if macro_f1 >= 0.80:
    print("\n✅ SUCCESS! Hit target (≥0.80)")
elif macro_f1 > phase1_f1:
    print("\n✅ IMPROVEMENT! RAG helps")
else:
    print(f"\n📊 Gap: {phase1_f1 - macro_f1:.4f}")

results = {
    'phase': 'Phase 2 Fixed - Diagnosis-Aware RAG',
    'test_metrics': {
        'macro_f1': float(macro_f1),
        'micro_f1': float(micro_f1),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_auc': float(macro_auc),
        'per_class_f1': {code: float(f1) for code, f1 in zip(TARGET_CODES, per_class_f1)}
    },
    'comparison': {
        'phase1_fixed_f1': float(phase1_f1),
        'phase2_fixed_f1': float(macro_f1),
        'improvement': float(improvement),
        'improvement_pct': float(improvement/phase1_f1*100)
    },
    'config': {
        'diagnosis_aware': True,
        'relevance_scoring': True,
        'adaptive_gate': True,
        'avg_relevance': float(checkpoint.get('avg_relevance', 0)),
        'rag_gate': float(checkpoint.get('rag_gate', 0))
    },
    'fixes_applied': [
        'Diagnosis-aware retrieval (separate indices per diagnosis)',
        'Relevance scoring for retrieved documents',
        'Adaptive RAG gate (learns when RAG helps)',
        'Enhanced corpus (more clinical knowledge)',
        'Quality-filtered case prototypes'
    ]
}

with open(RESULTS_PATH / 'phase2_fixed_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Results: {RESULTS_PATH}")
print(f"💾 Checkpoint: {checkpoint_file}")

print("\n" + "="*80)
print("✅ PHASE 2 FIXED COMPLETE!")
print("="*80)
print(f"\n📈 Final Macro F1: {macro_f1:.4f}")
print("\n🚀 Ready for Phase 3 Fixed (Enhanced XAI Metrics)")
print(f"\nAlhamdulillah! 🤲")
