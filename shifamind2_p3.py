#!/usr/bin/env python3
"""
================================================================================
SHIFAMIND2 PHASE 3: RAG with FAISS (Top-50 ICD-10)
================================================================================
Author: Mohammed Sameer Syed
University of Arizona - MS in AI Capstone

CHANGES FROM SHIFAMIND1_P3:
1. ✅ Uses Top-50 ICD-10 codes from Phase 1/2
2. ✅ Loads from SAME run folder
3. ✅ Fresh evidence store (no reuse)
4. ✅ Evidence corpus with Top-50 clinical knowledge
5. ✅ Gated fusion with 40% RAG cap

Architecture:
- Load Phase 2 checkpoint (concept bottleneck + GraphSAGE)
- Build FAISS index with sentence-transformers
- Create evidence corpus from clinical knowledge + MIMIC prototypes (Top-50)
- Gated fusion for RAG integration
- Diagnosis-focused training

Target Metrics:
- Diagnosis F1: >0.80

================================================================================
"""

print("="*80)
print("🚀 SHIFAMIND2 PHASE 3 - RAG WITH TOP-50 FAISS")
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
from transformers import (
    AutoTokenizer, AutoModel,
    get_linear_schedule_with_warmup
)

from sentence_transformers import SentenceTransformer

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    print("⚠️  FAISS not available")
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
# CONFIGURATION: LOAD FROM PHASE 2
# ============================================================================

print("\n" + "="*80)
print("⚙️  CONFIGURATION: LOADING FROM PHASE 2")
print("="*80)

BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')
SHIFAMIND2_BASE = BASE_PATH / '10_ShifaMind'

run_folders = sorted([d for d in SHIFAMIND2_BASE.glob('run_*') if d.is_dir()], reverse=True)

if not run_folders:
    print("❌ No Phase 2 run found!")
    sys.exit(1)

OUTPUT_BASE = run_folders[0]
print(f"📁 Using run folder: {OUTPUT_BASE.name}")

PHASE2_CHECKPOINT = OUTPUT_BASE / 'checkpoints' / 'phase2' / 'phase2_best.pt'
if not PHASE2_CHECKPOINT.exists():
    print(f"❌ Phase 2 checkpoint not found!")
    sys.exit(1)

checkpoint = torch.load(PHASE2_CHECKPOINT, map_location='cpu', weights_only=False)
phase2_config = checkpoint['config']
TOP_50_CODES = phase2_config['top_50_codes']
timestamp = phase2_config['timestamp']

print(f"✅ Loaded Phase 2 config:")
print(f"   Timestamp: {timestamp}")
print(f"   Top-50 codes: {len(TOP_50_CODES)}")

SHARED_DATA_PATH = OUTPUT_BASE / 'shared_data'
CHECKPOINT_PATH = OUTPUT_BASE / 'checkpoints' / 'phase3'
RESULTS_PATH = OUTPUT_BASE / 'results' / 'phase3'
EVIDENCE_PATH = OUTPUT_BASE / 'evidence_store'

for path in [CHECKPOINT_PATH, RESULTS_PATH, EVIDENCE_PATH]:
    path.mkdir(parents=True, exist_ok=True)

with open(SHARED_DATA_PATH / 'concept_list.json', 'r') as f:
    ALL_CONCEPTS = json.load(f)

print(f"\n🧠 Concepts: {len(ALL_CONCEPTS)}")

# RAG hyperparameters
RAG_TOP_K = 3
RAG_THRESHOLD = 0.7
RAG_GATE_MAX = 0.4
PROTOTYPES_PER_DIAGNOSIS = 20

# Training hyperparameters
LAMBDA_DX = 2.0
LAMBDA_ALIGN = 0.5
LAMBDA_CONCEPT = 0.3
LEARNING_RATE = 5e-6
EPOCHS = 5
BATCH_SIZE = 8

print(f"\n⚖️  Loss Weights:")
print(f"   λ_dx:      {LAMBDA_DX}")
print(f"   λ_align:   {LAMBDA_ALIGN}")
print(f"   λ_concept: {LAMBDA_CONCEPT}")

# ============================================================================
# BUILD EVIDENCE CORPUS (TOP-50 CLINICAL KNOWLEDGE)
# ============================================================================

print("\n" + "="*80)
print("📚 BUILDING EVIDENCE CORPUS (TOP-50)")
print("="*80)

def build_evidence_corpus_top50(top_50_codes):
    """
    Build evidence corpus for Top-50 diagnoses
    1. Clinical knowledge (curated for top diagnoses)
    2. Case prototypes from MIMIC
    """
    print("\n📖 Building evidence corpus...")

    corpus = []

    # Part 1: Generic clinical knowledge (expandable for Top-50)
    # In production, this would be loaded from medical databases
    clinical_knowledge_base = {
        # Respiratory (J codes)
        'J': 'Respiratory conditions: assess cough, dyspnea, chest imaging, oxygen saturation',
        'J18': 'Pneumonia diagnosis requires fever, cough, infiltrates on imaging',
        'J44': 'COPD: chronic airflow limitation, emphysema, chronic bronchitis',
        'J96': 'Respiratory failure: hypoxia, hypercapnia, requires oxygen support',

        # Cardiac (I codes)
        'I': 'Cardiovascular disease: assess chest pain, dyspnea, edema, cardiac markers',
        'I50': 'Heart failure: dyspnea, edema, elevated BNP, reduced EF on echo',
        'I25': 'Ischemic heart disease: angina, troponin, EKG changes',
        'I21': 'MI: acute chest pain, troponin elevation, ST changes',

        # Infection (A codes)
        'A': 'Infectious disease: fever, cultures, antibiotics',
        'A41': 'Sepsis: organ dysfunction, hypotension, lactate >2, positive cultures',

        # Renal (N codes)
        'N': 'Renal disease: creatinine, BUN, urine output',
        'N17': 'Acute kidney injury: rapid creatinine rise, oliguria',
        'N18': 'Chronic kidney disease: GFR <60, proteinuria',

        # Metabolic (E codes)
        'E': 'Endocrine/metabolic: glucose, electrolytes, hormone levels',
        'E11': 'Type 2 diabetes: hyperglycemia, A1c >6.5%, insulin resistance',
        'E87': 'Electrolyte disorders: sodium, potassium, calcium imbalance',

        # GI (K codes)
        'K': 'GI disease: abdominal pain, nausea, imaging',
        'K80': 'Cholelithiasis: RUQ pain, ultrasound showing stones',

        # Mental health (F codes)
        'F': 'Mental health: psychiatric assessment, mood, cognition',

        # Injury (S/T codes)
        'S': 'Injury/trauma: mechanism, imaging, stabilization',
        'T': 'Poisoning/external causes: toxicology, supportive care',
    }

    print("\n📝 Adding clinical knowledge...")
    for code in top_50_codes:
        # Match by chapter (first letter) or specific code prefix
        matched = False
        for key, knowledge in clinical_knowledge_base.items():
            if code.startswith(key):
                corpus.append({
                    'text': f"{code}: {knowledge}",
                    'diagnosis': code,
                    'source': 'clinical_knowledge'
                })
                matched = True
                break

        if not matched:
            corpus.append({
                'text': f"{code}: Diagnosis code requiring clinical correlation",
                'diagnosis': code,
                'source': 'clinical_knowledge'
            })

    print(f"   Added {len(corpus)} clinical knowledge passages")

    # Part 2: Case prototypes from MIMIC
    print(f"\n🏥 Sampling {PROTOTYPES_PER_DIAGNOSIS} case prototypes per diagnosis...")

    with open(SHARED_DATA_PATH / 'train_split.pkl', 'rb') as f:
        df_train = pickle.load(f)

    for idx, dx_code in enumerate(top_50_codes):
        # Find positive samples for this diagnosis
        code_column_exists = dx_code in df_train.columns
        if code_column_exists:
            positive_samples = df_train[df_train[dx_code] == 1]
        else:
            # Try to find from labels list
            if 'labels' in df_train.columns:
                code_idx = top_50_codes.index(dx_code)
                positive_samples = df_train[df_train['labels'].apply(
                    lambda x: x[code_idx] == 1 if isinstance(x, list) and len(x) > code_idx else False
                )]
            else:
                positive_samples = pd.DataFrame()

        n_samples = min(len(positive_samples), PROTOTYPES_PER_DIAGNOSIS)
        if n_samples > 0:
            sampled = positive_samples.sample(n=n_samples, random_state=SEED)

            for _, row in sampled.iterrows():
                text = str(row['text'])[:500]
                corpus.append({
                    'text': text,
                    'diagnosis': dx_code,
                    'source': 'mimic_prototype'
                })

        if (idx + 1) % 10 == 0:
            print(f"   Processed {idx + 1}/{len(top_50_codes)} diagnoses...")

    print(f"\n✅ Evidence corpus built:")
    print(f"   Total passages: {len(corpus)}")
    print(f"   Clinical knowledge: {len([c for c in corpus if c['source'] == 'clinical_knowledge'])}")
    print(f"   MIMIC prototypes: {len([c for c in corpus if c['source'] == 'mimic_prototype'])}")

    return corpus

evidence_corpus = build_evidence_corpus_top50(TOP_50_CODES)

with open(EVIDENCE_PATH / 'evidence_corpus_top50.json', 'w') as f:
    json.dump(evidence_corpus, f, indent=2)

print(f"💾 Saved corpus to: {EVIDENCE_PATH / 'evidence_corpus_top50.json'}")

# ============================================================================
# FAISS RETRIEVER
# ============================================================================

print("\n" + "="*80)
print("🔍 BUILDING FAISS RETRIEVER")
print("="*80)

class SimpleRAG:
    """Simple RAG using FAISS + sentence-transformers"""
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', top_k=3, threshold=0.7):
        print(f"\n🤖 Initializing RAG with {model_name}...")
        self.encoder = SentenceTransformer(model_name)
        self.top_k = top_k
        self.threshold = threshold
        self.index = None
        self.documents = []
        print(f"✅ RAG encoder loaded")

    def build_index(self, documents: List[Dict]):
        print(f"\n🔨 Building FAISS index from {len(documents)} documents...")
        self.documents = documents
        texts = [doc['text'] for doc in documents]

        print("   Encoding documents...")
        embeddings = self.encoder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        embeddings = embeddings.astype('float32')

        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        print(f"✅ FAISS index built:")
        print(f"   Dimension: {dimension}")
        print(f"   Total vectors: {self.index.ntotal}")

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

if not FAISS_AVAILABLE:
    print("⚠️  FAISS not available - RAG disabled")
    rag = None
else:
    rag = SimpleRAG(top_k=RAG_TOP_K, threshold=RAG_THRESHOLD)
    rag.build_index(evidence_corpus)

# ============================================================================
# SHIFAMIND2 PHASE 3 MODEL
# ============================================================================

print("\n" + "="*80)
print("🏗️  BUILDING SHIFAMIND2 PHASE 3 MODEL")
print("="*80)

class ShifaMind2Phase3(nn.Module):
    """ShifaMind2 with RAG integration (Top-50)"""
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

    def forward(self, input_ids, attention_mask, concept_embeddings, input_texts=None):
        batch_size = input_ids.shape[0]

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        pooled_bert = hidden_states.mean(dim=1)

        # RAG retrieval and fusion
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
            gate = gate * RAG_GATE_MAX

            fused_representation = pooled_bert + gate * rag_context
        else:
            fused_representation = pooled_bert

        fused_states = fused_representation.unsqueeze(1).expand(-1, hidden_states.shape[1], -1)

        # Concept bottleneck
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

        return {
            'logits': diagnosis_logits,
            'concept_logits': concept_logits,
            'concept_scores': torch.sigmoid(concept_logits),
            'gate_values': gate
        }

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

if PHASE2_CHECKPOINT.exists():
    print(f"\n📥 Loading Phase 2 checkpoint...")
    checkpoint = torch.load(PHASE2_CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print("✅ Loaded Phase 2 weights (partial)")

print(f"\n✅ ShifaMind2 Phase 3 model initialized")
print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# TRAINING SETUP
# ============================================================================

print("\n" + "="*80)
print("⚙️  TRAINING SETUP")
print("="*80)

with open(SHARED_DATA_PATH / 'train_split.pkl', 'rb') as f:
    df_train = pickle.load(f)
with open(SHARED_DATA_PATH / 'val_split.pkl', 'rb') as f:
    df_val = pickle.load(f)
with open(SHARED_DATA_PATH / 'test_split.pkl', 'rb') as f:
    df_test = pickle.load(f)

train_concept_labels = np.load(SHARED_DATA_PATH / 'train_concept_labels.npy')
val_concept_labels = np.load(SHARED_DATA_PATH / 'val_concept_labels.npy')
test_concept_labels = np.load(SHARED_DATA_PATH / 'test_concept_labels.npy')

print(f"\n📊 Data loaded:")
print(f"   Train: {len(df_train)} samples")
print(f"   Val:   {len(df_val)} samples")
print(f"   Test:  {len(df_test)} samples")

class RAGDataset(Dataset):
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

train_dataset = RAGDataset(df_train, tokenizer, train_concept_labels)
val_dataset = RAGDataset(df_val, tokenizer, val_concept_labels)
test_dataset = RAGDataset(df_test, tokenizer, test_concept_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

class MultiObjectiveLoss(nn.Module):
    def __init__(self, lambda_dx, lambda_align, lambda_concept):
        super().__init__()
        self.lambda_dx = lambda_dx
        self.lambda_align = lambda_align
        self.lambda_concept = lambda_concept
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs, dx_labels, concept_labels):
        loss_dx = self.bce(outputs['logits'], dx_labels)

        dx_probs = torch.sigmoid(outputs['logits'])
        concept_scores = outputs['concept_scores']
        loss_align = torch.abs(dx_probs.unsqueeze(-1) - concept_scores.unsqueeze(1)).mean()

        loss_concept = self.bce(outputs['concept_logits'], concept_labels)

        total_loss = (
            self.lambda_dx * loss_dx +
            self.lambda_align * loss_align +
            self.lambda_concept * loss_concept
        )

        return total_loss, {
            'loss_dx': loss_dx.item(),
            'loss_align': loss_align.item(),
            'loss_concept': loss_concept.item(),
            'total_loss': total_loss.item()
        }

criterion = MultiObjectiveLoss(LAMBDA_DX, LAMBDA_ALIGN, LAMBDA_CONCEPT)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

print("✅ Training setup complete")

# ============================================================================
# TRAINING LOOP
# ============================================================================

print("\n" + "="*80)
print("🏋️  TRAINING PHASE 3 (RAG-ENHANCED)")
print("="*80)

best_val_f1 = 0.0
history = {'train_loss': [], 'val_loss': [], 'val_f1': []}

concept_embeddings = concept_embedding_layer.weight.detach()

for epoch in range(EPOCHS):
    print(f"\n📍 Epoch {epoch+1}/{EPOCHS}")

    model.train()
    train_losses = []

    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        concept_labels = batch['concept_labels'].to(device)
        texts = batch['text']

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask, concept_embeddings, input_texts=texts)
        loss, loss_components = criterion(outputs, labels, concept_labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        train_losses.append(loss.item())
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_train_loss = np.mean(train_losses)
    history['train_loss'].append(avg_train_loss)

    # Validation
    model.eval()
    val_losses = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            concept_labels = batch['concept_labels'].to(device)
            texts = batch['text']

            outputs = model(input_ids, attention_mask, concept_embeddings, input_texts=texts)
            loss, _ = criterion(outputs, labels, concept_labels)

            val_losses.append(loss.item())

            preds = (torch.sigmoid(outputs['logits']) > 0.5).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    avg_val_loss = np.mean(val_losses)
    val_f1 = f1_score(all_labels, all_preds, average='macro')

    history['val_loss'].append(avg_val_loss)
    history['val_f1'].append(val_f1)

    print(f"   Train Loss: {avg_train_loss:.4f}")
    print(f"   Val Loss:   {avg_val_loss:.4f}")
    print(f"   Val F1:     {val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_f1': best_val_f1,
            'concept_embeddings': concept_embeddings,
            'evidence_corpus': evidence_corpus,
            'config': {
                'num_concepts': len(ALL_CONCEPTS),
                'num_diagnoses': len(TOP_50_CODES),
                'rag_config': {
                    'top_k': RAG_TOP_K,
                    'threshold': RAG_THRESHOLD,
                    'gate_max': RAG_GATE_MAX
                },
                'top_50_codes': TOP_50_CODES,
                'timestamp': timestamp
            }
        }, CHECKPOINT_PATH / 'phase3_best.pt')
        print(f"   ✅ Saved best model (F1: {best_val_f1:.4f})")

# ============================================================================
# EVALUATION
# ============================================================================

print("\n" + "="*80)
print("📊 FINAL EVALUATION")
print("="*80)

checkpoint = torch.load(CHECKPOINT_PATH / 'phase3_best.pt', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        texts = batch['text']

        outputs = model(input_ids, attention_mask, concept_embeddings, input_texts=texts)

        probs = torch.sigmoid(outputs['logits']).cpu().numpy()
        preds = (probs > 0.5).astype(int)

        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

all_preds = np.vstack(all_preds)
all_labels = np.vstack(all_labels)

macro_f1 = f1_score(all_labels, all_preds, average='macro')
micro_f1 = f1_score(all_labels, all_preds, average='micro')
per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

print(f"\n🎯 Diagnosis Performance (Top-50):")
print(f"   Macro F1: {macro_f1:.4f}")
print(f"   Micro F1: {micro_f1:.4f}")

results = {
    'phase': 'ShifaMind2 Phase 3 - RAG with FAISS (Top-50)',
    'timestamp': timestamp,
    'run_folder': str(OUTPUT_BASE),
    'diagnosis_metrics': {
        'macro_f1': float(macro_f1),
        'micro_f1': float(micro_f1),
        'per_class_f1': {code: float(f1) for code, f1 in zip(TOP_50_CODES, per_class_f1)}
    },
    'architecture': 'Concept Bottleneck + GraphSAGE + FAISS RAG (Top-50)',
    'rag_config': {
        'method': 'FAISS + sentence-transformers',
        'top_k': RAG_TOP_K,
        'threshold': RAG_THRESHOLD,
        'gate_max': RAG_GATE_MAX,
        'corpus_size': len(evidence_corpus)
    },
    'training_history': history
}

with open(RESULTS_PATH / 'results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Results saved to: {RESULTS_PATH / 'results.json'}")
print(f"💾 Best model saved to: {CHECKPOINT_PATH / 'phase3_best.pt'}")

print("\n" + "="*80)
print("✅ SHIFAMIND2 PHASE 3 COMPLETE!")
print("="*80)
print(f"\n📍 Run folder: {OUTPUT_BASE}")
print(f"   Macro F1: {macro_f1:.4f} | Micro F1: {micro_f1:.4f}")
print("\nNext: Run shifamind2_p4.py (XAI metrics)")
print("\nAlhamdulillah! 🤲")
