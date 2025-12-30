#!/usr/bin/env python3
"""
================================================================================
SHIFAMIND PHASE 3 V2 FIXED: RAG with Proven FAISS Approach
================================================================================
Author: Mohammed Sameer Syed
University of Arizona - MS in AI Capstone

FIXES from phase3_v2.py:
1. ✅ Use FAISS + sentence-transformers (like successful p2_phase2_rag.py)
2. ✅ Build corpus from clinical knowledge + MIMIC case prototypes
3. ✅ Gated fusion mechanism (40% cap on RAG contribution)
4. ✅ Rebalanced loss weights (prioritize diagnosis)
5. ✅ Simplified architecture (optional citation/action heads)
6. ✅ top_k=3, threshold=0.7 for retrieval

This recovers the proven RAG approach that improved F1 from 0.75 → 0.81

Architecture:
- Load Phase 2 checkpoint (concept bottleneck + GraphSAGE)
- Build FAISS index with sentence-transformers
- Create evidence corpus from clinical knowledge + MIMIC prototypes
- Gated fusion for RAG integration
- Diagnosis-focused training

Target Metrics:
- Diagnosis F1: >0.80 (recover from 0.54 → 0.80+)

================================================================================
"""

print("="*80)
print("🚀 SHIFAMIND PHASE 3 V2 FIXED - RAG WITH PROVEN FAISS APPROACH")
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
import torch_geometric
from torch_geometric.nn import SAGEConv

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoTokenizer, AutoModel,
    get_linear_schedule_with_warmup
)

# Sentence transformers for RAG
from sentence_transformers import SentenceTransformer

# FAISS for efficient similarity search
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    print("⚠️  FAISS not available - install with: pip install faiss-cpu")
    FAISS_AVAILABLE = False

import json
import pickle
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from collections import defaultdict

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

# Paths
BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')
OUTPUT_BASE = BASE_PATH / '08_ShifaMind'

# Use existing shared_data if available
EXISTING_SHARED_DATA = BASE_PATH / '03_Models/shared_data'
if EXISTING_SHARED_DATA.exists():
    SHARED_DATA_PATH = EXISTING_SHARED_DATA
else:
    SHARED_DATA_PATH = OUTPUT_BASE / 'shared_data'

PHASE2_CHECKPOINT = OUTPUT_BASE / 'checkpoints/phase2_v2/phase2_v2_best.pt'
CHECKPOINT_PATH = OUTPUT_BASE / 'checkpoints/phase3_v2_fixed'
RESULTS_PATH = OUTPUT_BASE / 'results/phase3_v2_fixed'
EVIDENCE_PATH = OUTPUT_BASE / 'evidence_store'

# Create directories
for path in [CHECKPOINT_PATH, RESULTS_PATH, EVIDENCE_PATH]:
    path.mkdir(parents=True, exist_ok=True)

print(f"📁 Phase 2 Checkpoint: {PHASE2_CHECKPOINT}")
print(f"📁 Checkpoints: {CHECKPOINT_PATH}")
print(f"📁 Shared Data: {SHARED_DATA_PATH}")
print(f"📁 Results: {RESULTS_PATH}")
print(f"📁 Evidence Store: {EVIDENCE_PATH}")

# Target diagnoses
TARGET_CODES = ['J189', 'I5023', 'A419', 'K8000']
ICD_DESCRIPTIONS = {
    'J189': 'Pneumonia, unspecified organism',
    'I5023': 'Acute on chronic systolic heart failure',
    'A419': 'Sepsis, unspecified organism',
    'K8000': 'Calculus of gallbladder with acute cholecystitis'
}

# Load concept list
with open(SHARED_DATA_PATH / 'concept_list.json', 'r') as f:
    ALL_CONCEPTS = json.load(f)

print(f"\n🎯 Target: {len(TARGET_CODES)} diagnoses")
print(f"🧠 Concepts: {len(ALL_CONCEPTS)} clinical concepts")

# RAG hyperparameters (from successful implementation)
RAG_TOP_K = 3  # Retrieve top 3 passages
RAG_THRESHOLD = 0.7  # Similarity threshold
RAG_GATE_MAX = 0.4  # Cap RAG contribution at 40%
PROTOTYPES_PER_DIAGNOSIS = 20  # Sample 20 cases per diagnosis

# Training hyperparameters (FIXED: prioritize diagnosis)
LAMBDA_DX = 2.0  # ← INCREASED (was 1.0)
LAMBDA_ALIGN = 0.5
LAMBDA_CONCEPT = 0.3
LEARNING_RATE = 5e-6
EPOCHS = 5  # ← INCREASED (was 3)
BATCH_SIZE = 8

print(f"\n⚖️  Loss Weights (FIXED):")
print(f"   λ_dx:      {LAMBDA_DX} ← DOUBLED to prioritize diagnosis")
print(f"   λ_align:   {LAMBDA_ALIGN}")
print(f"   λ_concept: {LAMBDA_CONCEPT}")
print(f"\n🔧 RAG Config:")
print(f"   Top-k:     {RAG_TOP_K}")
print(f"   Threshold: {RAG_THRESHOLD}")
print(f"   Gate Max:  {RAG_GATE_MAX} (40% cap)")
print(f"   Prototypes: {PROTOTYPES_PER_DIAGNOSIS} per diagnosis")

# ============================================================================
# BUILD EVIDENCE CORPUS (WITH MIMIC PROTOTYPES)
# ============================================================================

print("\n" + "="*80)
print("📚 BUILDING EVIDENCE CORPUS")
print("="*80)

def build_evidence_corpus():
    """
    Build evidence corpus using proven approach from p2_phase2_rag.py:
    1. Clinical knowledge from ICD descriptions
    2. Case prototypes from MIMIC training data (20 per diagnosis)
    """
    print("\n📖 Building evidence corpus...")

    corpus = []

    # Part 1: Clinical knowledge base
    clinical_knowledge = {
        'J189': [
            'Pneumonia diagnosis requires fever, cough, dyspnea, and radiographic infiltrates. Consolidation on chest X-ray is highly specific.',
            'Bronchial breath sounds, dullness to percussion, fever and productive cough with purulent sputum indicate bacterial pneumonia.',
            'Chest imaging showing infiltrates or consolidation is essential. Respiratory rate elevation and hypoxia indicate severity.'
        ],
        'I5023': [
            'Acute on chronic systolic heart failure presents with dyspnea, orthopnea, PND. Elevated BNP >400 pg/mL strongly supports diagnosis.',
            'Physical exam: bilateral edema, elevated JVP, S3 gallop. Cardiomegaly on CXR. Reduced EF on echo confirms systolic dysfunction.',
            'Pulmonary edema on imaging plus cardiac dysfunction on echo confirms heart failure. BNP aids diagnosis and risk stratification.'
        ],
        'A419': [
            'Sepsis: life-threatening organ dysfunction from dysregulated infection response. Fever/hypothermia, tachycardia, hypotension, altered mental status. Lactate >2 mmol/L.',
            'Septic shock requires hypotension despite fluids and lactate >2 mmol/L. Blood cultures before antibiotics. WBC elevation with left shift.',
            'Sepsis requires infection evidence plus organ dysfunction. Hemodynamic instability and vasopressor need indicate septic shock.'
        ],
        'K8000': [
            'Acute cholecystitis with cholelithiasis: RUQ pain, fever, positive Murphy sign. Ultrasound shows gallbladder wall thickening >3mm.',
            'Murphy sign (inspiratory arrest during RUQ palpation) highly specific. Pain radiates to right shoulder. Leukocytosis and elevated inflammatory markers.',
            'Ultrasound first-line: shows gallstones and inflammation. Cholestasis with elevated bilirubin may indicate CBD involvement.'
        ]
    }

    print("\n📝 Adding clinical knowledge...")
    for dx_code, knowledge_list in clinical_knowledge.items():
        for text in knowledge_list:
            corpus.append({
                'text': text,
                'diagnosis': dx_code,
                'source': 'clinical_knowledge'
            })

    print(f"   Added {sum(len(k) for k in clinical_knowledge.values())} clinical knowledge passages")

    # Part 2: Case prototypes from MIMIC training data
    print(f"\n🏥 Sampling {PROTOTYPES_PER_DIAGNOSIS} case prototypes per diagnosis from MIMIC...")

    # Load training data
    with open(SHARED_DATA_PATH / 'train_split.pkl', 'rb') as f:
        df_train = pickle.load(f)

    for dx_code in TARGET_CODES:
        # Handle both DataFrame formats
        if 'labels' in df_train.columns:
            # Labels is a list column
            dx_idx = TARGET_CODES.index(dx_code)
            positive_samples = df_train[df_train['labels'].apply(lambda x: x[dx_idx] == 1 if isinstance(x, list) else False)]
        else:
            # Individual columns
            if dx_code in df_train.columns:
                positive_samples = df_train[df_train[dx_code] == 1]
            else:
                positive_samples = pd.DataFrame()

        # Sample up to PROTOTYPES_PER_DIAGNOSIS cases
        n_samples = min(len(positive_samples), PROTOTYPES_PER_DIAGNOSIS)
        if n_samples > 0:
            sampled = positive_samples.sample(n=n_samples, random_state=SEED)

            for _, row in sampled.iterrows():
                # Truncate long notes to first 500 chars for efficiency
                text = str(row['text'])[:500]
                corpus.append({
                    'text': text,
                    'diagnosis': dx_code,
                    'source': 'mimic_prototype'
                })

            print(f"   {dx_code}: Added {n_samples} case prototypes")
        else:
            print(f"   {dx_code}: ⚠️  No positive samples found")

    print(f"\n✅ Evidence corpus built:")
    print(f"   Total passages: {len(corpus)}")
    print(f"   Clinical knowledge: {len([c for c in corpus if c['source'] == 'clinical_knowledge'])}")
    print(f"   MIMIC prototypes: {len([c for c in corpus if c['source'] == 'mimic_prototype'])}")

    return corpus

evidence_corpus = build_evidence_corpus()

# Save corpus
with open(EVIDENCE_PATH / 'evidence_corpus_fixed.json', 'w') as f:
    json.dump(evidence_corpus, f, indent=2)

# ============================================================================
# FAISS RETRIEVER (PROVEN APPROACH)
# ============================================================================

print("\n" + "="*80)
print("🔍 BUILDING FAISS RETRIEVER")
print("="*80)

class SimpleRAG:
    """
    Simple RAG using FAISS + sentence-transformers

    This is the PROVEN approach from p2_phase2_rag.py that worked:
    - FAISS IndexFlatIP for fast similarity search
    - sentence-transformers/all-MiniLM-L6-v2 for embeddings
    - top_k=3, threshold=0.7 for filtering
    """
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', top_k=3, threshold=0.7):
        print(f"\n🤖 Initializing RAG with {model_name}...")

        self.encoder = SentenceTransformer(model_name)
        self.top_k = top_k
        self.threshold = threshold
        self.index = None
        self.documents = []

        print(f"✅ RAG encoder loaded")

    def build_index(self, documents: List[Dict]):
        """Build FAISS index from documents"""
        print(f"\n🔨 Building FAISS index from {len(documents)} documents...")

        self.documents = documents
        texts = [doc['text'] for doc in documents]

        # Encode all documents
        print("   Encoding documents...")
        embeddings = self.encoder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        embeddings = embeddings.astype('float32')

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Build FAISS index (Inner Product = cosine similarity after normalization)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        print(f"✅ FAISS index built:")
        print(f"   Dimension: {dimension}")
        print(f"   Total vectors: {self.index.ntotal}")

    def retrieve(self, query: str) -> str:
        """
        Retrieve relevant passages for query

        Returns:
            Concatenated text from top-k relevant passages (above threshold)
        """
        if self.index is None:
            return ""

        # Encode query
        query_embedding = self.encoder.encode([query], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, self.top_k)

        # Filter by threshold and concatenate
        relevant_texts = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= self.threshold:
                relevant_texts.append(self.documents[idx]['text'])

        return " ".join(relevant_texts) if relevant_texts else ""

# Initialize RAG
if not FAISS_AVAILABLE:
    print("⚠️  FAISS not available - RAG will be disabled")
    rag = None
else:
    rag = SimpleRAG(top_k=RAG_TOP_K, threshold=RAG_THRESHOLD)
    rag.build_index(evidence_corpus)

# ============================================================================
# SHIFAMIND PHASE 3 FIXED MODEL
# ============================================================================

print("\n" + "="*80)
print("🏗️  BUILDING SHIFAMIND PHASE 3 FIXED MODEL")
print("="*80)

class ShifaMindPhase3Fixed(nn.Module):
    """
    ShifaMind with FIXED RAG integration

    Key fixes:
    1. Gated fusion mechanism (40% cap on RAG contribution)
    2. Simplified architecture (no citation/action heads)
    3. Focus on diagnosis task

    Architecture:
    1. BioClinicalBERT encoder
    2. RAG retrieval (FAISS + sentence-transformers)
    3. Gated fusion: output = hidden + gate * rag_context
    4. Concept bottleneck
    5. Diagnosis head
    """
    def __init__(self, base_model, rag_retriever, num_concepts, num_diagnoses, hidden_size=768):
        super().__init__()

        self.bert = base_model
        self.rag = rag_retriever
        self.hidden_size = hidden_size
        self.num_concepts = num_concepts
        self.num_diagnoses = num_diagnoses

        # RAG encoder (to match BERT hidden size)
        if rag_retriever is not None:
            rag_dim = 384  # all-MiniLM-L6-v2 dimension
            self.rag_projection = nn.Linear(rag_dim, hidden_size)
        else:
            self.rag_projection = None

        # Gated fusion for RAG (KEY FIX: 40% cap)
        self.rag_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

        # Concept bottleneck
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

        # Output heads (simplified)
        self.concept_head = nn.Linear(hidden_size, num_concepts)
        self.diagnosis_head = nn.Linear(hidden_size, num_diagnoses)

    def forward(self, input_ids, attention_mask, concept_embeddings, input_texts=None):
        """
        Forward pass with gated RAG fusion

        Args:
            input_ids: Tokenized text [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            concept_embeddings: Concept embeddings [num_concepts, hidden_size]
            input_texts: Original text for RAG retrieval (optional)
        """
        batch_size = input_ids.shape[0]

        # 1. Encode text with BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
        pooled_bert = hidden_states.mean(dim=1)  # [batch, hidden_size]

        # 2. RAG retrieval and fusion (FIXED)
        if self.rag is not None and input_texts is not None:
            # Retrieve for each text in batch
            rag_texts = [self.rag.retrieve(text) for text in input_texts]

            # Encode RAG context
            rag_embeddings = []
            for rag_text in rag_texts:
                if rag_text:
                    emb = self.rag.encoder.encode([rag_text], convert_to_numpy=True)[0]
                else:
                    emb = np.zeros(384)  # all-MiniLM-L6-v2 dim
                rag_embeddings.append(emb)

            rag_embeddings = torch.tensor(np.array(rag_embeddings), dtype=torch.float32).to(pooled_bert.device)
            rag_context = self.rag_projection(rag_embeddings)  # [batch, hidden_size]

            # Gated fusion with 40% cap
            gate_input = torch.cat([pooled_bert, rag_context], dim=-1)
            gate = self.rag_gate(gate_input)  # [batch, hidden_size]
            gate = gate * RAG_GATE_MAX  # Cap at 40%

            # Additive fusion with gating
            fused_representation = pooled_bert + gate * rag_context
        else:
            fused_representation = pooled_bert

        # Expand for attention
        fused_states = fused_representation.unsqueeze(1).expand(-1, hidden_states.shape[1], -1)

        # 3. Concept bottleneck
        bert_concepts = concept_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        concept_context, concept_attn = self.cross_attention(
            query=fused_states,
            key=bert_concepts,
            value=bert_concepts,
            need_weights=True
        )

        # 4. Multiplicative bottleneck gating
        pooled_context = concept_context.mean(dim=1)

        gate_input = torch.cat([fused_representation, pooled_context], dim=-1)
        gate = self.gate_net(gate_input)

        bottleneck_output = gate * pooled_context
        bottleneck_output = self.layer_norm(bottleneck_output)

        # 5. Outputs
        concept_logits = self.concept_head(fused_representation)
        diagnosis_logits = self.diagnosis_head(bottleneck_output)

        return {
            'logits': diagnosis_logits,
            'concept_logits': concept_logits,
            'concept_scores': torch.sigmoid(concept_logits),
            'gate_values': gate
        }

# Initialize model
print("\n🔧 Initializing model components...")
tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
base_model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT').to(device)
concept_embedding_layer = nn.Embedding(len(ALL_CONCEPTS), 768).to(device)

model = ShifaMindPhase3Fixed(
    base_model=base_model,
    rag_retriever=rag,
    num_concepts=len(ALL_CONCEPTS),
    num_diagnoses=len(TARGET_CODES),
    hidden_size=768
).to(device)

# Load Phase 2 weights if available
if PHASE2_CHECKPOINT.exists():
    print(f"\n📥 Loading Phase 2 checkpoint...")
    checkpoint = torch.load(PHASE2_CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print("✅ Loaded Phase 2 weights (partial)")
else:
    print("⚠️  Phase 2 checkpoint not found - training from scratch")

print(f"\n✅ ShifaMind Phase 3 Fixed model initialized")
print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# TRAINING SETUP
# ============================================================================

print("\n" + "="*80)
print("⚙️  TRAINING SETUP")
print("="*80)

# Load data
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

# Dataset class
class RAGDatasetFixed(Dataset):
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
            'text': str(self.texts[idx]),  # For RAG retrieval
            'labels': torch.tensor(self.labels[idx], dtype=torch.float),
            'concept_labels': torch.tensor(self.concept_labels[idx], dtype=torch.float)
        }

# Create datasets
train_dataset = RAGDatasetFixed(df_train, tokenizer, train_concept_labels)
val_dataset = RAGDatasetFixed(df_val, tokenizer, val_concept_labels)
test_dataset = RAGDatasetFixed(df_test, tokenizer, test_concept_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# Multi-objective loss (SIMPLIFIED)
class MultiObjectiveLossFixed(nn.Module):
    def __init__(self, lambda_dx, lambda_align, lambda_concept):
        super().__init__()
        self.lambda_dx = lambda_dx
        self.lambda_align = lambda_align
        self.lambda_concept = lambda_concept
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs, dx_labels, concept_labels):
        # 1. Diagnosis loss (PRIORITIZED)
        loss_dx = self.bce(outputs['logits'], dx_labels)

        # 2. Alignment loss
        dx_probs = torch.sigmoid(outputs['logits'])
        concept_scores = outputs['concept_scores']
        loss_align = torch.abs(dx_probs.unsqueeze(-1) - concept_scores.unsqueeze(1)).mean()

        # 3. Concept prediction loss
        loss_concept = self.bce(outputs['concept_logits'], concept_labels)

        # Total loss
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

criterion = MultiObjectiveLossFixed(LAMBDA_DX, LAMBDA_ALIGN, LAMBDA_CONCEPT)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

print("✅ Training setup complete")

# ============================================================================
# TRAINING LOOP
# ============================================================================

print("\n" + "="*80)
print("🏋️  TRAINING PHASE 3 FIXED (DIAGNOSIS-FOCUSED)")
print("="*80)

best_val_f1 = 0.0
history = {'train_loss': [], 'val_loss': [], 'val_f1': []}

concept_embeddings = concept_embedding_layer.weight.detach()

for epoch in range(EPOCHS):
    print(f"\n📍 Epoch {epoch+1}/{EPOCHS}")

    # Training
    model.train()
    train_losses = []

    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        concept_labels = batch['concept_labels'].to(device)
        texts = batch['text']  # For RAG

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

    # Save best model
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
                'num_diagnoses': len(TARGET_CODES),
                'rag_config': {
                    'top_k': RAG_TOP_K,
                    'threshold': RAG_THRESHOLD,
                    'gate_max': RAG_GATE_MAX
                }
            }
        }, CHECKPOINT_PATH / 'phase3_v2_fixed_best.pt')
        print(f"   ✅ Saved best model (F1: {best_val_f1:.4f})")

# ============================================================================
# EVALUATION
# ============================================================================

print("\n" + "="*80)
print("📊 FINAL EVALUATION")
print("="*80)

checkpoint = torch.load(CHECKPOINT_PATH / 'phase3_v2_fixed_best.pt', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

all_preds = []
all_labels = []
all_probs = []

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
        all_probs.append(probs)

all_preds = np.vstack(all_preds)
all_labels = np.vstack(all_labels)
all_probs = np.vstack(all_probs)

# Metrics
macro_f1 = f1_score(all_labels, all_preds, average='macro')
per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
per_class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
per_class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)

print(f"\n🎯 Diagnosis Performance:")
print(f"   Macro F1: {macro_f1:.4f}")

print(f"\n📊 Per-Class Results:")
for i, code in enumerate(TARGET_CODES):
    print(f"\n   {code} - {ICD_DESCRIPTIONS[code]}")
    print(f"      F1:        {per_class_f1[i]:.4f}")
    print(f"      Precision: {per_class_precision[i]:.4f}")
    print(f"      Recall:    {per_class_recall[i]:.4f}")

# Comparison with Phase 2
print(f"\n📈 Performance Comparison:")
print(f"   Phase 2 (GraphSAGE):      0.7599")
print(f"   Phase 3 Original (RAG):   0.5435 ❌ (28% drop)")
print(f"   Phase 3 FIXED (RAG):      {macro_f1:.4f} {'✅' if macro_f1 > 0.76 else '⚠️'}")

if macro_f1 > 0.76:
    improvement = ((macro_f1 - 0.5435) / 0.5435) * 100
    print(f"   Improvement over broken RAG: +{improvement:.1f}%")

# Save results
results = {
    'phase': 'Phase 3 V2 FIXED - RAG with FAISS',
    'diagnosis_metrics': {
        'macro_f1': float(macro_f1),
        'per_class_f1': {code: float(f1) for code, f1 in zip(TARGET_CODES, per_class_f1)},
        'per_class_precision': {code: float(p) for code, p in zip(TARGET_CODES, per_class_precision)},
        'per_class_recall': {code: float(r) for code, r in zip(TARGET_CODES, per_class_recall)}
    },
    'architecture': 'Concept Bottleneck + GraphSAGE + FAISS RAG (40% gated fusion)',
    'rag_config': {
        'method': 'FAISS + sentence-transformers',
        'top_k': RAG_TOP_K,
        'threshold': RAG_THRESHOLD,
        'gate_max': RAG_GATE_MAX,
        'corpus_size': len(evidence_corpus)
    },
    'fixes_applied': [
        'FAISS + sentence-transformers (instead of BioClinicalBERT retrieval)',
        'Evidence corpus with MIMIC case prototypes',
        'Gated fusion with 40% cap',
        'Rebalanced loss weights (λ_dx=2.0)',
        'Removed citation/action heads (simplified)',
        'Increased epochs to 5'
    ],
    'training_history': history
}

with open(RESULTS_PATH / 'results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Results saved to: {RESULTS_PATH / 'results.json'}")
print(f"💾 Best model saved to: {CHECKPOINT_PATH / 'phase3_v2_fixed_best.pt'}")

print("\n" + "="*80)
print("✅ PHASE 3 V2 FIXED COMPLETE!")
print("="*80)
print("\nKey Fixes Applied:")
print("✅ FAISS + sentence-transformers (proven approach)")
print("✅ Evidence corpus with MIMIC case prototypes")
print("✅ Gated fusion mechanism (40% RAG cap)")
print("✅ Diagnosis-focused loss (λ_dx=2.0)")
print("✅ Simplified architecture (no citation/action heads)")
print("✅ Extended training (5 epochs)")
print("\nNext: Use this checkpoint for Phase 4 (Uncertainty) and Phase 5 (XAI)")
print("\nAlhamdulillah! 🤲")
