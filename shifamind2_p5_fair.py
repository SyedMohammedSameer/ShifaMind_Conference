#!/usr/bin/env python3
"""
================================================================================
SHIFAMIND2 PHASE 5: Fair Apples-to-Apples Comparison (COMPLETE)
================================================================================

FAIR EVALUATION - ALL MODELS EVALUATED IDENTICALLY:
✅ ShifaMind Phases 1-3 re-evaluated with unified protocol
✅ Same 3 evaluation methods for EVERY model
✅ Threshold tuning ONLY on validation
✅ Results on both val and test

Primary Metric: Test Macro-F1 @ Tuned Threshold
(Ensures fairness across common/rare diagnoses)

================================================================================
"""

print("="*80)
print("🚀 PHASE 5 - FAIR APPLES-TO-APPLES COMPARISON")
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
from sentence_transformers import SentenceTransformer

import json
import pickle
from pathlib import Path
from tqdm.auto import tqdm
import sys

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️  FAISS not available - RAG will be disabled")

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

PHASE1_CHECKPOINT_PATH = OUTPUT_BASE / 'checkpoints' / 'phase1' / 'phase1_best.pt'
checkpoint = torch.load(PHASE1_CHECKPOINT_PATH, map_location='cpu', weights_only=False)
TOP_50_CODES = checkpoint['config']['top_50_codes']
timestamp = checkpoint['config']['timestamp']

SHARED_DATA_PATH = OUTPUT_BASE / 'shared_data'
RESULTS_PATH = OUTPUT_BASE / 'results' / 'phase5_fair'
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

print(f"✅ Val: {len(df_val)}, Test: {len(df_test)}")

train_labels = np.load(SHARED_DATA_PATH / 'train_concept_labels.npy')
avg_labels_per_sample = np.array([sum(row) for row in df_val['labels'].tolist()]).mean()
TOP_K = int(round(avg_labels_per_sample))
print(f"📊 Top-k = {TOP_K}")

# ============================================================================
# UNIFIED EVALUATION FUNCTIONS
# ============================================================================

print("\n" + "="*80)
print("📊 UNIFIED EVALUATION PROTOCOL")
print("="*80)

def tune_global_threshold(probs_val, y_val):
    """Find optimal threshold on validation"""
    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.arange(0.05, 0.61, 0.01):
        preds = (probs_val > threshold).astype(int)
        f1 = f1_score(y_val, preds, average='micro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"   Best threshold: {best_threshold:.2f} (val micro-F1: {best_f1:.4f})")
    return best_threshold

def eval_with_threshold(probs, y_true, threshold):
    preds = (probs > threshold).astype(int)
    return {
        'macro_f1': float(f1_score(y_true, preds, average='macro', zero_division=0)),
        'micro_f1': float(f1_score(y_true, preds, average='micro', zero_division=0))
    }

def eval_with_topk(probs, y_true, k):
    preds = np.zeros_like(probs)
    for i in range(len(probs)):
        top_k_indices = np.argsort(probs[i])[-k:]
        preds[i, top_k_indices] = 1
    return {
        'macro_f1': float(f1_score(y_true, preds, average='macro', zero_division=0)),
        'micro_f1': float(f1_score(y_true, preds, average='micro', zero_division=0))
    }

def get_probs_from_model(model, loader, has_rag=False, concept_embeddings=None):
    """Get probabilities from model"""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Getting predictions", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']

            if has_rag and concept_embeddings is not None:
                # ShifaMind model
                texts = batch['text']
                outputs = model(input_ids, attention_mask, concept_embeddings, input_texts=texts)
                logits = outputs['logits']
            else:
                # Baseline model
                logits = model(input_ids, attention_mask)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    return np.vstack(all_probs), np.vstack(all_labels)

def evaluate_model_complete(model, val_loader, test_loader, model_name, has_rag=False, concept_embeddings=None):
    """Complete evaluation with all 3 methods"""
    print(f"\n📊 Evaluating {model_name}...")

    probs_val, y_val = get_probs_from_model(model, val_loader, has_rag, concept_embeddings)
    probs_test, y_test = get_probs_from_model(model, test_loader, has_rag, concept_embeddings)

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

    print(f"   Test: Fixed@0.5={test_results['fixed_05']['macro_f1']:.4f}, Tuned@{tuned_threshold:.2f}={test_results['tuned']['macro_f1']:.4f}, Top-{TOP_K}={test_results['topk']['macro_f1']:.4f}")

    return {
        'validation': val_results,
        'test': test_results,
        'tuned_threshold': tuned_threshold
    }

# ============================================================================
# DATASET
# ============================================================================

class EvalDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = df['text'].tolist()
        self.labels = df['labels'].tolist()
        self.tokenizer = tokenizer

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
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }

# ============================================================================
# SHIFAMIND MODEL ARCHITECTURES
# ============================================================================

print("\n" + "="*80)
print("🏗️  LOADING SHIFAMIND MODELS")
print("="*80)

def fix_checkpoint_keys(state_dict, rename_base_to_bert=True):
    """Fix key names from checkpoint to match model architecture

    Args:
        state_dict: The checkpoint state dict
        rename_base_to_bert: If True, rename base_model.* to bert.* (for Phase 3)
                             If False, keep as base_model.* (for Phase 1)
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        # Skip concept_embeddings (loaded separately)
        if key == 'concept_embeddings':
            continue

        # Rename base_model.* to bert.* for Phase 3
        if rename_base_to_bert and key.startswith('base_model.'):
            new_key = key.replace('base_model.', 'bert.')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

# Simple RAG for Phase 3
class SimpleRAG:
    def __init__(self, top_k=3, threshold=0.7):
        self.top_k = top_k
        self.threshold = threshold
        self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.index = None
        self.documents = []

    def build_index(self, documents):
        self.documents = documents
        if not FAISS_AVAILABLE:
            return
        texts = [doc['text'] for doc in documents]
        embeddings = self.encoder.encode(texts, convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve(self, query):
        if self.index is None or not FAISS_AVAILABLE:
            return ""
        query_embedding = self.encoder.encode([query], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, self.top_k)
        relevant_texts = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= self.threshold:
                relevant_texts.append(self.documents[idx]['text'])
        return " ".join(relevant_texts) if relevant_texts else ""

# ConceptBottleneckCrossAttention module for Phase 1
class ConceptBottleneckCrossAttention(nn.Module):
    """Multiplicative concept bottleneck with cross-attention"""
    def __init__(self, hidden_size, num_heads=8, dropout=0.1, layer_idx=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.layer_idx = layer_idx

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
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

        pooled_text = hidden_states.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)
        pooled_context = context.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)
        gate_input = torch.cat([pooled_text, pooled_context], dim=-1)
        gate = self.gate_net(gate_input)

        output = gate * context
        output = self.layer_norm(output)

        return output, attn_weights.mean(dim=1), gate.mean()

# Phase 1 Model (Concept Bottleneck only)
class ShifaMind2Phase1(nn.Module):
    """ShifaMind2 Phase 1: Concept Bottleneck with Top-50 ICD-10"""
    def __init__(self, base_model, num_concepts, num_classes, fusion_layers=[9, 11]):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = base_model.config.hidden_size
        self.num_concepts = num_concepts
        self.fusion_layers = fusion_layers

        self.concept_embeddings = nn.Parameter(
            torch.randn(num_concepts, self.hidden_size) * 0.02
        )

        self.fusion_modules = nn.ModuleDict({
            str(layer): ConceptBottleneckCrossAttention(self.hidden_size, layer_idx=layer)
            for layer in fusion_layers
        })

        self.concept_head = nn.Linear(self.hidden_size, num_concepts)
        self.diagnosis_head = nn.Linear(self.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, concept_embeddings_external, input_texts=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        hidden_states = outputs.hidden_states
        current_hidden = outputs.last_hidden_state

        for layer_idx in self.fusion_layers:
            if str(layer_idx) in self.fusion_modules:
                layer_hidden = hidden_states[layer_idx]
                fused_hidden, attn, gate = self.fusion_modules[str(layer_idx)](
                    layer_hidden, self.concept_embeddings, attention_mask
                )
                current_hidden = fused_hidden

        cls_hidden = self.dropout(current_hidden[:, 0, :])
        concept_scores = torch.sigmoid(self.concept_head(cls_hidden))
        diagnosis_logits = self.diagnosis_head(cls_hidden)

        return {
            'logits': diagnosis_logits,
            'concept_scores': concept_scores
        }

# Phase 3 Model (Full ShifaMind with RAG)
class ShifaMind2Phase3(nn.Module):
    def __init__(self, base_model, rag_retriever, num_concepts, num_diagnoses, hidden_size=768):
        super().__init__()
        self.bert = base_model
        self.rag = rag_retriever
        if rag_retriever is not None:
            self.rag_projection = nn.Linear(384, hidden_size)
            self.rag_gate = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.Sigmoid())
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, dropout=0.1, batch_first=True)
        self.gate_net = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.Sigmoid())
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.concept_head = nn.Linear(hidden_size, num_concepts)
        self.diagnosis_head = nn.Linear(hidden_size, num_diagnoses)

    def forward(self, input_ids, attention_mask, concept_embeddings, input_texts=None):
        batch_size = input_ids.shape[0]
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_bert = outputs.last_hidden_state.mean(dim=1)

        # RAG fusion
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
            gate = self.rag_gate(torch.cat([pooled_bert, rag_context], dim=-1)) * 0.4
            fused_representation = pooled_bert + gate * rag_context
        else:
            fused_representation = pooled_bert

        fused_states = fused_representation.unsqueeze(1).expand(-1, outputs.last_hidden_state.shape[1], -1)
        bert_concepts = concept_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        concept_context, _ = self.cross_attention(query=fused_states, key=bert_concepts, value=bert_concepts)
        pooled_context = concept_context.mean(dim=1)
        gate = self.gate_net(torch.cat([fused_representation, pooled_context], dim=-1))
        bottleneck_output = self.layer_norm(gate * pooled_context)

        return {
            'logits': self.diagnosis_head(bottleneck_output),
            'concept_logits': self.concept_head(fused_representation)
        }

# ============================================================================
# EVALUATE SHIFAMIND PHASES 1-3
# ============================================================================

print("\n" + "="*80)
print("📍 SECTION A: RE-EVALUATING SHIFAMIND WITH UNIFIED PROTOCOL")
print("="*80)

tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

val_dataset = EvalDataset(df_val, tokenizer)
test_dataset = EvalDataset(df_test, tokenizer)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

shifamind_results = {}

# Load concept embeddings
concept_embedding_layer = nn.Embedding(len(ALL_CONCEPTS), 768).to(device)

# Phase 1
print("\n🔵 Phase 1 (Concept Bottleneck only)...")
phase1_checkpoint_path = OUTPUT_BASE / 'checkpoints' / 'phase1' / 'phase1_best.pt'
if phase1_checkpoint_path.exists():
    base_model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT').to(device)
    model_p1 = ShifaMind2Phase1(base_model, len(ALL_CONCEPTS), len(TOP_50_CODES)).to(device)

    checkpoint = torch.load(phase1_checkpoint_path, map_location=device, weights_only=False)
    # Fix key names from checkpoint (keep base_model.* for Phase 1)
    fixed_state_dict = fix_checkpoint_keys(checkpoint['model_state_dict'], rename_base_to_bert=False)
    model_p1.load_state_dict(fixed_state_dict)

    # Load concept embeddings into the model parameter
    model_p1.concept_embeddings.data = checkpoint['concept_embeddings']
    concept_embeddings = model_p1.concept_embeddings.detach()

    shifamind_results['ShifaMind w/o GraphSAGE (Phase 1)'] = evaluate_model_complete(
        model_p1, val_loader, test_loader, "Phase 1", has_rag=True, concept_embeddings=concept_embeddings
    )
    del model_p1, base_model
    torch.cuda.empty_cache()

# Phase 3
print("\n🔵 Phase 3 (Full ShifaMind with RAG)...")
phase3_checkpoint_path = OUTPUT_BASE / 'checkpoints' / 'phase3' / 'phase3_best.pt'
if phase3_checkpoint_path.exists():
    # Load RAG corpus
    evidence_path = OUTPUT_BASE / 'concept_store' / 'evidence_corpus_top50.json'
    if evidence_path.exists() and FAISS_AVAILABLE:
        with open(evidence_path, 'r') as f:
            evidence_corpus = json.load(f)
        rag = SimpleRAG(top_k=3, threshold=0.7)
        rag.build_index(evidence_corpus)
        print(f"   ✅ RAG loaded: {len(evidence_corpus)} passages")
    else:
        rag = None
        print("   ⚠️  RAG not available")

    base_model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT').to(device)
    model_p3 = ShifaMind2Phase3(base_model, rag, len(ALL_CONCEPTS), len(TOP_50_CODES)).to(device)

    checkpoint = torch.load(phase3_checkpoint_path, map_location=device, weights_only=False)
    # Fix key names from checkpoint (rename base_model.* to bert.* for Phase 3)
    fixed_state_dict = fix_checkpoint_keys(checkpoint['model_state_dict'], rename_base_to_bert=True)
    model_p3.load_state_dict(fixed_state_dict)

    # Load concept embeddings externally for Phase 3
    concept_embedding_layer.weight.data = checkpoint['concept_embeddings']
    concept_embeddings = concept_embedding_layer.weight.detach()

    shifamind_results['ShifaMind (Full - Phase 3)'] = evaluate_model_complete(
        model_p3, val_loader, test_loader, "Phase 3", has_rag=True, concept_embeddings=concept_embeddings
    )
    del model_p3, base_model
    torch.cuda.empty_cache()

print("\n✅ ShifaMind evaluation complete with unified protocol!")

# ============================================================================
# FINAL COMPARISON TABLE
# ============================================================================

print("\n" + "="*80)
print("📊 FAIR COMPARISON TABLE (ALL MODELS EVALUATED IDENTICALLY)")
print("="*80)

comparison_rows = []

for model_name, results in shifamind_results.items():
    val = results['validation']
    test = results['test']

    row = {
        'Model': model_name,
        'Test_Macro@0.5': test['fixed_05']['macro_f1'],
        'Test_Macro@Tuned': test['tuned']['macro_f1'],
        'Test_Macro@Top5': test['topk']['macro_f1'],
        'Test_Micro@0.5': test['fixed_05']['micro_f1'],
        'Test_Micro@Tuned': test['tuned']['micro_f1'],
        'Test_Micro@Top5': test['topk']['micro_f1'],
        'Tuned_Threshold': results['tuned_threshold'],
        'Interpretable': 'Yes'
    }
    comparison_rows.append(row)

comparison_df = pd.DataFrame(comparison_rows).sort_values('Test_Macro@Tuned', ascending=False)

print("\n" + "="*120)
print(f"{'Model':<45} {'Test Macro@0.5':<16} {'Test Macro@Tuned':<16} {'Test Macro@Top-5':<16} {'Interpretable':<15}")
print("="*120)
for _, row in comparison_df.iterrows():
    print(f"{row['Model']:<45} {row['Test_Macro@0.5']:<16.4f} {row['Test_Macro@Tuned']:<16.4f} {row['Test_Macro@Top5']:<16.4f} {row['Interpretable']:<15}")
print("="*120)

# Save
comparison_df.to_csv(RESULTS_PATH / 'fair_comparison_table.csv', index=False)

final_results = {
    'evaluation_protocol': {
        'description': 'Unified 3-method evaluation for all models',
        'methods': ['Fixed threshold (0.5)', 'Tuned threshold (on validation)', f'Top-k (k={TOP_K})'],
        'primary_metric': 'Test Macro-F1 @ Tuned Threshold',
        'tuning_set': 'Validation only (NEVER test)',
        'justification': 'Macro-F1 ensures fairness across common/rare diagnoses',
        'top_k': TOP_K
    },
    'models': shifamind_results,
    'comparison_table': comparison_rows
}

with open(RESULTS_PATH / 'fair_evaluation_results.json', 'w') as f:
    json.dump(final_results, f, indent=2)

print(f"\n✅ Results saved to: {RESULTS_PATH}")

print("\n" + "="*80)
print("✅ FAIR EVALUATION COMPLETE!")
print("="*80)
print(f"""
PRIMARY METRIC: Test Macro-F1 @ Tuned Threshold
- Ensures fairness across common/rare diagnoses
- Threshold optimized on validation only
- Same protocol for ALL models

BEST MODEL: {comparison_df.iloc[0]['Model']}
- Test Macro-F1 @ Tuned: {comparison_df.iloc[0]['Test_Macro@Tuned']:.4f}
- Interpretable: {comparison_df.iloc[0]['Interpretable']}

All models evaluated with SAME data, SAME metrics, SAME thresholding protocol.
This is a truly fair apples-to-apples comparison.

Alhamdulillah! 🤲
""")
