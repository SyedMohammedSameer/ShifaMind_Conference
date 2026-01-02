#!/usr/bin/env python3
"""
================================================================================
SHIFAMIND2 PHASE 2: GraphSAGE + Concept Linker (Top-50 ICD-10)
================================================================================
Author: Mohammed Sameer Syed
University of Arizona - MS in AI Capstone

CHANGES FROM SHIFAMIND1_P2:
1. ✅ Uses Top-50 ICD-10 codes from Phase 1
2. ✅ Loads from SAME run folder (no new timestamp)
3. ✅ Fresh concept store (no reuse from old runs)
4. ✅ Medical ontology with Top-50 diagnoses
5. ✅ Same 120 global concepts

Architecture:
1. BioClinicalBERT encoder (from Phase 1)
2. GraphSAGE encoder for medical ontology with Top-50 codes
3. Concept Linker for entity recognition
4. Enhanced concept embeddings from knowledge graph
5. Ontology-aware concept bottleneck

Target Metrics:
- Diagnosis F1: >0.75
- Concept F1: >0.75 (improved with ontology)

================================================================================
"""

print("="*80)
print("🚀 SHIFAMIND2 PHASE 2 - GRAPHSAGE + TOP-50 ONTOLOGY")
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
from torch_geometric.data import Data

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from transformers import (
    AutoTokenizer, AutoModel,
    get_linear_schedule_with_warmup
)

import json
import pickle
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import networkx as nx
import re
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
# CONFIGURATION: LOAD FROM PHASE 1
# ============================================================================

print("\n" + "="*80)
print("⚙️  CONFIGURATION: LOADING FROM PHASE 1")
print("="*80)

# Find the most recent Phase 1 run
BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')
SHIFAMIND2_BASE = BASE_PATH / '10_ShifaMind'

# List all run folders
run_folders = sorted([d for d in SHIFAMIND2_BASE.glob('run_*') if d.is_dir()], reverse=True)

if not run_folders:
    print("❌ No Phase 1 run found! Please run shifamind2_p1.py first.")
    sys.exit(1)

# Use most recent run
OUTPUT_BASE = run_folders[0]
print(f"📁 Using run folder: {OUTPUT_BASE.name}")

# Verify Phase 1 checkpoint exists
PHASE1_CHECKPOINT = OUTPUT_BASE / 'checkpoints' / 'phase1' / 'phase1_best.pt'
if not PHASE1_CHECKPOINT.exists():
    print(f"❌ Phase 1 checkpoint not found at: {PHASE1_CHECKPOINT}")
    print("   Please run shifamind2_p1.py first.")
    sys.exit(1)

# Load Phase 1 config
checkpoint = torch.load(PHASE1_CHECKPOINT, map_location='cpu', weights_only=False)
phase1_config = checkpoint['config']
TOP_50_CODES = phase1_config['top_50_codes']
timestamp = phase1_config['timestamp']

print(f"✅ Loaded Phase 1 config:")
print(f"   Timestamp: {timestamp}")
print(f"   Top-50 codes: {len(TOP_50_CODES)} diagnoses")
print(f"   Num concepts: {phase1_config['num_concepts']}")

# Paths (same run folder)
SHARED_DATA_PATH = OUTPUT_BASE / 'shared_data'
CHECKPOINT_PATH = OUTPUT_BASE / 'checkpoints' / 'phase2'
RESULTS_PATH = OUTPUT_BASE / 'results' / 'phase2'
CONCEPT_STORE_PATH = OUTPUT_BASE / 'concept_store'

# Create Phase 2 directories
for path in [CHECKPOINT_PATH, RESULTS_PATH, CONCEPT_STORE_PATH]:
    path.mkdir(parents=True, exist_ok=True)

print(f"\n📁 Phase 2 Paths:")
print(f"   Checkpoint: {CHECKPOINT_PATH}")
print(f"   Results: {RESULTS_PATH}")
print(f"   Concept Store: {CONCEPT_STORE_PATH}")

# Load concept list
with open(SHARED_DATA_PATH / 'concept_list.json', 'r') as f:
    ALL_CONCEPTS = json.load(f)

print(f"\n🧠 Concepts: {len(ALL_CONCEPTS)} clinical concepts")

# GraphSAGE hyperparameters
GRAPH_HIDDEN_DIM = 256
GRAPH_LAYERS = 2
GRAPHSAGE_AGGREGATION = 'mean'

# Training hyperparameters
LAMBDA_DX = 1.0
LAMBDA_ALIGN = 0.5
LAMBDA_CONCEPT = 0.3
LEARNING_RATE = 1e-5
EPOCHS = 3

print(f"\n🕸️  GraphSAGE Config:")
print(f"   Hidden Dim: {GRAPH_HIDDEN_DIM}")
print(f"   Layers: {GRAPH_LAYERS}")
print(f"   Aggregation: {GRAPHSAGE_AGGREGATION}")

# ============================================================================
# BUILD MEDICAL KNOWLEDGE GRAPH (TOP-50 CODES)
# ============================================================================

print("\n" + "="*80)
print("🕸️  BUILDING MEDICAL KNOWLEDGE GRAPH (TOP-50)")
print("="*80)

def build_medical_ontology_top50(top_50_codes, all_concepts):
    """
    Build medical knowledge graph from Top-50 ICD-10 codes and clinical concepts

    Structure:
    - 50 diagnosis nodes (ICD-10 codes)
    - 120 concept nodes (clinical concepts)
    - Edges: concept -> diagnosis (indicates relationship)
    - Hierarchical/co-occurrence edges between diagnoses
    """
    print("\n📊 Building knowledge graph...")

    G = nx.DiGraph()

    # Add diagnosis nodes
    for code in top_50_codes:
        G.add_node(code, node_type='diagnosis')

    # Add concept nodes
    for concept in all_concepts:
        G.add_node(concept, node_type='concept')

    # Build concept-diagnosis edges based on keyword matching
    # (In production, this would use UMLS/SNOMED-CT mappings)
    print("\n🔗 Creating concept-diagnosis edges...")

    # Medical domain knowledge: concept -> likely diagnoses
    concept_diagnosis_patterns = {
        # Respiratory
        'pneumonia': ['J', 'J1', 'J18', 'J44', 'J96'],  # Respiratory codes
        'lung': ['J', 'C34'],
        'respiratory': ['J'],
        'dyspnea': ['J', 'I50', 'I25'],
        'cough': ['J'],
        'hypoxia': ['J', 'I50'],

        # Cardiac
        'cardiac': ['I'],
        'heart': ['I'],
        'failure': ['I50'],
        'infarction': ['I21', 'I22', 'I25'],
        'ischemia': ['I', 'G45'],
        'edema': ['I50', 'R60'],

        # Infection/Sepsis
        'sepsis': ['A', 'R65'],
        'infection': ['A', 'B', 'J', 'N39'],
        'fever': ['A', 'R50'],
        'bacteremia': ['A'],

        # Renal
        'renal': ['N'],
        'kidney': ['N'],
        'creatinine': ['N17', 'N18', 'N19'],

        # Metabolic
        'diabetes': ['E', 'E10', 'E11'],
        'hyperglycemia': ['E', 'R73'],
        'hypoglycemia': ['E'],

        # GI
        'gastrointestinal': ['K'],
        'abdominal': ['K', 'R10'],
        'nausea': ['R11'],
        'vomiting': ['R11'],

        # Neuro
        'confusion': ['F', 'R41'],
        'altered': ['F', 'R40', 'R41'],
        'stroke': ['I6', 'G45'],

        # Hematologic
        'anemia': ['D', 'D50', 'D51'],
        'thrombocytopenia': ['D69'],
        'hemorrhage': ['I', 'K', 'R58'],
    }

    edges_added = 0
    for concept in all_concepts:
        concept_lower = concept.lower()
        # Check direct matches
        if concept_lower in concept_diagnosis_patterns:
            patterns = concept_diagnosis_patterns[concept_lower]
            for code in top_50_codes:
                for pattern in patterns:
                    if code.startswith(pattern):
                        G.add_edge(concept, code, edge_type='indicates', weight=1.0)
                        edges_added += 1
                        break

    print(f"   Added {edges_added} concept-diagnosis edges")

    # Add hierarchical relationships between diagnoses
    # Group by ICD-10 chapter (first letter)
    print("\n🔗 Creating diagnosis similarity edges...")
    chapter_groups = defaultdict(list)
    for code in top_50_codes:
        chapter = code[0] if code else 'X'
        chapter_groups[chapter].append(code)

    similarity_edges = 0
    for chapter, codes in chapter_groups.items():
        # Codes in same chapter are related
        for i, code1 in enumerate(codes):
            for code2 in codes[i+1:]:
                G.add_edge(code1, code2, edge_type='similar_chapter', weight=0.5)
                G.add_edge(code2, code1, edge_type='similar_chapter', weight=0.5)
                similarity_edges += 2

    print(f"   Added {similarity_edges} diagnosis similarity edges")

    # Add concept similarity edges (common symptom/finding)
    common_symptom_groups = [
        ['fever', 'infection', 'sepsis'],
        ['dyspnea', 'hypoxia', 'respiratory'],
        ['chest', 'cardiac', 'heart'],
        ['pain', 'abdominal'],
        ['confusion', 'altered', 'neurologic'],
    ]

    concept_edges = 0
    for group in common_symptom_groups:
        valid_group = [c for c in group if c in all_concepts]
        for i, c1 in enumerate(valid_group):
            for c2 in valid_group[i+1:]:
                if c1 in G and c2 in G:
                    G.add_edge(c1, c2, edge_type='related_symptom', weight=0.3)
                    G.add_edge(c2, c1, edge_type='related_symptom', weight=0.3)
                    concept_edges += 2

    print(f"   Added {concept_edges} concept similarity edges")

    print(f"\n✅ Knowledge graph built:")
    print(f"   Nodes: {G.number_of_nodes()}")
    print(f"   Edges: {G.number_of_edges()}")
    print(f"   - Diagnosis nodes: {len([n for n in G.nodes if G.nodes[n].get('node_type') == 'diagnosis'])}")
    print(f"   - Concept nodes: {len([n for n in G.nodes if G.nodes[n].get('node_type') == 'concept'])}")

    return G

ontology_graph = build_medical_ontology_top50(TOP_50_CODES, ALL_CONCEPTS)

# Convert NetworkX to PyTorch Geometric format
def nx_to_pyg(G, concept_list):
    """Convert NetworkX graph to PyTorch Geometric Data object"""

    all_nodes = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}

    edge_index = []
    edge_attr = []
    for u, v, data in G.edges(data=True):
        edge_index.append([node_to_idx[u], node_to_idx[v]])
        edge_attr.append(data.get('weight', 1.0))

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(-1)

    num_nodes = len(all_nodes)
    x = torch.randn(num_nodes, GRAPH_HIDDEN_DIM)

    node_types = []
    for node in all_nodes:
        if G.nodes[node].get('node_type') == 'diagnosis':
            node_types.append(0)
        else:
            node_types.append(1)
    node_type_mask = torch.tensor(node_types, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.node_type_mask = node_type_mask
    data.node_to_idx = node_to_idx
    data.idx_to_node = {idx: node for node, idx in node_to_idx.items()}

    return data

graph_data = nx_to_pyg(ontology_graph, ALL_CONCEPTS)
print(f"\n✅ Converted to PyTorch Geometric:")
print(f"   Nodes: {graph_data.x.shape[0]}")
print(f"   Edges: {graph_data.edge_index.shape[1]}")
print(f"   Node features: {graph_data.x.shape[1]}")

# Save graph
import pickle as pkl
with open(CONCEPT_STORE_PATH / 'medical_ontology_top50.gpickle', 'wb') as f:
    pkl.dump(ontology_graph, f)

print(f"💾 Saved ontology to: {CONCEPT_STORE_PATH / 'medical_ontology_top50.gpickle'}")

# ============================================================================
# GRAPHSAGE ENCODER
# ============================================================================

print("\n" + "="*80)
print("🏗️  GRAPHSAGE ENCODER")
print("="*80)

class GraphSAGEEncoder(nn.Module):
    """GraphSAGE encoder for learning concept embeddings from medical ontology"""
    def __init__(self, in_channels, hidden_channels, num_layers=2, aggr='mean'):
        super().__init__()

        self.num_layers = num_layers
        self.convs = nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))

        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))

        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = self.dropout(x)

        return x

graph_encoder = GraphSAGEEncoder(
    in_channels=GRAPH_HIDDEN_DIM,
    hidden_channels=GRAPH_HIDDEN_DIM,
    num_layers=GRAPH_LAYERS,
    aggr=GRAPHSAGE_AGGREGATION
).to(device)

print(f"✅ GraphSAGE encoder initialized")
print(f"   Parameters: {sum(p.numel() for p in graph_encoder.parameters()):,}")

# ============================================================================
# ENHANCED CONCEPT BOTTLENECK (PHASE 1 + GRAPHSAGE)
# ============================================================================

print("\n" + "="*80)
print("🏗️  LOADING PHASE 1 + ADDING GRAPHSAGE")
print("="*80)

class ShifaMind2Phase2(nn.Module):
    """
    ShifaMind2 Phase 2: Enhanced with GraphSAGE-enriched concepts (Top-50)

    Architecture:
    1. BioClinicalBERT encoder (from Phase 1)
    2. GraphSAGE encoder for ontology-based concept embeddings
    3. Concept bottleneck with cross-attention
    4. Multi-head outputs (diagnosis Top-50, concepts)
    """
    def __init__(self, base_model, graph_encoder, graph_data, num_concepts, num_diagnoses, hidden_size=768):
        super().__init__()

        self.bert = base_model
        self.graph_encoder = graph_encoder
        self.hidden_size = hidden_size
        self.num_concepts = num_concepts
        self.num_diagnoses = num_diagnoses

        # Store graph data
        self.register_buffer('graph_x', graph_data.x)
        self.register_buffer('graph_edge_index', graph_data.edge_index)
        self.graph_node_to_idx = graph_data.node_to_idx
        self.graph_idx_to_node = graph_data.idx_to_node

        # Concept embedding fusion (combine BERT + GraphSAGE)
        self.concept_fusion = nn.Sequential(
            nn.Linear(hidden_size + GRAPH_HIDDEN_DIM, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Cross-attention for concept bottleneck
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Gating network (multiplicative bottleneck)
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

        # Output heads
        self.concept_head = nn.Linear(hidden_size, num_concepts)
        self.diagnosis_head = nn.Linear(hidden_size, num_diagnoses)

    def get_graph_concept_embeddings(self, concept_indices):
        """Get GraphSAGE embeddings for specific concepts"""
        graph_embeddings = self.graph_encoder(self.graph_x, self.graph_edge_index)

        concept_embeds = []
        for concept in ALL_CONCEPTS:
            if concept in self.graph_node_to_idx:
                idx = self.graph_node_to_idx[concept]
                concept_embeds.append(graph_embeddings[idx])
            else:
                concept_embeds.append(torch.zeros(GRAPH_HIDDEN_DIM, device=self.graph_x.device))

        return torch.stack(concept_embeds)

    def forward(self, input_ids, attention_mask, concept_embeddings_bert):
        """
        Forward pass with GraphSAGE-enhanced concepts

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            concept_embeddings_bert: [num_concepts, hidden_size]
        """
        batch_size = input_ids.shape[0]

        # 1. Encode text with BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        # 2. Get GraphSAGE concept embeddings
        graph_concept_embeds = self.get_graph_concept_embeddings(None)

        # 3. Fuse BERT + GraphSAGE concept embeddings
        bert_concepts = concept_embeddings_bert.unsqueeze(0).expand(batch_size, -1, -1)
        graph_concepts = graph_concept_embeds.unsqueeze(0).expand(batch_size, -1, -1)

        fused_input = torch.cat([bert_concepts, graph_concepts], dim=-1)
        enhanced_concepts = self.concept_fusion(fused_input)

        # 4. Cross-attention: text attends to enhanced concepts
        context, attn_weights = self.cross_attention(
            query=hidden_states,
            key=enhanced_concepts,
            value=enhanced_concepts,
            need_weights=True
        )

        # 5. Multiplicative bottleneck gating
        pooled_text = hidden_states.mean(dim=1)
        pooled_context = context.mean(dim=1)

        gate_input = torch.cat([pooled_text, pooled_context], dim=-1)
        gate = self.gate_net(gate_input)

        bottleneck_output = gate * pooled_context
        bottleneck_output = self.layer_norm(bottleneck_output)

        # 6. Output heads
        concept_logits = self.concept_head(pooled_text)
        diagnosis_logits = self.diagnosis_head(bottleneck_output)

        return {
            'logits': diagnosis_logits,
            'concept_logits': concept_logits,
            'concept_scores': torch.sigmoid(concept_logits),
            'gate_values': gate,
            'attention_weights': attn_weights,
            'bottleneck_output': bottleneck_output
        }

# Initialize model
print("\n🔧 Initializing BioClinicalBERT...")
tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
base_model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT').to(device)

concept_embedding_layer = nn.Embedding(len(ALL_CONCEPTS), 768).to(device)

model = ShifaMind2Phase2(
    base_model=base_model,
    graph_encoder=graph_encoder,
    graph_data=graph_data,
    num_concepts=len(ALL_CONCEPTS),
    num_diagnoses=len(TOP_50_CODES),
    hidden_size=768
).to(device)

# Load Phase 1 weights if available
print(f"\n📥 Loading Phase 1 checkpoint...")
checkpoint = torch.load(PHASE1_CHECKPOINT, map_location=device, weights_only=False)
try:
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print("✅ Loaded Phase 1 weights (partial)")
except Exception as e:
    print(f"⚠️  Could not load Phase 1 weights: {e}")

print(f"\n✅ ShifaMind2 Phase 2 model initialized")
print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# TRAINING SETUP
# ============================================================================

print("\n" + "="*80)
print("⚙️  TRAINING SETUP")
print("="*80)

# Load data splits from Phase 1
with open(SHARED_DATA_PATH / 'train_split.pkl', 'rb') as f:
    df_train = pickle.load(f)
with open(SHARED_DATA_PATH / 'val_split.pkl', 'rb') as f:
    df_val = pickle.load(f)
with open(SHARED_DATA_PATH / 'test_split.pkl', 'rb') as f:
    df_test = pickle.load(f)

train_concept_labels = np.load(SHARED_DATA_PATH / 'train_concept_labels.npy')
val_concept_labels = np.load(SHARED_DATA_PATH / 'val_concept_labels.npy')
test_concept_labels = np.load(SHARED_DATA_PATH / 'test_concept_labels.npy')

print(f"\n✅ Loaded data splits:")
print(f"   Train: {len(df_train):,}")
print(f"   Val: {len(df_val):,}")
print(f"   Test: {len(df_test):,}")

# Dataset class
class ConceptDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, concept_labels):
        self.texts = texts
        self.labels = labels
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
            'labels': torch.tensor(self.labels[idx], dtype=torch.float),
            'concept_labels': torch.tensor(self.concept_labels[idx], dtype=torch.float)
        }

train_dataset = ConceptDataset(df_train['text'].tolist(), df_train['labels'].tolist(), tokenizer, train_concept_labels)
val_dataset = ConceptDataset(df_val['text'].tolist(), df_val['labels'].tolist(), tokenizer, val_concept_labels)
test_dataset = ConceptDataset(df_test['text'].tolist(), df_test['labels'].tolist(), tokenizer, test_concept_labels)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

# Loss function
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

print("✅ Training setup complete")

# ============================================================================
# TRAINING LOOP
# ============================================================================

print("\n" + "="*80)
print("🏋️  TRAINING PHASE 2 (GRAPHSAGE-ENHANCED)")
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

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask, concept_embeddings)
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

            outputs = model(input_ids, attention_mask, concept_embeddings)
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
            'graph_data': graph_data,
            'concept_embeddings': concept_embeddings,
            'config': {
                'num_concepts': len(ALL_CONCEPTS),
                'num_diagnoses': len(TOP_50_CODES),
                'graph_hidden_dim': GRAPH_HIDDEN_DIM,
                'graph_layers': GRAPH_LAYERS,
                'top_50_codes': TOP_50_CODES,
                'timestamp': timestamp
            }
        }, CHECKPOINT_PATH / 'phase2_best.pt')
        print(f"   ✅ Saved best model (F1: {best_val_f1:.4f})")

# ============================================================================
# EVALUATION
# ============================================================================

print("\n" + "="*80)
print("📊 FINAL EVALUATION")
print("="*80)

checkpoint = torch.load(CHECKPOINT_PATH / 'phase2_best.pt', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask, concept_embeddings)

        preds = (torch.sigmoid(outputs['logits']) > 0.5).cpu().numpy()

        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

all_preds = np.vstack(all_preds)
all_labels = np.vstack(all_labels)

macro_f1 = f1_score(all_labels, all_preds, average='macro')
micro_f1 = f1_score(all_labels, all_preds, average='micro')
per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

print(f"\n🎯 Diagnosis Performance (Top-50):")
print(f"   Macro F1:    {macro_f1:.4f}")
print(f"   Micro F1:    {micro_f1:.4f}")

print(f"\n📊 Top-10 Best Performing Diagnoses:")
top_10_best = sorted(zip(TOP_50_CODES, per_class_f1), key=lambda x: x[1], reverse=True)[:10]
for rank, (code, f1) in enumerate(top_10_best, 1):
    print(f"   {rank}. {code}: {f1:.4f}")

results = {
    'phase': 'ShifaMind2 Phase 2 - GraphSAGE + Top-50 Ontology',
    'timestamp': timestamp,
    'run_folder': str(OUTPUT_BASE),
    'diagnosis_metrics': {
        'macro_f1': float(macro_f1),
        'micro_f1': float(micro_f1),
        'per_class_f1': {code: float(f1) for code, f1 in zip(TOP_50_CODES, per_class_f1)}
    },
    'architecture': 'Concept Bottleneck + GraphSAGE Ontology (Top-50)',
    'graph_stats': {
        'nodes': ontology_graph.number_of_nodes(),
        'edges': ontology_graph.number_of_edges(),
        'hidden_dim': GRAPH_HIDDEN_DIM,
        'layers': GRAPH_LAYERS
    },
    'training_history': history
}

with open(RESULTS_PATH / 'results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Results saved to: {RESULTS_PATH / 'results.json'}")
print(f"💾 Best model saved to: {CHECKPOINT_PATH / 'phase2_best.pt'}")

print("\n" + "="*80)
print("✅ SHIFAMIND2 PHASE 2 COMPLETE!")
print("="*80)
print(f"\n📍 Run folder: {OUTPUT_BASE}")
print(f"   Macro F1: {macro_f1:.4f} | Micro F1: {micro_f1:.4f}")
print("\nNext: Run shifamind2_p3.py (RAG) with this run folder")
print("\nAlhamdulillah! 🤲")
