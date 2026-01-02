# ShifaMind Phase 1: Concept Bottleneck Model - Handoff Documentation

## Project Overview

**Goal**: Scale ShifaMind from 4 ICD-10 codes to 50 codes for ICML-level contribution using Concept Bottleneck Models (CBM) for interpretable clinical diagnosis prediction.

**Dataset**: MIMIC-IV discharge notes (~114K samples, 49 valid ICD codes after filtering)

**Architecture**: BioClinicalBERT → Cross-attention with 72 clinical concepts → Concept prediction → Diagnosis prediction (through concept bottleneck)

**Current Branch**: `claude/setup-shifamind-project-2WNXg`

---

## What We Did: Complete Journey

### Initial State (4 ICD Codes - Working)
- 4 codes: I10, E119, I4891, J449
- Validation Diagnosis F1: ~0.45-0.55 ✅
- Everything working perfectly

### Scaling Attempt (50 ICD Codes - BROKE)
**Changes made:**
1. Updated to top 50 ICD codes from MIMIC-IV
2. Created `ShifaMind200p1.py` (standalone Phase 1 file)
3. Created `prepare_mimic_data_50codes.py` (preprocesses raw MIMIC-IV data)

**Result**: Validation Diagnosis F1 = 0.0000 ❌ (complete failure)

---

## Problems Encountered & Fixes Applied

### Problem 1: Zero Diagnosis F1 (Original)
**Symptom**: Model predicted nothing (all zeros)

**Root Cause**: Broken alignment loss
```python
# BROKEN CODE:
concept_contrib = concept_activations.mean(dim=1)  # Average ALL 72 concepts
dx_present = (dx_labels.sum(dim=1) > 0).float()    # 1.0 if ANY diagnosis
loss_align = F.mse_loss(concept_contrib, dx_present)
# This forced ALL concepts → 1.0 when any diagnosis present → no discrimination!
```

**Fix Attempt 1**: Disabled alignment loss → F1 still 0.0000
**Fix Attempt 2**: Lowered threshold 0.5 → 0.3 → F1 still 0.0000

### Problem 2: Missing Class Imbalance Handling
**Root Cause**: No pos_weight for BCEWithLogitsLoss with 50 imbalanced codes

**Fix Applied**:
```python
pos_counts = train_labels.sum(axis=0)
neg_counts = len(train_labels) - pos_counts
pos_weight = neg_counts / pos_counts
bce_dx = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

### Problem 3: Zero-Frequency Code (Z20822)
**Symptom**: Code Z20822 had 0 samples → pos_weight = 8 billion → model over-predicted everything (98% positive)

**Fix Applied**:
```python
# Filter codes with < 100 samples
MIN_SAMPLES = 100
valid_codes_mask = pos_counts >= MIN_SAMPLES
# Removed Z20822, kept 49 valid codes
```

**Result after fixes**: Diagnosis F1 = 0.1876 ✅ (no longer zero!)
**BUT**: Still predicting 100% positive (should be ~10%)

### Problem 4: Probability Clustering
**Symptom**: All diagnosis probabilities clustered around 0.5 (std=0.021)
```
Diagnosis probs: mean=0.4989, std=0.0212, min=0.41, max=0.61
→ With threshold 0.2, everything > 0.41 predicted positive → 100% positive rate
```

**Root Cause**: Temperature scaling (tau=2.0) over-compressed concept signal
```python
concept_activations = torch.sigmoid(concept_logits / 2.0)  # Over-compressed
```

**Fix Applied (CURRENT STATE)**:
```python
concept_activations = torch.sigmoid(concept_logits)  # Removed /tau
```

---

## Current Code State (Latest Commit: aa848e5)

### Key Files
1. **`ShifaMind200p1.py`** - Main Phase 1 training script (49 ICD codes)
2. **`prepare_mimic_data_50codes.py`** - Preprocesses MIMIC-IV raw data
3. **`mimic_dx_data.csv`** - Preprocessed data (114,633 samples, 50 codes)

### Architecture (ShifaMind200p1.py)
```python
class ConceptBottleneckModel:
    # 1. BioClinicalBERT encoder
    # 2. Learnable concept embeddings (72 concepts)
    # 3. Cross-attention (concepts attend to text)
    # 4. Concept prediction head
    # 5. Concept activations = sigmoid(concept_logits)  ← NO TEMPERATURE
    # 6. Diagnosis head (MLP) takes concept_activations as input
```

### Loss Function (Multi-Objective)
```python
L_total = λ_dx·L_dx + λ_concept·L_concept + λ_prior·L_prior + λ_w·L_w_sparse

Components:
- L_dx: BCEWithLogitsLoss with pos_weight (handles imbalance)
- L_concept: BCEWithLogitsLoss with label smoothing (0→0.05, 1→0.95)
- L_prior: Sparsity prior (mean activation → 15%)
- L_w_sparse: L1 regularization on diagnosis weights

Hyperparameters:
- λ_dx = 1.0
- λ_concept = 0.6
- λ_prior = 0.2
- λ_w_sparse = 1e-3
- Target sparsity = 15%
- Diagnosis threshold = 0.2
```

### Data Stats (After Filtering)
- **49 valid ICD codes** (Z20822 removed)
- **Train**: 80,243 samples
- **Val**: 17,195 samples
- **Test**: 17,195 samples
- **Avg labels per sample**: 5.31
- **pos_weight range**: 1.60 to 24.87 (reasonable)

---

## IMMEDIATE NEXT STEPS

### 1. Restart Training from Scratch
**Important**: Previous runs used broken configurations. Start fresh.

**Command**:
```bash
python ShifaMind200p1.py
```

### 2. Watch Epoch 1 Metrics Carefully

**✅ SUCCESS CRITERIA** (indicating fix worked):
```
Diagnosis probabilities:
  mean = 0.25-0.35, std = 0.10-0.20  ← GOOD SPREAD (was std=0.02)

Diagnosis predictions (positive rate):
  ~10-15%  ← REALISTIC (was 100%)

Diagnosis F1:
  0.30-0.45  ← STRONG (was 0.19)

Concept F1:
  0.30-0.40  ← WORKING
```

**❌ FAILURE INDICATORS** (need Phase 2):
```
Diagnosis probabilities:
  std < 0.05  ← Still clustering

Diagnosis predictions (positive rate):
  > 50%  ← Still over-predicting

Diagnosis F1:
  < 0.25  ← Not improving
```

### 3. Target Performance (By Epoch 5)

**Realistic targets for 49-code interpretable CBM**:
- **Micro-F1**: 0.40-0.50
- **Macro-F1**: 0.25-0.35
- **Diagnosis positive rate**: ~10% (matching label distribution)
- **Concept sparsity**: mean activation ~15%

**Black-box baseline (BioClinicalBERT) typically gets**: Micro-F1 ~0.55-0.65
**CBM trade-off**: ~10-15 point F1 drop for interpretability is acceptable

---

## IF CURRENT FIX DOESN'T WORK: Phase 2 Plan

### Option: Linear Diagnosis Head (Recommended by AI Consultant)

**Problem**: MLP diagnosis head might be too complex for compressed [0,1] concept inputs

**Solution**: Replace with linear layer
```python
# Current (MLP):
self.diagnosis_head = nn.Sequential(
    nn.Linear(num_concepts, hidden_dim // 2),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(hidden_dim // 2, num_diagnoses)
)

# Phase 2 (Linear):
self.diagnosis_head = nn.Linear(num_concepts, num_diagnoses)

# Optional: Nonnegative weights for monotonicity
class DxHead(nn.Module):
    def __init__(self, num_concepts, num_dx):
        super().__init__()
        self.W_raw = nn.Parameter(torch.randn(num_dx, num_concepts) * 0.01)
        self.b = nn.Parameter(torch.zeros(num_dx))

    def forward(self, c):
        W = F.softplus(self.W_raw)  # Force nonnegative
        return c @ W.t() + self.b
```

**Benefits**:
- Perfect interpretability: dx_logits = Σ(c_i × W_ij) + b
- More stable for compressed inputs
- Each diagnosis depends on specific concepts (sparse)

**Implementation**: Only do this if temperature removal doesn't work

---

## Technical Details & Gotchas

### Data Path Structure
```
/content/drive/MyDrive/ShifaMind/
├── mimic_dx_data.csv (preprocessed data)
├── 09_ShifaMind/
│   ├── checkpoints/phase1/ (model checkpoints)
│   ├── shared_data/ (train/val/test splits, concept labels)
│   └── results/phase1/ (phase1_results.json)
└── 01_Raw_Datasets/Extracted/ (MIMIC-IV raw files)
```

### Concept Labeling
- **Method**: Keyword-based search (noisy but fast)
- **72 clinical concepts**: hypertension, diabetes, chest pain, fever, etc.
- **Processing time**: ~5-10 minutes for 80K samples (chunked)
- **Label smoothing applied**: Accounts for noise (0→0.05, 1→0.95)

### Model Size
- **Parameters**: ~111M (BioClinicalBERT backbone)
- **Training time**: ~45-50 min/epoch on GPU
- **Memory**: Batch size 8 fits on most GPUs

### Important Code Locations
- **Model definition**: `ShifaMind200p1.py:398-506`
- **Loss function**: `ShifaMind200p1.py:521-587`
- **pos_weight calculation**: `ShifaMind200p1.py:258-297`
- **Training loop**: `ShifaMind200p1.py:617-708`

---

## Common Issues & Solutions

### Issue: "FileNotFoundError: mimic_dx_data.csv"
**Solution**: Run `prepare_mimic_data_50codes.py` first to preprocess raw MIMIC-IV data

### Issue: High GPU memory usage
**Solution**: Reduce batch size from 8 to 4 in line 177

### Issue: Training very slow
**Solution**: Normal - 45-50 min/epoch is expected for 80K samples

### Issue: All concepts predicting 1.0
**Solution**: Check sparsity prior is working (loss_prior should be in logs)

### Issue: Diagnosis F1 still 0
**Solution**: Check diagnosis positive rate - if 0%, increase threshold; if 100%, decrease threshold or check pos_weight

---

## Expected Training Output (What You Should See)

```
================================================================================
Epoch 1/5
================================================================================
Training: 100% 10031/10031 [47:00<00:00, 4.2it/s]
📉 Train Loss: ~1.3-1.5
   • Diagnosis:  ~1.0-1.2
   • Concept:    ~0.3-0.4
   • Sparsity:   ~0.02-0.04
   • Weight L1:  ~0.05

Validation: 100% 2150/2150 [07:00<00:00, 5.1it/s]
📊 Validation Results:
   Loss:        ~1.2-1.4
   Diagnosis F1:  0.30-0.45  ✅ TARGET
   Concept F1:    0.30-0.40

🔍 Prediction Statistics:
   Diagnosis predictions (positive rate): 0.08-0.15  ✅ (expected: ~0.11)
   Concept predictions (positive rate):   0.12-0.18

🔬 Raw Value Distributions:
   Diagnosis probabilities: mean=0.25-0.35, std=0.10-0.20  ✅ GOOD SPREAD
   Concept activations:     mean=0.15-0.25, std=0.15-0.20
   ✅ Best model saved! (F1: 0.3X)
```

---

## Questions to Ask If Things Don't Work

1. **What's the diagnosis probability std?** (Need > 0.10 for good variance)
2. **What's the positive prediction rate?** (Should match ~10% label rate)
3. **Are concepts sparse?** (Mean activation ~15-20%)
4. **Is pos_weight reasonable?** (Should be 1.6 to ~25, not millions)
5. **Any zero-sample codes?** (All codes should have ≥100 samples)

---

## Summary for Next Session

**Current State**:
- Code is ready with all fixes applied (commit: aa848e5)
- Temperature scaling removed (last fix)
- Awaiting fresh training run to validate

**Immediate Task**:
1. Start training from scratch
2. Monitor Epoch 1 for success criteria
3. If successful → continue to Epoch 5
4. If unsuccessful → implement Phase 2 (linear diagnosis head)

**Success Metric**:
- Diagnosis F1 > 0.30 with realistic positive rate (~10%)

**Branch**: `claude/setup-shifamind-project-2WNXg`

---

## Files Modified in This Session

1. `ShifaMind200p1.py` - Complete rewrite for 50 codes with all fixes
2. `prepare_mimic_data_50codes.py` - Created for data preprocessing
3. All commits on branch `claude/setup-shifamind-project-2WNXg`

**Key Commits**:
- `90178c6` - Phase 1 fixes (pos_weight, sparsity, temperature)
- `da2b22f` - Zero-frequency code filtering
- `aa848e5` - Temperature scaling removal (LATEST)

---

## Contact Points for Debugging

If F1 is still low after Epoch 1:
1. Check diagnosis probability distributions (need spread, not clustering)
2. Verify positive rate matches label distribution
3. Consider Phase 2: linear diagnosis head
4. May need threshold calibration per-class

Good luck! The fixes should work - we've addressed all known issues systematically. 🚀
