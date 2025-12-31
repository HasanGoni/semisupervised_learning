# DINOv2 Self-Supervised Domain Adaptation - Implementation Plan

## Overview

Complete the DINOv2 domain adaptation implementation in `00_core.ipynb` following literate programming principles. The goal is to adapt DINOv2 to Electron Microscopy images using self-supervised DINO training for later segmentation tasks.

## Current State

**Already Implemented:**
- ✓ MultiCropDataset class (2 global + 8 local crops)
- ✓ Data augmentation pipelines
- ✓ DINOv2 model loading (dinov2_vitl14, 1024-dim)
- ✓ Visualization utilities
- ✓ Dataset: 1,642 Electron Microscopy training images

**Missing:**
- DataLoader integration
- Student-teacher model wrapper
- DINO loss function
- Training loop with EMA updates
- Checkpoint management
- Feature extraction utilities

## Implementation Strategy

Follow literate programming pattern for each section:
1. **Exploration cell** (no export) - Test, print outputs, understand behavior
2. **Export cell** (`#| export`) - Create clean, modular function
3. **Verification cell** (no export) - Test the function

### Critical File

- **Primary:** `/home/hasan/Schreibtisch/projects/git_data/semisupervised_learning/nbs/00_core.ipynb`
  - All implementation happens in this notebook
  - Cells marked `#| export` auto-generate production code

## Implementation Sections

### Section 1: DataLoader Setup (3 cells)

**Purpose:** Wrap MultiCropDataset with PyTorch DataLoader

**Key Function:**
```python
create_dataloader(dataset, batch_size=16, shuffle=True, num_workers=4)
```

**Cells:**
1. Explore: Test basic DataLoader, inspect batch shapes
2. Export: Create `create_dataloader()` function
3. Verify: Test function with different batch sizes

---

### Section 2: Model Architecture (7 cells)

**Purpose:** Build student-teacher framework with projection heads

**Key Classes:**
- `DINOHead(in_dim, hidden_dim, out_dim)` - MLP projection head with bottleneck
- `DINOModel(backbone, head)` - Wrapper combining backbone + head
- `create_dino_models(backbone, freeze_backbone, device)` - Factory function

**Cells:**
1. Explore: Test DINOv2 forward pass, understand output shapes
2. Export: Create `DINOHead` class (3-layer MLP with normalization)
3. Verify: Test projection head output
4. Export: Create `DINOModel` wrapper class
5. Explore: Create student and teacher models, test EMA
6. Export: Create `create_dino_models()` factory function
7. Verify: Test model creation with different configurations

**Key Decision:** Make `freeze_backbone` configurable (recommend starting with `True` for safety)

---

### Section 3: DINO Loss Function (3 cells)

**Purpose:** Implement asymmetric student-teacher loss with centering

**Key Class:**
```python
DINOLoss(out_dim=256, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9)
```

**Features:**
- Cross-entropy between student/teacher predictions
- Temperature scaling (sharp teacher, smooth student)
- Center momentum (EMA buffer for prototype centering)
- Student processes all crops, teacher only global crops

**Cells:**
1. Explore: Understand softmax with temperature scaling
2. Export: Create `DINOLoss` class with centering
3. Verify: Test loss computation with example tensors

---

### Section 4: Training Utilities (6 cells)

**Purpose:** Optimizer, scheduler, and EMA update helpers

**Key Functions:**
- `create_optimizer(model, lr, weight_decay)` - AdamW with separate bias weight decay
- `WarmupCosineSchedule` - Cosine annealing with linear warmup
- `update_teacher_ema(student, teacher, momentum)` - EMA parameter update

**Cells:**
1. Explore: Test AdamW optimizer setup
2. Export: Create `create_optimizer()` function
3. Explore: Test cosine LR schedule, visualize curve
4. Export: Create `WarmupCosineSchedule` class
5. Export: Create `update_teacher_ema()` function
6. Verify: Test EMA update mechanism

---

### Section 5: Training Loop (2 cells)

**Purpose:** Complete training infrastructure

**Key Functions:**
- `train_one_epoch(student, teacher, train_loader, loss_fn, optimizer, device)` - Single epoch
- `train_dino(student, teacher, ..., n_epochs)` - Full training loop

**Features:**
- Gradient clipping (prevents instability)
- Teacher EMA update every step
- Progress bars (tqdm)
- Periodic checkpointing
- Loss tracking

**Cells:**
1. Export: Create `train_one_epoch()` function
2. Export: Create `train_dino()` function

---

### Section 6: Checkpoint Management (3 cells)

**Purpose:** Save/load training state

**Key Functions:**
- `save_checkpoint(path, student, teacher, optimizer, scheduler, epoch, loss)`
- `load_checkpoint(path, student, teacher, optimizer, scheduler, device)`

**Cells:**
1. Export: Create `save_checkpoint()` function
2. Export: Create `load_checkpoint()` function
3. Verify: Test save/load cycle with temporary file

---

### Section 7: Feature Extraction (2 cells)

**Purpose:** Extract embeddings from trained models

**Key Functions:**
- `extract_features(model, dataloader, device, use_head=False)` - Batch feature extraction
- `save_features(features, path, labels)` - Save features to disk

**Cells:**
1. Export: Create `extract_features()` function
2. Export: Create `save_features()` function

---

### Section 8: Complete Training Script (1 cell)

**Purpose:** Show how all components work together

**Cell:**
1. Example: Complete end-to-end training script with:
   - Dataset and DataLoader creation
   - Model initialization
   - Loss and optimizer setup
   - Training execution
   - Loss plotting

---

### Section 9: Evaluation Utilities (2 cells - Optional)

**Purpose:** Monitoring and visualization helpers

**Key Functions:**
- `plot_loss_history(loss_history, save_path)` - Plot training curve
- `visualize_features_tsne(features, labels, n_samples)` - t-SNE visualization

**Cells:**
1. Export: Create `plot_loss_history()` function
2. Export: Create `visualize_features_tsne()` function

---

## Configuration & Hyperparameters

**Recommended Starting Values:**
```python
# Data
batch_size = 16  # Adjust based on GPU memory
num_workers = 4
n_global_crops = 2
n_local_crops = 8

# Model
freeze_backbone = True  # Start conservative, try False if needed
out_dim = 256  # Projection dimension

# Training
n_epochs = 100
base_lr = 1e-4  # Conservative for fine-tuning
weight_decay = 0.04
warmup_epochs = 10

# Loss
teacher_temp = 0.04  # Sharp teacher
student_temp = 0.1   # Smooth student
center_momentum = 0.9

# EMA
teacher_momentum = 0.996

# Checkpointing
save_every = 10  # Save every 10 epochs
```

---

## Expected Training Time

- **Batch processing:** ~5-10 seconds/batch (GPU-dependent)
- **Batches per epoch:** ~103 (1,642 images / 16 batch size)
- **Per epoch:** ~8-17 minutes
- **100 epochs:** ~14-28 hours (overnight run)

**Recommendation:** Start with 5-10 epoch test run to verify setup before full training

---

## Troubleshooting Guide

### GPU Memory Issues
- Reduce `batch_size` (16 → 8 → 4)
- Reduce `n_local_crops` (8 → 4)
- Reduce `num_workers` if CPU memory is tight

### Loss Not Decreasing
- Check teacher is updating: print parameter norms
- Verify center is updating: print center values
- Lower learning rate (1e-4 → 5e-5)
- Increase warmup period (10 → 20 epochs)
- Try `freeze_backbone=True` first

### NaN Loss
- Lower learning rate
- Check gradient clipping is enabled (default: 3.0)
- Verify temperature values are reasonable

### Slow Training
- Increase `num_workers` (4 → 8)
- Enable `pin_memory=True` in DataLoader
- Profile to identify bottleneck

---

## Post-Training Workflow

After training completes:

1. **Load best checkpoint:**
```python
student, teacher = create_dino_models(dinov2, device=device)
load_checkpoint('checkpoints/checkpoint_epoch_100.pth', student, teacher)
```

2. **Extract features:**
```python
features = extract_features(teacher, train_loader, device=device, use_head=False)
save_features(features, 'features_train.pth')
```

3. **Visualize:**
```python
plot_loss_history(loss_history, save_path='training_loss.png')
visualize_features_tsne(features, labels=labels)
```

4. **Save adapted backbone for Phase 2 (segmentation):**
```python
torch.save(teacher.backbone.state_dict(), 'dinov2_microscopy_adapted.pth')
```

---

## Success Criteria

- ✓ Loss decreases over epochs (expect: ~2.0 → ~0.5-1.0)
- ✓ No NaN or unstable loss values
- ✓ Checkpoints save successfully every 10 epochs
- ✓ Can extract features from final model
- ✓ t-SNE shows clustering structure in features
- ✓ Adapted model ready for downstream segmentation tasks

---

## Implementation Order Summary

1. **DataLoader** (easiest, needed for all testing)
2. **Model Architecture** (student-teacher setup)
3. **DINO Loss** (core algorithm)
4. **Training Utilities** (optimizer, scheduler, EMA)
5. **Training Loop** (integrates everything)
6. **Checkpoint Management** (save/resume capability)
7. **Feature Extraction** (post-training utility)
8. **Example Script** (end-to-end demonstration)
9. **Evaluation** (optional monitoring tools)

**Total Implementation:** 26 cells following literate programming pattern
**Estimated Time:** 3-4 hours for notebook development
