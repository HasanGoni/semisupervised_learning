# DINOv2 Domain Adaptation - TODO List

## Project Status: Implementation Complete ✅

**Last Updated:** 2025-12-30

---

## Completed Tasks ✅

### 1. ✅ DataLoader Setup (3 cells)
- [x] Cell 1A: Explore basic DataLoader
- [x] Cell 1B: Export `create_dataloader()` function
- [x] Cell 1C: Verify DataLoader functionality

### 2. ✅ Model Architecture - Student/Teacher (7 cells)
- [x] Cell 2A: Explore DINOv2 forward pass
- [x] Cell 2B: Export `DINOHead` class
- [x] Cell 2C: Verify projection head
- [x] Cell 2D: Export `DINOModel` wrapper class
- [x] Cell 2E: Explore student/teacher creation
- [x] Cell 2F: Export `create_dino_models()` factory
- [x] Cell 2G: Verify model creation

### 3. ✅ DINO Loss Function (3 cells)
- [x] Cell 3A: Explore softmax with temperature
- [x] Cell 3B: Export `DINOLoss` class
- [x] Cell 3C: Verify loss computation

### 4. ✅ Training Utilities (6 cells)
- [x] Cell 4A: Explore optimizer setup
- [x] Cell 4B: Export `create_optimizer()` function
- [x] Cell 4C: Explore cosine LR schedule
- [x] Cell 4D: Export `WarmupCosineSchedule` class
- [x] Cell 4E: Export `update_teacher_ema()` function
- [x] Cell 4F: Verify EMA update

### 5. ✅ Training Loop (2 cells)
- [x] Cell 5A: Export `train_one_epoch()` function
- [x] Cell 5B: Export `train_dino()` function

### 6. ✅ Checkpoint Management (3 cells)
- [x] Cell 6A: Export `save_checkpoint()` function
- [x] Cell 6B: Export `load_checkpoint()` function
- [x] Cell 6C: Verify checkpoint save/load

### 7. ✅ Feature Extraction (2 cells)
- [x] Cell 7A: Export `extract_features()` function
- [x] Cell 7B: Export `save_features()` function

### 8. ✅ Complete Training Script (1 cell)
- [x] Cell 8A: Complete end-to-end training example

### 9. ✅ Evaluation Utilities (7 cells)
- [x] Cell 9A: Export `plot_loss_history()` function
- [x] Cell 9B: Export `visualize_features_tsne()` function
- [x] Cell 9C: Export `compute_sample_losses()` helper function
- [x] Cell 9D: Export `plot_top_losses()` function for error analysis
- [x] Cell 9E: Example usage of `plot_top_losses()`
- [x] Cell 9F: Export `visualize_dataloader_batch()` function
- [x] Cell 9G: Example usage of `visualize_dataloader_batch()`

---

## Current Tasks 🚀

### Phase 1: Test & Validate

1. **Reorder Notebook Sections**
   - [ ] Sections are currently backwards (9→1)
   - [ ] Should be reordered to (1→9) for readability
   - Priority: Medium
   - Estimated time: 5 minutes

2. **Export Module**
   - [ ] Run `nbdev_export()` to generate `semisupervised_learning/core.py`
   - [ ] Verify all functions are exported correctly
   - Priority: High
   - Estimated time: 2 minutes

3. **Test Basic Functionality**
   - [ ] Run all cells in notebook sequentially
   - [ ] Verify no errors in exploration cells
   - [ ] Check model creation works
   - [ ] Verify loss computation
   - [ ] Test DataLoader visualization: `visualize_dataloader_batch(train_loader)`
   - Priority: High
   - Estimated time: 10 minutes

### Phase 2: Initial Training Run

4. **Quick Test Training (5 epochs)**
   - [ ] Uncomment Section 8 training script
   - [ ] Set `n_epochs=5`, `batch_size=8` (conservative)
   - [ ] Set `freeze_backbone=True` (faster initial test)
   - [ ] Run training
   - [ ] Verify:
     - Loss decreases
     - No CUDA OOM errors
     - Checkpoints save successfully
   - Priority: High
   - Estimated time: 30-60 minutes

5. **Analyze Initial Results**
   - [ ] Plot loss curve
   - [ ] Check if loss is decreasing
   - [ ] Verify checkpoint files exist
   - [ ] Note GPU memory usage
   - Priority: High

### Phase 3: Full Training

6. **Optimize Hyperparameters**
   - [ ] Adjust batch_size based on GPU memory
   - [ ] Decide: freeze_backbone (True/False)
   - [ ] Set final n_epochs (recommended: 100)
   - Priority: Medium

7. **Full Training Run**
   - [ ] Update training script with final hyperparameters
   - [ ] Run 100 epoch training (overnight)
   - [ ] Monitor loss periodically
   - Priority: High
   - Estimated time: 14-28 hours

8. **Evaluate Trained Model**
   - [ ] Load best checkpoint
   - [ ] Extract features from dataset
   - [ ] Visualize features with t-SNE
   - [ ] Check feature clustering quality
   - Priority: High

### Phase 4: Prepare for Segmentation (Future)

9. **Save Adapted Model**
   - [ ] Extract and save adapted backbone
   - [ ] Document model performance
   - [ ] Save features for downstream tasks
   - Priority: Medium

10. **Documentation**
    - [ ] Document final hyperparameters used
    - [ ] Note training time and GPU usage
    - [ ] Record final loss values
    - [ ] Add notes to this TODO
    - Priority: Low

---

## Known Issues & Notes

### Notebook Organization
- ⚠️ **Sections are in reverse order** (9→8→7...→1)
  - Not critical for functionality
  - Makes reading harder
  - Should reorder for better workflow

### Dependencies
- ✅ All required packages installed:
  - torch, torchvision, xformers
  - datasets (HuggingFace)
  - cv_tools (local package)
  - fastcore, nbdev
  - tqdm, matplotlib, numpy

### Potential Issues to Watch

1. **GPU Memory**
   - 1,642 images × batch_size=16 × 10 crops = large memory footprint
   - Symptoms: CUDA OOM error
   - Solution: Reduce batch_size or n_local_crops

2. **Training Stability**
   - Watch for NaN loss
   - Solution: Lower learning rate, check gradient clipping

3. **Slow Training**
   - First epoch may be slower (compilation)
   - Solution: Increase num_workers if CPU-bound

---

## Hyperparameter Settings

### Current Recommendations

```python
# Data
batch_size = 16  # Reduce to 8 or 4 if OOM
num_workers = 4  # Increase to 8 if CPU available
n_global_crops = 2
n_local_crops = 8  # Reduce to 4 if memory issues

# Model
freeze_backbone = True  # Start here, try False later
out_dim = 256

# Training
n_epochs = 100  # Start with 5 for testing
base_lr = 1e-4
weight_decay = 0.04
warmup_epochs = 10

# Loss
teacher_temp = 0.04
student_temp = 0.1
center_momentum = 0.9

# EMA
teacher_momentum = 0.996

# Other
gradient_clip = 3.0
save_every = 10
```

### Tested Configurations

| Config | Batch Size | Freeze Backbone | Status | Notes |
|--------|------------|-----------------|--------|-------|
| Test 1 | TBD | TBD | Pending | Initial test run |

*(Update this table with your experimental results)*

---

## Next Steps After Training

1. **Analyze Top Losses (Error Analysis)**
   ```python
   # Find which images had highest losses during training
   plot_top_losses(student, teacher, torch_ds_trn, loss_fn, device, n_samples=10)

   # Also check validation set
   torch_ds_tst = MultiCropDataset(ds_tst, n_global=2, n_local=8)
   plot_top_losses(student, teacher, torch_ds_tst, loss_fn, device, n_samples=10)
   ```

2. **Extract Features**
   ```python
   features = extract_features(teacher, train_loader, device=device, use_head=False)
   save_features(features, 'dinov2_microscopy_features.pth')
   ```

3. **Visualize**
   ```python
   plot_loss_history(loss_history, save_path='training_loss.png')
   visualize_features_tsne(features, labels=labels)
   ```

4. **Save Adapted Backbone**
   ```python
   torch.save(teacher.backbone.state_dict(), 'dinov2_microscopy_adapted.pth')
   ```

4. **Phase 2: Segmentation Fine-tuning**
   - Load adapted backbone
   - Add segmentation head
   - Train on segmentation dataset
   - Evaluate segmentation performance

---

## Questions & Research

- [ ] Optimal batch size for your GPU?
- [ ] Best freeze_backbone setting for your data?
- [ ] How many epochs needed for convergence?
- [ ] Quality of learned features (t-SNE clustering)?
- [ ] Performance gain vs pretrained DINOv2?

---

## Resources

- **Plan file:** `PLAN.md` (this directory)
- **Notebook:** `nbs/00_core.ipynb`
- **Exported module:** `semisupervised_learning/core.py` (generated)
- **Original plan:** `~/.claude/plans/cached-marinating-turtle.md`

---

## Training Log

### Session 1: [Date]
- Configuration:
- Results:
- Notes:

*(Add your training sessions here)*
