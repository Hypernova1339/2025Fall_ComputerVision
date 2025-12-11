# YOLO Training Guide for Jinx Ability Detection

This guide covers how to train the YOLOv8 model for detecting Jinx's abilities (W and R) in League of Legends gameplay footage.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Hardware Requirements](#hardware-requirements)
3. [Training Configurations](#training-configurations)
4. [Preparing Your Dataset](#preparing-your-dataset)
5. [Running Training](#running-training)
6. [Monitoring Training](#monitoring-training)
7. [Running Experiments](#running-experiments)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Quick training with auto device detection (recommended)
python src/training/train_yolo.py \
    --data configs/jinx_abilities.yaml \
    --config configs/training_config.yaml \
    --name my_experiment
```

---

## Hardware Requirements

### Recommended (Fast Training)
- **GPU**: NVIDIA CUDA-compatible GPU (RTX 3060+) OR Apple Silicon Mac (M1/M2/M3)
- **RAM**: 16GB+
- **Expected Time**: 15-45 minutes for 100 epochs

### Minimum (Slow Training)
- **CPU**: Any modern CPU
- **RAM**: 8GB+
- **Expected Time**: 4-6 hours for 100 epochs

The training script automatically detects the best available device:
1. **NVIDIA CUDA** (fastest)
2. **Apple MPS** (fast on Apple Silicon)
3. **CPU** (slowest, fallback)

Override with `--device`:
```bash
--device 0      # Use first CUDA GPU
--device mps    # Force Apple MPS
--device cpu    # Force CPU (not recommended)
```

---

## Training Configurations

Three pre-configured training profiles are available:

### 1. Default Configuration (`training_config.yaml`)
```bash
python src/training/train_yolo.py \
    --data configs/jinx_abilities.yaml \
    --config configs/training_config.yaml
```
- **Model**: yolov8n (nano)
- **Image Size**: 640px
- **Epochs**: 100
- **Use Case**: General purpose, balanced speed/accuracy

### 2. Speed Configuration (`training_speed.yaml`)
```bash
python src/training/train_yolo.py \
    --data configs/jinx_abilities.yaml \
    --config configs/training_speed.yaml
```
- **Model**: yolov8n (nano)
- **Image Size**: 480px (smaller)
- **Epochs**: 50
- **Optimizations**: Caching, rectangular training, reduced augmentation
- **Use Case**: Fast iteration during development, ~15-30 min on GPU

### 3. Accuracy Configuration (`training_accuracy.yaml`)
```bash
python src/training/train_yolo.py \
    --data configs/jinx_abilities.yaml \
    --config configs/training_accuracy.yaml
```
- **Model**: yolov8s (small, more capacity)
- **Image Size**: 640px
- **Epochs**: 150
- **Optimizations**: Tuned augmentation, copy-paste for rare classes
- **Use Case**: Final model training, best mAP, ~1-2 hours on GPU

---

## Preparing Your Dataset

### 1. Label Your Images
Use the bounding box labeler tool:
```bash
python tools/bbox_labeler.py --input data/images/unlabeled/
```

### 2. Split into Train/Val
```bash
python tools/prepare_dataset.py --val-ratio 0.2 --clean
```

### 3. Address Class Imbalance (Important!)
The dataset likely has class imbalance (many W casts, few R impacts). Use oversampling:
```bash
python tools/prepare_dataset.py --val-ratio 0.2 --oversample --clean
```

Options:
- `--oversample`: Enable oversampling of rare classes
- `--oversample-target 100`: Target 100 samples per class
- `--oversample-ratio 0.5`: At least 50% of max class count (default)

### 4. Verify Dataset
```bash
python tools/verify_dataset.py --config configs/jinx_abilities.yaml
```

---

## Running Training

### Basic Training
```bash
python src/training/train_yolo.py \
    --data configs/jinx_abilities.yaml \
    --epochs 100 \
    --name jinx_v1
```

### With Config File (Recommended)
```bash
python src/training/train_yolo.py \
    --data configs/jinx_abilities.yaml \
    --config configs/training_config.yaml \
    --name jinx_v1
```

### All Training Options
```
--data          Path to dataset YAML (required)
--config        Path to training config YAML
--model         Base model (yolov8n.pt, yolov8s.pt, etc.)
--epochs        Number of training epochs
--batch         Batch size
--imgsz         Image size
--device        Device (cpu, mps, 0, 1, etc.)
--patience      Early stopping patience
--workers       Data loader workers
--cache         Cache images in RAM
--project       Output project directory
--name          Experiment name
--resume        Resume from checkpoint path
```

### Resume Training
```bash
python src/training/train_yolo.py \
    --data configs/jinx_abilities.yaml \
    --resume runs/detect/jinx_v1/weights/last.pt
```

---

## Monitoring Training

### During Training
Training progress is displayed in the terminal:
- Current epoch and batch
- Loss values (box_loss, cls_loss, dfl_loss)
- Validation metrics (precision, recall, mAP)

### After Training
Results are saved to `runs/detect/<experiment_name>/`:
```
runs/detect/jinx_v1/
├── args.yaml           # Training arguments
├── results.csv         # Per-epoch metrics
├── results.png         # Training curves plot
├── confusion_matrix.png
├── labels.jpg          # Label distribution
├── train_batch*.jpg    # Sample training batches
├── val_batch*.jpg      # Sample validation batches
└── weights/
    ├── best.pt         # Best model (by mAP)
    └── last.pt         # Last checkpoint
```

### Key Metrics to Watch
| Metric | Description | Target |
|--------|-------------|--------|
| mAP50 | Mean Average Precision @ IoU=0.5 | > 0.6 |
| mAP50-95 | Mean AP across IoU thresholds | > 0.4 |
| Precision | True positives / all detections | > 0.7 |
| Recall | True positives / all ground truth | > 0.6 |

### TensorBoard
If TensorBoard is enabled:
```bash
tensorboard --logdir runs/detect
```

---

## Running Experiments

Use the experiment runner for ablation studies:

```bash
# Make executable (once)
chmod +x tools/run_experiments.sh

# See available experiments
./tools/run_experiments.sh help

# Run specific experiments
./tools/run_experiments.sh baseline
./tools/run_experiments.sh speed
./tools/run_experiments.sh accuracy
./tools/run_experiments.sh small_obj
./tools/run_experiments.sh larger_model

# Run ALL experiments (takes several hours)
./tools/run_experiments.sh all
```

### Experiment Grid

| Experiment | Model | ImgSz | Config | Purpose |
|------------|-------|-------|--------|---------|
| baseline | yolov8n | 640 | default | Standard training |
| speed | yolov8n | 480 | training_speed | Fast iteration |
| accuracy | yolov8s | 640 | training_accuracy | Best mAP |
| small_obj | yolov8n | 800 | default | Better for tiny projectiles |
| larger_model | yolov8s | 640 | default | More model capacity |

---

## Troubleshooting

### Training is Very Slow
**Cause**: Training is running on CPU instead of GPU.

**Fix**:
1. Check device detection output at start of training
2. For NVIDIA: Ensure CUDA toolkit is installed
3. For Mac: Ensure you have Apple Silicon and latest PyTorch
4. Force MPS: `--device mps`

### Out of Memory (OOM)
**Cause**: Batch size too large for GPU memory.

**Fix**:
```bash
--batch 8    # Reduce batch size
--imgsz 480  # Reduce image size
```

### mAP is Very Low (<20%)
**Causes**:
1. Not enough training epochs
2. Severe class imbalance
3. Poor quality labels

**Fixes**:
1. Let training run longer (check early stopping)
2. Use `--oversample` when preparing dataset
3. Re-check labels with `python tools/bbox_labeler.py`

### Per-Class AP is Very Uneven
**Cause**: Class imbalance (common for R_IMPACT).

**Fix**:
```bash
# Oversample rare classes
python tools/prepare_dataset.py --oversample --clean

# Use copy-paste augmentation (in accuracy config)
python src/training/train_yolo.py --config configs/training_accuracy.yaml
```

### Training Loss Not Decreasing
**Causes**:
1. Learning rate too high
2. Data corruption
3. Label errors

**Fixes**:
1. Use config file with tuned LR: `--config configs/training_config.yaml`
2. Verify dataset: `python tools/verify_dataset.py --config configs/jinx_abilities.yaml`
3. Check sample images in training output

---

## Best Practices

1. **Always use a config file** - Don't rely on CLI defaults
2. **Enable caching** - Use `--cache` for significant speedup
3. **Address class imbalance** - Use `--oversample` when preparing data
4. **Monitor per-class metrics** - Overall mAP can hide poor rare-class performance
5. **Start with speed config** - Iterate quickly, then use accuracy config for final model
6. **Save experiments** - Use unique `--name` for each experiment

---

## Example Workflow

```bash
# 1. Prepare dataset with oversampling
python tools/prepare_dataset.py --val-ratio 0.2 --oversample --clean

# 2. Verify dataset
python tools/verify_dataset.py --config configs/jinx_abilities.yaml

# 3. Quick experiment (fast)
python src/training/train_yolo.py \
    --data configs/jinx_abilities.yaml \
    --config configs/training_speed.yaml \
    --name jinx_quick

# 4. Check results, iterate on labels if needed

# 5. Final training (accurate)
python src/training/train_yolo.py \
    --data configs/jinx_abilities.yaml \
    --config configs/training_accuracy.yaml \
    --name jinx_final

# 6. Best model is at: runs/detect/jinx_final/weights/best.pt
```




