# Jinx Ability Detection

A computer vision system for detecting **Jinx's W (Zap!)** and **R (Super Mega Death Rocket!)** abilities in League of Legends gameplay video.

## Overview

This project uses **YOLOv8** for frame-level detection of ability visual effects, combined with temporal aggregation to identify complete ability cast events and classify them as hits or misses.

### Detected Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | `JINX_W_PROJECTILE` | Zap! laser beam in flight |
| 1 | `JINX_W_IMPACT` | Zap! hit effect on target |
| 2 | `JINX_R_ROCKET` | Super Mega Death Rocket projectile |
| 3 | `JINX_R_IMPACT` | Rocket explosion VFX |

## Project Structure

```
├── configs/                 # Configuration files
│   ├── jinx_abilities.yaml  # YOLO dataset config
│   └── training_config.yaml # Training hyperparameters
├── data/
│   ├── raw_videos/          # Full gameplay VODs
│   ├── clips/               # Extracted ability clips
│   ├── images/              # Frame images (train/val)
│   ├── labels/              # YOLO annotations (train/val)
│   └── events/              # Event timestamps (events.csv)
├── src/
│   ├── data/                # Data processing modules
│   ├── training/            # Model training
│   ├── inference/           # Detection inference
│   ├── events/              # Event aggregation logic
│   └── utils/               # Utility functions
├── tools/                   # CLI scripts
├── notebooks/               # Jupyter notebooks
├── runs/                    # Training outputs
└── outputs/                 # Inference results
```

## Setup

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended)

### Installation

```bash
# Clone the repository
cd "NYU CV final project"

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install as package (editable)
pip install -e .
```

## Usage

### 1. Prepare Event Timestamps

Create `data/events/events.csv` with ability cast timestamps:

```csv
video_id,ability,cast_time_sec,notes
video_1_long,W,45.2,hit enemy bot
video_1_long,R,120.3,snipe kill mid
```

### 2. Extract Clips from VOD

```bash
python tools/make_event_clips.py \
    --video data/raw_videos/gameplay.mp4 \
    --events data/events/events.csv \
    --output data/clips/
```

### 3. Extract Frames from Clips

```bash
python tools/extract_frames.py \
    --input data/clips/ \
    --output data/images/unlabeled/ \
    --every 3
```

### 4. Label Frames

Use [LabelImg](https://github.com/HumanSignal/labelImg) with YOLO format:

```bash
labelImg data/images/unlabeled/ configs/classes.txt
```

### 5. Prepare Dataset

```bash
# Split into train/val
python tools/split_train_val.py \
    --images data/images/unlabeled/ \
    --labels data/labels/unlabeled/ \
    --val-ratio 0.2

# Verify dataset integrity
python tools/verify_dataset.py --config configs/jinx_abilities.yaml
```

### 6. Train Model

```bash
python src/training/train_yolo.py \
    --data configs/jinx_abilities.yaml \
    --config configs/training_config.yaml \
    --epochs 100
```

### 7. Run Inference

```bash
python tools/run_inference.py \
    --video path/to/new_gameplay.mp4 \
    --weights runs/detect/jinx_abilities/weights/best.pt \
    --output outputs/
```

## Event Aggregation

The system uses a temporal state machine to convert frame-level detections into ability events:

1. **IDLE → CAST**: Projectile class detected
2. **CAST → HIT**: Impact class detected within time window
3. **CAST → MISS**: Time window expires without impact

Output format (`events.jsonl`):
```json
{"event_id": 1, "ability": "W", "cast_time": 45.2, "hit": true, "confidence": 0.92}
```

## Labeling Guidelines

### What to Label

- **W Projectile**: The entire visible laser beam
- **W Impact**: The hit spark/effect at point of contact
- **R Rocket**: The missile body (not the trail)
- **R Impact**: The explosion effect

### Tips

- Label every frame where the VFX is clearly visible
- Use tight bounding boxes
- Include negative frames (no abilities) for training
- Focus on frames near ability casts (high-signal data)

## Results

| Metric | Value |
|--------|-------|
| mAP@50 | TBD |
| mAP@50-95 | TBD |
| Cast Detection Accuracy | TBD |
| Hit/Miss Classification | TBD |

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- NYU Computer Vision Course





