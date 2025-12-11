# Jinx Ability Labeling Guide

This guide explains how to annotate Jinx's W and R abilities for training the detection model.

## Step 1: Create events.csv

Watch your gameplay video and log each W and R ability cast.

### CSV Format

```csv
video_id,ability,cast_time_sec,notes
video_1_long,W,45.2,hit enemy adc
video_1_long,W,78.5,missed - into fog
video_1_long,R,120.3,snipe kill on mid
video_1_long,R,245.8,missed execute
```

### Fields

| Field | Description | Example |
|-------|-------------|---------|
| `video_id` | Video filename without extension | `video_1_long` |
| `ability` | W or R (uppercase) | `W` |
| `cast_time_sec` | Time when ability is cast (seconds) | `45.2` |
| `notes` | Optional context | `hit enemy adc` |

### Tips for Logging

1. Use VLC or similar player with timestamp display
2. Pause at the moment Jinx casts the ability
3. Round to 0.1 second precision (e.g., 45.2, not 45.234)
4. Include whether it was a hit or miss in notes

---

## Step 2: Extract Clips and Frames

After creating events.csv, run the extraction pipeline:

```bash
# Extract clips around each event
python tools/make_event_clips.py \
    --video data/raw_videos/video_1_long.mp4 \
    --events data/events/events.csv \
    --output data/clips/

# Extract frames from clips
python tools/extract_frames.py \
    --input data/clips/ \
    --output data/images/unlabeled/ \
    --every 3
```

---

## Step 3: Label with LabelImg

### Setup

```bash
pip install labelImg
labelImg data/images/unlabeled/ configs/classes.txt data/labels/unlabeled/
```

### Classes to Label

| ID | Class | What to Look For |
|----|-------|------------------|
| 0 | JINX_W_PROJECTILE | The Zap! laser beam in flight |
| 1 | JINX_W_IMPACT | The electric/spark effect when Zap! hits |
| 2 | JINX_R_ROCKET | The rocket missile in flight |
| 3 | JINX_R_IMPACT | The explosion when rocket hits |

### Bounding Box Guidelines

#### W Projectile (Class 0)
- Draw box around the entire visible laser beam
- Include the full length from start to end
- Box should be elongated/narrow

```
    ┌─────────────────────────────┐
    │ ═══════════════════════════ │  ← W beam
    └─────────────────────────────┘
```

#### W Impact (Class 1)
- Draw box around the hit spark/electric effect
- Usually appears at the point of contact
- Relatively small box

```
        ┌───────┐
        │  ✦✧✦  │  ← Impact spark
        └───────┘
```

#### R Rocket (Class 2)
- Draw box around the rocket body
- Do NOT include the trail/exhaust
- Rocket grows larger as it travels

```
        ┌───────────┐
        │    🚀     │  ← Rocket body only
        └───────────┘
```

#### R Impact (Class 3)
- Draw box around the explosion effect
- Include the full explosion radius
- Usually a large box

```
    ┌───────────────────┐
    │                   │
    │    💥 BOOM 💥     │  ← Full explosion
    │                   │
    └───────────────────┘
```

### Labeling Strategy

**DO Label:**
- Every frame where VFX is clearly visible
- Multiple objects if both W and R are on screen
- Frames where ability is partially visible

**DON'T Label:**
- Frames with no W/R abilities visible (keep as negatives)
- Ambiguous/blurry frames
- Jinx's other abilities (Q, E)

### Negative Samples

Keep ~20-30% of frames unlabeled as negative samples:
- Frames between ability casts
- Frames with other champions' abilities
- General gameplay without Jinx W/R

These help the model learn what NOT to detect.

---

## Step 4: Quality Check

Before training, verify your labels:

```bash
python tools/verify_dataset.py --config configs/jinx_abilities.yaml
```

This checks:
- All images have matching labels (or are intentional negatives)
- Class IDs are valid (0-3 only)
- Bounding boxes are normalized (0-1 range)
- No corrupted files

---

## Example Labeling Session

1. **Open LabelImg** with the unlabeled folder
2. **Set format** to YOLO (View → Auto Save mode)
3. **For each frame:**
   - Press `W` to create box
   - Draw around VFX
   - Select class from dropdown
   - Press `D` for next image
4. **Skip frames** with no W/R abilities (they become negatives)
5. **Save frequently** (Ctrl+S)

---

## Recommended Quantities

For initial training:

| Class | Min Samples | Target |
|-------|-------------|--------|
| JINX_W_PROJECTILE | 50 | 100+ |
| JINX_W_IMPACT | 30 | 75+ |
| JINX_R_ROCKET | 50 | 100+ |
| JINX_R_IMPACT | 30 | 75+ |
| Negatives | 50 | 100+ |

**Total: ~250-500 labeled frames for first iteration**

---

## Common Mistakes to Avoid

1. **Too loose boxes** - Keep boxes tight around VFX
2. **Missing partial VFX** - Label even if ability is at edge of screen
3. **Wrong class** - Double-check W vs R
4. **Overlapping boxes** - Each VFX instance gets its own box
5. **Labeling other abilities** - Only W and R, not Q/E/autos





