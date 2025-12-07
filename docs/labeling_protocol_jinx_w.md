# Fast labeling protocol (Jinx W)

Goal: finish a high-quality starter dataset in ~30-45 minutes with balanced positives and negatives for Jinx W projectile and impact detection.

## Phase 1 — Prep (5 minutes)
- Pick a single 15-30s clip that contains multiple Jinx W casts.
- Extract frames (example):  
  `python scripts/extract_frames.py --video data/raw_videos/jinx_clip.mp4 --out data/images/train --every 3`  
  This yields ~150-300 frames quickly.
- Open LabelImg in YOLO format. Save labels to `data/labels/train`.

## Phase 2 — Triage first (critical)
- Do not label every frame. Select 40-60 useful frames into a temp folder like `data/images/train_selected/`.
- Target mix: 20-30 projectile mid-flight, 10-15 cast moments, 5-10 clear impacts, 10-15 negative frames (no projectile or impact).

## Phase 3 — Label in high-impact order (10-15 minutes)
1) Negative frames (~2 minutes): open and save with no boxes.  
2) Projectile frames (~8-10 minutes): one tight box around the full projectile. Label as class `0` (JINX_W_PROJECTILE). Skip blurry or edge clips. Aim for 20-30 good samples.  
3) Impact frames (~5 minutes): box around explosion/sparks. Label as class `1` (JINX_W_IMPACT). 5-10 samples are enough to teach appearance.

## Phase 4 — Organize (2 minutes)
- Move labeled images to `data/images/train` and labels to `data/labels/train`.
- Create a 10% validation split (e.g., 6-10 frames) under `data/images/val` and `data/labels/val`, keeping image/label pairs together.

## Phase 5 — Verify (3 minutes)
- Every labeled image has a matching `.txt` file; negatives do not.
- Class IDs are only `0` or `1`.
- `configs/jinx_w.yaml` paths point to your dataset folders.

## Phase 6 — Train
- Kick off training immediately after labeling:  
  `python src/training/train.py --config configs/jinx_w.yaml --epochs 50 --imgsz 640`
- Training takes time but no supervision; this dataset size is enough for a functional demo model.
