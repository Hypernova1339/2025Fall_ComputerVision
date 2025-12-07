# League of Legends Ability VFX Detector

Computer vision project skeleton for detecting League of Legends champion ability visual effects (VFX) on the main screen. The initial scope targets Jinx W (Zap!) projectiles and impacts using a lightweight YOLO-style detector, with room to extend to other abilities and champions.

## Repo structure
- `configs/` — data/model configs (`jinx_w.yaml` for YOLO dataset layout).
- `docs/` — labeling protocol and detection scope notes.
- `scripts/` — utilities such as frame extraction.
- `src/data/` — data utilities imported by scripts and training.
- `src/training/` — training entrypoints.
- `src/inference/` — inference entrypoints.
- `src/models/` — model registry or helper definitions.
- `data/` — expected dataset root (kept empty; ignored by git).
- `outputs/` — checkpoints, logs, and predictions (ignored by git).

## Quickstart
1. Install Python 3.10+ and FFmpeg (for video decoding) on your machine.
2. Install deps:
   ```
   pip install -r requirements.txt
   ```
3. Arrange data in YOLO format (paths relative to repo):
   ```
   data/
     raw_videos/             # source clips (ignored by git)
     images/
       train/
       val/
     labels/
       train/
       val/
   ```
4. Extract frames from a clip (every 3rd frame into train images):
   ```
   python scripts/extract_frames.py --video data/raw_videos/jinx_clip.mp4 --out data/images/train --every 3
   ```
5. Label frames in LabelImg (YOLO format) with two classes:
   - `0`: JINX_W_PROJECTILE
   - `1`: JINX_W_IMPACT
   Save labels into `data/labels/train`. Move ~10% to `val` for validation.
   See `docs/labeling_protocol_jinx_w.md` for a fast triage-first workflow.

## Verify dataset structure (optional sanity check)
```
python src/data/verify_dataset.py --images data/images/train --labels data/labels/train --names-file configs/jinx_w.yaml
```

## Block 2 — Train the detector
Run YOLO training (defaults to `runs/detect/train/weights/best.pt`):
```
python src/jinx_w_pipeline.py train \
  --data-config configs/jinx_w.yaml \
  --epochs 50 \
  --imgsz 640 \
  --model-size yolov8n.pt
```
Outputs: `runs/detect/train/weights/best.pt`, results plots, metrics.

## Block 3 — Inference + events
Generate overlay video and VFX events from a gameplay clip:
```
python src/jinx_w_pipeline.py infer \
  --weights runs/detect/train/weights/best.pt \
  --video gameplay.mp4 \
  --out-video outputs/jinx_w_overlay.mp4 \
  --out-events outputs/jinx_w_events.jsonl
```
Events log includes cast time, hit/no-hit (based on impact near projectile end), and screen location. Overlay video is copied to `outputs/jinx_w_overlay.mp4`.

## Extending
- Add more abilities or champions: create a new data config in `configs/`, add class names, and collect labels.
- Integrate minimap/HUD fusion later by adding modules under `src/` and wiring them in `src/pipeline.py`.
- For labeling tips and VFX scope rationale, read `docs/vfx_detection_scope.md`.
