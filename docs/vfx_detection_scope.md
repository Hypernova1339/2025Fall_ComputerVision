# VFX detection scope

YOLO-based detector to localize ability visual effects (VFX) in the main game viewport. Designed to complement minimap/HUD cues by providing spatial evidence of projectiles and impacts.

## Motivation
- Minimap trajectories alone cannot tell if narrow skillshots thread through or clip targets; main-screen VFX provide the hit-or-miss evidence.
- HUD signals when an ability is used; VFX show what happened on-screen, especially when the camera moves.

## Initial scope (v1)
- Champion: Jinx only.
- Classes: `JINX_W_PROJECTILE` (core), `JINX_W_IMPACT` (optional), `JINX_R_PROJECTILE` (stretch), `JINX_R_IMPACT` (stretch).
- Negative frames: similar game context with no relevant VFX to reduce false positives.

## Labeling schema
- YOLO bounding boxes in screen coordinates on sampled frames.
- Projectiles: tight box around the full streak, not just the brightest core.
- Impacts: box the explosion/spark when clearly visible.

## Model and training
- Lightweight YOLO (nano/small) suited for elongated objects.
- Input downscaled (e.g., 1920x1080 -> 640x360) for speed.
- Augmentations: mild scale/flip/brightness; avoid heavy distortions that bend projectile shape.
- Standard YOLO losses; train on clips with repeated casts and balanced negatives.

## Runtime fusion (future extension)
- Per-frame detections form short VFX tracks (start/end time, trajectory, class).
- Fuse VFX tracks with minimap tracks to infer cast hit vs miss.
- HUD-based fallback for Jinx R when the projectile is off-screen (watch cooldown/icon flip).

## Event log (future extension)
- Fields: champion, ability, timestamp, screen-space track (if any), minimap hit/miss, source (VFX detector, HUD detector, or both).
- Serves downstream analytics and higher-level classifiers.
