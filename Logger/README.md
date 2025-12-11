# Ability Event Logger (Jinx-focused)

This folder describes how to fabricate a cast/hit log from VFX detections, using the four classes:
- `JINX_W_PROJECTILE`
- `JINX_W_IMPACT`
- `JINX_R_PROJECTILE`
- `JINX_R_IMPACT`

## Inputs
1) Per-frame YOLO detections with boxes and classes (from `jinx_w_pipeline.py infer` or a similar runner).
2) Video FPS to convert frames → seconds.

## Event construction
- **Projectile track**: Link consecutive `JINX_W_PROJECTILE` or `JINX_R_PROJECTILE` boxes into a track by proximity and small frame gaps.
- **Cast time**: First frame of the projectile track.
- **Hit detection**: If an `IMPACT` box appears near the final projectile position within a short window (e.g., 3–6 frames), mark `is_hit_vfx = true` and record impact time/location.

## Output format (JSONL)
One event per line, now including an impact target hint:
```json
{
  "event_id": 12,
  "ability": "JINX_W",
  "cast_time_sec": 104.53,
  "is_hit_vfx": true,
  "hit_time_sec": 105.01,
  "hit_location_screen": [883.4, 472.1],
  "impact_target": "champion",  // or "minion" or "unknown"
  "source": ["VFX_DETECTOR"]
}
```

## Quick flow
1) Run inference to produce per-frame detections + overlay:
   ```
   python src/jinx_w_pipeline.py infer \
     --weights runs/detect/train/weights/best.pt \
     --video data/raw_videos/gameplay.mp4 \
     --out-video outputs/jinx_w_overlay.mp4 \
     --out-events outputs/jinx_w_events.jsonl
   ```
2) The JSONL log is your fabricated ability event log (cast → hit/no-hit). The overlay video is optional for review.

## Extending to Jinx R
- Track `JINX_R_PROJECTILE` and match to `JINX_R_IMPACT` the same way.
- If R leaves the screen, supplement with HUD cues (cooldown flip) to emit `is_hit_vfx = null` events with `source = ["HUD_DETECTOR"]`.

## Champion vs. minion impact notes
- The pipeline exposes `impact_target` but currently sets it to `"unknown"` unless you fuse with a target detector.
- To populate it, add a lightweight classifier/detector for on-screen champions/minions and choose the closest target to the impact center; otherwise leave `"unknown"`.
