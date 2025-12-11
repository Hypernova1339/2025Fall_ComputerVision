Use this prompt to compile a cast/hit event log from YOLO VFX detections:

```
You are given per-frame YOLO detections for League of Legends VFX with classes:
- 0: JINX_W_PROJECTILE
- 1: JINX_W_IMPACT
- 2: JINX_R_PROJECTILE
- 3: JINX_R_IMPACT

Goal: produce JSONL events with fields:
  event_id, ability (JINX_W or JINX_R), cast_time_sec, is_hit_vfx (true/false/null),
  hit_time_sec (nullable), hit_location_screen [x, y] (nullable),
  impact_target ("champion" | "minion" | "unknown"), source ["VFX_DETECTOR"].

Rules:
1) Link consecutive projectile detections into a track if the center moves smoothly and frame gaps <= 2.
2) Cast time = first frame of the track (convert via fps).
3) A hit occurs if an IMPACT appears within 3–6 frames after the projectile ends and within ~80 px of the last projectile center.
4) Impact target: set to "unknown" unless an external target detector links the impact to a champion/minion; otherwise choose the closest detected target class at impact.
5) If no impact is found, set is_hit_vfx = false and hit_* = null (impact_target = "unknown").
6) Number events sequentially (event_id).

Return JSONL (one event per line). Do not include any other text.
```

Context: detections come from `python src/jinx_w_pipeline.py infer ...`, and FPS is taken from the video metadata.
