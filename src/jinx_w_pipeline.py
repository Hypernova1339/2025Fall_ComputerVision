"""
Unified CLI for training and running the Jinx W VFX detector.

Usage:
  Train:
    python src/jinx_w_pipeline.py train --data-config configs/jinx_w.yaml --epochs 50 --imgsz 640 --model-size yolov8n.pt

  Inference + events:
    python src/jinx_w_pipeline.py infer --weights runs/detect/train/weights/best.pt --video gameplay.mp4 --out-video outputs/jinx_w_overlay.mp4 --out-events outputs/jinx_w_events.jsonl
"""
import argparse
import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
from ultralytics import YOLO


# --------------------
# Helpers
# --------------------
@dataclass
class Detection:
    frame_idx: int
    cls: int
    conf: float
    center: tuple[float, float]


@dataclass
class ProjectileTrack:
    track_id: int
    start_frame: int
    detections: list[Detection] = field(default_factory=list)

    @property
    def last_det(self) -> Detection:
        return self.detections[-1]

    def add(self, det: Detection) -> None:
        self.detections.append(det)


def box_center_xyxy(box: Iterable[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)


def euclidean(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# --------------------
# Training
# --------------------
def run_train(args: argparse.Namespace) -> None:
    project = Path(args.project)
    project.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model_size)
    results = model.train(
        data=args.data_config,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(project),
        name=args.run_name,
        exist_ok=True,
    )
    final_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"Training complete. Best weights: {final_weights}")


# --------------------
# Inference + events
# --------------------
def load_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()
    return fps if fps > 0 else 30.0


def track_projectiles_and_impacts(
    results_stream: Iterable[Any],
    fps: float,
    hit_radius: float,
    max_gap: int,
    impact_window: int,
) -> tuple[list[dict[str, Any]], Path | None]:
    tracks: list[ProjectileTrack] = []
    active_track: ProjectileTrack | None = None
    impacts: list[Detection] = []
    track_id = 0
    save_dir: Path | None = None

    for frame_idx, result in enumerate(results_stream):
        if save_dir is None and hasattr(result, "save_dir"):
            try:
                save_dir = Path(result.save_dir)
            except TypeError:
                save_dir = None
        projectile_dets: list[Detection] = []
        impact_dets: list[Detection] = []
        boxes = result.boxes
        cls_list = boxes.cls.tolist() if boxes.cls is not None else []
        conf_list = boxes.conf.tolist() if boxes.conf is not None else []
        xyxy_list = boxes.xyxy.tolist() if boxes.xyxy is not None else []

        for cls_id, conf, box in zip(cls_list, conf_list, xyxy_list):
            center = box_center_xyxy(box)
            det = Detection(frame_idx=frame_idx, cls=int(cls_id), conf=float(conf), center=center)
            if det.cls == 0:
                projectile_dets.append(det)
            elif det.cls == 1:
                impact_dets.append(det)

        # Update impacts list
        impacts.extend(impact_dets)

        # Sort projectile dets by confidence (desc) for deterministic assignment
        projectile_dets.sort(key=lambda d: d.conf, reverse=True)

        # Link or create track
        if active_track and projectile_dets:
            candidate = projectile_dets[0]
            gap_ok = (candidate.frame_idx - active_track.last_det.frame_idx) <= max_gap
            dist_ok = euclidean(candidate.center, active_track.last_det.center) <= hit_radius
            if gap_ok and dist_ok:
                active_track.add(candidate)
                projectile_dets = projectile_dets[1:]
            else:
                tracks.append(active_track)
                active_track = None

        if not active_track and projectile_dets:
            active_track = ProjectileTrack(track_id=track_id, start_frame=projectile_dets[0].frame_idx)
            active_track.add(projectile_dets[0])
            track_id += 1

    # Finalize last active track
    if active_track:
        tracks.append(active_track)

    events: list[dict[str, Any]] = []
    for idx, track in enumerate(tracks):
        last_det = track.last_det
        hit = None
        for imp in impacts:
            if 0 <= (imp.frame_idx - last_det.frame_idx) <= impact_window:
                if euclidean(imp.center, last_det.center) <= hit_radius:
                    hit = imp
                    break

        event = {
            "event_id": idx,
            "ability": "JINX_W",
            "cast_time_sec": track.start_frame / fps,
            "is_hit_vfx": hit is not None,
            "hit_time_sec": (hit.frame_idx / fps) if hit else None,
            "hit_location_screen": hit.center if hit else None,
            "impact_target": "unknown" if hit else None,  # placeholder; requires additional target classifier
            "source": ["VFX_DETECTOR"],
            "track_len_frames": len(track.detections),
        }
        events.append(event)

    return events, save_dir


def run_infer(args: argparse.Namespace) -> None:
    video_path = Path(args.video)
    out_events = Path(args.out_events)
    out_events.parent.mkdir(parents=True, exist_ok=True)
    project = Path(args.project)
    project.mkdir(parents=True, exist_ok=True)

    fps = load_fps(video_path)
    model = YOLO(str(args.weights))

    save_dir: Path | None = None
    results_stream = model.predict(
        source=str(video_path),
        stream=True,
        save=True,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        project=str(project),
        name=args.run_name,
        exist_ok=True,
    )

    # Consume stream while tracking
    events, save_dir = track_projectiles_and_impacts(
        results_stream=results_stream,
        fps=fps,
        hit_radius=args.hit_radius,
        max_gap=args.max_gap,
        impact_window=args.impact_window,
    )

    # Write events JSONL
    with out_events.open("w", encoding="utf-8") as f:
        for evt in events:
            f.write(json.dumps(evt) + "\n")
    print(f"Wrote events to {out_events}")

    # Locate saved overlay video and copy/rename to requested path
    # Ultralytics saves to project/run_name/{source_stem}.mp4
    run_dir = save_dir if save_dir else (project / args.run_name)
    candidates = list(run_dir.glob("*.mp4")) if run_dir.exists() else []
    if candidates and args.out_video:
        out_video = Path(args.out_video)
        out_video.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidates[0], out_video)
        print(f"Saved overlay video to {out_video}")
    elif not candidates:
        print("No overlay video found in prediction output.")


# --------------------
# CLI
# --------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Jinx W VFX detector pipeline (train and inference).")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train
    p_train = subparsers.add_parser("train", help="Train YOLO VFX detector.")
    p_train.add_argument("--data-config", type=str, default="configs/jinx_w.yaml")
    p_train.add_argument("--model-size", type=str, default="yolov8n.pt", help="Base model weights or YAML.")
    p_train.add_argument("--epochs", type=int, default=50)
    p_train.add_argument("--imgsz", type=int, default=640)
    p_train.add_argument("--batch", type=int, default=16)
    p_train.add_argument("--device", type=str, default="auto")
    p_train.add_argument("--workers", type=int, default=4)
    p_train.add_argument("--project", type=Path, default=Path("runs/detect"))
    p_train.add_argument("--run-name", type=str, default="train")

    # Infer
    p_infer = subparsers.add_parser("infer", help="Run inference + event generation.")
    p_infer.add_argument("--weights", type=Path, required=True, help="Path to trained weights (.pt).")
    p_infer.add_argument("--video", type=str, required=True, help="Video path for inference.")
    p_infer.add_argument("--out-video", type=str, default="outputs/jinx_w_overlay.mp4", help="Overlay video path.")
    p_infer.add_argument("--out-events", type=str, default="outputs/jinx_w_events.jsonl", help="Event log JSONL path.")
    p_infer.add_argument("--imgsz", type=int, default=640)
    p_infer.add_argument("--conf", type=float, default=0.25)
    p_infer.add_argument("--iou", type=float, default=0.45)
    p_infer.add_argument("--hit-radius", type=float, default=80.0, help="Pixel radius to match impact to projectile end.")
    p_infer.add_argument("--impact-window", type=int, default=6, help="Frames after projectile end to search for impact.")
    p_infer.add_argument("--max-gap", type=int, default=2, help="Max frame gap to continue a projectile track.")
    p_infer.add_argument("--project", type=Path, default=Path("runs/detect"))
    p_infer.add_argument("--run-name", type=str, default="infer")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        run_train(args)
    elif args.command == "infer":
        run_infer(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
