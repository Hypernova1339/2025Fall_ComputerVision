"""
Jinx W/R VFX Detection Pipeline

Main pipeline for training and inference on Jinx ability VFX detection.
Supports both W ("Zap!") and R ("Super Mega Death Rocket!") abilities.

Usage:
    python -m src.jinx_wr_pipeline train --data-config configs/jinx_wr.yaml --model-size n --epochs 50
    python -m src.jinx_wr_pipeline infer --weights outputs/train/weights/best.pt --video input.mp4 --out-video output.mp4 --out-events events.jsonl
"""
import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Class constants (must not change)
PROJECTILE_CLASSES = {0, 2}  # JINX_W_PROJECTILE, JINX_R_PROJECTILE
IMPACT_CLASSES = {1, 3}  # JINX_W_IMPACT, JINX_R_IMPACT
IMPACT_FOR_PROJECTILE = {
    0: 1,  # W projectile → W impact
    2: 3,  # R projectile → R impact
}

CLASS_NAMES = {
    0: "JINX_W_PROJECTILE",
    1: "JINX_W_IMPACT",
    2: "JINX_R_PROJECTILE",
    3: "JINX_R_IMPACT",
}

ABILITY_FOR_CLASS = {
    0: "W",
    1: "W",
    2: "R",
    3: "R",
}


@dataclass
class Detection:
    """Single detection bounding box."""

    cls_id: int
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    center: Tuple[float, float]  # cx, cy

    @classmethod
    def from_yolo_result(cls, result) -> List["Detection"]:
        """Convert Ultralytics result to Detection objects."""
        detections = []
        if result.boxes is None:
            return detections

        boxes = result.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            detections.append(
                cls(
                    cls_id=cls_id,
                    confidence=conf,
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    center=(float(cx), float(cy)),
                )
            )
        return detections


@dataclass
class ImpactDetection:
    """Impact detection with class ID for type matching."""

    cls_id: int
    confidence: float
    center: Tuple[float, float]
    frame: int
    time_sec: float


@dataclass
class ProjectileTrack:
    """Tracked projectile with trajectory and hit association."""

    track_id: int
    cls_id: int
    cast_frame: int
    end_frame: int
    cast_time_sec: float
    end_time_sec: float
    trajectory: List[Tuple[float, float]] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    last_bbox: Optional[Tuple[float, float, float, float]] = None  # x1, y1, x2, y2
    impact: Optional[ImpactDetection] = None

    @property
    def ability(self) -> str:
        """Get ability name (W or R)."""
        return ABILITY_FOR_CLASS[self.cls_id]

    @property
    def class_name(self) -> str:
        """Get class name."""
        return CLASS_NAMES[self.cls_id]

    @property
    def avg_confidence(self) -> float:
        """Calculate average confidence."""
        return float(np.mean(self.confidences)) if self.confidences else 0.0

    @property
    def is_hit_vfx(self) -> bool:
        """Check if projectile hit (has associated impact)."""
        return self.impact is not None

    def to_event_dict(self, event_id: int) -> dict:
        """Convert track to event log dictionary."""
        event = {
            "event_id": event_id,
            "champion": "Jinx",
            "ability": self.ability,
            "class_id": self.cls_id,
            "class_name": self.class_name,
            "cast_frame": self.cast_frame,
            "end_frame": self.end_frame,
            "cast_time_sec": round(self.cast_time_sec, 2),
            "end_time_sec": round(self.end_time_sec, 2),
            "avg_confidence": round(self.avg_confidence, 2),
            "trajectory": [[float(x), float(y)] for x, y in self.trajectory],
            "is_hit_vfx": self.is_hit_vfx,
            "hit_frame": self.impact.frame if self.impact else None,
            "hit_time_sec": round(self.impact.time_sec, 2) if self.impact else None,
            "hit_location_screen": [float(self.impact.center[0]), float(self.impact.center[1])]
            if self.impact
            else None,
            "hit_confidence": round(self.impact.confidence, 2) if self.impact else None,
            "source": ["VFX_DETECTOR"],
        }
        return event


class IOUTracker:
    """Simple IOU-based tracker for projectiles."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 5):
        """
        Initialize tracker.

        Args:
            iou_threshold: Minimum IOU to associate detection with track
            max_age: Maximum frames a track can go without updates before being removed
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: Dict[int, ProjectileTrack] = {}
        self.next_track_id = 0

    def update(
        self,
        detections: List[Detection],
        frame: int,
        time_sec: float,
    ) -> List[ProjectileTrack]:
        """
        Update tracks with new detections.

        Args:
            detections: List of projectile detections (filtered to PROJECTILE_CLASSES)
            frame: Current frame number
            time_sec: Current time in seconds

        Returns:
            List of active tracks
        """
        # Filter to projectile classes only
        projectile_detections = [d for d in detections if d.cls_id in PROJECTILE_CLASSES]

        # Match detections to existing tracks
        matched_tracks = set()
        matched_detections = set()

        for track_id, track in self.tracks.items():
            if track.end_frame < frame - self.max_age:
                continue  # Track is too old

            best_iou = 0.0
            best_detection_idx = None

            for idx, det in enumerate(projectile_detections):
                if idx in matched_detections:
                    continue
                if det.cls_id != track.cls_id:
                    continue  # Must match class

                # Calculate IOU with last bounding box
                if track.last_bbox is not None:
                    iou = calculate_iou(track.last_bbox, det.bbox)
                else:
                    # New track, use a default IOU to allow matching
                    iou = 0.5

                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_detection_idx = idx

            if best_detection_idx is not None:
                det = projectile_detections[best_detection_idx]
                track.trajectory.append(det.center)
                track.confidences.append(det.confidence)
                track.last_bbox = det.bbox
                track.end_frame = frame
                track.end_time_sec = time_sec
                matched_tracks.add(track_id)
                matched_detections.add(best_detection_idx)

        # Create new tracks for unmatched detections
        for idx, det in enumerate(projectile_detections):
            if idx not in matched_detections:
                new_track = ProjectileTrack(
                    track_id=self.next_track_id,
                    cls_id=det.cls_id,
                    cast_frame=frame,
                    end_frame=frame,
                    cast_time_sec=time_sec,
                    end_time_sec=time_sec,
                    trajectory=[det.center],
                    confidences=[det.confidence],
                )
                self.tracks[self.next_track_id] = new_track
                self.next_track_id += 1

        # Return active tracks
        active_tracks = [
            track
            for track_id, track in self.tracks.items()
            if track.end_frame >= frame - self.max_age
        ]
        return active_tracks

    def get_finished_tracks(self, current_frame: int) -> List[ProjectileTrack]:
        """Get tracks that have finished (not seen for max_age frames)."""
        finished = [
            track
            for track_id, track in self.tracks.items()
            if track.end_frame < current_frame - self.max_age
        ]
        # Remove finished tracks
        self.tracks = {
            track_id: track
            for track_id, track in self.tracks.items()
            if track.end_frame >= current_frame - self.max_age
        }
        return finished


def calculate_iou(bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float]) -> float:
    """Calculate Intersection over Union (IOU) between two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def associate_impacts_with_tracks(
    tracks: List[ProjectileTrack],
    impacts: List[ImpactDetection],
    max_time_gap: float = 2.0,
    max_space_dist: float = 100.0,
) -> None:
    """
    Associate impact detections with projectile tracks.

    Args:
        tracks: List of finished projectile tracks
        impacts: List of impact detections
        max_time_gap: Maximum time gap (seconds) between track end and impact
        max_space_dist: Maximum spatial distance (pixels) between track end and impact
    """
    for track in tracks:
        if track.is_hit_vfx:
            continue  # Already associated

        expected_impact_cls = IMPACT_FOR_PROJECTILE.get(track.cls_id)
        if expected_impact_cls is None:
            continue

        # Find matching impact
        best_impact = None
        best_score = 0.0

        for impact in impacts:
            if impact.cls_id != expected_impact_cls:
                continue

            # Check time gap
            time_gap = abs(impact.time_sec - track.end_time_sec)
            if time_gap > max_time_gap:
                continue

            # Check spatial distance
            if track.trajectory:
                last_pos = track.trajectory[-1]
                space_dist = np.sqrt(
                    (impact.center[0] - last_pos[0]) ** 2 + (impact.center[1] - last_pos[1]) ** 2
                )
                if space_dist > max_space_dist:
                    continue

                # Score based on time and space proximity
                score = 1.0 / (1.0 + time_gap + space_dist / 50.0)
                if score > best_score:
                    best_score = score
                    best_impact = impact

        if best_impact is not None:
            track.impact = best_impact
            # Remove impact from list to avoid double association
            impacts.remove(best_impact)


def train_command(args: argparse.Namespace) -> None:
    """Train YOLO model on Jinx W/R VFX dataset."""
    logger.info("Starting training...")
    logger.info(f"Data config: {args.data_config}")
    logger.info(f"Model size: {args.model_size}")
    logger.info(f"Epochs: {args.epochs}")
    
    model_name = f"yolov8{args.model_size}.pt"
    logger.info(f"Loading base model: {model_name}")
    model = YOLO(model_name)

    results = model.train(
        data=str(args.data_config),
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
    )

    final_weights = Path(results.save_dir) / "weights" / "best.pt"
    logger.info(f"Training complete. Best weights: {final_weights}")


def infer_command(args: argparse.Namespace) -> None:
    """Run inference with tracking and event logging."""
    logger.info("Starting inference...")
    logger.info(f"Weights: {args.weights}")
    logger.info(f"Video: {args.video}")

    # Load model
    model = YOLO(str(args.weights))

    # Open video
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    logger.info(f"Video: {width}x{height} @ {fps:.2f} FPS, {total_frames} frames")

    # Setup video writer if output requested
    video_writer = None
    if args.out_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(args.out_video), fourcc, fps, (width, height))
        logger.info(f"Writing annotated video to: {args.out_video}")

    # Setup event log
    event_log = []
    event_id_counter = 0

    # Tracker and impact storage
    tracker = IOUTracker(iou_threshold=args.track_iou, max_age=args.max_age)
    all_impacts: List[ImpactDetection] = []
    finished_tracks: List[ProjectileTrack] = []

    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            time_sec = frame_idx / fps if fps > 0 else 0.0

            # Run YOLO inference
            results = model.predict(
                frame,
                conf=args.conf,
                imgsz=640,
                verbose=False,
            )
            result = results[0]

            # Convert to Detection objects
            detections = Detection.from_yolo_result(result)

            # Separate projectiles and impacts
            projectile_detections = [d for d in detections if d.cls_id in PROJECTILE_CLASSES]
            impact_detections = [d for d in detections if d.cls_id in IMPACT_CLASSES]

            # Store impacts
            for det in impact_detections:
                all_impacts.append(
                    ImpactDetection(
                        cls_id=det.cls_id,
                        confidence=det.confidence,
                        center=det.center,
                        frame=frame_idx,
                        time_sec=time_sec,
                    )
                )

            # Update tracker
            active_tracks = tracker.update(projectile_detections, frame_idx, time_sec)

            # Get finished tracks
            finished = tracker.get_finished_tracks(frame_idx)
            finished_tracks.extend(finished)

            # Draw annotations
            annotated_frame = frame.copy()

            # Draw all detections
            for det in detections:
                x1, y1, x2, y2 = det.bbox
                cls_name = CLASS_NAMES.get(det.cls_id, f"Class_{det.cls_id}")
                label = f"{cls_name} {det.confidence:.2f}"

                # Color by class
                if det.cls_id in PROJECTILE_CLASSES:
                    color = (0, 255, 0)  # Green for projectiles
                else:
                    color = (0, 0, 255)  # Red for impacts

                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(
                    annotated_frame,
                    label,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

            # Draw active tracks
            for track in active_tracks:
                if len(track.trajectory) > 1:
                    pts = np.array(track.trajectory, dtype=np.int32)
                    cv2.polylines(annotated_frame, [pts], False, (255, 255, 0), 2)

            if video_writer:
                video_writer.write(annotated_frame)

            frame_idx += 1

            if frame_idx % 100 == 0:
                logger.info(f"Processed {frame_idx}/{total_frames} frames")

    finally:
        cap.release()
        if video_writer:
            video_writer.release()
            logger.info(f"Video written to: {args.out_video}")

    # Process any remaining active tracks
    remaining_tracks = list(tracker.tracks.values())
    finished_tracks.extend(remaining_tracks)

    # Associate impacts with tracks
    logger.info(f"Associating {len(all_impacts)} impacts with {len(finished_tracks)} tracks...")
    associate_impacts_with_tracks(
        finished_tracks,
        all_impacts,
        max_time_gap=args.max_time_gap,
        max_space_dist=args.max_space_dist,
    )

    # Generate event log
    logger.info("Generating event log...")
    MIN_TRACK_LEN = 3          # at least 3 frames
    MIN_AVG_CONF = 0.4         # empirical; adjust as needed

    for track in finished_tracks:
        if len(track.trajectory) < MIN_TRACK_LEN:
            continue
        if track.avg_confidence < MIN_AVG_CONF:
            continue

        event = track.to_event_dict(event_id_counter)
        event_log.append(event)
        event_id_counter += 1

    # Write event log
    if args.out_events:
        with open(args.out_events, "w") as f:
            for event in event_log:
                f.write(json.dumps(event) + "\n")
        logger.info(f"Event log written to: {args.out_events} ({len(event_log)} events)")

    logger.info("Inference complete!")


def main() -> None:
    """Main entry point with subcommands."""
    parser = argparse.ArgumentParser(
        description="Jinx W/R VFX Detection Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    subparsers.required = True

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train YOLO model")
    train_parser.add_argument(
        "--data-config",
        type=Path,
        required=True,
        help="Path to YOLO dataset config YAML",
    )
    train_parser.add_argument(
        "--model-size",
        type=str,
        default="n",
        choices=["n", "s", "m", "l", "x"],
        help="YOLOv8 model size",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    train_parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for training",
    )
    train_parser.add_argument(
        "--project",
        type=Path,
        default=Path("outputs"),
        help="Project directory for training outputs",
    )
    train_parser.add_argument(
        "--name",
        type=str,
        default="jinx_wr",
        help="Run name",
    )

    # Infer subcommand
    infer_parser = subparsers.add_parser("infer", help="Run inference with tracking")
    infer_parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to trained YOLO weights (.pt)",
    )
    infer_parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to input video",
    )
    infer_parser.add_argument(
        "--out-video",
        type=Path,
        default=None,
        help="Path to output annotated video (optional)",
    )
    infer_parser.add_argument(
        "--out-events",
        type=Path,
        required=True,
        help="Path to output JSONL event log",
    )
    infer_parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections",
    )
    infer_parser.add_argument(
        "--track-iou",
        type=float,
        default=0.3,
        help="IOU threshold for track association",
    )
    infer_parser.add_argument(
        "--max-age",
        type=int,
        default=5,
        help="Maximum frames a track can go without updates",
    )
    infer_parser.add_argument(
        "--max-time-gap",
        type=float,
        default=2.0,
        help="Maximum time gap (seconds) between track end and impact",
    )
    infer_parser.add_argument(
        "--max-space-dist",
        type=float,
        default=100.0,
        help="Maximum spatial distance (pixels) between track end and impact",
    )

    args = parser.parse_args()

    if args.command == "train":
        train_command(args)
    elif args.command == "infer":
        infer_command(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

