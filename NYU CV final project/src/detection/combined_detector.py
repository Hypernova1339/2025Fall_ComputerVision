"""
Combined UI + VFX detection pipeline.

Integrates UI cooldown detection with world VFX detection for robust
ability detection and hit/miss classification.

Detection Strategy:
1. UI Detection: Monitors ability bar for cooldown transitions
   - READY → COOLDOWN = ability was cast (exact timing)
   
2. VFX Detection: YOLO model detects projectiles and impacts
   - Confirms ability in the game world
   - Detects hit (impact seen) vs miss (no impact)

3. Event Aggregation: Combines both signals
   - UI provides precise cast times
   - VFX provides hit/miss classification
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import json

from .ui_detector import UIDetector, CooldownEvent, AbilityState


@dataclass
class CombinedEvent:
    """
    Ability event detected from combined UI + VFX signals.
    """
    event_id: int
    ability: str  # "W" or "R"
    
    # Timing
    cast_frame: int
    cast_time_sec: float
    impact_frame: Optional[int] = None
    impact_time_sec: Optional[float] = None
    
    # Detection sources
    ui_detected: bool = False      # Cast detected from UI
    vfx_projectile_detected: bool = False  # Projectile seen in world
    vfx_impact_detected: bool = False      # Impact seen in world
    
    # Classification
    hit: Optional[bool] = None
    
    # Confidence scores
    ui_confidence: float = 0.0
    vfx_projectile_confidence: float = 0.0
    vfx_impact_confidence: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "ability": self.ability,
            "cast_frame": self.cast_frame,
            "cast_time": self.cast_time_sec,
            "impact_frame": self.impact_frame,
            "impact_time": self.impact_time_sec,
            "hit": self.hit,
            "sources": {
                "ui": self.ui_detected,
                "vfx_projectile": self.vfx_projectile_detected,
                "vfx_impact": self.vfx_impact_detected,
            },
            "confidence": {
                "ui": self.ui_confidence,
                "vfx_projectile": self.vfx_projectile_confidence,
                "vfx_impact": self.vfx_impact_confidence,
                "overall": self.overall_confidence,
            }
        }
    
    @property
    def overall_confidence(self) -> float:
        """Calculate overall detection confidence."""
        scores = []
        if self.ui_detected:
            scores.append(self.ui_confidence)
        if self.vfx_projectile_detected:
            scores.append(self.vfx_projectile_confidence)
        if self.vfx_impact_detected:
            scores.append(self.vfx_impact_confidence)
        return sum(scores) / len(scores) if scores else 0.0


# Class ID mappings for the 8-class model
CLASS_IDS = {
    # UI classes
    "UI_W_READY": 0,
    "UI_W_COOLDOWN": 1,
    "UI_R_READY": 2,
    "UI_R_COOLDOWN": 3,
    # VFX classes
    "VFX_W_PROJECTILE": 4,
    "VFX_W_IMPACT": 5,
    "VFX_R_ROCKET": 6,
    "VFX_R_IMPACT": 7,
}

# Reverse mapping
CLASS_NAMES = {v: k for k, v in CLASS_IDS.items()}


class CombinedDetector:
    """
    Combined detection pipeline using UI + VFX signals.
    """
    
    # Time windows for impact detection (seconds after cast)
    IMPACT_WINDOWS = {
        "W": (0.1, 1.5),   # W impact expected within 0.1-1.5s
        "R": (0.2, 10.0),  # R can travel for a long time
    }
    
    def __init__(
        self,
        yolo_weights: Optional[str | Path] = None,
        use_ui_detection: bool = True,
        use_vfx_detection: bool = True,
        conf_threshold: float = 0.25,
    ):
        """
        Initialize combined detector.
        
        Args:
            yolo_weights: Path to trained YOLO weights (None = UI only)
            use_ui_detection: Enable UI-based detection
            use_vfx_detection: Enable VFX-based detection
            conf_threshold: Confidence threshold for detections
        """
        self.yolo_weights = Path(yolo_weights) if yolo_weights else None
        self.use_ui_detection = use_ui_detection
        self.use_vfx_detection = use_vfx_detection
        self.conf_threshold = conf_threshold
        
        # Detectors
        self.ui_detector: Optional[UIDetector] = None
        self.yolo_model = None
        
        # State tracking
        self.events: List[CombinedEvent] = []
        self.event_counter = 0
        self.pending_events: Dict[str, CombinedEvent] = {}  # ability -> pending event
        
    def load(self) -> None:
        """Load detection models."""
        if self.use_ui_detection:
            self.ui_detector = UIDetector()
            
        if self.use_vfx_detection and self.yolo_weights:
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO(str(self.yolo_weights))
                print(f"Loaded YOLO model: {self.yolo_weights}")
            except ImportError:
                print("Warning: Ultralytics not installed, VFX detection disabled")
                self.use_vfx_detection = False
            except Exception as e:
                print(f"Warning: Could not load YOLO model: {e}")
                self.use_vfx_detection = False
                
    def reset(self) -> None:
        """Reset detection state."""
        self.events = []
        self.event_counter = 0
        self.pending_events = {}
        
        if self.ui_detector:
            self.ui_detector.prev_w_state = AbilityState.UNKNOWN
            self.ui_detector.prev_r_state = AbilityState.UNKNOWN
            
    def process_frame(
        self,
        frame,
        frame_idx: int,
        fps: float = 60.0
    ) -> List[CombinedEvent]:
        """
        Process a single frame with combined detection.
        
        Args:
            frame: Video frame (BGR numpy array)
            frame_idx: Frame index
            fps: Video FPS
            
        Returns:
            List of newly completed events
        """
        import numpy as np
        
        timestamp = frame_idx / fps
        completed_events = []
        
        # --- UI Detection ---
        if self.use_ui_detection and self.ui_detector:
            ui_events = self.ui_detector.detect_frame(frame, frame_idx, fps)
            
            for ui_event in ui_events:
                # New ability cast detected from UI
                self.event_counter += 1
                event = CombinedEvent(
                    event_id=self.event_counter,
                    ability=ui_event.ability,
                    cast_frame=ui_event.frame_idx,
                    cast_time_sec=ui_event.timestamp_sec,
                    ui_detected=True,
                    ui_confidence=ui_event.confidence,
                )
                self.pending_events[ui_event.ability] = event
                
        # --- VFX Detection ---
        vfx_detections = {}  # class_id -> (confidence, bbox)
        
        if self.use_vfx_detection and self.yolo_model:
            results = self.yolo_model(frame, conf=self.conf_threshold, verbose=False)
            
            for result in results:
                if result.boxes is None:
                    continue
                    
                for i in range(len(result.boxes)):
                    class_id = int(result.boxes.cls[i].cpu().numpy())
                    conf = float(result.boxes.conf[i].cpu().numpy())
                    bbox = result.boxes.xyxy[i].cpu().numpy()
                    
                    # Keep highest confidence detection per class
                    if class_id not in vfx_detections or conf > vfx_detections[class_id][0]:
                        vfx_detections[class_id] = (conf, bbox)
                        
        # Process VFX detections
        w_projectile = vfx_detections.get(CLASS_IDS["VFX_W_PROJECTILE"])
        w_impact = vfx_detections.get(CLASS_IDS["VFX_W_IMPACT"])
        r_rocket = vfx_detections.get(CLASS_IDS["VFX_R_ROCKET"])
        r_impact = vfx_detections.get(CLASS_IDS["VFX_R_IMPACT"])
        
        # --- Event Aggregation ---
        
        # Check for W events
        if "W" in self.pending_events:
            event = self.pending_events["W"]
            time_since_cast = timestamp - event.cast_time_sec
            min_time, max_time = self.IMPACT_WINDOWS["W"]
            
            # Update with VFX detections
            if w_projectile:
                event.vfx_projectile_detected = True
                event.vfx_projectile_confidence = max(
                    event.vfx_projectile_confidence, w_projectile[0]
                )
                
            if w_impact and time_since_cast >= min_time:
                # Impact detected - HIT
                event.vfx_impact_detected = True
                event.vfx_impact_confidence = w_impact[0]
                event.impact_frame = frame_idx
                event.impact_time_sec = timestamp
                event.hit = True
                
                # Event complete
                self.events.append(event)
                completed_events.append(event)
                del self.pending_events["W"]
                
            elif time_since_cast > max_time:
                # Timeout - MISS
                event.hit = False
                self.events.append(event)
                completed_events.append(event)
                del self.pending_events["W"]
                
        # Check for R events
        if "R" in self.pending_events:
            event = self.pending_events["R"]
            time_since_cast = timestamp - event.cast_time_sec
            min_time, max_time = self.IMPACT_WINDOWS["R"]
            
            if r_rocket:
                event.vfx_projectile_detected = True
                event.vfx_projectile_confidence = max(
                    event.vfx_projectile_confidence, r_rocket[0]
                )
                
            if r_impact and time_since_cast >= min_time:
                # Impact detected - HIT
                event.vfx_impact_detected = True
                event.vfx_impact_confidence = r_impact[0]
                event.impact_frame = frame_idx
                event.impact_time_sec = timestamp
                event.hit = True
                
                self.events.append(event)
                completed_events.append(event)
                del self.pending_events["R"]
                
            elif time_since_cast > max_time:
                # Timeout - MISS
                event.hit = False
                self.events.append(event)
                completed_events.append(event)
                del self.pending_events["R"]
                
        # --- VFX-only detection (no UI signal) ---
        # Create events if VFX detected without prior UI detection
        
        if w_projectile and "W" not in self.pending_events:
            # W projectile seen without UI detection
            self.event_counter += 1
            event = CombinedEvent(
                event_id=self.event_counter,
                ability="W",
                cast_frame=frame_idx,
                cast_time_sec=timestamp,
                vfx_projectile_detected=True,
                vfx_projectile_confidence=w_projectile[0],
            )
            self.pending_events["W"] = event
            
        if r_rocket and "R" not in self.pending_events:
            # R rocket seen without UI detection
            self.event_counter += 1
            event = CombinedEvent(
                event_id=self.event_counter,
                ability="R",
                cast_frame=frame_idx,
                cast_time_sec=timestamp,
                vfx_projectile_detected=True,
                vfx_projectile_confidence=r_rocket[0],
            )
            self.pending_events["R"] = event
            
        return completed_events
        
    def finalize(self, final_timestamp: float) -> List[CombinedEvent]:
        """Finalize any pending events at end of video."""
        finalized = []
        
        for ability, event in list(self.pending_events.items()):
            event.hit = False  # No impact seen
            self.events.append(event)
            finalized.append(event)
            
        self.pending_events.clear()
        return finalized
        
    def get_summary(self) -> dict:
        """Get detection summary."""
        w_events = [e for e in self.events if e.ability == "W"]
        r_events = [e for e in self.events if e.ability == "R"]
        
        def summarize(events: List[CombinedEvent]) -> dict:
            if not events:
                return {"total": 0, "hits": 0, "misses": 0, "hit_rate": 0.0}
            hits = sum(1 for e in events if e.hit)
            return {
                "total": len(events),
                "hits": hits,
                "misses": len(events) - hits,
                "hit_rate": hits / len(events),
                "ui_detected": sum(1 for e in events if e.ui_detected),
                "vfx_detected": sum(1 for e in events if e.vfx_projectile_detected),
            }
            
        return {
            "total_events": len(self.events),
            "W": summarize(w_events),
            "R": summarize(r_events),
        }
        
    def save_events(self, output_path: str | Path) -> None:
        """Save events to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "events": [e.to_dict() for e in self.events],
            "summary": self.get_summary(),
        }
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
            
        print(f"Saved {len(self.events)} events to: {output_path}")


def run_combined_detection(
    video_path: str | Path,
    yolo_weights: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
    use_ui: bool = True,
    use_vfx: bool = True,
    frame_stride: int = 1,
    show_progress: bool = True,
) -> List[CombinedEvent]:
    """
    Run combined detection on a video.
    
    Args:
        video_path: Path to video file
        yolo_weights: Path to YOLO weights (optional)
        output_dir: Output directory for results
        use_ui: Enable UI detection
        use_vfx: Enable VFX detection
        frame_stride: Process every N frames
        show_progress: Show progress bar
        
    Returns:
        List of detected events
    """
    import cv2
    from tqdm import tqdm
    
    video_path = Path(video_path)
    
    # Create detector
    detector = CombinedDetector(
        yolo_weights=yolo_weights,
        use_ui_detection=use_ui,
        use_vfx_detection=use_vfx,
    )
    detector.load()
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing: {video_path.name}")
    print(f"  FPS: {fps}, Frames: {total_frames}")
    print(f"  UI detection: {use_ui}")
    print(f"  VFX detection: {use_vfx}")
    
    # Process frames
    pbar = tqdm(total=total_frames // frame_stride, desc="Detecting", disable=not show_progress)
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % frame_stride == 0:
            detector.process_frame(frame, frame_idx, fps)
            pbar.update(1)
            
        frame_idx += 1
        
    pbar.close()
    cap.release()
    
    # Finalize
    final_time = frame_idx / fps
    detector.finalize(final_time)
    
    # Print summary
    summary = detector.get_summary()
    print(f"\nDetection Results:")
    print(f"  Total events: {summary['total_events']}")
    print(f"  W: {summary['W']['total']} ({summary['W']['hits']} hits, {summary['W']['misses']} misses)")
    print(f"  R: {summary['R']['total']} ({summary['R']['hits']} hits, {summary['R']['misses']} misses)")
    
    # Save if output specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{video_path.stem}_combined_events.json"
        detector.save_events(output_path)
        
    return detector.events





