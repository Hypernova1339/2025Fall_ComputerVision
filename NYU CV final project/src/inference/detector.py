"""YOLO detector for Jinx ability detection."""

from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Detection:
    """Single detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2) in pixels
    bbox_normalized: tuple  # (x1, y1, x2, y2) normalized 0-1
    
    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": float(self.confidence),
            "bbox": [float(x) for x in self.bbox],
            "bbox_normalized": [float(x) for x in self.bbox_normalized],
        }


@dataclass
class FrameDetections:
    """Detections for a single frame."""
    frame_idx: int
    timestamp_sec: float
    detections: List[Detection] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "timestamp_sec": self.timestamp_sec,
            "detections": [d.to_dict() for d in self.detections],
        }
        
    @property
    def has_w_projectile(self) -> bool:
        return any(d.class_id == 0 for d in self.detections)
        
    @property
    def has_w_impact(self) -> bool:
        return any(d.class_id == 1 for d in self.detections)
        
    @property
    def has_r_rocket(self) -> bool:
        return any(d.class_id == 2 for d in self.detections)
        
    @property
    def has_r_impact(self) -> bool:
        return any(d.class_id == 3 for d in self.detections)


CLASS_NAMES = {
    0: "JINX_W_PROJECTILE",
    1: "JINX_W_IMPACT",
    2: "JINX_R_ROCKET",
    3: "JINX_R_IMPACT",
}


class JinxDetector:
    """YOLO-based detector for Jinx abilities."""
    
    def __init__(
        self,
        weights_path: str | Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: Optional[str] = None
    ):
        """
        Initialize detector.
        
        Args:
            weights_path: Path to trained YOLO weights
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            device: Device to run on (None for auto)
        """
        self.weights_path = Path(weights_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = None
        
    def load(self) -> None:
        """Load the model."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "Ultralytics not installed. Run: pip install ultralytics"
            )
            
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {self.weights_path}")
            
        self.model = YOLO(str(self.weights_path))
        
        if self.device:
            self.model.to(self.device)
            
    def detect_frame(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
        fps: float = 60.0
    ) -> FrameDetections:
        """
        Run detection on a single frame.
        
        Args:
            frame: Input frame (BGR numpy array)
            frame_idx: Frame index for tracking
            fps: Video FPS for timestamp calculation
            
        Returns:
            FrameDetections object
        """
        if self.model is None:
            self.load()
            
        # Run inference
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        # Parse results
        detections = []
        height, width = frame.shape[:2]
        
        for result in results:
            boxes = result.boxes
            
            if boxes is None:
                continue
                
            for i in range(len(boxes)):
                # Get box coordinates (xyxy format)
                box = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = box
                
                # Get class and confidence
                class_id = int(boxes.cls[i].cpu().numpy())
                confidence = float(boxes.conf[i].cpu().numpy())
                
                # Create detection
                detection = Detection(
                    class_id=class_id,
                    class_name=CLASS_NAMES.get(class_id, f"class_{class_id}"),
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    bbox_normalized=(
                        x1 / width,
                        y1 / height,
                        x2 / width,
                        y2 / height
                    )
                )
                detections.append(detection)
                
        return FrameDetections(
            frame_idx=frame_idx,
            timestamp_sec=frame_idx / fps,
            detections=detections
        )
        
    def detect_batch(
        self,
        frames: List[np.ndarray],
        start_frame_idx: int = 0,
        fps: float = 60.0
    ) -> List[FrameDetections]:
        """
        Run detection on a batch of frames.
        
        Args:
            frames: List of input frames
            start_frame_idx: Starting frame index
            fps: Video FPS
            
        Returns:
            List of FrameDetections
        """
        if self.model is None:
            self.load()
            
        # Run batch inference
        results = self.model(
            frames,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        all_detections = []
        
        for i, result in enumerate(results):
            frame_idx = start_frame_idx + i
            frame = frames[i]
            height, width = frame.shape[:2]
            
            detections = []
            boxes = result.boxes
            
            if boxes is not None:
                for j in range(len(boxes)):
                    box = boxes.xyxy[j].cpu().numpy()
                    x1, y1, x2, y2 = box
                    
                    class_id = int(boxes.cls[j].cpu().numpy())
                    confidence = float(boxes.conf[j].cpu().numpy())
                    
                    detection = Detection(
                        class_id=class_id,
                        class_name=CLASS_NAMES.get(class_id, f"class_{class_id}"),
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                        bbox_normalized=(
                            x1 / width,
                            y1 / height,
                            x2 / width,
                            y2 / height
                        )
                    )
                    detections.append(detection)
                    
            all_detections.append(FrameDetections(
                frame_idx=frame_idx,
                timestamp_sec=frame_idx / fps,
                detections=detections
            ))
            
        return all_detections
        
    def get_class_detections(
        self,
        frame_detections: FrameDetections,
        class_id: int
    ) -> List[Detection]:
        """Get detections for a specific class."""
        return [d for d in frame_detections.detections if d.class_id == class_id]





