"""Visualization utilities for detections and events."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


# Class colors (BGR format for OpenCV)
# Matches jinx_abilities.yaml class definitions
CLASS_COLORS = {
    0: (255, 200, 100),    # JINX_IDLE - Light blue/white
    1: (255, 100, 100),    # VFX_W_PROJECTILE - Blue (Zap laser)
    2: (100, 255, 100),    # VFX_W_IMPACT - Green (Zap hit)
    3: (100, 100, 255),    # VFX_R_ROCKET - Red (Rocket)
    4: (0, 200, 255),      # VFX_R_IMPACT - Orange (Explosion)
}

CLASS_NAMES = {
    0: "IDLE",
    1: "W_PROJ",
    2: "W_IMPACT",
    3: "R_ROCKET",
    4: "R_IMPACT",
}


@dataclass
class Detection:
    """Single detection result."""
    class_id: int
    confidence: float
    x1: float  # Normalized or pixel coordinates
    y1: float
    x2: float
    y2: float
    
    @property
    def class_name(self) -> str:
        return CLASS_NAMES.get(self.class_id, f"class_{self.class_id}")
    
    def to_pixels(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """Convert normalized coordinates to pixel coordinates."""
        return (
            int(self.x1 * width),
            int(self.y1 * height),
            int(self.x2 * width),
            int(self.y2 * height),
        )


def draw_detections(
    frame: np.ndarray,
    detections: List[Detection],
    normalized: bool = True,
    thickness: int = 2,
    font_scale: float = 0.6,
    show_confidence: bool = True
) -> np.ndarray:
    """
    Draw detection bounding boxes on a frame.
    
    Args:
        frame: Input frame (BGR format)
        detections: List of Detection objects
        normalized: Whether coordinates are normalized (0-1)
        thickness: Box line thickness
        font_scale: Label font scale
        show_confidence: Whether to show confidence scores
        
    Returns:
        Frame with drawn detections
    """
    frame = frame.copy()
    height, width = frame.shape[:2]
    
    for det in detections:
        # Get pixel coordinates
        if normalized:
            x1, y1, x2, y2 = det.to_pixels(width, height)
        else:
            x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
            
        # Get color for class
        color = CLASS_COLORS.get(det.class_id, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label
        label = det.class_name
        if show_confidence:
            label += f" {det.confidence:.2f}"
            
        # Draw label background
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        cv2.rectangle(
            frame,
            (x1, y1 - label_h - baseline - 5),
            (x1 + label_w, y1),
            color,
            -1  # Filled
        )
        
        # Draw label text
        cv2.putText(
            frame,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),  # Black text
            thickness
        )
        
    return frame


def draw_timeline(
    events: List[Dict],
    duration_sec: float,
    width: int = 1200,
    height: int = 200,
    output_path: Optional[str | Path] = None
) -> np.ndarray:
    """
    Draw a timeline visualization of ability events.
    
    Args:
        events: List of event dicts with 'ability', 'cast_time', 'hit' keys
        duration_sec: Total video duration in seconds
        width: Output image width
        height: Output image height
        output_path: Optional path to save the image
        
    Returns:
        Timeline image as numpy array
    """
    # Create white background
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Draw timeline axis
    margin = 50
    timeline_y = height // 2
    cv2.line(img, (margin, timeline_y), (width - margin, timeline_y), (0, 0, 0), 2)
    
    # Draw time markers
    num_markers = 10
    for i in range(num_markers + 1):
        x = margin + int((width - 2 * margin) * i / num_markers)
        time_sec = duration_sec * i / num_markers
        
        # Tick mark
        cv2.line(img, (x, timeline_y - 5), (x, timeline_y + 5), (0, 0, 0), 1)
        
        # Time label
        label = f"{int(time_sec // 60)}:{int(time_sec % 60):02d}"
        cv2.putText(
            img, label, (x - 15, timeline_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1
        )
        
    # Draw events
    w_y_offset = -30  # W events above timeline
    r_y_offset = 30   # R events below timeline
    
    for event in events:
        ability = event.get("ability", "W")
        cast_time = event.get("cast_time", 0)
        hit = event.get("hit", None)
        
        # Calculate x position
        x = margin + int((width - 2 * margin) * cast_time / duration_sec)
        y_offset = w_y_offset if ability == "W" else r_y_offset
        
        # Color based on hit/miss
        if hit is True:
            color = (0, 200, 0)  # Green for hit
        elif hit is False:
            color = (0, 0, 200)  # Red for miss
        else:
            color = (200, 200, 0)  # Yellow for unknown
            
        # Draw event marker
        cv2.circle(img, (x, timeline_y + y_offset), 8, color, -1)
        cv2.circle(img, (x, timeline_y + y_offset), 8, (0, 0, 0), 1)
        
        # Draw ability label
        cv2.putText(
            img, ability, (x - 5, timeline_y + y_offset - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1
        )
        
    # Draw legend
    legend_x = margin
    legend_y = 30
    
    cv2.putText(img, "Legend:", (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # W marker
    cv2.putText(img, "W (above)", (legend_x + 80, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # R marker
    cv2.putText(img, "R (below)", (legend_x + 160, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Hit/miss colors
    cv2.circle(img, (legend_x + 260, legend_y - 5), 6, (0, 200, 0), -1)
    cv2.putText(img, "Hit", (legend_x + 270, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    cv2.circle(img, (legend_x + 310, legend_y - 5), 6, (0, 0, 200), -1)
    cv2.putText(img, "Miss", (legend_x + 320, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Title
    cv2.putText(
        img, "Jinx Ability Timeline",
        (width // 2 - 100, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
    )
    
    # Save if path provided
    if output_path:
        cv2.imwrite(str(output_path), img)
        
    return img


def create_detection_video(
    input_video: str | Path,
    detections_by_frame: Dict[int, List[Detection]],
    output_path: str | Path,
    show_progress: bool = True
) -> None:
    """
    Create a video with detection overlays.
    
    Args:
        input_video: Path to input video
        detections_by_frame: Dict mapping frame index to list of detections
        output_path: Path for output video
        show_progress: Whether to show progress bar
    """
    from .video_utils import VideoReader, VideoWriter
    from tqdm import tqdm
    
    with VideoReader(input_video) as reader:
        info = reader.info
        
        with VideoWriter.from_video_info(output_path, info) as writer:
            frame_iter = reader.iter_frames()
            
            if show_progress:
                frame_iter = tqdm(frame_iter, total=info.frame_count, desc="Creating video")
                
            for frame_idx, frame in frame_iter:
                detections = detections_by_frame.get(frame_idx, [])
                annotated = draw_detections(frame, detections)
                writer.write(annotated)


