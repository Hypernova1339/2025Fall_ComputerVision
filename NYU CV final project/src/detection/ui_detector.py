"""
UI-based ability cooldown detector.

Detects when Jinx's W and R abilities are cast by monitoring the ability bar UI.
When an ability is used, the icon goes on cooldown (grayed out with timer).

This provides a reliable signal for ability cast times since the UI is always
in a fixed position and has consistent appearance.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class AbilityState(Enum):
    """State of an ability."""
    READY = "ready"      # Available to use (bright icon)
    COOLDOWN = "cooldown"  # On cooldown (grayed out)
    UNKNOWN = "unknown"   # Cannot determine


@dataclass
class UIRegion:
    """Defines a UI region to monitor."""
    name: str
    x1: int
    y1: int
    x2: int
    y2: int
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
        
    def crop(self, frame: np.ndarray) -> np.ndarray:
        """Crop this region from a frame."""
        return frame[self.y1:self.y2, self.x1:self.x2]
        
    def scale(self, scale_x: float, scale_y: float) -> "UIRegion":
        """Scale region coordinates."""
        return UIRegion(
            name=self.name,
            x1=int(self.x1 * scale_x),
            y1=int(self.y1 * scale_y),
            x2=int(self.x2 * scale_x),
            y2=int(self.y2 * scale_y),
        )


@dataclass
class CooldownEvent:
    """Detected cooldown event."""
    ability: str  # "W" or "R"
    frame_idx: int
    timestamp_sec: float
    confidence: float


# Default UI regions for 1920x1080 resolution
# Calibrated from actual gameplay footage
DEFAULT_REGIONS_1080P = {
    # W and R ability icons - used to detect cooldown state
    "w_icon": UIRegion("w_icon", 848, 989, 897, 1029),
    "r_icon": UIRegion("r_icon", 933, 990, 982, 1036),
}


class UIDetector:
    """
    Detects ability usage from the UI cooldown indicators.
    
    Uses pixel analysis to detect when ability icons transition from
    ready (bright) to cooldown (grayed out) state.
    """
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (1920, 1080),
        regions: Optional[Dict[str, UIRegion]] = None,
        saturation_threshold: float = 0.3,
        brightness_threshold: float = 0.4,
    ):
        """
        Initialize UI detector.
        
        Args:
            resolution: Video resolution (width, height)
            regions: Custom UI regions (or None for defaults)
            saturation_threshold: Below this = cooldown (grayed out)
            brightness_threshold: Below this = cooldown (darkened)
        """
        self.resolution = resolution
        self.saturation_threshold = saturation_threshold
        self.brightness_threshold = brightness_threshold
        
        # Scale regions if not 1080p
        if regions:
            self.regions = regions
        else:
            scale_x = resolution[0] / 1920
            scale_y = resolution[1] / 1080
            self.regions = {
                name: region.scale(scale_x, scale_y)
                for name, region in DEFAULT_REGIONS_1080P.items()
            }
            
        # State tracking
        self.prev_w_state = AbilityState.UNKNOWN
        self.prev_r_state = AbilityState.UNKNOWN
        self.cooldown_events: List[CooldownEvent] = []
        
    def analyze_icon_state(self, icon_crop: np.ndarray) -> Tuple[AbilityState, float]:
        """
        Analyze an ability icon crop to determine if it's on cooldown.
        
        When an ability is on cooldown:
        - The icon is grayed out (low saturation)
        - There's often a dark overlay
        - A cooldown timer number may be visible
        
        Args:
            icon_crop: Cropped image of the ability icon (BGR)
            
        Returns:
            Tuple of (state, confidence)
        """
        if icon_crop.size == 0:
            return AbilityState.UNKNOWN, 0.0
            
        # Convert to HSV for saturation analysis
        hsv = cv2.cvtColor(icon_crop, cv2.COLOR_BGR2HSV)
        
        # Get mean saturation and value (brightness)
        mean_saturation = np.mean(hsv[:, :, 1]) / 255.0
        mean_brightness = np.mean(hsv[:, :, 2]) / 255.0
        
        # Calculate a combined score
        # Low saturation AND low brightness = on cooldown
        # High saturation OR high brightness = ready
        
        # Cooldown detection logic
        is_low_saturation = mean_saturation < self.saturation_threshold
        is_low_brightness = mean_brightness < self.brightness_threshold
        
        if is_low_saturation and is_low_brightness:
            # Likely on cooldown (grayed out)
            confidence = 1.0 - (mean_saturation + mean_brightness) / 2
            return AbilityState.COOLDOWN, min(confidence, 1.0)
        elif mean_saturation > 0.4 or mean_brightness > 0.5:
            # Likely ready (colorful/bright)
            confidence = (mean_saturation + mean_brightness) / 2
            return AbilityState.READY, min(confidence, 1.0)
        else:
            # Uncertain
            return AbilityState.UNKNOWN, 0.5
            
    def detect_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        fps: float = 60.0
    ) -> List[CooldownEvent]:
        """
        Detect ability state changes in a single frame.
        
        Args:
            frame: Video frame (BGR)
            frame_idx: Frame index
            fps: Video FPS for timestamp calculation
            
        Returns:
            List of new cooldown events detected
        """
        events = []
        timestamp = frame_idx / fps
        
        # Analyze W icon
        w_crop = self.regions["w_icon"].crop(frame)
        w_state, w_conf = self.analyze_icon_state(w_crop)
        
        # Detect W cast (transition from ready to cooldown)
        if self.prev_w_state == AbilityState.READY and w_state == AbilityState.COOLDOWN:
            event = CooldownEvent(
                ability="W",
                frame_idx=frame_idx,
                timestamp_sec=timestamp,
                confidence=w_conf,
            )
            events.append(event)
            self.cooldown_events.append(event)
            
        self.prev_w_state = w_state
        
        # Analyze R icon
        r_crop = self.regions["r_icon"].crop(frame)
        r_state, r_conf = self.analyze_icon_state(r_crop)
        
        # Detect R cast (transition from ready to cooldown)
        if self.prev_r_state == AbilityState.READY and r_state == AbilityState.COOLDOWN:
            event = CooldownEvent(
                ability="R",
                frame_idx=frame_idx,
                timestamp_sec=timestamp,
                confidence=r_conf,
            )
            events.append(event)
            self.cooldown_events.append(event)
            
        self.prev_r_state = r_state
        
        return events
        
    def process_video(
        self,
        video_path: str | Path,
        show_progress: bool = True,
        sample_rate: int = 1,
    ) -> List[CooldownEvent]:
        """
        Process entire video and detect all ability casts.
        
        Args:
            video_path: Path to video file
            show_progress: Show progress bar
            sample_rate: Process every N frames (1 = all frames)
            
        Returns:
            List of all detected cooldown events
        """
        from tqdm import tqdm
        
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Update resolution if different
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (width, height) != self.resolution:
            print(f"Adjusting regions for resolution: {width}x{height}")
            scale_x = width / 1920
            scale_y = height / 1080
            self.regions = {
                name: region.scale(scale_x, scale_y)
                for name, region in DEFAULT_REGIONS_1080P.items()
            }
            self.resolution = (width, height)
            
        # Reset state
        self.prev_w_state = AbilityState.UNKNOWN
        self.prev_r_state = AbilityState.UNKNOWN
        self.cooldown_events = []
        
        print(f"Processing video: {video_path.name}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Frames: {total_frames}")
        
        # Process frames
        pbar = tqdm(total=total_frames // sample_rate, desc="Detecting UI", disable=not show_progress)
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % sample_rate == 0:
                self.detect_frame(frame, frame_idx, fps)
                pbar.update(1)
                
            frame_idx += 1
            
        pbar.close()
        cap.release()
        
        print(f"\nDetected {len(self.cooldown_events)} ability casts:")
        w_count = sum(1 for e in self.cooldown_events if e.ability == "W")
        r_count = sum(1 for e in self.cooldown_events if e.ability == "R")
        print(f"  W casts: {w_count}")
        print(f"  R casts: {r_count}")
        
        return self.cooldown_events
        
    def visualize_regions(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw UI regions on a frame for debugging/calibration.
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with regions drawn
        """
        vis = frame.copy()
        
        for name, region in self.regions.items():
            # Draw rectangle
            color = (0, 255, 0) if "icon" in name else (255, 0, 0)
            cv2.rectangle(vis, (region.x1, region.y1), (region.x2, region.y2), color, 2)
            
            # Draw label
            cv2.putText(vis, name, (region.x1, region.y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                       
        return vis
        
    def calibrate_regions(
        self,
        video_path: str | Path,
        output_path: Optional[str | Path] = None
    ) -> None:
        """
        Interactive tool to calibrate UI regions.
        
        Opens a window showing the first frame with current regions.
        Use this to verify/adjust region positions.
        
        Args:
            video_path: Path to video file
            output_path: Optional path to save calibration image
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
            
        # Read first frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise RuntimeError("Could not read first frame")
            
        # Draw regions
        vis = self.visualize_regions(frame)
        
        # Show
        cv2.imshow("UI Region Calibration", vis)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save if requested
        if output_path:
            cv2.imwrite(str(output_path), vis)
            print(f"Saved calibration image to: {output_path}")


def export_events_to_csv(
    events: List[CooldownEvent],
    video_id: str,
    output_path: str | Path
) -> None:
    """
    Export detected events to CSV format compatible with the labeling pipeline.
    
    Args:
        events: List of CooldownEvent objects
        video_id: Video identifier
        output_path: Output CSV path
    """
    import csv
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = ["video_id", "ability", "cast_time_sec", "result", "notes"]
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for event in sorted(events, key=lambda e: e.timestamp_sec):
            writer.writerow({
                "video_id": video_id,
                "ability": event.ability,
                "cast_time_sec": f"{event.timestamp_sec:.1f}",
                "result": "",  # Need manual labeling for hit/miss
                "notes": f"auto-detected (conf={event.confidence:.2f})",
            })
            
    print(f"Exported {len(events)} events to: {output_path}")

