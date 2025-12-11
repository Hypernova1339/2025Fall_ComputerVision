#!/usr/bin/env python3
"""
Live Detection Viewer - Real-time YOLO detection overlay on video playback.

Plays a video with your trained model running inference on each frame,
displaying bounding boxes and confidence scores as an overlay.

Usage:
    python tools/live_detection_viewer.py \
        --video "data/raw_videos/video_18min.mp4" \
        --weights runs/detect/jinx_speed_v1/weights/best.pt

Controls:
    Space       - Play/Pause
    Left Arrow  - Rewind 5 seconds
    Right Arrow - Forward 5 seconds
    Up Arrow    - Increase playback speed
    Down Arrow  - Decrease playback speed
    S           - Step forward one frame (while paused)
    D           - Toggle detection display
    C           - Toggle confidence scores
    R           - Reset to beginning
    Q / Esc     - Quit
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Class configuration (matches jinx_abilities.yaml)
CLASS_NAMES = {
    0: "IDLE",
    1: "W_PROJ",
    2: "W_IMPACT", 
    3: "R_ROCKET",
    4: "R_IMPACT",
}

# Colors (BGR format for OpenCV) - bright, visible colors
CLASS_COLORS = {
    0: (200, 200, 200),   # IDLE - Gray
    1: (255, 150, 50),    # W_PROJ - Cyan/Blue (Zap laser)
    2: (50, 255, 50),     # W_IMPACT - Green (Zap hit)
    3: (50, 50, 255),     # R_ROCKET - Red (Rocket)
    4: (0, 165, 255),     # R_IMPACT - Orange (Explosion)
}


class LiveDetectionViewer:
    """Interactive video player with real-time YOLO detection overlay."""
    
    def __init__(
        self,
        video_path: str,
        weights_path: str,
        conf_threshold: float = 0.25,
        device: str = None,
        window_scale: float = 0.75
    ):
        """
        Initialize the viewer.
        
        Args:
            video_path: Path to input video
            weights_path: Path to YOLO weights file
            conf_threshold: Minimum confidence for detections
            device: Device for inference (None for auto)
            window_scale: Scale factor for display window
        """
        self.video_path = Path(video_path)
        self.weights_path = Path(weights_path)
        self.conf_threshold = conf_threshold
        self.window_scale = window_scale
        
        # State
        self.paused = True  # Start paused
        self.show_detections = True
        self.show_confidence = True
        self.playback_speed = 1.0
        self.frame_idx = 0
        
        # Load model
        print(f"Loading model: {self.weights_path}")
        self._load_model(device)
        
        # Open video
        print(f"Opening video: {self.video_path}")
        self._open_video()
        
    def _load_model(self, device: str = None):
        """Load YOLO model."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("Ultralytics not installed. Run: pip install ultralytics")
        
        self.model = YOLO(str(self.weights_path))
        
        # Auto-detect device
        if device is None:
            import torch
            if torch.cuda.is_available():
                device = 0
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        print(f"  Using device: {device}")
        
    def _open_video(self):
        """Open video file."""
        self.cap = cv2.VideoCapture(str(self.video_path))
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration_sec = self.total_frames / self.fps
        
        print(f"  Resolution: {self.width}x{self.height}")
        print(f"  FPS: {self.fps}")
        print(f"  Duration: {self.duration_sec:.1f}s ({self.total_frames} frames)")
        
        # Calculate display size
        self.display_width = int(self.width * self.window_scale)
        self.display_height = int(self.height * self.window_scale)
        
    def _detect_frame(self, frame: np.ndarray) -> list:
        """Run detection on a single frame."""
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False
        )
        
        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    
                    detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': conf,
                        'class_id': cls_id,
                        'class_name': CLASS_NAMES.get(cls_id, f"cls_{cls_id}")
                    })
        
        return detections
    
    def _draw_detections(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """Draw detection boxes and labels on frame."""
        frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cls_id = det['class_id']
            cls_name = det['class_name']
            conf = det['confidence']
            
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            if self.show_confidence:
                label = f"{cls_name} {conf:.2f}"
            else:
                label = cls_name
            
            # Draw label background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            cv2.rectangle(
                frame,
                (x1, y1 - label_h - baseline - 5),
                (x1 + label_w + 5, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame,
                label,
                (x1 + 2, y1 - baseline - 2),
                font,
                font_scale,
                (0, 0, 0),
                thickness
            )
        
        return frame
    
    def _draw_hud(self, frame: np.ndarray, detections: list, inference_time: float) -> np.ndarray:
        """Draw heads-up display with status info."""
        h, w = frame.shape[:2]
        
        # Status text
        current_time = self.frame_idx / self.fps
        status = "PAUSED" if self.paused else f"PLAYING {self.playback_speed:.1f}x"
        
        time_str = f"{int(current_time // 60):02d}:{int(current_time % 60):02d}"
        total_str = f"{int(self.duration_sec // 60):02d}:{int(self.duration_sec % 60):02d}"
        
        # Background panel for HUD
        panel_height = 80
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Status line
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"{status} | {time_str} / {total_str}", 
                    (10, 25), font, 0.7, (255, 255, 255), 2)
        
        # Detection info
        det_text = f"Detections: {len(detections)} | Inference: {inference_time*1000:.0f}ms"
        cv2.putText(frame, det_text, (10, 50), font, 0.6, (200, 200, 200), 1)
        
        # Detection summary by class
        class_counts = {}
        for det in detections:
            name = det['class_name']
            class_counts[name] = class_counts.get(name, 0) + 1
        
        summary_parts = [f"{name}:{count}" for name, count in sorted(class_counts.items())]
        if summary_parts:
            summary = " | ".join(summary_parts)
            cv2.putText(frame, summary, (10, 72), font, 0.5, (100, 255, 100), 1)
        
        # Controls hint (right side)
        hint = "Space:Play/Pause | Arrows:Seek | Q:Quit"
        hint_size = cv2.getTextSize(hint, font, 0.4, 1)[0]
        cv2.putText(frame, hint, (w - hint_size[0] - 10, 25), font, 0.4, (150, 150, 150), 1)
        
        # Progress bar
        bar_y = panel_height - 5
        bar_width = w - 20
        progress = self.frame_idx / max(self.total_frames, 1)
        
        cv2.rectangle(frame, (10, bar_y - 3), (10 + bar_width, bar_y), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, bar_y - 3), (10 + int(bar_width * progress), bar_y), (0, 200, 0), -1)
        
        return frame
    
    def _seek(self, delta_frames: int):
        """Seek forward or backward by delta frames."""
        new_frame = max(0, min(self.total_frames - 1, self.frame_idx + delta_frames))
        self.frame_idx = new_frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
    
    def run(self):
        """Main viewer loop."""
        window_name = f"Jinx Ability Detection - {self.video_path.name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.display_width, self.display_height)
        
        print("\n" + "=" * 60)
        print("LIVE DETECTION VIEWER")
        print("=" * 60)
        print("\nControls:")
        print("  Space       - Play/Pause")
        print("  Left/Right  - Seek ±5 seconds")
        print("  Up/Down     - Speed ±0.25x")
        print("  S           - Step one frame (when paused)")
        print("  D           - Toggle detection boxes")
        print("  C           - Toggle confidence scores")
        print("  R           - Reset to beginning")
        print("  Q / Esc     - Quit")
        print("\nStarting paused. Press Space to play.")
        print("=" * 60 + "\n")
        
        # Read first frame
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read first frame")
            return
        
        frame_time = 1.0 / self.fps  # Time per frame at 1x speed
        last_frame_time = time.time()
        
        while True:
            current_time = time.time()
            
            # Handle playback timing
            if not self.paused:
                elapsed = current_time - last_frame_time
                frames_to_advance = int(elapsed * self.fps * self.playback_speed)
                
                if frames_to_advance >= 1:
                    # Read next frame(s)
                    for _ in range(frames_to_advance):
                        ret, frame = self.cap.read()
                        self.frame_idx += 1
                        
                        if not ret or self.frame_idx >= self.total_frames:
                            # End of video - loop or pause
                            self.frame_idx = 0
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = self.cap.read()
                            break
                    
                    last_frame_time = current_time
            
            # Run detection
            inference_start = time.time()
            if self.show_detections:
                detections = self._detect_frame(frame)
            else:
                detections = []
            inference_time = time.time() - inference_start
            
            # Draw overlays
            display_frame = frame.copy()
            if self.show_detections and detections:
                display_frame = self._draw_detections(display_frame, detections)
            display_frame = self._draw_hud(display_frame, detections, inference_time)
            
            # Resize for display
            display_frame = cv2.resize(display_frame, (self.display_width, self.display_height))
            
            # Show frame
            cv2.imshow(window_name, display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or Esc
                break
            elif key == ord(' '):  # Space - play/pause
                self.paused = not self.paused
                last_frame_time = time.time()
            elif key == ord('s'):  # S - step frame
                if self.paused:
                    ret, frame = self.cap.read()
                    if ret:
                        self.frame_idx += 1
            elif key == ord('d'):  # D - toggle detections
                self.show_detections = not self.show_detections
            elif key == ord('c'):  # C - toggle confidence
                self.show_confidence = not self.show_confidence
            elif key == ord('r'):  # R - reset
                self.frame_idx = 0
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            elif key == 81 or key == 2:  # Left arrow
                self._seek(-int(5 * self.fps))  # Back 5 seconds
                ret, frame = self.cap.read()
            elif key == 83 or key == 3:  # Right arrow
                self._seek(int(5 * self.fps))  # Forward 5 seconds
                ret, frame = self.cap.read()
            elif key == 82 or key == 0:  # Up arrow
                self.playback_speed = min(4.0, self.playback_speed + 0.25)
            elif key == 84 or key == 1:  # Down arrow
                self.playback_speed = max(0.25, self.playback_speed - 0.25)
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("\nViewer closed.")


def main():
    parser = argparse.ArgumentParser(
        description="Live video viewer with real-time YOLO detection overlay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
    Space       - Play/Pause
    Left/Right  - Seek ±5 seconds  
    Up/Down     - Speed ±0.25x
    S           - Step frame (when paused)
    D           - Toggle detection boxes
    C           - Toggle confidence scores
    R           - Reset to beginning
    Q / Esc     - Quit

Examples:
    python tools/live_detection_viewer.py \\
        --video "data/raw_videos/video_18min.mp4" \\
        --weights runs/detect/jinx_speed_v1/weights/best.pt

    # With higher confidence threshold
    python tools/live_detection_viewer.py \\
        --video gameplay.mp4 \\
        --weights runs/detect/jinx_speed_v1/weights/best.pt \\
        --conf 0.4
        """
    )
    
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file"
    )
    
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to YOLO weights file (.pt)"
    )
    
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cpu, mps, 0, 1, etc. (default: auto)"
    )
    
    parser.add_argument(
        "--scale",
        type=float,
        default=0.75,
        help="Window scale factor (default: 0.75)"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    video_path = Path(args.video)
    weights_path = Path(args.weights)
    
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)
    
    if not weights_path.exists():
        print(f"Error: Weights not found: {weights_path}")
        sys.exit(1)
    
    # Create and run viewer
    viewer = LiveDetectionViewer(
        video_path=args.video,
        weights_path=args.weights,
        conf_threshold=args.conf,
        device=args.device,
        window_scale=args.scale
    )
    
    viewer.run()


if __name__ == "__main__":
    main()




