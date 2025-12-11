#!/usr/bin/env python3
"""
Interactive UI Region Calibration Tool.

Draw bounding boxes around the W and R ability icons to calibrate
the UI detection regions.

Usage:
    python tools/calibrate_ui_regions.py --video "data/raw_videos/video 1 (long).mp4"

Controls:
    - Click and drag to draw a box
    - Press 'w' to assign current box to W icon
    - Press 'r' to assign current box to R icon
    - Press 'c' to clear all boxes
    - Press 's' to save and exit
    - Press 'q' to quit without saving
    - Press 'n' to go to next frame
    - Press 'p' to go to previous frame
"""

import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class RegionCalibrator:
    """Interactive tool to draw UI detection regions."""
    
    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        self.cap = None
        self.frame = None
        self.display_frame = None
        self.frame_idx = 0
        self.total_frames = 0
        self.fps = 60.0
        
        # Drawing state
        self.drawing = False
        self.start_point = None
        self.current_box = None
        
        # Defined regions
        self.regions = {
            "w_icon": None,  # (x1, y1, x2, y2)
            "r_icon": None,
            "ability_bar": None,
        }
        
        # Colors
        self.colors = {
            "w_icon": (0, 255, 0),      # Green
            "r_icon": (0, 0, 255),       # Red
            "ability_bar": (255, 0, 0),  # Blue
            "current": (255, 255, 0),    # Yellow (current drawing)
        }
        
        self.window_name = "UI Region Calibrator"
        
    def load_video(self) -> bool:
        """Load the video file."""
        if not self.video_path.exists():
            print(f"Error: Video not found: {self.video_path}")
            return False
            
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            print(f"Error: Could not open video: {self.video_path}")
            return False
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 60.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Loaded: {self.video_path.name}")
        print(f"  Resolution: {self.width}x{self.height}")
        print(f"  FPS: {self.fps}")
        
        return True
        
    def read_frame(self, idx: int) -> bool:
        """Read a specific frame."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, self.frame = self.cap.read()
        if ret:
            self.frame_idx = idx
        return ret
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing boxes."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.current_box = None
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_box = (
                    min(self.start_point[0], x),
                    min(self.start_point[1], y),
                    max(self.start_point[0], x),
                    max(self.start_point[1], y),
                )
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.start_point:
                self.current_box = (
                    min(self.start_point[0], x),
                    min(self.start_point[1], y),
                    max(self.start_point[0], x),
                    max(self.start_point[1], y),
                )
                
    def draw_overlay(self) -> np.ndarray:
        """Draw boxes and instructions on frame."""
        display = self.frame.copy()
        
        # Draw saved regions
        for name, box in self.regions.items():
            if box:
                color = self.colors.get(name, (255, 255, 255))
                cv2.rectangle(display, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(display, name, (box[0], box[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                           
        # Draw current box being drawn
        if self.current_box:
            cv2.rectangle(display, 
                         (self.current_box[0], self.current_box[1]),
                         (self.current_box[2], self.current_box[3]),
                         self.colors["current"], 2)
                         
        # Draw instructions
        instructions = [
            f"Frame: {self.frame_idx}/{self.total_frames}",
            "",
            "Draw box, then press:",
            "  [W] - Assign to W icon",
            "  [R] - Assign to R icon", 
            "  [B] - Assign to ability bar",
            "",
            "[C] Clear all  [S] Save & exit",
            "[N/P] Next/Prev frame  [Q] Quit",
        ]
        
        y_offset = 30
        for line in instructions:
            cv2.putText(display, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
            
        # Show region status
        y_offset += 10
        for name in ["w_icon", "r_icon"]:
            status = "SET" if self.regions[name] else "NOT SET"
            color = (0, 255, 0) if self.regions[name] else (0, 0, 255)
            cv2.putText(display, f"{name}: {status}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 20
            
        return display
        
    def save_regions(self, output_path: str = "configs/ui_regions.json"):
        """Save calibrated regions to JSON file."""
        output_path = project_root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "resolution": [self.width, self.height],
            "regions": {}
        }
        
        for name, box in self.regions.items():
            if box:
                data["regions"][name] = {
                    "x1": box[0],
                    "y1": box[1],
                    "x2": box[2],
                    "y2": box[3],
                }
                
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
            
        print(f"\nSaved regions to: {output_path}")
        print(f"Resolution: {self.width}x{self.height}")
        for name, box in self.regions.items():
            if box:
                print(f"  {name}: ({box[0]}, {box[1]}) -> ({box[2]}, {box[3]})")
                
        # Also print Python code to update ui_detector.py
        print("\n" + "="*50)
        print("Copy this to update ui_detector.py DEFAULT_REGIONS_1080P:")
        print("="*50)
        print("DEFAULT_REGIONS_1080P = {")
        for name, box in self.regions.items():
            if box:
                # Scale to 1080p if not already
                scale_x = 1920 / self.width
                scale_y = 1080 / self.height
                scaled = (
                    int(box[0] * scale_x),
                    int(box[1] * scale_y),
                    int(box[2] * scale_x),
                    int(box[3] * scale_y),
                )
                print(f'    "{name}": UIRegion("{name}", {scaled[0]}, {scaled[1]}, {scaled[2]}, {scaled[3]}),')
        print("}")
        
    def run(self):
        """Run the calibration tool."""
        if not self.load_video():
            return
            
        # Start at 10 seconds in (usually past loading screen)
        start_frame = int(10 * self.fps)
        self.read_frame(min(start_frame, self.total_frames - 1))
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("\n" + "="*50)
        print("UI REGION CALIBRATOR")
        print("="*50)
        print("1. Draw a box around the W ability icon")
        print("2. Press 'W' to assign it")
        print("3. Draw a box around the R ability icon")
        print("4. Press 'R' to assign it")
        print("5. Press 'S' to save and exit")
        print("="*50 + "\n")
        
        while True:
            display = self.draw_overlay()
            cv2.imshow(self.window_name, display)
            
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q') or key == 27:  # Quit
                print("Exiting without saving...")
                break
                
            elif key == ord('s'):  # Save and exit
                if self.regions["w_icon"] and self.regions["r_icon"]:
                    self.save_regions()
                    break
                else:
                    print("Please define both W and R icon regions first!")
                    
            elif key == ord('w'):  # Assign to W
                if self.current_box:
                    self.regions["w_icon"] = self.current_box
                    print(f"W icon region set: {self.current_box}")
                    self.current_box = None
                else:
                    print("Draw a box first!")
                    
            elif key == ord('r'):  # Assign to R
                if self.current_box:
                    self.regions["r_icon"] = self.current_box
                    print(f"R icon region set: {self.current_box}")
                    self.current_box = None
                else:
                    print("Draw a box first!")
                    
            elif key == ord('b'):  # Assign to ability bar
                if self.current_box:
                    self.regions["ability_bar"] = self.current_box
                    print(f"Ability bar region set: {self.current_box}")
                    self.current_box = None
                    
            elif key == ord('c'):  # Clear all
                self.regions = {"w_icon": None, "r_icon": None, "ability_bar": None}
                self.current_box = None
                print("Cleared all regions")
                
            elif key == ord('n'):  # Next frame
                new_idx = min(self.frame_idx + int(self.fps), self.total_frames - 1)
                self.read_frame(new_idx)
                
            elif key == ord('p'):  # Previous frame
                new_idx = max(self.frame_idx - int(self.fps), 0)
                self.read_frame(new_idx)
                
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive UI region calibration tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Instructions:
    1. The video will open showing a frame from the gameplay
    2. Click and drag to draw a box around the W ability icon
    3. Press 'W' to assign that box to the W icon
    4. Click and drag to draw a box around the R ability icon
    5. Press 'R' to assign that box to the R icon
    6. Press 'S' to save the regions and exit

The regions will be saved to configs/ui_regions.json and the code
to update ui_detector.py will be printed to the console.
        """
    )
    
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to video file"
    )
    
    args = parser.parse_args()
    
    calibrator = RegionCalibrator(args.video)
    calibrator.run()


if __name__ == "__main__":
    main()





