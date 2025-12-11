#!/usr/bin/env python3
"""
Simple Bounding Box Labeling Tool

Click two corners to draw a bounding box. Fast and simple.

Usage:
    python tools/bbox_labeler.py --input data/images/unlabeled/

Controls:
    Left Click (1st)   - Set first corner of bbox
    Left Click (2nd)   - Set second corner, create bbox
    Click Red X        - Delete specific box
    Right Click        - Cancel current box / Delete last box
    Z                  - Undo last box
    C                  - Clear all boxes on current image
    Left Arrow         - Previous image
    Right Arrow        - Next image
    Space / Enter      - Save and go to next image
    S                  - Save current image
    N                  - Skip image (no boxes)
    Q / Esc            - Quit
"""

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
import cv2
import numpy as np


# Class mapping: folder name -> (class_id, class_name)
FOLDER_TO_CLASS = {
    'idle': (0, 'JINX_IDLE'),
    'w_cast': (1, 'VFX_W_PROJECTILE'),
    'w_impact': (2, 'VFX_W_IMPACT'),
    'r_cast': (3, 'VFX_R_ROCKET'),
    'r_impact': (4, 'VFX_R_IMPACT'),
}

# Colors for boxes (BGR)
BOX_COLOR = (0, 255, 0)  # Green
PREVIEW_COLOR = (0, 255, 255)  # Yellow
POINT_COLOR = (0, 0, 255)  # Red
DELETE_BTN_COLOR = (0, 0, 255)  # Red for delete button
DELETE_BTN_SIZE = 18  # Size of delete button


@dataclass
class ImageItem:
    path: Path
    folder: str
    class_id: int
    class_name: str
    
    @property
    def label_filename(self) -> str:
        return self.path.stem + '.txt'


@dataclass 
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int
    
    def to_yolo(self, img_w: int, img_h: int) -> Tuple[int, float, float, float, float]:
        """Convert to YOLO format (class_id, x_center, y_center, width, height) normalized."""
        x_center = ((self.x1 + self.x2) / 2) / img_w
        y_center = ((self.y1 + self.y2) / 2) / img_h
        width = abs(self.x2 - self.x1) / img_w
        height = abs(self.y2 - self.y1) / img_h
        return (self.class_id, x_center, y_center, width, height)


class BBoxLabeler:
    WINDOW_NAME = "BBox Labeler - Click two corners"
    
    def __init__(self, input_dir: Path, output_dir: Optional[Path] = None, resume: bool = True, show_all: bool = False):
        self.input_dir = input_dir
        self.output_dir = output_dir or input_dir.parent / 'labels'
        self.resume = resume
        self.show_all = show_all
        
        self.images: List[ImageItem] = []
        self.current_idx = 0
        self.current_image: Optional[np.ndarray] = None
        self.boxes: List[BBox] = []
        
        # For drawing
        self.first_point: Optional[Tuple[int, int]] = None
        self.mouse_pos: Tuple[int, int] = (0, 0)
        self.scale = 1.0
        self.delete_buttons: List[Tuple[int, int, int, int, int]] = []  # (x1, y1, x2, y2, box_idx)
        
    def load_images(self):
        """Load all images."""
        self.images = []
        
        for folder_name, (class_id, class_name) in FOLDER_TO_CLASS.items():
            folder_path = self.input_dir / folder_name
            if not folder_path.exists():
                continue
                
            label_folder = self.output_dir / folder_name
            label_folder.mkdir(parents=True, exist_ok=True)
            
            for ext in ['*.jpg', '*.png', '*.jpeg']:
                for img_path in sorted(folder_path.glob(ext)):
                    label_path = label_folder / (img_path.stem + '.txt')
                    # Skip already-labeled images only if resume=True and show_all=False
                    if self.resume and not self.show_all and label_path.exists():
                        continue
                    self.images.append(ImageItem(
                        path=img_path,
                        folder=folder_name,
                        class_id=class_id,
                        class_name=class_name
                    ))
                    
        print(f"Found {len(self.images)} images to label")
        
    def load_current(self):
        """Load current image and any existing labels."""
        if not self.images or self.current_idx >= len(self.images):
            return False
            
        item = self.images[self.current_idx]
        self.current_image = cv2.imread(str(item.path))
        
        if self.current_image is None:
            return False
            
        h, w = self.current_image.shape[:2]
        self.scale = min(1400 / max(h, w), 1.0)
        self.boxes = []
        self.first_point = None
        
        # Load existing labels if they exist
        label_path = self.output_dir / item.folder / item.label_filename
        if label_path.exists():
            self.load_labels(label_path, w, h, item.class_id)
            
        return True
        
    def load_labels(self, label_path: Path, img_w: int, img_h: int, default_class_id: int):
        """Load existing YOLO labels from file."""
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        xc, yc, bw, bh = map(float, parts[1:5])
                        
                        # Convert from YOLO format to pixel coordinates
                        x1 = int((xc - bw / 2) * img_w)
                        y1 = int((yc - bh / 2) * img_h)
                        x2 = int((xc + bw / 2) * img_w)
                        y2 = int((yc + bh / 2) * img_h)
                        
                        self.boxes.append(BBox(x1, y1, x2, y2, class_id))
            print(f"Loaded {len(self.boxes)} existing boxes")
        except Exception as e:
            print(f"Error loading labels: {e}")
        
    def to_img_coords(self, x: int, y: int) -> Tuple[int, int]:
        """Convert display coords to image coords."""
        return int(x / self.scale), int(y / self.scale)
        
    def to_display_coords(self, x: int, y: int) -> Tuple[int, int]:
        """Convert image coords to display coords."""
        return int(x * self.scale), int(y * self.scale)
        
    def render(self) -> np.ndarray:
        """Render current view."""
        if self.current_image is None:
            return np.zeros((100, 400, 3), dtype=np.uint8)
            
        item = self.images[self.current_idx]
        h, w = self.current_image.shape[:2]
        
        # Copy and scale
        if self.scale < 1.0:
            display = cv2.resize(self.current_image, (int(w * self.scale), int(h * self.scale)))
        else:
            display = self.current_image.copy()
            
        dh, dw = display.shape[:2]
        
        # Draw existing boxes with delete buttons
        self.delete_buttons = []
        for idx, box in enumerate(self.boxes):
            p1 = self.to_display_coords(box.x1, box.y1)
            p2 = self.to_display_coords(box.x2, box.y2)
            cv2.rectangle(display, p1, p2, BOX_COLOR, 2)
            
            # Draw delete button (red X) in top-right corner
            btn_x1 = p2[0] - DELETE_BTN_SIZE
            btn_y1 = p1[1]
            btn_x2 = p2[0]
            btn_y2 = p1[1] + DELETE_BTN_SIZE
            
            # Store button position for click detection
            self.delete_buttons.append((btn_x1, btn_y1, btn_x2, btn_y2, idx))
            
            # Draw button background
            cv2.rectangle(display, (btn_x1, btn_y1), (btn_x2, btn_y2), DELETE_BTN_COLOR, -1)
            
            # Draw X
            margin = 4
            cv2.line(display, (btn_x1 + margin, btn_y1 + margin), 
                    (btn_x2 - margin, btn_y2 - margin), (255, 255, 255), 2)
            cv2.line(display, (btn_x2 - margin, btn_y1 + margin), 
                    (btn_x1 + margin, btn_y2 - margin), (255, 255, 255), 2)
            
        # Draw first point if set
        if self.first_point:
            fp_display = self.to_display_coords(*self.first_point)
            cv2.circle(display, fp_display, 5, POINT_COLOR, -1)
            
            # Draw preview rectangle to mouse
            cv2.rectangle(display, fp_display, self.mouse_pos, PREVIEW_COLOR, 1)
            
        # Info bar
        bar_h = 70
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (dw, bar_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
        
        # Text
        cv2.putText(display, f"[{self.current_idx + 1}/{len(self.images)}] {item.folder} -> {item.class_name}",
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(display, f"Boxes: {len(self.boxes)} | {item.path.name}",
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Status
        if self.first_point:
            cv2.putText(display, "Click second corner...", (dw - 200, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Help
        cv2.putText(display, "Click: Draw box | Click X: Delete box | Z: Undo | C: Clear | Arrows: Nav | Q: Quit",
                   (10, dh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        return display
        
    def check_delete_button_click(self, x: int, y: int) -> int:
        """Check if click is on a delete button. Returns box index or -1."""
        for btn_x1, btn_y1, btn_x2, btn_y2, box_idx in self.delete_buttons:
            if btn_x1 <= x <= btn_x2 and btn_y1 <= y <= btn_y2:
                return box_idx
        return -1
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse."""
        self.mouse_pos = (x, y)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # First check if clicking on a delete button
            delete_idx = self.check_delete_button_click(x, y)
            if delete_idx >= 0:
                deleted_box = self.boxes.pop(delete_idx)
                print(f"Deleted box {delete_idx + 1}: ({deleted_box.x1}, {deleted_box.y1}) -> ({deleted_box.x2}, {deleted_box.y2})")
                self.first_point = None  # Cancel any pending box
                return
            
            ix, iy = self.to_img_coords(x, y)
            
            if self.first_point is None:
                # First click - set first corner
                self.first_point = (ix, iy)
            else:
                # Second click - create box
                x1, y1 = self.first_point
                x2, y2 = ix, iy
                
                # Ensure proper order (top-left to bottom-right)
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # Only add if box is big enough
                if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                    item = self.images[self.current_idx]
                    self.boxes.append(BBox(x1, y1, x2, y2, item.class_id))
                    print(f"Added box: ({x1}, {y1}) -> ({x2}, {y2})")
                    
                self.first_point = None
                
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.first_point:
                # Cancel current box drawing
                self.first_point = None
                print("Cancelled box - right-click to undo last box")
            elif self.boxes:
                # Delete last box
                self.boxes.pop()
                print("Removed last box")
                
    def save_labels(self):
        """Save current boxes."""
        if not self.images:
            return
            
        item = self.images[self.current_idx]
        h, w = self.current_image.shape[:2]
        
        label_path = self.output_dir / item.folder / item.label_filename
        
        with open(label_path, 'w') as f:
            for box in self.boxes:
                cid, xc, yc, bw, bh = box.to_yolo(w, h)
                f.write(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                
        print(f"Saved: {label_path.name} ({len(self.boxes)} boxes)")
        
    def skip_image(self):
        """Skip without saving."""
        if not self.images:
            return
        item = self.images[self.current_idx]
        label_path = self.output_dir / item.folder / item.label_filename
        label_path.touch()
        print(f"Skipped: {item.path.name}")
    
    def confirm_quit(self) -> bool:
        """Show confirmation dialog before quitting."""
        # Create confirmation overlay
        if self.current_image is None:
            return True
            
        h, w = self.current_image.shape[:2]
        overlay = np.zeros((int(h * self.scale), int(w * self.scale), 3), dtype=np.uint8)
        overlay[:] = (30, 30, 30)
        
        dh, dw = overlay.shape[:2]
        
        # Draw confirmation box
        box_w, box_h = 400, 150
        box_x = (dw - box_w) // 2
        box_y = (dh - box_h) // 2
        
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (60, 60, 60), -1)
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (100, 100, 100), 2)
        
        # Text
        cv2.putText(overlay, "Quit labeling?", (box_x + 100, box_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(overlay, "Y = Yes, quit    N = No, continue", (box_x + 50, box_y + 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.imshow(self.WINDOW_NAME, overlay)
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('y') or key == ord('Y'):
                return True
            elif key == ord('n') or key == ord('N') or key == 27:
                return False
        
    def run(self, start_idx: int = 0):
        """Main loop."""
        self.load_images()
        
        if not self.images:
            print("No images to label!")
            return
            
        # Set starting index
        self.current_idx = min(start_idx, len(self.images) - 1)
        
        cv2.namedWindow(self.WINDOW_NAME)
        cv2.setMouseCallback(self.WINDOW_NAME, self.mouse_callback)
        
        self.load_current()
        
        print("\n" + "=" * 60)
        print("BBOX LABELER")
        print("=" * 60)
        print("Left click: Set corner (click twice to make box)")
        print("Right click: Cancel / Delete last box")
        print("Z: Undo | C: Clear all | Arrows: Navigate")
        print("Space/Enter: Save & Next | S: Save | N: Skip | Q: Quit")
        print("=" * 60 + "\n")
        
        running = True
        while running:
            display = self.render()
            cv2.imshow(self.WINDOW_NAME, display)
            
            key = cv2.waitKey(30) & 0xFF
            
            if key == 255:
                continue
                
            if key == ord('q') or key == 27:
                if self.confirm_quit():
                    running = False
                
            elif key == ord('z'):
                if self.boxes:
                    self.boxes.pop()
                    print("Undo - removed last box")
                    
            elif key == ord('c'):
                self.boxes.clear()
                self.first_point = None
                print("Cleared all boxes")
                
            elif key == ord('s'):
                self.save_labels()
                
            elif key == ord('n'):
                self.skip_image()
                if self.current_idx < len(self.images) - 1:
                    self.current_idx += 1
                    self.load_current()
                else:
                    print("All done!")
                    running = False
                    
            elif key == ord(' ') or key == 13:
                self.save_labels()
                if self.current_idx < len(self.images) - 1:
                    self.current_idx += 1
                    self.load_current()
                else:
                    print("All images labeled!")
                    running = False
                    
            # Arrow keys (auto-save before navigating)
            elif key == 2 or key == 81:  # Left
                if self.current_idx > 0:
                    if self.boxes:
                        self.save_labels()
                    self.current_idx -= 1
                    self.load_current()
                    print(f"Image {self.current_idx + 1}/{len(self.images)}")
                    
            elif key == 3 or key == 83:  # Right
                if self.current_idx < len(self.images) - 1:
                    if self.boxes:
                        self.save_labels()
                    self.current_idx += 1
                    self.load_current()
                    print(f"Image {self.current_idx + 1}/{len(self.images)}")
                    
        cv2.destroyAllWindows()
        print(f"\nLabels saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Simple bbox labeling tool")
    parser.add_argument("--input", type=str, default="data/images/unlabeled/")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--show-all", action="store_true", help="Show all images including already-labeled ones")
    parser.add_argument("--start", type=int, default=0, help="Start from image index (0-based)")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: {input_dir} not found")
        sys.exit(1)
        
    labeler = BBoxLabeler(
        input_dir=input_dir,
        output_dir=Path(args.output) if args.output else None,
        resume=not args.no_resume,
        show_all=args.show_all
    )
    labeler.run(start_idx=args.start)


if __name__ == "__main__":
    main()

