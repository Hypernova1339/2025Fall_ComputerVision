#!/usr/bin/env python3
"""
Brush-to-BBox Labeling Tool

Paint over subjects with a brush and auto-generate YOLO bounding boxes.
Classes are auto-assigned based on folder name.

Usage:
    python tools/brush_labeler.py --input data/images/unlabeled/

Controls:
    Left Click + Drag  - Paint (red, 30% opacity)
    Right Click        - Erase paint
    ]                  - Increase brush size
    [                  - Decrease brush size
    Cmd+Z / Z          - Undo last stroke
    C                  - Clear all paint
    Left Arrow         - Previous image
    Right Arrow        - Next image  
    Space / Enter      - Save bbox and go to next image
    N                  - Skip image (no label)
    S                  - Save current progress
    Q / Esc            - Save and quit
"""

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass, field
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


@dataclass
class ImageItem:
    """Represents an image to label."""
    path: Path
    folder: str
    class_id: int
    class_name: str
    
    @property
    def label_filename(self) -> str:
        return self.path.stem + '.txt'


class BrushLabeler:
    """Interactive brush-to-bbox labeling tool."""
    
    WINDOW_NAME = "Brush Labeler"
    
    def __init__(
        self,
        input_dir: Path,
        output_dir: Optional[Path] = None,
        resume: bool = True
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir or input_dir.parent / 'labels'
        self.resume = resume
        
        # Brush settings
        self.brush_size = 25
        self.brush_min = 5
        self.brush_max = 150
        self.brush_opacity = 0.3
        
        # State
        self.images: List[ImageItem] = []
        self.current_idx = 0
        self.current_image: Optional[np.ndarray] = None
        self.mask: Optional[np.ndarray] = None
        self.undo_stack: List[np.ndarray] = []
        self.drawing = False
        self.erasing = False
        self.last_pt: Optional[Tuple[int, int]] = None
        self.cursor_pos: Tuple[int, int] = (0, 0)
        self.needs_redraw = True
        
        # Display cache
        self.display_cache: Optional[np.ndarray] = None
        self.scale = 1.0
        
    def load_images(self):
        """Load all images from input directory."""
        self.images = []
        
        for folder_name, (class_id, class_name) in FOLDER_TO_CLASS.items():
            folder_path = self.input_dir / folder_name
            if not folder_path.exists():
                continue
                
            # Create output folder
            label_folder = self.output_dir / folder_name
            label_folder.mkdir(parents=True, exist_ok=True)
            
            for ext in ['*.jpg', '*.png', '*.jpeg']:
                for img_path in sorted(folder_path.glob(ext)):
                    # Check if already labeled
                    label_path = label_folder / (img_path.stem + '.txt')
                    if self.resume and label_path.exists():
                        continue
                        
                    self.images.append(ImageItem(
                        path=img_path,
                        folder=folder_name,
                        class_id=class_id,
                        class_name=class_name
                    ))
                    
        print(f"Found {len(self.images)} images to label")
        
    def load_current_image(self):
        """Load current image."""
        if not self.images or self.current_idx >= len(self.images):
            return False
            
        item = self.images[self.current_idx]
        self.current_image = cv2.imread(str(item.path))
        
        if self.current_image is None:
            print(f"Error loading: {item.path}")
            return False
            
        h, w = self.current_image.shape[:2]
        
        # Scale to fit screen (max 1400px)
        self.scale = min(1400 / max(h, w), 1.0)
        
        # Init mask
        self.mask = np.zeros((h, w), dtype=np.uint8)
        self.undo_stack = []
        self.needs_redraw = True
        
        return True
        
    def render(self) -> np.ndarray:
        """Render current view."""
        if self.current_image is None:
            return np.zeros((100, 400, 3), dtype=np.uint8)
            
        h, w = self.current_image.shape[:2]
        item = self.images[self.current_idx]
        
        # Start with original image
        display = self.current_image.copy()
        
        # Apply red paint overlay where mask > 0
        if self.mask is not None:
            red_overlay = np.zeros_like(display)
            red_overlay[:, :, 2] = 255  # Red channel
            
            mask_float = (self.mask / 255.0 * self.brush_opacity)
            mask_3ch = np.stack([mask_float] * 3, axis=-1)
            
            display = (display * (1 - mask_3ch) + red_overlay * mask_3ch).astype(np.uint8)
        
        # Scale image
        if self.scale < 1.0:
            new_w, new_h = int(w * self.scale), int(h * self.scale)
            display = cv2.resize(display, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        dh, dw = display.shape[:2]
        
        # Draw info bar at top
        bar_h = 60
        cv2.rectangle(display, (0, 0), (dw, bar_h), (20, 20, 20), -1)
        
        # Progress
        cv2.putText(display, f"[{self.current_idx + 1}/{len(self.images)}]", 
                   (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Class info
        cv2.putText(display, f"{item.folder} -> {item.class_name}", 
                   (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Brush size
        cv2.putText(display, f"Brush: {self.brush_size}", 
                   (dw - 100, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Filename
        cv2.putText(display, item.path.name, 
                   (dw - 200, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Draw brush cursor
        cx, cy = self.cursor_pos
        brush_display_size = max(1, int(self.brush_size * self.scale))
        cv2.circle(display, (cx, cy), brush_display_size, (255, 255, 255), 1)
        
        # Help text at bottom
        cv2.putText(display, "Space:Save+Next | Arrows:Navigate | Z:Undo | C:Clear | N:Skip | Q:Quit", 
                   (10, dh - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)
        
        return display
        
    def paint(self, x: int, y: int, erase: bool = False):
        """Paint at display coordinates."""
        if self.mask is None or self.current_image is None:
            return
            
        # Convert to image coordinates
        ix = int(x / self.scale)
        iy = int(y / self.scale)
        
        h, w = self.current_image.shape[:2]
        ix = max(0, min(w - 1, ix))
        iy = max(0, min(h - 1, iy))
        
        val = 0 if erase else 255
        
        # Draw circle
        cv2.circle(self.mask, (ix, iy), self.brush_size, val, -1)
        
        # Connect to last point for smooth strokes
        if self.last_pt is not None:
            cv2.line(self.mask, self.last_pt, (ix, iy), val, self.brush_size * 2)
            
        self.last_pt = (ix, iy)
        self.needs_redraw = True
        
    def save_undo(self):
        """Save mask state for undo."""
        if self.mask is not None:
            self.undo_stack.append(self.mask.copy())
            if len(self.undo_stack) > 30:
                self.undo_stack.pop(0)
                
    def undo(self):
        """Undo last stroke."""
        if self.undo_stack:
            self.mask = self.undo_stack.pop()
            self.needs_redraw = True
            print("Undo")
            
    def clear(self):
        """Clear mask."""
        if self.mask is not None:
            self.save_undo()
            self.mask.fill(0)
            self.needs_redraw = True
            print("Cleared")
            
    def compute_bboxes(self) -> List[Tuple[int, float, float, float, float]]:
        """Get YOLO bboxes from mask."""
        if self.mask is None or self.current_image is None:
            return []
            
        h, w = self.current_image.shape[:2]
        item = self.images[self.current_idx]
        
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < 10 or bh < 10:  # Skip tiny
                continue
                
            # YOLO format (normalized)
            xc = (x + bw / 2) / w
            yc = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            
            bboxes.append((item.class_id, xc, yc, nw, nh))
            
        return bboxes
        
    def save_label(self):
        """Save current label."""
        if not self.images:
            return
            
        item = self.images[self.current_idx]
        bboxes = self.compute_bboxes()
        
        label_path = self.output_dir / item.folder / item.label_filename
        
        with open(label_path, 'w') as f:
            for cid, xc, yc, nw, nh in bboxes:
                f.write(f"{cid} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")
                
        print(f"Saved: {label_path.name} ({len(bboxes)} boxes)")
        
    def skip_image(self):
        """Skip without label."""
        if not self.images:
            return
        item = self.images[self.current_idx]
        label_path = self.output_dir / item.folder / item.label_filename
        label_path.touch()
        print(f"Skipped: {item.path.name}")
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse."""
        self.cursor_pos = (x, y)
        self.needs_redraw = True
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.save_undo()
            self.drawing = True
            self.last_pt = None
            self.paint(x, y, erase=False)
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.save_undo()
            self.erasing = True
            self.last_pt = None
            self.paint(x, y, erase=True)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.paint(x, y, erase=False)
            elif self.erasing:
                self.paint(x, y, erase=True)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.last_pt = None
            
        elif event == cv2.EVENT_RBUTTONUP:
            self.erasing = False
            self.last_pt = None
            
    def next_image(self):
        """Go to next image."""
        if self.current_idx < len(self.images) - 1:
            self.current_idx += 1
            self.load_current_image()
            print(f"Image {self.current_idx + 1}/{len(self.images)}")
            
    def prev_image(self):
        """Go to previous image."""
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_current_image()
            print(f"Image {self.current_idx + 1}/{len(self.images)}")
            
    def run(self):
        """Main loop."""
        self.load_images()
        
        if not self.images:
            print("No images to label!")
            return
            
        cv2.namedWindow(self.WINDOW_NAME)
        cv2.setMouseCallback(self.WINDOW_NAME, self.mouse_callback)
        
        self.load_current_image()
        
        print("\n" + "=" * 50)
        print("BRUSH LABELER")
        print("=" * 50)
        print("Left-click: Paint | Right-click: Erase")
        print("[ ]: Brush size | Z: Undo | C: Clear")
        print("Arrows: Navigate | Space: Save & Next")
        print("=" * 50 + "\n")
        
        running = True
        while running:
            # Render and show
            display = self.render()
            cv2.imshow(self.WINDOW_NAME, display)
            
            # Wait for key (use longer wait when not drawing for less CPU)
            wait_time = 1 if (self.drawing or self.erasing) else 30
            key = cv2.waitKey(wait_time) & 0xFF
            
            if key == 255:  # No key
                continue
                
            # Debug: uncomment to see key codes
            # if key != 255: print(f"Key: {key}")
            
            if key == ord('q') or key == 27:  # Q or Esc
                running = False
                
            elif key == ord(']'):
                self.brush_size = min(self.brush_max, self.brush_size + 5)
                print(f"Brush: {self.brush_size}")
                self.needs_redraw = True
                
            elif key == ord('['):
                self.brush_size = max(self.brush_min, self.brush_size - 5)
                print(f"Brush: {self.brush_size}")
                self.needs_redraw = True
                
            elif key == ord('z'):  # Z for undo (Cmd+Z sends just 'z' on Mac)
                self.undo()
                
            elif key == ord('c'):
                self.clear()
                
            elif key == ord('n'):
                self.skip_image()
                self.next_image()
                if self.current_idx >= len(self.images) - 1:
                    print("\nAll done!")
                    running = False
                    
            elif key == ord('s'):
                self.save_label()
                
            elif key == ord(' ') or key == 13:  # Space or Enter
                self.save_label()
                self.current_idx += 1
                if self.current_idx >= len(self.images):
                    print("\nAll images labeled!")
                    running = False
                else:
                    self.load_current_image()
                    
            # Arrow keys (Mac OpenCV key codes)
            elif key == 0 or key == 63232:  # Up arrow (unused)
                pass
            elif key == 1 or key == 63233:  # Down arrow (unused)
                pass
            elif key == 2 or key == 63234 or key == 81:  # Left arrow
                self.prev_image()
            elif key == 3 or key == 63235 or key == 83:  # Right arrow
                self.next_image()
                
        cv2.destroyAllWindows()
        print(f"\nLabels saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Brush-to-BBox labeling tool")
    
    parser.add_argument("--input", type=str, default="data/images/unlabeled/",
                       help="Input directory with image folders")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory for labels")
    parser.add_argument("--no-resume", action="store_true",
                       help="Don't skip already labeled images")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else None
    
    if not input_dir.exists():
        print(f"Error: {input_dir} not found")
        sys.exit(1)
        
    labeler = BrushLabeler(
        input_dir=input_dir,
        output_dir=output_dir,
        resume=not args.no_resume
    )
    
    labeler.run()


if __name__ == "__main__":
    main()
