#!/usr/bin/env python3
"""
Annotation Reviewer - View labeled images with bounding boxes drawn

Usage:
    python tools/review_annotations.py --images data/images/train --labels data/labels/train

Controls:
    Left/Right Arrow  - Previous/Next image
    Space            - Next image
    Q / Esc          - Quit
    S                - Save current view as PNG
    F                - Filter by class (cycle through classes)
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np

# Class names and colors (BGR format for OpenCV)
CLASS_INFO = {
    0: ("JINX_IDLE", (150, 150, 150)),       # Grey
    1: ("VFX_W_PROJECTILE", (255, 150, 50)),  # Blue
    2: ("VFX_W_IMPACT", (50, 255, 50)),       # Green
    3: ("VFX_R_ROCKET", (50, 50, 255)),       # Red
    4: ("VFX_R_IMPACT", (50, 255, 255)),      # Yellow
}


def load_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """Load YOLO format labels from a text file."""
    labels = []
    if not label_path.exists():
        return labels
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                labels.append((class_id, x_center, y_center, width, height))
    
    return labels


def draw_boxes(image: np.ndarray, labels: List[Tuple[int, float, float, float, float]]) -> np.ndarray:
    """Draw bounding boxes on the image."""
    h, w = image.shape[:2]
    annotated = image.copy()
    
    for class_id, x_center, y_center, box_w, box_h in labels:
        # Convert normalized coords to pixel coords
        x1 = int((x_center - box_w / 2) * w)
        y1 = int((y_center - box_h / 2) * h)
        x2 = int((x_center + box_w / 2) * w)
        y2 = int((y_center + box_h / 2) * h)
        
        # Get class info
        class_name, color = CLASS_INFO.get(class_id, (f"Class_{class_id}", (255, 255, 255)))
        
        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label_text = f"{class_name}"
        (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), color, -1)
        
        # Draw label text
        cv2.putText(annotated, label_text, (x1 + 2, y1 - 4), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return annotated


def main():
    parser = argparse.ArgumentParser(description="Review annotated images with bounding boxes")
    parser.add_argument("--images", type=str, default="data/images/unlabeled",
                        help="Path to images directory (can contain subfolders)")
    parser.add_argument("--labels", type=str, default="data/images/labels", 
                        help="Path to labels directory (can contain subfolders)")
    parser.add_argument("--filter-class", type=int, default=None,
                        help="Only show images with this class ID")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Start at this image index")
    
    args = parser.parse_args()
    
    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        sys.exit(1)
    
    # Find all images (recursively search subfolders)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    all_images = sorted([f for f in images_dir.rglob('*') 
                         if f.suffix.lower() in image_extensions])
    
    if not all_images:
        print(f"No images found in {images_dir}")
        sys.exit(1)
    
    def get_label_path(img_path: Path, labels_dir: Path, images_dir: Path) -> Path:
        """Find the corresponding label file for an image."""
        # Try to match subfolder structure (e.g., images/unlabeled/idle/ -> labels/idle/)
        try:
            relative_path = img_path.relative_to(images_dir)
            label_path = labels_dir / relative_path.with_suffix('.txt')
            if label_path.exists():
                return label_path
        except ValueError:
            pass
        # Fallback: try flat label directory
        return labels_dir / f"{img_path.stem}.txt"
    
    # Filter by class if specified
    if args.filter_class is not None:
        filtered_images = []
        for img_path in all_images:
            label_path = get_label_path(img_path, labels_dir, images_dir)
            labels = load_labels(label_path)
            if any(l[0] == args.filter_class for l in labels):
                filtered_images.append(img_path)
        all_images = filtered_images
        print(f"Filtered to {len(all_images)} images containing class {args.filter_class}")
    
    print(f"\nFound {len(all_images)} images")
    print("\nControls:")
    print("  Left/Right Arrow - Previous/Next image")
    print("  Space           - Next image")
    print("  1-5             - Filter by class (1=IDLE, 2=W_PROJ, 3=W_IMPACT, 4=R_ROCKET, 5=R_IMPACT)")
    print("  0               - Show all classes")
    print("  S               - Save current view")
    print("  Q / Esc         - Quit")
    
    current_idx = min(args.start_index, len(all_images) - 1)
    current_filter = args.filter_class
    
    cv2.namedWindow("Annotation Review", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Annotation Review", 1280, 720)
    
    while True:
        if not all_images:
            print("No images match the current filter")
            break
            
        img_path = all_images[current_idx]
        label_path = get_label_path(img_path, labels_dir, images_dir)
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Failed to load: {img_path}")
            current_idx = (current_idx + 1) % len(all_images)
            continue
        
        # Load and draw labels
        labels = load_labels(label_path)
        annotated = draw_boxes(image, labels)
        
        # Add info overlay
        info_text = f"[{current_idx + 1}/{len(all_images)}] {img_path.name}"
        cv2.putText(annotated, info_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated, info_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
        
        # Show label count
        label_counts = {}
        for l in labels:
            class_name = CLASS_INFO.get(l[0], (f"Class_{l[0]}", None))[0]
            label_counts[class_name] = label_counts.get(class_name, 0) + 1
        
        count_text = "Labels: " + ", ".join(f"{k}:{v}" for k, v in label_counts.items())
        if not labels:
            count_text = "Labels: None"
        cv2.putText(annotated, count_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated, count_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Show filter status
        if current_filter is not None:
            filter_name = CLASS_INFO.get(current_filter, (f"Class_{current_filter}", None))[0]
            filter_text = f"Filter: {filter_name}"
        else:
            filter_text = "Filter: All classes"
        cv2.putText(annotated, filter_text, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        cv2.imshow("Annotation Review", annotated)
        
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q') or key == 27:  # Q or Esc
            break
        elif key == 83 or key == 3 or key == ord(' '):  # Right arrow or Space
            current_idx = (current_idx + 1) % len(all_images)
        elif key == 81 or key == 2:  # Left arrow
            current_idx = (current_idx - 1) % len(all_images)
        elif key == ord('s'):  # Save
            save_path = Path("outputs") / f"review_{img_path.stem}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), annotated)
            print(f"Saved: {save_path}")
        elif key in [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
            # Filter by class
            filter_class = int(chr(key)) - 1 if chr(key) != '0' else None
            current_filter = filter_class
            
            # Re-filter images
            if current_filter is not None:
                filtered_images = []
                for img_p in sorted([f for f in images_dir.rglob('*') 
                                     if f.suffix.lower() in image_extensions]):
                    lbl_path = get_label_path(img_p, labels_dir, images_dir)
                    lbls = load_labels(lbl_path)
                    if any(l[0] == current_filter for l in lbls):
                        filtered_images.append(img_p)
                all_images = filtered_images
                class_name = CLASS_INFO.get(current_filter, (f"Class_{current_filter}", None))[0]
                print(f"Filtered to {len(all_images)} images with {class_name}")
            else:
                all_images = sorted([f for f in images_dir.rglob('*') 
                                     if f.suffix.lower() in image_extensions])
                print(f"Showing all {len(all_images)} images")
            current_idx = 0 if all_images else 0
    
    cv2.destroyAllWindows()
    print("\nDone!")


if __name__ == "__main__":
    main()

