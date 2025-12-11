#!/usr/bin/env python3
"""
Run combined UI + VFX detection on gameplay video.

This tool uses both UI cooldown detection and YOLO-based VFX detection
for robust ability detection and hit/miss classification.

Usage:
    # UI detection only (no trained model needed)
    python tools/run_combined_detection.py \
        --video "data/raw_videos/video 1 (long).mp4" \
        --output outputs/ \
        --ui-only

    # Combined UI + VFX detection
    python tools/run_combined_detection.py \
        --video "data/raw_videos/video 1 (long).mp4" \
        --weights runs/detect/jinx_abilities/weights/best.pt \
        --output outputs/

    # VFX detection only
    python tools/run_combined_detection.py \
        --video "data/raw_videos/video 1 (long).mp4" \
        --weights runs/detect/jinx_abilities/weights/best.pt \
        --output outputs/ \
        --vfx-only
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.detection.combined_detector import run_combined_detection


def main():
    parser = argparse.ArgumentParser(
        description="Run combined UI + VFX detection on gameplay video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # UI-only detection (no model needed)
    python tools/run_combined_detection.py \\
        --video "data/raw_videos/video 1 (long).mp4" \\
        --output outputs/ \\
        --ui-only

    # Full combined detection
    python tools/run_combined_detection.py \\
        --video "data/raw_videos/video 1 (long).mp4" \\
        --weights runs/detect/jinx_abilities_v2/weights/best.pt \\
        --output outputs/

    # Faster processing (skip frames)
    python tools/run_combined_detection.py \\
        --video "data/raw_videos/video 1 (long).mp4" \\
        --output outputs/ \\
        --ui-only \\
        --stride 2
        """
    )
    
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to video file"
    )
    
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to YOLO weights for VFX detection"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--ui-only",
        action="store_true",
        help="Use only UI detection (no YOLO model needed)"
    )
    
    parser.add_argument(
        "--vfx-only",
        action="store_true",
        help="Use only VFX detection (requires YOLO weights)"
    )
    
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Process every N frames (default: 1 = all frames)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress bar"
    )
    
    args = parser.parse_args()
    
    # Validate args
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)
        
    if args.vfx_only and not args.weights:
        print("Error: --vfx-only requires --weights")
        sys.exit(1)
        
    # Determine detection modes
    use_ui = not args.vfx_only
    use_vfx = not args.ui_only and args.weights is not None
    
    if not use_ui and not use_vfx:
        print("Error: At least one detection mode must be enabled")
        sys.exit(1)
        
    # Run detection
    events = run_combined_detection(
        video_path=video_path,
        yolo_weights=args.weights,
        output_dir=args.output,
        use_ui=use_ui,
        use_vfx=use_vfx,
        frame_stride=args.stride,
        show_progress=not args.quiet,
    )
    
    print(f"\n{'='*50}")
    print("Detection complete!")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()





