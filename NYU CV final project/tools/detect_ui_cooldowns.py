#!/usr/bin/env python3
"""
Detect ability casts from UI cooldown indicators.

This tool analyzes the ability bar UI to automatically detect when
Jinx's W and R abilities are cast, without needing manual labeling.

Usage:
    # Detect casts and export to CSV
    python tools/detect_ui_cooldowns.py \
        --video "data/raw_videos/video 1 (long).mp4" \
        --output data/events/auto_events.csv

    # Calibrate UI regions (verify they're positioned correctly)
    python tools/detect_ui_cooldowns.py \
        --video "data/raw_videos/video 1 (long).mp4" \
        --calibrate
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.detection.ui_detector import UIDetector, export_events_to_csv


def main():
    parser = argparse.ArgumentParser(
        description="Detect ability casts from UI cooldown indicators",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Auto-detect ability casts
    python tools/detect_ui_cooldowns.py \\
        --video "data/raw_videos/video 1 (long).mp4" \\
        --output data/events/auto_events.csv

    # Calibrate UI regions (check if regions are correct)
    python tools/detect_ui_cooldowns.py \\
        --video "data/raw_videos/video 1 (long).mp4" \\
        --calibrate

    # Process every 2nd frame (faster)
    python tools/detect_ui_cooldowns.py \\
        --video "data/raw_videos/video 1 (long).mp4" \\
        --output data/events/auto_events.csv \\
        --sample-rate 2

Notes:
    - The UI regions are calibrated for standard 1920x1080 gameplay
    - If detection seems off, use --calibrate to verify region positions
    - Results should be reviewed manually for hit/miss classification
        """
    )
    
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to video file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file for detected events"
    )
    
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Show calibration image with UI regions"
    )
    
    parser.add_argument(
        "--save-calibration",
        type=str,
        default=None,
        help="Save calibration image to file"
    )
    
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=1,
        help="Process every N frames (default: 1 = all frames)"
    )
    
    parser.add_argument(
        "--saturation-threshold",
        type=float,
        default=0.3,
        help="Saturation threshold for cooldown detection (default: 0.3)"
    )
    
    parser.add_argument(
        "--brightness-threshold",
        type=float,
        default=0.4,
        help="Brightness threshold for cooldown detection (default: 0.4)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress bar"
    )
    
    args = parser.parse_args()
    
    # Validate video path
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)
        
    # Create detector
    detector = UIDetector(
        saturation_threshold=args.saturation_threshold,
        brightness_threshold=args.brightness_threshold,
    )
    
    if args.calibrate:
        # Calibration mode
        print("Opening calibration view...")
        print("Check that the green boxes are positioned over the W and R icons.")
        print("Press any key to close the window.")
        detector.calibrate_regions(video_path, args.save_calibration)
        return
        
    if not args.output:
        print("Error: --output is required for detection mode")
        print("Use --calibrate for calibration mode")
        sys.exit(1)
        
    # Process video
    events = detector.process_video(
        video_path,
        show_progress=not args.quiet,
        sample_rate=args.sample_rate,
    )
    
    if not events:
        print("No ability casts detected.")
        print("Tips:")
        print("  - Run with --calibrate to verify UI region positions")
        print("  - Try adjusting --saturation-threshold or --brightness-threshold")
        return
        
    # Export to CSV
    video_id = video_path.stem
    export_events_to_csv(events, video_id, args.output)
    
    print(f"\n{'='*50}")
    print("Next steps:")
    print("  1. Review the detected events in the CSV")
    print("  2. Use tools/label_video.py to add hit/miss labels")
    print("  3. Or manually edit the CSV to add 'hit' or 'miss' in the result column")


if __name__ == "__main__":
    main()





