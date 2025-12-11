#!/usr/bin/env python3
"""
Extract frames from video files for labeling.

Usage:
    # Extract from a single video
    python tools/extract_frames.py \
        --input data/clips/gameplay_W_000.mp4 \
        --output data/images/unlabeled/ \
        --every 3

    # Extract from all videos in a directory
    python tools/extract_frames.py \
        --input data/clips/ \
        --output data/images/unlabeled/ \
        --every 3
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.frame_extractor import FrameExtractor


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from videos for labeling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Extract every 3rd frame from a video
    python tools/extract_frames.py \\
        --input data/clips/gameplay_W_000.mp4 \\
        --output data/images/unlabeled/ \\
        --every 3

    # Extract from all clips in a directory
    python tools/extract_frames.py \\
        --input data/clips/ \\
        --output data/images/unlabeled/ \\
        --every 3

    # Extract every frame (for detailed analysis)
    python tools/extract_frames.py \\
        --input video.mp4 \\
        --output frames/ \\
        --every 1

    # Extract as PNG with custom prefix
    python tools/extract_frames.py \\
        --input video.mp4 \\
        --output frames/ \\
        --every 5 \\
        --format png \\
        --prefix my_video
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input video file or directory containing videos"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for extracted frames"
    )
    
    parser.add_argument(
        "--every",
        type=int,
        default=3,
        help="Extract every N-th frame (default: 3)"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        default="jpg",
        choices=["jpg", "jpeg", "png"],
        help="Output image format (default: jpg)"
    )
    
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality 1-100 (default: 95)"
    )
    
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Custom filename prefix (default: video filename)"
    )
    
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting frame index (default: 0)"
    )
    
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Ending frame index (default: all frames)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress bar"
    )
    
    args = parser.parse_args()
    
    # Validate input
    input_path = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: Input not found: {input_path}")
        sys.exit(1)
        
    # Create extractor
    extractor = FrameExtractor(
        frame_stride=args.every,
        output_format=args.format,
        jpeg_quality=args.quality
    )
    
    # Extract frames
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if input_path.is_file():
        # Single video
        print(f"Extracting frames from: {input_path}")
        print(f"Output directory: {output_dir}")
        print(f"Frame stride: every {args.every} frame(s)")
        
        extracted = extractor.extract_from_video(
            video_path=input_path,
            output_dir=output_dir,
            prefix=args.prefix,
            start_frame=args.start,
            end_frame=args.end,
            show_progress=not args.quiet
        )
        
    elif input_path.is_dir():
        # Directory of videos
        print(f"Extracting frames from videos in: {input_path}")
        print(f"Output directory: {output_dir}")
        print(f"Frame stride: every {args.every} frame(s)")
        
        extracted = extractor.extract_from_directory(
            video_dir=input_path,
            output_dir=output_dir,
            show_progress=not args.quiet
        )
        
    else:
        print(f"Error: Invalid input path: {input_path}")
        sys.exit(1)
        
    print(f"\n{'='*50}")
    print(f"Total frames extracted: {len(extracted)}")
    print(f"Output location: {output_dir}")


if __name__ == "__main__":
    main()





