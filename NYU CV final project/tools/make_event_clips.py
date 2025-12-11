#!/usr/bin/env python3
"""
Extract short video clips around ability cast events.

Usage:
    python tools/make_event_clips.py \
        --video data/raw_videos/gameplay.mp4 \
        --events data/events/events.csv \
        --output data/clips/

This will create clips like:
    data/clips/gameplay_W_000.mp4
    data/clips/gameplay_W_001.mp4
    data/clips/gameplay_R_000.mp4
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.clip_extractor import ClipExtractor, AbilityEvent


def main():
    parser = argparse.ArgumentParser(
        description="Extract video clips around ability events",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Extract clips from a single video
    python tools/make_event_clips.py \\
        --video data/raw_videos/video_1_long.mp4 \\
        --events data/events/events.csv \\
        --output data/clips/

    # Use OpenCV instead of FFmpeg
    python tools/make_event_clips.py \\
        --video data/raw_videos/video_1_long.mp4 \\
        --events data/events/events.csv \\
        --output data/clips/ \\
        --no-ffmpeg

    # Custom clip windows
    python tools/make_event_clips.py \\
        --video data/raw_videos/video_1_long.mp4 \\
        --events data/events/events.csv \\
        --output data/clips/ \\
        --w-window -1.0 3.0 \\
        --r-window -1.0 5.0
        """
    )
    
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file"
    )
    
    parser.add_argument(
        "--events",
        type=str,
        required=True,
        help="Path to events.csv file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for clips"
    )
    
    parser.add_argument(
        "--w-window",
        type=float,
        nargs=2,
        default=[-0.5, 2.0],
        metavar=("PRE", "POST"),
        help="Time window for W clips: seconds before and after cast (default: -0.5 2.0)"
    )
    
    parser.add_argument(
        "--r-window",
        type=float,
        nargs=2,
        default=[-0.5, 4.0],
        metavar=("PRE", "POST"),
        help="Time window for R clips: seconds before and after cast (default: -0.5 4.0)"
    )
    
    parser.add_argument(
        "--no-ffmpeg",
        action="store_true",
        help="Use OpenCV instead of FFmpeg (slower but no FFmpeg dependency)"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        default="mp4",
        choices=["mp4", "avi", "mov"],
        help="Output video format (default: mp4)"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    video_path = Path(args.video)
    events_path = Path(args.events)
    output_dir = Path(args.output)
    
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
        
    if not events_path.exists():
        print(f"Error: Events file not found: {events_path}")
        sys.exit(1)
        
    # Create extractor with custom windows
    clip_windows = {
        "W": (args.w_window[0], args.w_window[1]),
        "R": (args.r_window[0], args.r_window[1]),
    }
    
    extractor = ClipExtractor(
        clip_windows=clip_windows,
        output_format=args.format
    )
    
    # Load events
    print(f"Loading events from: {events_path}")
    events = extractor.load_events(events_path)
    
    # Filter events for this video
    video_id = video_path.stem
    video_events = [e for e in events if e.video_id == video_id or e.video_id.replace("_", " ") == video_id.replace("_", " ")]
    
    if not video_events:
        print(f"Warning: No events found for video_id '{video_id}'")
        print(f"Available video_ids in events.csv: {set(e.video_id for e in events)}")
        sys.exit(1)
        
    print(f"Found {len(video_events)} events for video: {video_id}")
    print(f"  W events: {sum(1 for e in video_events if e.ability == 'W')}")
    print(f"  R events: {sum(1 for e in video_events if e.ability == 'R')}")
    
    # Extract clips
    output_dir.mkdir(parents=True, exist_ok=True)
    
    use_ffmpeg = not args.no_ffmpeg
    
    extracted = []
    for i, event in enumerate(video_events):
        start_sec, end_sec = extractor.get_clip_times(event)
        
        output_name = f"{video_id}_{event.ability}_{i:03d}.{args.format}"
        output_path = output_dir / output_name
        
        print(f"\n[{i+1}/{len(video_events)}] Extracting: {output_name}")
        print(f"  Time: {start_sec:.1f}s - {end_sec:.1f}s (duration: {end_sec-start_sec:.1f}s)")
        print(f"  Notes: {event.notes or 'none'}")
        
        if use_ffmpeg:
            success = extractor.extract_clip_ffmpeg(
                video_path, output_path, start_sec, end_sec
            )
        else:
            success = extractor.extract_clip_opencv(
                video_path, output_path, start_sec, end_sec
            )
            
        if success:
            extracted.append(output_path)
            print(f"  ✓ Saved: {output_path}")
        else:
            print(f"  ✗ Failed to extract clip")
            
    print(f"\n{'='*50}")
    print(f"Extracted {len(extracted)}/{len(video_events)} clips to {output_dir}")


if __name__ == "__main__":
    main()





