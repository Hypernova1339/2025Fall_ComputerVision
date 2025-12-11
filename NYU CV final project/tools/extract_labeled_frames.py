#!/usr/bin/env python3
"""
Extract specific frames from video based on events.csv annotations.

This script reads the labeled events and extracts:
- All cast frames (for W/R projectile detection)
- All impact frames (for W/R impact detection)
- All IDLE frames (negative samples)

Usage:
    python tools/extract_labeled_frames.py \
        --video "data/raw_videos/video 1 (long).mp4" \
        --events data/events/events.csv \
        --output data/images/unlabeled/
"""

import argparse
import csv
import sys
from pathlib import Path
from collections import defaultdict
import cv2
from tqdm import tqdm


def load_events(events_csv: Path, video_id: str) -> dict:
    """
    Load events from CSV and organize by frame type.
    
    Returns:
        Dict with keys: 'w_cast', 'w_impact', 'r_cast', 'r_impact', 'idle'
        Each value is a list of frame numbers.
    """
    frames = {
        'w_cast': [],
        'w_impact': [],
        'r_cast': [],
        'r_impact': [],
        'idle': [],
    }
    
    with open(events_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('video_id') != video_id:
                continue
                
            ability = row.get('ability', '').upper()
            cast_frames_str = row.get('cast_frames', '')
            impact_frame_str = row.get('impact_frame', '')
            
            # Parse cast frames
            if cast_frames_str:
                cast_frames = [int(f) for f in cast_frames_str.split(';') if f]
                
                if ability == 'W':
                    frames['w_cast'].extend(cast_frames)
                elif ability == 'R':
                    frames['r_cast'].extend(cast_frames)
                elif ability == 'IDLE':
                    frames['idle'].extend(cast_frames)
                    
            # Parse impact frame
            if impact_frame_str:
                impact_frame = int(impact_frame_str)
                if ability == 'W':
                    frames['w_impact'].append(impact_frame)
                elif ability == 'R':
                    frames['r_impact'].append(impact_frame)
                    
    return frames


def extract_frames(
    video_path: Path,
    frame_dict: dict,
    output_dir: Path,
    quality: int = 95
) -> dict:
    """
    Extract frames from video and save as images.
    
    Args:
        video_path: Path to video file
        frame_dict: Dict mapping frame types to frame numbers
        output_dir: Output directory for images
        quality: JPEG quality (1-100)
        
    Returns:
        Dict with counts of extracted frames per type
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
        
    # Get all unique frames with their types
    frame_to_types = defaultdict(list)
    for frame_type, frame_list in frame_dict.items():
        for frame_num in frame_list:
            frame_to_types[frame_num].append(frame_type)
            
    # Sort frames for efficient sequential reading
    all_frames = sorted(frame_to_types.keys())
    
    if not all_frames:
        print("No frames to extract!")
        return {}
        
    print(f"Extracting {len(all_frames)} unique frames...")
    
    # Create output subdirectories
    for frame_type in frame_dict.keys():
        (output_dir / frame_type).mkdir(parents=True, exist_ok=True)
        
    counts = defaultdict(int)
    
    # Extract frames
    for frame_num in tqdm(all_frames, desc="Extracting"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Warning: Could not read frame {frame_num}")
            continue
            
        # Save to each relevant type folder
        for frame_type in frame_to_types[frame_num]:
            filename = f"{frame_type}_{frame_num:06d}.jpg"
            output_path = output_dir / frame_type / filename
            
            cv2.imwrite(
                str(output_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, quality]
            )
            counts[frame_type] += 1
            
    cap.release()
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(
        description="Extract labeled frames from video based on events.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output structure:
    output_dir/
        w_cast/       - W projectile frames (draw box around beam)
        w_impact/     - W impact frames (draw box around hit effect)
        r_cast/       - R rocket frames (draw box around missile)
        r_impact/     - R impact frames (draw box around explosion)
        idle/         - Negative samples (no boxes needed)

Examples:
    python tools/extract_labeled_frames.py \\
        --video "data/raw_videos/video 1 (long).mp4" \\
        --events data/events/events.csv \\
        --output data/images/unlabeled/
        """
    )
    
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to video file"
    )
    
    parser.add_argument(
        "--events",
        type=str,
        default="data/events/events.csv",
        help="Path to events CSV file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/images/unlabeled/",
        help="Output directory for extracted frames"
    )
    
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality 1-100 (default: 95)"
    )
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    events_path = Path(args.events)
    output_dir = Path(args.output)
    
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)
        
    if not events_path.exists():
        print(f"Error: Events file not found: {events_path}")
        sys.exit(1)
        
    # Get video ID from filename
    video_id = video_path.stem
    
    print(f"Video: {video_path}")
    print(f"Events: {events_path}")
    print(f"Output: {output_dir}")
    print(f"Video ID: {video_id}")
    print()
    
    # Load events
    print("Loading events...")
    frames = load_events(events_path, video_id)
    
    print(f"\nFrames to extract:")
    print(f"  W cast frames:   {len(frames['w_cast'])}")
    print(f"  W impact frames: {len(frames['w_impact'])}")
    print(f"  R cast frames:   {len(frames['r_cast'])}")
    print(f"  R impact frames: {len(frames['r_impact'])}")
    print(f"  IDLE frames:     {len(frames['idle'])}")
    total = sum(len(v) for v in frames.values())
    print(f"  TOTAL:           {total}")
    print()
    
    # Extract frames
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = extract_frames(video_path, frames, output_dir, args.quality)
    
    print(f"\n{'='*50}")
    print("Extraction complete!")
    print(f"{'='*50}")
    print(f"\nExtracted frames:")
    for frame_type, count in sorted(counts.items()):
        print(f"  {frame_type}: {count} frames -> {output_dir / frame_type}/")
    print(f"\nTotal: {sum(counts.values())} frames")
    print(f"\nNext step: Use LabelImg to draw bounding boxes")
    print(f"  labelImg {output_dir}/w_cast configs/classes.txt {output_dir}/../labels/")


if __name__ == "__main__":
    main()





