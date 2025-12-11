#!/usr/bin/env python3
"""
Create detection overlay clips from specific video segments.

Extracts clips, runs inference, and applies speed adjustments.
"""

import argparse
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Class configuration
CLASS_NAMES = {
    0: "IDLE",
    1: "W_PROJ",
    2: "W_IMPACT",
    3: "R_ROCKET",
    4: "R_IMPACT",
}

CLASS_COLORS = {
    0: (200, 200, 200),
    1: (255, 150, 50),
    2: (50, 255, 50),
    3: (50, 50, 255),
    4: (0, 165, 255),
}


def time_to_seconds(time_str: str) -> float:
    """Convert MM:SS or HH:MM:SS to seconds."""
    parts = time_str.split(':')
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    else:
        return float(time_str)


def draw_detections(frame, detections, show_confidence=True):
    """Draw detection boxes on frame."""
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cls_id = det['class_id']
        cls_name = det['class_name']
        conf = det['confidence']
        
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Label
        label = f"{cls_name} {conf:.2f}" if show_confidence else cls_name
        font = cv2.FONT_HERSHEY_SIMPLEX
        (lw, lh), baseline = cv2.getTextSize(label, font, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), font, 0.6, (0, 0, 0), 2)
    
    return frame


def create_clip_with_detections(
    video_path: str,
    weights_path: str,
    start_time: str,
    end_time: str,
    output_path: str,
    speed: float = 1.0,
    conf_threshold: float = 0.25
):
    """
    Create a clip with detection overlays.
    
    Args:
        video_path: Source video
        weights_path: YOLO weights
        start_time: Start time (MM:SS)
        end_time: End time (MM:SS)
        output_path: Output file path
        speed: Playback speed (0.5 = half speed / slow motion)
        conf_threshold: Detection confidence threshold
    """
    from ultralytics import YOLO
    import torch
    
    # Load model
    print(f"\nCreating clip: {start_time} - {end_time} at {speed}x speed")
    model = YOLO(weights_path)
    
    # Auto-detect device
    if torch.cuda.is_available():
        device = 0
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    # Convert times to seconds
    start_sec = time_to_seconds(start_time)
    end_sec = time_to_seconds(end_time)
    duration = end_sec - start_sec
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate output FPS for slow motion effect
    output_fps = fps * speed  # e.g., 60 * 0.5 = 30 fps output for slow motion
    
    # Seek to start position
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    total_frames = end_frame - start_frame
    
    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
    
    print(f"  Source: {width}x{height} @ {fps}fps")
    print(f"  Output: {output_fps}fps (speed={speed}x)")
    print(f"  Frames: {total_frames}")
    
    # Process frames
    frame_count = 0
    pbar = tqdm(total=total_frames, desc="  Processing")
    
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        results = model.predict(frame, conf=conf_threshold, device=device, verbose=False)
        
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
        
        # Draw detections
        annotated = draw_detections(frame.copy(), detections)
        
        # Add timestamp overlay
        current_time = start_sec + (frame_count / fps)
        time_text = f"{int(current_time // 60):02d}:{current_time % 60:05.2f}"
        cv2.putText(annotated, time_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (255, 255, 255), 2)
        cv2.putText(annotated, f"Speed: {speed}x", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 1)
        
        # Write frame
        out.write(annotated)
        
        frame_count += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    print(f"  Saved: {output_path}")
    
    # Convert to compatible format with ffmpeg
    temp_path = output_path.replace('.mp4', '_temp.mp4')
    subprocess.run([
        'mv', output_path, temp_path
    ], check=True)
    
    subprocess.run([
        'ffmpeg', '-y', '-i', temp_path,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        output_path
    ], capture_output=True, check=True)
    
    subprocess.run(['rm', temp_path], check=True)
    print(f"  Converted to H.264: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create detection overlay clips")
    parser.add_argument("--video", type=str, required=True, help="Source video")
    parser.add_argument("--weights", type=str, required=True, help="YOLO weights")
    parser.add_argument("--output-dir", type=str, default="outputs/clips", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    
    args = parser.parse_args()
    
    # Define clips to create
    clips = [
        ("5:24", "5:26", 0.8, "clip_5m24s_0.8x"),
        ("3:01", "3:19", 0.5, "clip_3m01s_0.5x"),
        ("8:53", "9:03", 0.5, "clip_8m53s_0.5x"),
        ("9:23", "9:37", 0.5, "clip_9m23s_0.5x"),
        ("10:04", "10:17", 0.5, "clip_10m04s_0.5x"),
        ("11:46", "11:53", 0.5, "clip_11m46s_0.5x"),
    ]
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("CREATING DETECTION CLIPS")
    print("=" * 60)
    print(f"Video: {args.video}")
    print(f"Weights: {args.weights}")
    print(f"Output: {output_dir}")
    print(f"Clips to create: {len(clips)}")
    
    for start, end, speed, name in clips:
        output_path = str(output_dir / f"{name}.mp4")
        create_clip_with_detections(
            args.video,
            args.weights,
            start,
            end,
            output_path,
            speed=speed,
            conf_threshold=args.conf
        )
    
    print("\n" + "=" * 60)
    print("ALL CLIPS CREATED!")
    print("=" * 60)
    print(f"\nClips saved to: {output_dir}")


if __name__ == "__main__":
    main()




