"""
Frame extraction tool for creating YOLO-ready image folders.
Extracts frames from gameplay videos at a specified interval.

Example:
    python -m src.extract_frames --video data/raw_videos/jinx_clip.mp4 --out data/images/train --every 3
    
    python -m src.extract_frames \
  --video data/raw_videos/jinx_clip.mp4 \
  --out data/images/train \
  --every 3
"""
import argparse
from pathlib import Path
from typing import Optional

import cv2


def extract_frames(
    video_path: Path,
    out_dir: Path,
    every: int = 1,
    max_frames: Optional[int] = None,
) -> int:
    """
    Extract frames from a video at a fixed stride.

    Args:
        video_path: Path to source video file
        out_dir: Output directory for extracted frames
        every: Keep every Nth frame (e.g., every=3 keeps 1 out of 3 frames)
        max_frames: Optional maximum number of frames to extract

    Returns:
        Number of frames written

    Raises:
        RuntimeError: If video cannot be opened
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"[INFO] Video: {video_path}")
    print(f"[INFO] FPS: {fps:.2f}, Total frames: {total_frames}")
    print(f"[INFO] Extracting every {every} frame(s) to: {out_dir}")

    written = 0
    frame_index = 0
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % every == 0:
            frame_name = out_dir / f"frame_{frame_id:06d}.jpg"
            cv2.imwrite(str(frame_name), frame)
            written += 1
            frame_id += 1

            if max_frames is not None and written >= max_frames:
                break

        frame_index += 1

    cap.release()
    print(f"[DONE] Extracted {written} frames to {out_dir}")
    return written


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract frames from video at a fixed stride.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to source video file",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for frames",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=1,
        help="Keep every Nth frame (e.g., 3 = keep 1 out of 3 frames)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on number of frames to save",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    saved = extract_frames(args.video, args.out, args.every, args.max_frames)
    print(f"Successfully saved {saved} frames to {args.out}")


if __name__ == "__main__":
    main()

