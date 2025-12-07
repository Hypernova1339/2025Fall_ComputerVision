"""
Simple frame extractor for creating YOLO-ready image folders.
Example:
python scripts/extract_frames.py --video data/raw_videos/jinx_clip.mp4 --out data/images/train --every 3
"""
import argparse
from pathlib import Path

import cv2


def extract_frames(video_path: Path, out_dir: Path, every: int, max_frames: int | None = None) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    written = 0
    index = 0
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % every == 0:
            frame_name = out_dir / f"frame_{frame_id:06d}.jpg"
            cv2.imwrite(str(frame_name), frame)
            written += 1
            frame_id += 1
            if max_frames is not None and written >= max_frames:
                break
        index += 1

    cap.release()
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames from video at a fixed stride.")
    parser.add_argument("--video", type=Path, required=True, help="Path to source video file.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for frames.")
    parser.add_argument("--every", type=int, default=3, help="Keep every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on number of frames to save.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved = extract_frames(args.video, args.out, args.every, args.max_frames)
    print(f"Saved {saved} frames to {args.out}")


if __name__ == "__main__":
    main()
