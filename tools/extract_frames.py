import cv2
import os
import argparse


def extract_frames(video_path: str, out_dir: str, every: int = 5) -> None:
    """
    Extract frames from a video every N frames and save as .jpg images.

    Args:
        video_path: Path to the input video file.
        out_dir: Directory where extracted frames will be saved.
        every: Save every N-th frame (e.g., every=3 saves 1 out of 3 frames).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"[INFO] Video: {video_path}")
    print(f"[INFO] Total frames: {total_frames}")
    print(f"[INFO] Saving every {every} frame(s) to: {out_dir}")

    saved = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every == 0:
            filename = os.path.join(out_dir, f"frame_{saved:05d}.jpg")
            cv2.imwrite(filename, frame)
            saved += 1

        frame_idx += 1

    cap.release()
    print(f"[DONE] Extracted {saved} frames to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract frames from a video.")
    parser.add_argument(
        "--video",
        required=True,
        help="Path to input video file (e.g., videoplayback.mp4)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory to save extracted frames (e.g., data/images/train)",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=5,
        help="Save every N-th frame (default: 5)",
    )

    args = parser.parse_args()
    extract_frames(args.video, args.out, args.every)


if __name__ == "__main__":
    main()

