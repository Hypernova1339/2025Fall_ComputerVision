"""
Inference helper for running YOLO VFX detector on images or video.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VFX detector inference.")
    parser.add_argument("--weights", type=Path, required=True, help="Trained YOLO weights (.pt).")
    parser.add_argument("--source", type=str, required=True, help="Image/video path or directory.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--project", type=Path, default=Path("outputs"), help="Root folder for predictions.")
    parser.add_argument("--run-name", type=str, default="preds", help="Subfolder for this run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.project.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        project=str(args.project),
        name=args.run_name,
        save=True,
        exist_ok=True,
    )
    print(f"Predictions saved to: {results[0].save_dir if results else 'n/a'}")


if __name__ == "__main__":
    main()
