"""
Training entrypoint for Jinx VFX detection using Ultralytics YOLO.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO model for LoL VFX detection.")
    parser.add_argument("--config", type=str, default="configs/jinx_w.yaml", help="YOLO dataset config path.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model weights or YAML.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto", help="CUDA device id or 'cpu'.")
    parser.add_argument("--project", type=Path, default=Path("outputs"), help="Training output root.")
    parser.add_argument("--run-name", type=str, default="jinx_w", help="Run subfolder inside project.")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.project.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    results = model.train(
        data=args.config,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=args.run_name,
        exist_ok=True,
    )
    final_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"Training complete. Best weights: {final_weights}")


if __name__ == "__main__":
    main()
