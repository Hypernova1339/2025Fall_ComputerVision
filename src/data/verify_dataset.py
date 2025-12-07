"""
Lightweight dataset sanity checks for YOLO-format labels.
Checks:
- Every label file has a matching image.
- Labels only use allowed class IDs.
- Report unlabeled images (negative frames) separately.
"""
import argparse
from pathlib import Path
from typing import Iterable


def load_allowed_classes(names_file: Path | None, default_classes: Iterable[int]) -> set[int]:
    if names_file is None:
        return set(default_classes)
    ids = set()
    with names_file.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, _ = line.split(":", 1)
                try:
                    ids.add(int(key.strip()))
                except ValueError:
                    continue
    return ids if ids else set(default_classes)


def read_label_ids(label_path: Path) -> set[int]:
    ids: set[int] = set()
    with label_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                ids.add(int(parts[0]))
            except (ValueError, IndexError):
                continue
    return ids


def verify_split(images_dir: Path, labels_dir: Path, allowed_ids: set[int]) -> dict:
    report: dict[str, list[str]] = {
        "missing_labels": [],
        "label_without_image": [],
        "invalid_class_ids": [],
    }
    image_stems = {p.stem for p in images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}

    for label_path in labels_dir.glob("*.txt"):
        if label_path.stem not in image_stems:
            report["label_without_image"].append(label_path.name)
            continue
        ids = read_label_ids(label_path)
        bad_ids = [str(i) for i in ids if i not in allowed_ids]
        if bad_ids:
            report["invalid_class_ids"].append(f"{label_path.name}: {','.join(bad_ids)}")

    for stem in image_stems:
        if not (labels_dir / f"{stem}.txt").exists():
            report["missing_labels"].append(f"{stem} (treated as negative)")

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify YOLO-format dataset splits.")
    parser.add_argument("--images", type=Path, required=True, help="Path to images split (e.g., data/images/train).")
    parser.add_argument("--labels", type=Path, required=True, help="Path to labels split (e.g., data/labels/train).")
    parser.add_argument(
        "--names-file",
        type=Path,
        default=Path("configs/jinx_w.yaml"),
        help="YAML-like file containing class IDs (first token on each 'id: name' line).",
    )
    parser.add_argument(
        "--default-classes",
        type=int,
        nargs="+",
        default=[0, 1],
        help="Fallback class IDs if names-file cannot be read.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    allowed = load_allowed_classes(args.names_file, args.default_classes)
    report = verify_split(args.images, args.labels, allowed)

    print(f"Allowed class IDs: {sorted(allowed)}")
    for key, items in report.items():
        print(f"{key}: {len(items)}")
        for item in items:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
