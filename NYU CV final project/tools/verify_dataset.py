#!/usr/bin/env python3
"""
Verify dataset integrity before training.

Usage:
    python tools/verify_dataset.py --config configs/jinx_abilities.yaml

Checks:
    - All images have matching labels (or are intentional negatives)
    - All label files have valid class IDs (0-3)
    - Bounding box coordinates are normalized (0-1)
    - Paths in config file exist
    - No corrupted image files
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_config(config_path: Path) -> dict:
    """Load YOLO dataset config."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def check_image_readable(image_path: Path) -> bool:
    """Check if image can be read by OpenCV."""
    try:
        import cv2
        img = cv2.imread(str(image_path))
        return img is not None
    except Exception:
        return False


def parse_yolo_label(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    Parse YOLO format label file.
    
    Returns:
        List of (class_id, x_center, y_center, width, height) tuples
    """
    labels = []
    
    with open(label_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Line {line_num}: Expected 5 values, got {len(parts)}")
                
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            labels.append((class_id, x_center, y_center, width, height))
            
    return labels


def verify_dataset(
    config_path: Path,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Verify dataset integrity.
    
    Returns:
        Dict with verification results
    """
    results = {
        "passed": True,
        "errors": [],
        "warnings": [],
        "stats": {}
    }
    
    # Load config
    print(f"Loading config: {config_path}")
    config = load_config(config_path)
    
    # Resolve paths relative to config file
    config_dir = config_path.parent
    data_path = (config_dir / config["path"]).resolve()
    train_images = data_path / config["train"]
    val_images = data_path / config["val"]
    train_labels = data_path / "labels" / "train"
    val_labels = data_path / "labels" / "val"
    
    # Check paths exist
    print("\n[1/5] Checking directory paths...")
    
    for name, path in [
        ("Data root", data_path),
        ("Train images", train_images),
        ("Val images", val_images),
        ("Train labels", train_labels),
        ("Val labels", val_labels),
    ]:
        if not path.exists():
            results["errors"].append(f"Directory not found: {name} ({path})")
            results["passed"] = False
        else:
            if verbose:
                print(f"  ✓ {name}: {path}")
                
    if not results["passed"]:
        print("\n✗ Path verification failed. Cannot continue.")
        return results
        
    # Get class info
    num_classes = len(config["names"])
    valid_classes = set(config["names"].keys())
    
    print(f"\n[2/5] Checking class definitions...")
    print(f"  Classes defined: {num_classes}")
    for class_id, class_name in config["names"].items():
        print(f"    {class_id}: {class_name}")
        
    # Verify train and val sets
    for split_name, images_dir, labels_dir in [
        ("train", train_images, train_labels),
        ("val", val_images, val_labels),
    ]:
        print(f"\n[3/5] Verifying {split_name} set...")
        
        # Find all images
        images = []
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            images.extend(images_dir.glob(f"*{ext}"))
            
        if not images:
            results["warnings"].append(f"No images found in {split_name} set")
            continue
            
        print(f"  Found {len(images)} images")
        
        # Check each image
        labeled_count = 0
        unlabeled_count = 0
        class_counts = {c: 0 for c in valid_classes}
        corrupted_images = []
        invalid_labels = []
        
        for image_path in images:
            # Check image is readable
            if not check_image_readable(image_path):
                corrupted_images.append(image_path)
                continue
                
            # Check for label file
            label_path = labels_dir / f"{image_path.stem}.txt"
            
            if not label_path.exists():
                unlabeled_count += 1
                continue
                
            # Parse and validate label
            try:
                labels = parse_yolo_label(label_path)
                
                if not labels:
                    unlabeled_count += 1
                    continue
                    
                labeled_count += 1
                
                for class_id, x, y, w, h in labels:
                    # Check class ID
                    if class_id not in valid_classes:
                        invalid_labels.append(
                            (label_path, f"Invalid class ID: {class_id}")
                        )
                        
                    else:
                        class_counts[class_id] += 1
                        
                    # Check coordinates
                    for val, name in [(x, "x"), (y, "y"), (w, "w"), (h, "h")]:
                        if not 0 <= val <= 1:
                            invalid_labels.append(
                                (label_path, f"Coordinate {name}={val} not in [0,1]")
                            )
                            
            except Exception as e:
                invalid_labels.append((label_path, str(e)))
                
        # Report results for this split
        print(f"  Labeled: {labeled_count}")
        print(f"  Unlabeled (negatives): {unlabeled_count}")
        print(f"  Class distribution:")
        for class_id in sorted(class_counts.keys()):
            class_name = config["names"][class_id]
            count = class_counts[class_id]
            print(f"    {class_id} ({class_name}): {count}")
            
        if corrupted_images:
            results["errors"].extend([
                f"Corrupted image: {p}" for p in corrupted_images
            ])
            results["passed"] = False
            print(f"  ✗ Corrupted images: {len(corrupted_images)}")
            
        if invalid_labels:
            for path, error in invalid_labels[:5]:  # Show first 5
                results["errors"].append(f"Invalid label {path}: {error}")
            results["passed"] = False
            print(f"  ✗ Invalid labels: {len(invalid_labels)}")
            
        # Store stats
        results["stats"][split_name] = {
            "total_images": len(images),
            "labeled": labeled_count,
            "unlabeled": unlabeled_count,
            "class_counts": class_counts,
            "corrupted": len(corrupted_images),
            "invalid_labels": len(invalid_labels),
        }
        
    # Check for class imbalance
    print(f"\n[4/5] Checking class balance...")
    
    if "train" in results["stats"]:
        counts = results["stats"]["train"]["class_counts"]
        if counts:
            min_count = min(counts.values())
            max_count = max(counts.values())
            
            if min_count == 0:
                results["warnings"].append("Some classes have zero samples")
                print(f"  ⚠ Warning: Some classes have zero samples!")
            elif max_count > 10 * min_count:
                results["warnings"].append(
                    f"Severe class imbalance: {max_count}/{min_count} = {max_count/min_count:.1f}x"
                )
                print(f"  ⚠ Warning: Severe class imbalance detected")
            else:
                print(f"  ✓ Class balance OK (ratio: {max_count/max(min_count,1):.1f}x)")
                
    # Check minimum dataset size
    print(f"\n[5/5] Checking dataset size...")
    
    if "train" in results["stats"]:
        train_labeled = results["stats"]["train"]["labeled"]
        if train_labeled < 50:
            results["warnings"].append(
                f"Very small training set ({train_labeled} labeled). Consider adding more data."
            )
            print(f"  ⚠ Warning: Small training set ({train_labeled} samples)")
        else:
            print(f"  ✓ Training set size OK ({train_labeled} labeled samples)")
            
    # Final summary
    print(f"\n{'='*50}")
    
    if results["passed"] and not results["errors"]:
        print("✓ Dataset verification PASSED")
    else:
        print("✗ Dataset verification FAILED")
        print("\nErrors:")
        for error in results["errors"][:10]:
            print(f"  - {error}")
        if len(results["errors"]) > 10:
            print(f"  ... and {len(results['errors']) - 10} more")
            
    if results["warnings"]:
        print("\nWarnings:")
        for warning in results["warnings"]:
            print(f"  - {warning}")
            
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Verify dataset integrity before training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic verification
    python tools/verify_dataset.py --config configs/jinx_abilities.yaml

    # Quiet mode (errors only)
    python tools/verify_dataset.py --config configs/jinx_abilities.yaml --quiet
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YOLO dataset config (e.g., configs/jinx_abilities.yaml)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only show errors and warnings"
    )
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
        
    results = verify_dataset(config_path, verbose=not args.quiet)
    
    if not results["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()





