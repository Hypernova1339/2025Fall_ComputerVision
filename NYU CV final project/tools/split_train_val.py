#!/usr/bin/env python3
"""
Split labeled dataset into train and validation sets.

Usage:
    python tools/split_train_val.py \
        --images data/images/unlabeled/ \
        --labels data/labels/unlabeled/ \
        --output data/ \
        --val-ratio 0.2
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset_splitter import DatasetSplitter


def main():
    parser = argparse.ArgumentParser(
        description="Split dataset into train and validation sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic split with default 80/20 ratio
    python tools/split_train_val.py \\
        --images data/images/unlabeled/ \\
        --labels data/labels/unlabeled/ \\
        --output data/

    # Custom validation ratio
    python tools/split_train_val.py \\
        --images data/images/unlabeled/ \\
        --labels data/labels/unlabeled/ \\
        --output data/ \\
        --val-ratio 0.15

    # Reproducible split with specific seed
    python tools/split_train_val.py \\
        --images data/images/unlabeled/ \\
        --labels data/labels/unlabeled/ \\
        --output data/ \\
        --seed 123

    # Split by individual frames (not by clip)
    python tools/split_train_val.py \\
        --images data/images/unlabeled/ \\
        --labels data/labels/unlabeled/ \\
        --output data/ \\
        --no-group-by-clip
        """
    )
    
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Directory containing images to split"
    )
    
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Directory containing YOLO label files"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output base directory (will create images/train, images/val, labels/train, labels/val)"
    )
    
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of data for validation (default: 0.2)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--no-group-by-clip",
        action="store_true",
        help="Split by individual frames instead of keeping clip frames together"
    )
    
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying (saves disk space)"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    output_dir = Path(args.output)
    
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        sys.exit(1)
        
    if not labels_dir.exists():
        print(f"Warning: Labels directory not found: {labels_dir}")
        print("Creating empty labels directory...")
        labels_dir.mkdir(parents=True, exist_ok=True)
        
    # Define output directories
    train_images_dir = output_dir / "images" / "train"
    train_labels_dir = output_dir / "labels" / "train"
    val_images_dir = output_dir / "images" / "val"
    val_labels_dir = output_dir / "labels" / "val"
    
    print(f"Dataset Split Configuration:")
    print(f"  Source images: {images_dir}")
    print(f"  Source labels: {labels_dir}")
    print(f"  Output base: {output_dir}")
    print(f"  Validation ratio: {args.val_ratio}")
    print(f"  Random seed: {args.seed}")
    print(f"  Group by clip: {not args.no_group_by_clip}")
    print(f"  Mode: {'move' if args.move else 'copy'}")
    print()
    
    # Create splitter
    splitter = DatasetSplitter(
        val_ratio=args.val_ratio,
        seed=args.seed,
        group_by_clip=not args.no_group_by_clip
    )
    
    # Find pairs
    pairs = splitter.find_image_label_pairs(images_dir, labels_dir)
    
    if not pairs:
        print("Error: No images found in source directory")
        sys.exit(1)
        
    print(f"Found {len(pairs)} images")
    labeled = sum(1 for _, lbl in pairs if lbl is not None)
    print(f"  Labeled: {labeled}")
    print(f"  Unlabeled (negatives): {len(pairs) - labeled}")
    print()
    
    # Split pairs
    train_pairs, val_pairs = splitter.split_pairs(pairs)
    
    # Create output directories
    for d in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    # Copy or move files
    if args.move:
        print("Moving files...")
        splitter.move_pairs_to_split(train_pairs, train_images_dir, train_labels_dir)
        splitter.move_pairs_to_split(val_pairs, val_images_dir, val_labels_dir)
    else:
        print("Copying files...")
        splitter.copy_pairs_to_split(train_pairs, train_images_dir, train_labels_dir)
        splitter.copy_pairs_to_split(val_pairs, val_images_dir, val_labels_dir)
        
    # Summary
    train_labeled = sum(1 for _, lbl in train_pairs if lbl is not None)
    val_labeled = sum(1 for _, lbl in val_pairs if lbl is not None)
    
    print()
    print(f"{'='*50}")
    print(f"Split Complete!")
    print()
    print(f"Training set:")
    print(f"  Images: {train_images_dir}")
    print(f"  Labels: {train_labels_dir}")
    print(f"  Total: {len(train_pairs)} ({train_labeled} labeled)")
    print()
    print(f"Validation set:")
    print(f"  Images: {val_images_dir}")
    print(f"  Labels: {val_labels_dir}")
    print(f"  Total: {len(val_pairs)} ({val_labeled} labeled)")


if __name__ == "__main__":
    main()





