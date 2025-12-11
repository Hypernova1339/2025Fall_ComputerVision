#!/usr/bin/env python3
"""
Prepare dataset for YOLO training.

Flattens images and labels from class subfolders into flat train/val directories.
Supports oversampling for class balancing.

Usage:
    python tools/prepare_dataset.py --val-ratio 0.2
    
    # With oversampling for rare classes (recommended)
    python tools/prepare_dataset.py --val-ratio 0.2 --oversample
    
    # Custom oversample target
    python tools/prepare_dataset.py --val-ratio 0.2 --oversample --oversample-target 100
"""

import argparse
import random
import shutil
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Optional, Dict


# Folder to class mapping (must match bbox_labeler.py)
FOLDER_TO_CLASS = {
    'idle': (0, 'JINX_IDLE'),
    'w_cast': (1, 'VFX_W_PROJECTILE'),
    'w_impact': (2, 'VFX_W_IMPACT'),
    'r_cast': (3, 'VFX_R_ROCKET'),
    'r_impact': (4, 'VFX_R_IMPACT'),
}


def find_image_label_pairs(
    images_base: Path,
    labels_base: Path
) -> List[Tuple[Path, Optional[Path], str]]:
    """
    Find all image/label pairs from class subfolders.
    
    Returns:
        List of (image_path, label_path, folder_name) tuples
    """
    pairs = []
    
    for folder_name in FOLDER_TO_CLASS.keys():
        images_folder = images_base / folder_name
        labels_folder = labels_base / folder_name
        
        if not images_folder.exists():
            print(f"  Warning: {images_folder} not found, skipping")
            continue
            
        # Find all images in this folder
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            for img_path in sorted(images_folder.glob(ext)):
                label_path = labels_folder / (img_path.stem + '.txt')
                
                if label_path.exists():
                    # Check if label file has content
                    if label_path.stat().st_size > 0:
                        pairs.append((img_path, label_path, folder_name))
                    else:
                        # Empty label = no detections, include as negative
                        pairs.append((img_path, None, folder_name))
                else:
                    # No label file = negative sample
                    pairs.append((img_path, None, folder_name))
                    
    return pairs


def split_by_event(
    pairs: List[Tuple[Path, Optional[Path], str]],
    val_ratio: float,
    seed: int
) -> Tuple[List, List]:
    """
    Split pairs into train/val, grouping by event (frame number prefix).
    
    This prevents data leakage by keeping frames from the same event together.
    """
    random.seed(seed)
    
    # Group by event ID (everything before the last underscore + frame number)
    events = defaultdict(list)
    for pair in pairs:
        img_path = pair[0]
        # Extract event ID: folder_frameNumber -> group by folder + rough frame range
        stem = img_path.stem
        folder = pair[2]
        
        # Group by folder and every 100 frames (to keep related frames together)
        try:
            frame_num = int(stem.split('_')[-1])
            event_id = f"{folder}_{frame_num // 100}"
        except ValueError:
            event_id = f"{folder}_{stem}"
            
        events[event_id].append(pair)
    
    # Shuffle event IDs and split
    event_ids = list(events.keys())
    random.shuffle(event_ids)
    
    val_count = int(len(event_ids) * val_ratio)
    val_event_ids = set(event_ids[:val_count])
    
    train_pairs = []
    val_pairs = []
    
    for event_id, event_pairs in events.items():
        if event_id in val_event_ids:
            val_pairs.extend(event_pairs)
        else:
            train_pairs.extend(event_pairs)
            
    return train_pairs, val_pairs


def calculate_class_distribution(
    pairs: List[Tuple[Path, Optional[Path], str]]
) -> Dict[str, int]:
    """Calculate the distribution of labeled samples per class folder."""
    counts = defaultdict(int)
    for img_path, label_path, folder_name in pairs:
        if label_path is not None:
            counts[folder_name] += 1
    return dict(counts)


def oversample_rare_classes(
    pairs: List[Tuple[Path, Optional[Path], str]],
    target_count: Optional[int] = None,
    min_ratio: float = 0.5
) -> List[Tuple[Path, Optional[Path], str]]:
    """
    Oversample underrepresented classes by duplicating samples.
    
    This helps address class imbalance which severely hurts detection of rare
    classes like VFX_R_IMPACT.
    
    Args:
        pairs: List of (image_path, label_path, folder_name) tuples
        target_count: Target sample count per class. If None, uses max class count * min_ratio.
        min_ratio: Minimum ratio relative to most common class (default 0.5 = 50%)
        
    Returns:
        Augmented list with oversampled rare class examples
    """
    # Separate labeled pairs by class
    class_pairs = defaultdict(list)
    unlabeled_pairs = []
    
    for pair in pairs:
        img_path, label_path, folder_name = pair
        if label_path is not None:
            class_pairs[folder_name].append(pair)
        else:
            unlabeled_pairs.append(pair)
    
    # Calculate target count
    if class_pairs:
        max_count = max(len(p) for p in class_pairs.values())
        if target_count is None:
            target_count = int(max_count * min_ratio)
    else:
        return pairs  # No labeled pairs, return as-is
    
    print(f"\n  Oversampling Configuration:")
    print(f"    Target samples per class: {target_count}")
    print(f"    Min ratio: {min_ratio}")
    
    # Oversample each class
    oversampled_pairs = []
    for folder_name, folder_pairs in class_pairs.items():
        current_count = len(folder_pairs)
        
        if current_count >= target_count:
            # No oversampling needed
            oversampled_pairs.extend(folder_pairs)
            print(f"    {folder_name}: {current_count} samples (no change)")
        else:
            # Need to oversample
            oversampled_pairs.extend(folder_pairs)  # Add originals
            
            # Calculate how many duplicates needed
            needed = target_count - current_count
            
            # Randomly sample with replacement from existing pairs
            random.seed(42)  # Reproducible
            duplicates = random.choices(folder_pairs, k=needed)
            oversampled_pairs.extend(duplicates)
            
            print(f"    {folder_name}: {current_count} -> {target_count} samples (+{needed} duplicates)")
    
    # Add back unlabeled pairs
    oversampled_pairs.extend(unlabeled_pairs)
    
    return oversampled_pairs


def copy_pairs(
    pairs: List[Tuple[Path, Optional[Path], str]],
    images_dest: Path,
    labels_dest: Path,
    prefix: str = ""
) -> dict:
    """
    Copy image/label pairs to destination directories.
    
    Handles duplicate filenames (from oversampling) by adding numeric suffixes.
    
    Returns stats dict.
    """
    images_dest.mkdir(parents=True, exist_ok=True)
    labels_dest.mkdir(parents=True, exist_ok=True)
    
    stats = {
        'total': 0,
        'labeled': 0,
        'by_class': defaultdict(int),
        'duplicates': 0
    }
    
    # Track used filenames to handle duplicates from oversampling
    used_names = set()
    
    for img_path, label_path, folder_name in pairs:
        # Generate unique filename (handle oversampling duplicates)
        base_name = img_path.stem
        ext = img_path.suffix
        
        if base_name in used_names:
            # Find a unique name by adding suffix
            counter = 1
            while f"{base_name}_dup{counter}" in used_names:
                counter += 1
            unique_name = f"{base_name}_dup{counter}"
            stats['duplicates'] += 1
        else:
            unique_name = base_name
            
        used_names.add(unique_name)
        
        # Copy image with unique name
        dest_img = images_dest / f"{unique_name}{ext}"
        shutil.copy2(img_path, dest_img)
        
        # Copy label if exists (with matching unique name)
        if label_path is not None:
            dest_label = labels_dest / f"{unique_name}.txt"
            shutil.copy2(label_path, dest_label)
            stats['labeled'] += 1
            stats['by_class'][folder_name] += 1
            
        stats['total'] += 1
        
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset for YOLO training"
    )
    parser.add_argument(
        "--images-src",
        type=str,
        default="data/images/unlabeled",
        help="Source directory with class subfolders containing images"
    )
    parser.add_argument(
        "--labels-src",
        type=str,
        default="data/images/labels",
        help="Source directory with class subfolders containing labels"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Output base directory"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation set ratio (default: 0.2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing train/val directories before copying"
    )
    parser.add_argument(
        "--oversample",
        action="store_true",
        help="Oversample underrepresented classes to balance dataset"
    )
    parser.add_argument(
        "--oversample-target",
        type=int,
        default=None,
        help="Target sample count per class when oversampling (default: 50%% of max class)"
    )
    parser.add_argument(
        "--oversample-ratio",
        type=float,
        default=0.5,
        help="Minimum ratio relative to largest class (default: 0.5)"
    )
    
    args = parser.parse_args()
    
    images_src = Path(args.images_src)
    labels_src = Path(args.labels_src)
    output_base = Path(args.output)
    
    # Output directories
    train_images = output_base / "images" / "train"
    train_labels = output_base / "labels" / "train"
    val_images = output_base / "images" / "val"
    val_labels = output_base / "labels" / "val"
    
    print("=" * 60)
    print("DATASET PREPARATION")
    print("=" * 60)
    print(f"\nSource images: {images_src}")
    print(f"Source labels: {labels_src}")
    print(f"Output base: {output_base}")
    print(f"Val ratio: {args.val_ratio}")
    print(f"Seed: {args.seed}")
    print(f"Oversample: {args.oversample}")
    
    # Clean if requested
    if args.clean:
        print("\nCleaning existing directories...")
        for d in [train_images, train_labels, val_images, val_labels]:
            if d.exists():
                shutil.rmtree(d)
                print(f"  Removed: {d}")
    
    # Find all pairs
    print("\nFinding image/label pairs...")
    pairs = find_image_label_pairs(images_src, labels_src)
    
    if not pairs:
        print("ERROR: No images found!")
        return 1
        
    print(f"  Found {len(pairs)} images")
    
    # Count by folder
    folder_counts = defaultdict(int)
    labeled_counts = defaultdict(int)
    for img, label, folder in pairs:
        folder_counts[folder] += 1
        if label is not None:
            labeled_counts[folder] += 1
            
    print("\n  By class folder:")
    for folder in FOLDER_TO_CLASS.keys():
        total = folder_counts.get(folder, 0)
        labeled = labeled_counts.get(folder, 0)
        print(f"    {folder}: {total} images ({labeled} labeled)")
    
    # Split
    print(f"\nSplitting dataset (val_ratio={args.val_ratio})...")
    train_pairs, val_pairs = split_by_event(pairs, args.val_ratio, args.seed)
    
    print(f"  Train: {len(train_pairs)} images")
    print(f"  Val: {len(val_pairs)} images")
    
    # Show class distribution before oversampling
    print("\n  Training set class distribution (before oversampling):")
    train_dist = calculate_class_distribution(train_pairs)
    for folder in FOLDER_TO_CLASS.keys():
        count = train_dist.get(folder, 0)
        print(f"    {folder}: {count} labeled samples")
    
    # Oversample if requested (only on training set!)
    if args.oversample:
        print("\n  Applying oversampling to training set...")
        train_pairs = oversample_rare_classes(
            train_pairs,
            target_count=args.oversample_target,
            min_ratio=args.oversample_ratio
        )
        print(f"\n  Train after oversampling: {len(train_pairs)} images")
    
    # Copy files
    print("\nCopying files...")
    
    print("  Copying training set...")
    train_stats = copy_pairs(train_pairs, train_images, train_labels)
    
    print("  Copying validation set...")
    val_stats = copy_pairs(val_pairs, val_images, val_labels)
    
    # Summary
    print("\n" + "=" * 60)
    print("DATASET PREPARED SUCCESSFULLY")
    print("=" * 60)
    
    print(f"\nTraining set: {train_images}")
    print(f"  Total images: {train_stats['total']}")
    print(f"  Labeled: {train_stats['labeled']}")
    if train_stats.get('duplicates', 0) > 0:
        print(f"  Oversampled duplicates: {train_stats['duplicates']}")
    print(f"  By class:")
    for folder, count in sorted(train_stats['by_class'].items()):
        class_id, class_name = FOLDER_TO_CLASS[folder]
        print(f"    {class_id} ({class_name}): {count}")
    
    print(f"\nValidation set: {val_images}")
    print(f"  Total images: {val_stats['total']}")
    print(f"  Labeled: {val_stats['labeled']}")
    print(f"  By class:")
    for folder, count in sorted(val_stats['by_class'].items()):
        class_id, class_name = FOLDER_TO_CLASS[folder]
        print(f"    {class_id} ({class_name}): {count}")
    
    print(f"\nLabels saved to:")
    print(f"  Train: {train_labels}")
    print(f"  Val: {val_labels}")
    
    print("\nNext steps:")
    print("  1. Verify dataset: python tools/verify_dataset.py --config configs/jinx_abilities.yaml")
    print("  2. Train model: python src/training/train_yolo.py --data configs/jinx_abilities.yaml")
    
    return 0


if __name__ == "__main__":
    exit(main())

