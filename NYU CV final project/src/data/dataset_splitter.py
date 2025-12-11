"""Split labeled data into train/val sets."""

import random
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
from collections import defaultdict


class DatasetSplitter:
    """Split image/label pairs into train and validation sets."""
    
    def __init__(
        self,
        val_ratio: float = 0.2,
        seed: int = 42,
        group_by_clip: bool = True
    ):
        """
        Initialize dataset splitter.
        
        Args:
            val_ratio: Fraction of data for validation (0.0-1.0)
            seed: Random seed for reproducibility
            group_by_clip: If True, keep frames from same clip together
        """
        self.val_ratio = val_ratio
        self.seed = seed
        self.group_by_clip = group_by_clip
        
    def _get_clip_id(self, filename: str) -> str:
        """
        Extract clip ID from filename.
        
        Assumes format: {clip_id}_frame_{number}.{ext}
        
        Args:
            filename: Image filename
            
        Returns:
            Clip ID string
        """
        # Remove extension and split by "_frame_"
        stem = Path(filename).stem
        parts = stem.rsplit("_frame_", 1)
        return parts[0] if len(parts) > 1 else stem
        
    def find_image_label_pairs(
        self,
        image_dir: str | Path,
        label_dir: str | Path,
        image_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png")
    ) -> List[Tuple[Path, Optional[Path]]]:
        """
        Find matching image and label file pairs.
        
        Args:
            image_dir: Directory containing images
            label_dir: Directory containing YOLO label files
            image_extensions: Valid image extensions
            
        Returns:
            List of (image_path, label_path) tuples
            label_path is None for unlabeled images (negatives)
        """
        image_dir = Path(image_dir)
        label_dir = Path(label_dir)
        
        pairs = []
        
        # Find all images
        images = []
        for ext in image_extensions:
            images.extend(image_dir.glob(f"*{ext}"))
            images.extend(image_dir.glob(f"*{ext.upper()}"))
            
        for image_path in sorted(images):
            # Look for corresponding label file
            label_path = label_dir / f"{image_path.stem}.txt"
            
            if label_path.exists():
                pairs.append((image_path, label_path))
            else:
                # Include as negative sample (no label)
                pairs.append((image_path, None))
                
        return pairs
        
    def split_pairs(
        self,
        pairs: List[Tuple[Path, Optional[Path]]]
    ) -> Tuple[List[Tuple[Path, Optional[Path]]], List[Tuple[Path, Optional[Path]]]]:
        """
        Split pairs into train and validation sets.
        
        Args:
            pairs: List of (image_path, label_path) tuples
            
        Returns:
            Tuple of (train_pairs, val_pairs)
        """
        random.seed(self.seed)
        
        if self.group_by_clip:
            # Group by clip ID to prevent frame leakage
            clips = defaultdict(list)
            for pair in pairs:
                clip_id = self._get_clip_id(pair[0].name)
                clips[clip_id].append(pair)
                
            # Split by clips
            clip_ids = list(clips.keys())
            random.shuffle(clip_ids)
            
            val_count = int(len(clip_ids) * self.val_ratio)
            val_clip_ids = set(clip_ids[:val_count])
            
            train_pairs = []
            val_pairs = []
            
            for clip_id, clip_pairs in clips.items():
                if clip_id in val_clip_ids:
                    val_pairs.extend(clip_pairs)
                else:
                    train_pairs.extend(clip_pairs)
        else:
            # Random split by individual frames
            shuffled = pairs.copy()
            random.shuffle(shuffled)
            
            val_count = int(len(shuffled) * self.val_ratio)
            val_pairs = shuffled[:val_count]
            train_pairs = shuffled[val_count:]
            
        return train_pairs, val_pairs
        
    def copy_pairs_to_split(
        self,
        pairs: List[Tuple[Path, Optional[Path]]],
        image_dest: str | Path,
        label_dest: str | Path
    ) -> int:
        """
        Copy image/label pairs to destination directories.
        
        Args:
            pairs: List of (image_path, label_path) tuples
            image_dest: Destination directory for images
            label_dest: Destination directory for labels
            
        Returns:
            Number of pairs copied
        """
        image_dest = Path(image_dest)
        label_dest = Path(label_dest)
        image_dest.mkdir(parents=True, exist_ok=True)
        label_dest.mkdir(parents=True, exist_ok=True)
        
        copied = 0
        
        for image_path, label_path in pairs:
            # Copy image
            shutil.copy2(image_path, image_dest / image_path.name)
            
            # Copy label if exists
            if label_path is not None:
                shutil.copy2(label_path, label_dest / label_path.name)
                
            copied += 1
            
        return copied
        
    def split_dataset(
        self,
        source_image_dir: str | Path,
        source_label_dir: str | Path,
        train_image_dir: str | Path,
        train_label_dir: str | Path,
        val_image_dir: str | Path,
        val_label_dir: str | Path
    ) -> dict:
        """
        Split dataset and copy to train/val directories.
        
        Args:
            source_image_dir: Source directory with all images
            source_label_dir: Source directory with all labels
            train_image_dir: Destination for training images
            train_label_dir: Destination for training labels
            val_image_dir: Destination for validation images
            val_label_dir: Destination for validation labels
            
        Returns:
            Dict with split statistics
        """
        # Find pairs
        pairs = self.find_image_label_pairs(source_image_dir, source_label_dir)
        
        if not pairs:
            raise ValueError(f"No images found in {source_image_dir}")
            
        # Split
        train_pairs, val_pairs = self.split_pairs(pairs)
        
        # Count labeled vs unlabeled
        train_labeled = sum(1 for _, lbl in train_pairs if lbl is not None)
        val_labeled = sum(1 for _, lbl in val_pairs if lbl is not None)
        
        print(f"Dataset split (seed={self.seed}, val_ratio={self.val_ratio}):")
        print(f"  Train: {len(train_pairs)} images ({train_labeled} labeled)")
        print(f"  Val:   {len(val_pairs)} images ({val_labeled} labeled)")
        
        # Copy files
        self.copy_pairs_to_split(train_pairs, train_image_dir, train_label_dir)
        self.copy_pairs_to_split(val_pairs, val_image_dir, val_label_dir)
        
        return {
            "total": len(pairs),
            "train_total": len(train_pairs),
            "train_labeled": train_labeled,
            "val_total": len(val_pairs),
            "val_labeled": val_labeled,
            "seed": self.seed,
            "val_ratio": self.val_ratio
        }
        
    def move_pairs_to_split(
        self,
        pairs: List[Tuple[Path, Optional[Path]]],
        image_dest: str | Path,
        label_dest: str | Path
    ) -> int:
        """
        Move (instead of copy) image/label pairs to destination.
        
        Args:
            pairs: List of (image_path, label_path) tuples
            image_dest: Destination directory for images
            label_dest: Destination directory for labels
            
        Returns:
            Number of pairs moved
        """
        image_dest = Path(image_dest)
        label_dest = Path(label_dest)
        image_dest.mkdir(parents=True, exist_ok=True)
        label_dest.mkdir(parents=True, exist_ok=True)
        
        moved = 0
        
        for image_path, label_path in pairs:
            # Move image
            shutil.move(str(image_path), str(image_dest / image_path.name))
            
            # Move label if exists
            if label_path is not None:
                shutil.move(str(label_path), str(label_dest / label_path.name))
                
            moved += 1
            
        return moved





