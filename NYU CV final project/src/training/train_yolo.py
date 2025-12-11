#!/usr/bin/env python3
"""
YOLO training module for Jinx ability detection.

Optimized for speed and accuracy with proper device detection (CUDA/MPS/CPU).

Usage:
    python src/training/train_yolo.py \
        --data configs/jinx_abilities.yaml \
        --epochs 100 \
        --name jinx_v1

    # With config file (recommended)
    python src/training/train_yolo.py \
        --data configs/jinx_abilities.yaml \
        --config configs/training_speed.yaml \
        --name jinx_fast
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def detect_best_device() -> str:
    """
    Auto-detect the best available device for training.
    
    Priority: CUDA GPU > Apple MPS > CPU
    
    Returns:
        Device string for YOLO training
    """
    try:
        import torch
        
        # Check for NVIDIA CUDA GPU
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"  [Device] CUDA GPU detected: {device_name}")
            return "0"  # Use first CUDA device
        
        # Check for Apple Silicon MPS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("  [Device] Apple MPS (Metal) detected")
            return "mps"
        
        # Fall back to CPU
        print("  [Device] No GPU detected, using CPU (training will be slow)")
        return "cpu"
        
    except ImportError:
        print("  [Device] PyTorch not found, defaulting to CPU")
        return "cpu"


class YOLOTrainer:
    """Wrapper for YOLO training with Ultralytics."""
    
    def __init__(
        self,
        data_config: str | Path,
        model: str = "yolov8n.pt",
        project: str = "runs/detect",
        name: str = "jinx_abilities"
    ):
        """
        Initialize YOLO trainer.
        
        Args:
            data_config: Path to dataset YAML config
            model: Base model (yolov8n.pt, yolov8s.pt, etc.)
            project: Output project directory
            name: Experiment name
        """
        self.data_config = Path(data_config)
        self.model_name = model
        self.project = project
        self.name = name
        self.model = None
        
    def load_model(self) -> None:
        """Load YOLO model."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "Ultralytics not installed. Run: pip install ultralytics"
            )
            
        print(f"Loading model: {self.model_name}")
        self.model = YOLO(self.model_name)
        
    def train(
        self,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        patience: int = 20,
        device: Optional[str] = None,
        workers: int = 8,
        resume: bool = False,
        **kwargs
    ) -> dict:
        """
        Train the model.
        
        Args:
            epochs: Number of training epochs
            imgsz: Image size
            batch: Batch size
            patience: Early stopping patience
            device: Device to use (None for auto-detect best available)
            workers: Number of data loader workers
            resume: Resume from last checkpoint
            **kwargs: Additional training arguments
            
        Returns:
            Training results dict
        """
        if self.model is None:
            self.load_model()
        
        # Auto-detect best device if not specified
        if device is None:
            device = detect_best_device()
        else:
            print(f"  [Device] Using specified device: {device}")
            
        print(f"\n{'='*50}")
        print("TRAINING CONFIGURATION")
        print(f"{'='*50}")
        print(f"  Data config: {self.data_config}")
        print(f"  Model: {self.model_name}")
        print(f"  Epochs: {epochs}")
        print(f"  Image size: {imgsz}")
        print(f"  Batch size: {batch}")
        print(f"  Patience: {patience}")
        print(f"  Device: {device}")
        print(f"  Workers: {workers}")
        
        # Log additional kwargs if present
        important_kwargs = ['cache', 'cos_lr', 'rect', 'mosaic', 'lr0', 'close_mosaic']
        active_opts = {k: v for k, v in kwargs.items() if k in important_kwargs}
        if active_opts:
            print(f"\n  Optimizations:")
            for k, v in active_opts.items():
                print(f"    {k}: {v}")
        print(f"{'='*50}\n")
        
        results = self.model.train(
            data=str(self.data_config),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=patience,
            device=device,
            workers=workers,
            project=self.project,
            name=self.name,
            resume=resume,
            verbose=True,
            plots=True,
            save=True,
            **kwargs
        )
        
        return results
        
    def validate(self, weights: Optional[str] = None) -> dict:
        """
        Run validation on the model.
        
        Args:
            weights: Path to weights file (None for current model)
            
        Returns:
            Validation metrics dict
        """
        if weights:
            from ultralytics import YOLO
            model = YOLO(weights)
        else:
            if self.model is None:
                raise RuntimeError("No model loaded. Train first or provide weights.")
            model = self.model
            
        results = model.val(data=str(self.data_config))
        return results
        
    def export(
        self,
        weights: str,
        format: str = "onnx",
        **kwargs
    ) -> str:
        """
        Export model to different format.
        
        Args:
            weights: Path to weights file
            format: Export format (onnx, torchscript, etc.)
            
        Returns:
            Path to exported model
        """
        from ultralytics import YOLO
        
        model = YOLO(weights)
        path = model.export(format=format, **kwargs)
        return path


def load_training_config(config_path: Path) -> dict:
    """Load training configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def extract_train_kwargs(config: dict) -> dict:
    """
    Extract all training kwargs from a config dictionary.
    
    Properly maps config file settings to Ultralytics YOLO parameters.
    
    Args:
        config: Parsed YAML config dictionary
        
    Returns:
        Dictionary of kwargs to pass to YOLO train()
    """
    kwargs = {}
    
    # Extract training hyperparameters
    if "training" in config:
        tc = config["training"]
        
        # Learning rate and optimizer settings
        if "learning_rate" in tc:
            kwargs["lr0"] = tc["learning_rate"]
        if "weight_decay" in tc:
            kwargs["weight_decay"] = tc["weight_decay"]
        if "momentum" in tc:
            kwargs["momentum"] = tc["momentum"]
        if "warmup_epochs" in tc:
            kwargs["warmup_epochs"] = tc["warmup_epochs"]
        if "warmup_momentum" in tc:
            kwargs["warmup_momentum"] = tc["warmup_momentum"]
            
        # Optimizer selection
        if "optimizer" in tc:
            kwargs["optimizer"] = tc["optimizer"]
            
        # Learning rate scheduler
        if tc.get("lr_scheduler") == "cosine":
            kwargs["cos_lr"] = True
    
    # Extract optimization settings
    if "optimizations" in config:
        opt = config["optimizations"]
        
        if "cache" in opt:
            kwargs["cache"] = opt["cache"]
        if "rect" in opt:
            kwargs["rect"] = opt["rect"]
        if "cos_lr" in opt:
            kwargs["cos_lr"] = opt["cos_lr"]
        if "close_mosaic" in opt:
            kwargs["close_mosaic"] = opt["close_mosaic"]
        if "amp" in opt:
            kwargs["amp"] = opt["amp"]
            
    # Extract augmentation parameters (direct mapping to YOLO)
    if "augmentation" in config:
        aug = config["augmentation"]
        
        # These map directly to Ultralytics augmentation params
        aug_params = [
            'hsv_h', 'hsv_s', 'hsv_v',  # Color augmentation
            'degrees', 'translate', 'scale', 'shear', 'perspective',  # Geometric
            'flipud', 'fliplr',  # Flips
            'mosaic', 'mixup', 'copy_paste',  # Advanced
            'erasing',  # Random erasing
        ]
        
        for param in aug_params:
            if param in aug:
                kwargs[param] = aug[param]
    
    # Extract validation settings
    if "validation" in config:
        val = config["validation"]
        
        if "conf_threshold" in val:
            kwargs["conf"] = val["conf_threshold"]
        if "iou_threshold" in val:
            kwargs["iou"] = val["iou_threshold"]
    
    # Extract model settings
    if "model" in config:
        model = config["model"]
        
        if "dropout" in model:
            kwargs["dropout"] = model["dropout"]
            
    return kwargs


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO model for Jinx ability detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training
    python src/training/train_yolo.py \\
        --data configs/jinx_abilities.yaml \\
        --epochs 100

    # Custom model and settings
    python src/training/train_yolo.py \\
        --data configs/jinx_abilities.yaml \\
        --model yolov8s.pt \\
        --epochs 150 \\
        --batch 32 \\
        --imgsz 640 \\
        --name jinx_v2

    # Resume training
    python src/training/train_yolo.py \\
        --data configs/jinx_abilities.yaml \\
        --resume runs/detect/jinx_v1/weights/last.pt

    # Use training config file
    python src/training/train_yolo.py \\
        --data configs/jinx_abilities.yaml \\
        --config configs/training_config.yaml
        """
    )
    
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset YAML config"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training config YAML (optional)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Base model (default: yolov8n.pt)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs (default: 100)"
    )
    
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)"
    )
    
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size (default: 640)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cpu, mps, 0, 1, etc. (default: auto-detect best)"
    )
    
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Cache images in RAM for faster training"
    )
    
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (default: 20)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of data loader workers (default: 8)"
    )
    
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Output project directory (default: runs/detect)"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default="jinx_abilities",
        help="Experiment name (default: jinx_abilities)"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint path"
    )
    
    args = parser.parse_args()
    
    # Load training config if provided
    train_kwargs = {}
    model_to_use = args.model
    epochs_to_use = args.epochs
    batch_to_use = args.batch
    imgsz_to_use = args.imgsz
    patience_to_use = args.patience
    workers_to_use = args.workers
    
    if args.config:
        print(f"\nLoading config: {args.config}")
        config = load_training_config(Path(args.config))
        
        # Extract all training kwargs using the comprehensive function
        train_kwargs = extract_train_kwargs(config)
        
        # Override defaults from config file (CLI args take precedence)
        if "model" in config and "base" in config["model"]:
            if args.model == "yolov8n.pt":  # Only override if using default
                model_to_use = config["model"]["base"]
                
        if "training" in config:
            tc = config["training"]
            if args.epochs == 100 and "epochs" in tc:  # Only override if using default
                epochs_to_use = tc["epochs"]
            if args.batch == 16 and "batch_size" in tc:
                batch_to_use = tc["batch_size"]
            if args.imgsz == 640 and "image_size" in tc:
                imgsz_to_use = tc["image_size"]
            if args.patience == 20 and "patience" in tc:
                patience_to_use = tc["patience"]
                
        if "optimizations" in config and "workers" in config["optimizations"]:
            if args.workers == 8:  # Only override if using default
                workers_to_use = config["optimizations"]["workers"]
                
        print(f"  Loaded {len(train_kwargs)} training parameters from config")
    
    # Add cache flag if specified via CLI
    if args.cache:
        train_kwargs["cache"] = True
            
    # Create trainer
    trainer = YOLOTrainer(
        data_config=args.data,
        model=model_to_use,
        project=args.project,
        name=args.name
    )
    
    # Handle resume
    if args.resume:
        trainer.model_name = args.resume
        train_kwargs["resume"] = True
        
    # Train
    try:
        results = trainer.train(
            epochs=epochs_to_use,
            imgsz=imgsz_to_use,
            batch=batch_to_use,
            patience=patience_to_use,
            device=args.device,
            workers=workers_to_use,
            **train_kwargs
        )
        
        print("\n" + "="*50)
        print("Training complete!")
        print(f"Results saved to: {args.project}/{args.name}")
        print(f"Best weights: {args.project}/{args.name}/weights/best.pt")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTraining failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


