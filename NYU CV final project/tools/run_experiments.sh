#!/bin/bash
# =============================================================================
# YOLO Training Experiments for Jinx Ability Detection
# =============================================================================
# This script runs a series of ablation experiments to find optimal settings.
#
# Usage:
#   chmod +x tools/run_experiments.sh
#   ./tools/run_experiments.sh [experiment_name]
#
# Or run specific experiments:
#   ./tools/run_experiments.sh baseline
#   ./tools/run_experiments.sh speed
#   ./tools/run_experiments.sh accuracy
#   ./tools/run_experiments.sh all
#
# Results will be saved to runs/detect/exp_<name>/
# Compare metrics in each experiment's results.csv
# =============================================================================

set -e  # Exit on error

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper function
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if virtual environment is active
if [ -z "$VIRTUAL_ENV" ]; then
    log_warn "Virtual environment not active. Activating..."
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    else
        log_error "Virtual environment not found. Please create one first."
        exit 1
    fi
fi

# Experiment functions
run_baseline() {
    log_info "Running BASELINE experiment (default settings, fixed device)..."
    python src/training/train_yolo.py \
        --data configs/jinx_abilities.yaml \
        --config configs/training_config.yaml \
        --epochs 100 \
        --name exp_baseline \
        --cache
}

run_speed() {
    log_info "Running SPEED experiment (optimized for fast iteration)..."
    python src/training/train_yolo.py \
        --data configs/jinx_abilities.yaml \
        --config configs/training_speed.yaml \
        --name exp_speed
}

run_accuracy() {
    log_info "Running ACCURACY experiment (optimized for best mAP)..."
    python src/training/train_yolo.py \
        --data configs/jinx_abilities.yaml \
        --config configs/training_accuracy.yaml \
        --name exp_accuracy
}

run_small_objects() {
    log_info "Running SMALL OBJECTS experiment (larger image size for tiny projectiles)..."
    python src/training/train_yolo.py \
        --data configs/jinx_abilities.yaml \
        --config configs/training_config.yaml \
        --imgsz 800 \
        --batch 8 \
        --epochs 80 \
        --name exp_small_obj \
        --cache
}

run_larger_model() {
    log_info "Running LARGER MODEL experiment (yolov8s instead of yolov8n)..."
    python src/training/train_yolo.py \
        --data configs/jinx_abilities.yaml \
        --config configs/training_config.yaml \
        --model yolov8s.pt \
        --epochs 100 \
        --name exp_yolov8s \
        --cache
}

# Print experiment summary
print_summary() {
    echo ""
    echo "============================================================"
    echo "EXPERIMENT SUMMARY"
    echo "============================================================"
    echo ""
    echo "| Experiment    | Model    | ImgSz | Config          | Notes                    |"
    echo "|---------------|----------|-------|-----------------|--------------------------|"
    echo "| baseline      | yolov8n  | 640   | default         | Standard training        |"
    echo "| speed         | yolov8n  | 480   | training_speed  | Fast iteration           |"
    echo "| accuracy      | yolov8s  | 640   | training_accuracy| Best mAP               |"
    echo "| small_obj     | yolov8n  | 800   | default         | Better for projectiles   |"
    echo "| larger_model  | yolov8s  | 640   | default         | More capacity            |"
    echo ""
    echo "Results saved to: runs/detect/exp_<name>/"
    echo ""
    echo "Key metrics to compare:"
    echo "  - mAP50, mAP50-95 (overall accuracy)"
    echo "  - Per-class AP (especially VFX_R_IMPACT)"
    echo "  - Training time (time column in results.csv)"
    echo "  - Inference speed (check val output)"
    echo ""
}

# Main logic
case "${1:-help}" in
    baseline)
        run_baseline
        ;;
    speed)
        run_speed
        ;;
    accuracy)
        run_accuracy
        ;;
    small_obj|small_objects)
        run_small_objects
        ;;
    larger_model|yolov8s)
        run_larger_model
        ;;
    all)
        log_info "Running ALL experiments sequentially..."
        echo ""
        print_summary
        echo ""
        read -p "This will take several hours. Continue? [y/N] " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            run_baseline
            run_speed
            run_accuracy
            run_small_objects
            run_larger_model
            log_info "All experiments complete!"
        else
            log_info "Cancelled."
        fi
        ;;
    help|--help|-h)
        echo "YOLO Training Experiments Runner"
        echo ""
        echo "Usage: $0 [experiment]"
        echo ""
        echo "Experiments:"
        echo "  baseline      - Default training with auto device detection"
        echo "  speed         - Speed-optimized (smaller images, fewer epochs)"
        echo "  accuracy      - Accuracy-optimized (larger model, tuned augmentation)"
        echo "  small_obj     - Larger image size (800px) for small projectiles"
        echo "  larger_model  - Use yolov8s instead of yolov8n"
        echo "  all           - Run all experiments sequentially"
        echo ""
        print_summary
        ;;
    *)
        log_error "Unknown experiment: $1"
        echo "Run '$0 help' for usage information."
        exit 1
        ;;
esac




