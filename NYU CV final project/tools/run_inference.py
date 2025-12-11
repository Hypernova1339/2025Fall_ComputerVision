#!/usr/bin/env python3
"""
Run full inference pipeline on a video.

This tool:
1. Runs YOLO detection on all frames
2. Aggregates detections into ability events
3. Outputs both frame-level and event-level results

Usage:
    python tools/run_inference.py \
        --video path/to/gameplay.mp4 \
        --weights runs/detect/jinx_abilities/weights/best.pt \
        --output outputs/
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.detector import JinxDetector
from src.inference.video_inference import VideoInference
from src.events.aggregator import EventAggregator
from src.events.hit_classifier import HitClassifier, analyze_ability_usage


def main():
    parser = argparse.ArgumentParser(
        description="Run full inference pipeline on gameplay video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic inference
    python tools/run_inference.py \\
        --video gameplay.mp4 \\
        --weights runs/detect/jinx_abilities/weights/best.pt \\
        --output outputs/

    # Skip every other frame for speed
    python tools/run_inference.py \\
        --video gameplay.mp4 \\
        --weights runs/detect/jinx_abilities/weights/best.pt \\
        --output outputs/ \\
        --stride 2

    # Higher confidence threshold
    python tools/run_inference.py \\
        --video gameplay.mp4 \\
        --weights runs/detect/jinx_abilities/weights/best.pt \\
        --output outputs/ \\
        --conf 0.4

    # Create annotated video
    python tools/run_inference.py \\
        --video gameplay.mp4 \\
        --weights runs/detect/jinx_abilities/weights/best.pt \\
        --output outputs/ \\
        --visualize
        """
    )
    
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file"
    )
    
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained YOLO weights (.pt file)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Process every N-th frame (default: 1 = all frames)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference (default: 8)"
    )
    
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)"
    )
    
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS (default: 0.45)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cpu, 0, 1, etc. (default: auto)"
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create annotated video with detections"
    )
    
    parser.add_argument(
        "--timeline",
        action="store_true",
        help="Create timeline visualization"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    video_path = Path(args.video)
    weights_path = Path(args.weights)
    output_dir = Path(args.output)
    
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)
        
    if not weights_path.exists():
        print(f"Error: Weights not found: {weights_path}")
        sys.exit(1)
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize detector
    print(f"\n{'='*60}")
    print("JINX ABILITY DETECTION INFERENCE")
    print(f"{'='*60}")
    print(f"\nVideo: {video_path}")
    print(f"Weights: {weights_path}")
    print(f"Output: {output_dir}")
    print(f"Frame stride: {args.stride}")
    print(f"Confidence threshold: {args.conf}")
    print()
    
    detector = JinxDetector(
        weights_path=weights_path,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device
    )
    
    # Run video inference
    video_inference = VideoInference(
        detector=detector,
        frame_stride=args.stride,
        batch_size=args.batch_size
    )
    
    video_name = video_path.stem
    detections_path = output_dir / f"{video_name}_detections.json"
    
    print("\n[1/3] Running frame-level detection...")
    frame_detections = video_inference.process_video(
        video_path,
        output_path=detections_path,
        show_progress=not args.quiet
    )
    
    # Aggregate into events
    print("\n[2/3] Aggregating detections into ability events...")
    
    aggregator = EventAggregator()
    
    # Get FPS for timestamp calculation
    from src.utils.video_utils import get_video_info
    video_info = get_video_info(video_path)
    fps = video_info.fps
    
    for fd in frame_detections:
        det_dicts = [
            {"class_id": d.class_id, "confidence": d.confidence}
            for d in fd.detections
        ]
        aggregator.process_frame(fd.frame_idx, fd.timestamp_sec, det_dicts)
        
    # Finalize pending events
    if frame_detections:
        final_time = frame_detections[-1].timestamp_sec
        aggregator.finalize(final_time)
        
    events = aggregator.get_events()
    
    # Save events
    events_path = output_dir / f"{video_name}_events.jsonl"
    aggregator.save_events(events_path)
    
    # Run hit classifier
    classifier = HitClassifier()
    analyses = classifier.classify_events(events)
    
    # Print summary
    summary = aggregator.get_summary()
    usage_stats = analyze_ability_usage(events)
    
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"\nTotal ability events detected: {summary['total_events']}")
    print(f"\nW (Zap!):")
    print(f"  Total casts: {summary['W']['total']}")
    print(f"  Hits: {summary['W']['hits']}")
    print(f"  Misses: {summary['W']['misses']}")
    print(f"  Hit rate: {summary['W']['hit_rate']:.1%}")
    print(f"\nR (Super Mega Death Rocket):")
    print(f"  Total casts: {summary['R']['total']}")
    print(f"  Hits: {summary['R']['hits']}")
    print(f"  Misses: {summary['R']['misses']}")
    print(f"  Hit rate: {summary['R']['hit_rate']:.1%}")
    
    # Save full results
    results_path = output_dir / f"{video_name}_results.json"
    results = {
        "video": str(video_path),
        "weights": str(weights_path),
        "settings": {
            "frame_stride": args.stride,
            "conf_threshold": args.conf,
            "iou_threshold": args.iou,
        },
        "summary": summary,
        "usage_stats": usage_stats,
        "events": [e.to_dict() for e in events],
    }
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to: {results_path}")
    
    # Optional visualizations
    if args.timeline and events:
        print("\n[3/3] Creating timeline visualization...")
        from src.utils.visualization import draw_timeline
        
        timeline_path = output_dir / f"{video_name}_timeline.png"
        event_dicts = [
            {"ability": e.ability, "cast_time": e.cast_time_sec, "hit": e.hit}
            for e in events
        ]
        draw_timeline(
            event_dicts,
            duration_sec=video_info.duration_sec,
            output_path=timeline_path
        )
        print(f"Timeline saved to: {timeline_path}")
        
    if args.visualize:
        print("\n[3/3] Creating annotated video...")
        from src.utils.visualization import Detection, create_detection_video
        
        # Convert to visualization format
        detections_by_frame = {}
        for fd in frame_detections:
            vis_dets = [
                Detection(
                    class_id=d.class_id,
                    confidence=d.confidence,
                    x1=d.bbox_normalized[0],
                    y1=d.bbox_normalized[1],
                    x2=d.bbox_normalized[2],
                    y2=d.bbox_normalized[3],
                )
                for d in fd.detections
            ]
            detections_by_frame[fd.frame_idx] = vis_dets
            
        annotated_path = output_dir / f"{video_name}_annotated.mp4"
        create_detection_video(
            video_path,
            detections_by_frame,
            annotated_path,
            show_progress=not args.quiet
        )
        print(f"Annotated video saved to: {annotated_path}")
        
    print(f"\n{'='*60}")
    print("INFERENCE COMPLETE")
    print(f"{'='*60}")
    print(f"\nOutput files:")
    print(f"  Detections: {detections_path}")
    print(f"  Events: {events_path}")
    print(f"  Results: {results_path}")


if __name__ == "__main__":
    main()





