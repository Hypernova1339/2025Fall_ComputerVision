"""Full video inference pipeline."""

import json
from pathlib import Path
from typing import List, Optional, Generator
from tqdm import tqdm

from .detector import JinxDetector, FrameDetections


class VideoInference:
    """Run detection on entire videos."""
    
    def __init__(
        self,
        detector: JinxDetector,
        frame_stride: int = 1,
        batch_size: int = 8
    ):
        """
        Initialize video inference.
        
        Args:
            detector: JinxDetector instance
            frame_stride: Process every N-th frame
            batch_size: Batch size for inference
        """
        self.detector = detector
        self.frame_stride = frame_stride
        self.batch_size = batch_size
        
    def process_video(
        self,
        video_path: str | Path,
        output_path: Optional[str | Path] = None,
        show_progress: bool = True
    ) -> List[FrameDetections]:
        """
        Process entire video and return detections.
        
        Args:
            video_path: Path to input video
            output_path: Optional path to save JSON results
            show_progress: Show progress bar
            
        Returns:
            List of FrameDetections for all processed frames
        """
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from src.utils.video_utils import VideoReader
        
        video_path = Path(video_path)
        
        with VideoReader(video_path) as reader:
            info = reader.info
            fps = info.fps
            total_frames = info.frame_count
            
            print(f"Processing video: {video_path.name}")
            print(f"  Resolution: {info.width}x{info.height}")
            print(f"  FPS: {fps}")
            print(f"  Total frames: {total_frames}")
            print(f"  Frame stride: {self.frame_stride}")
            print(f"  Frames to process: {total_frames // self.frame_stride}")
            
            all_detections = []
            
            # Create progress bar
            pbar = tqdm(
                total=total_frames // self.frame_stride,
                desc="Detecting",
                disable=not show_progress
            )
            
            # Process in batches
            batch_frames = []
            batch_indices = []
            
            for frame_idx, frame in reader.iter_frames(step=self.frame_stride):
                batch_frames.append(frame)
                batch_indices.append(frame_idx)
                
                if len(batch_frames) >= self.batch_size:
                    # Process batch
                    batch_detections = self.detector.detect_batch(
                        batch_frames,
                        start_frame_idx=batch_indices[0],
                        fps=fps
                    )
                    
                    # Fix frame indices for strided frames
                    for det, idx in zip(batch_detections, batch_indices):
                        det.frame_idx = idx
                        det.timestamp_sec = idx / fps
                        
                    all_detections.extend(batch_detections)
                    pbar.update(len(batch_frames))
                    
                    batch_frames = []
                    batch_indices = []
                    
            # Process remaining frames
            if batch_frames:
                batch_detections = self.detector.detect_batch(
                    batch_frames,
                    start_frame_idx=batch_indices[0],
                    fps=fps
                )
                
                for det, idx in zip(batch_detections, batch_indices):
                    det.frame_idx = idx
                    det.timestamp_sec = idx / fps
                    
                all_detections.extend(batch_detections)
                pbar.update(len(batch_frames))
                
            pbar.close()
            
        # Save results if path provided
        if output_path:
            self.save_detections(all_detections, output_path, video_path.name, fps)
            
        # Print summary
        total_dets = sum(len(fd.detections) for fd in all_detections)
        print(f"\nDetection summary:")
        print(f"  Frames processed: {len(all_detections)}")
        print(f"  Total detections: {total_dets}")
        
        # Count by class
        class_counts = {}
        for fd in all_detections:
            for det in fd.detections:
                class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
                
        for class_name, count in sorted(class_counts.items()):
            print(f"    {class_name}: {count}")
            
        return all_detections
        
    def save_detections(
        self,
        detections: List[FrameDetections],
        output_path: str | Path,
        video_name: str,
        fps: float
    ) -> None:
        """
        Save detections to JSON file.
        
        Args:
            detections: List of FrameDetections
            output_path: Output file path
            video_name: Source video name
            fps: Video FPS
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "video": video_name,
            "fps": fps,
            "frame_stride": self.frame_stride,
            "total_frames": len(detections),
            "frames": [fd.to_dict() for fd in detections]
        }
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
            
        print(f"Saved detections to: {output_path}")
        
    @staticmethod
    def load_detections(json_path: str | Path) -> dict:
        """Load detections from JSON file."""
        with open(json_path, "r") as f:
            return json.load(f)
            
    def stream_process(
        self,
        video_path: str | Path,
        show_progress: bool = True
    ) -> Generator[FrameDetections, None, None]:
        """
        Process video as a generator (for real-time processing).
        
        Args:
            video_path: Path to input video
            show_progress: Show progress bar
            
        Yields:
            FrameDetections for each processed frame
        """
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from src.utils.video_utils import VideoReader
        
        video_path = Path(video_path)
        
        with VideoReader(video_path) as reader:
            info = reader.info
            fps = info.fps
            
            pbar = tqdm(
                total=info.frame_count // self.frame_stride,
                desc="Detecting",
                disable=not show_progress
            )
            
            for frame_idx, frame in reader.iter_frames(step=self.frame_stride):
                detections = self.detector.detect_frame(frame, frame_idx, fps)
                yield detections
                pbar.update(1)
                
            pbar.close()





