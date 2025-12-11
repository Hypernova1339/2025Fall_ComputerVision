"""Extract frames from videos for training data."""

import cv2
from pathlib import Path
from typing import Iterator, List, Optional, Tuple
from tqdm import tqdm


class FrameExtractor:
    """Extract frames from video files."""
    
    def __init__(
        self,
        frame_stride: int = 3,
        output_format: str = "jpg",
        jpeg_quality: int = 95
    ):
        """
        Initialize frame extractor.
        
        Args:
            frame_stride: Save every N-th frame
            output_format: Output image format (jpg, png)
            jpeg_quality: JPEG quality (1-100)
        """
        self.frame_stride = frame_stride
        self.output_format = output_format
        self.jpeg_quality = jpeg_quality
        
    def extract_from_video(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        prefix: Optional[str] = None,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        show_progress: bool = True
    ) -> List[Path]:
        """
        Extract frames from a single video.
        
        Args:
            video_path: Path to video file
            output_dir: Directory for output frames
            prefix: Filename prefix (default: video stem)
            start_frame: Starting frame index
            end_frame: Ending frame index (None for all)
            show_progress: Show progress bar
            
        Returns:
            List of paths to extracted frames
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if prefix is None:
            prefix = video_path.stem
            
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if end_frame is None:
            end_frame = total_frames
            
        # Calculate number of frames to extract
        num_frames = (end_frame - start_frame) // self.frame_stride
        
        extracted = []
        frame_idx = start_frame
        saved_count = 0
        
        # Set starting position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Create progress bar
        pbar = tqdm(total=num_frames, desc=f"Extracting {video_path.name}", disable=not show_progress)
        
        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            if (frame_idx - start_frame) % self.frame_stride == 0:
                # Generate filename
                filename = f"{prefix}_frame_{saved_count:05d}.{self.output_format}"
                output_path = output_dir / filename
                
                # Save frame
                if self.output_format.lower() in ["jpg", "jpeg"]:
                    cv2.imwrite(
                        str(output_path),
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                    )
                else:
                    cv2.imwrite(str(output_path), frame)
                    
                extracted.append(output_path)
                saved_count += 1
                pbar.update(1)
                
            frame_idx += 1
            
        cap.release()
        pbar.close()
        
        print(f"Extracted {saved_count} frames to {output_dir}")
        return extracted
        
    def extract_from_directory(
        self,
        video_dir: str | Path,
        output_dir: str | Path,
        video_extensions: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv"),
        show_progress: bool = True
    ) -> List[Path]:
        """
        Extract frames from all videos in a directory.
        
        Args:
            video_dir: Directory containing videos
            output_dir: Directory for output frames
            video_extensions: Video file extensions to process
            show_progress: Show progress bar
            
        Returns:
            List of paths to all extracted frames
        """
        video_dir = Path(video_dir)
        output_dir = Path(output_dir)
        
        # Find all video files
        video_files = []
        for ext in video_extensions:
            video_files.extend(video_dir.glob(f"*{ext}"))
            video_files.extend(video_dir.glob(f"*{ext.upper()}"))
            
        if not video_files:
            print(f"No videos found in {video_dir}")
            return []
            
        print(f"Found {len(video_files)} videos")
        
        all_extracted = []
        for video_path in video_files:
            extracted = self.extract_from_video(
                video_path,
                output_dir,
                show_progress=show_progress
            )
            all_extracted.extend(extracted)
            
        return all_extracted
        
    def extract_specific_frames(
        self,
        video_path: str | Path,
        frame_indices: List[int],
        output_dir: str | Path,
        prefix: Optional[str] = None
    ) -> List[Path]:
        """
        Extract specific frames by index.
        
        Args:
            video_path: Path to video file
            frame_indices: List of frame indices to extract
            output_dir: Directory for output frames
            prefix: Filename prefix
            
        Returns:
            List of paths to extracted frames
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if prefix is None:
            prefix = video_path.stem
            
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
            
        extracted = []
        
        for i, frame_idx in enumerate(sorted(frame_indices)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                print(f"Warning: Could not read frame {frame_idx}")
                continue
                
            filename = f"{prefix}_frame_{frame_idx:05d}.{self.output_format}"
            output_path = output_dir / filename
            
            if self.output_format.lower() in ["jpg", "jpeg"]:
                cv2.imwrite(
                    str(output_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                )
            else:
                cv2.imwrite(str(output_path), frame)
                
            extracted.append(output_path)
            
        cap.release()
        return extracted
        
    def extract_at_timestamps(
        self,
        video_path: str | Path,
        timestamps_sec: List[float],
        output_dir: str | Path,
        prefix: Optional[str] = None
    ) -> List[Path]:
        """
        Extract frames at specific timestamps.
        
        Args:
            video_path: Path to video file
            timestamps_sec: List of timestamps in seconds
            output_dir: Directory for output frames
            prefix: Filename prefix
            
        Returns:
            List of paths to extracted frames
        """
        video_path = Path(video_path)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Convert timestamps to frame indices
        frame_indices = [int(t * fps) for t in timestamps_sec]
        
        return self.extract_specific_frames(
            video_path,
            frame_indices,
            output_dir,
            prefix
        )





