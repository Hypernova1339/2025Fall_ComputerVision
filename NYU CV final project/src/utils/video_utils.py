"""Video reading and writing utilities using OpenCV."""

import cv2
import numpy as np
from pathlib import Path
from typing import Iterator, Optional, Tuple
from dataclasses import dataclass


@dataclass
class VideoInfo:
    """Video metadata container."""
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float
    codec: str


class VideoReader:
    """Context manager for reading video frames."""
    
    def __init__(self, video_path: str | Path):
        """
        Initialize video reader.
        
        Args:
            video_path: Path to video file
        """
        self.video_path = Path(video_path)
        self.cap: Optional[cv2.VideoCapture] = None
        self._info: Optional[VideoInfo] = None
        
    def __enter__(self) -> "VideoReader":
        self.open()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def open(self) -> None:
        """Open the video file."""
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")
            
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")
            
        # Cache video info
        self._info = VideoInfo(
            width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=self.cap.get(cv2.CAP_PROP_FPS),
            frame_count=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            duration_sec=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) / max(self.cap.get(cv2.CAP_PROP_FPS), 1),
            codec=self._get_codec()
        )
        
    def _get_codec(self) -> str:
        """Get the video codec as a string."""
        if self.cap is None:
            return ""
        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        return "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
    def close(self) -> None:
        """Release the video capture."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
    @property
    def info(self) -> VideoInfo:
        """Get video metadata."""
        if self._info is None:
            raise RuntimeError("Video not opened. Call open() first.")
        return self._info
        
    @property
    def is_opened(self) -> bool:
        """Check if video is opened."""
        return self.cap is not None and self.cap.isOpened()
        
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single frame."""
        if self.cap is None:
            raise RuntimeError("Video not opened. Call open() first.")
        ret, frame = self.cap.read()
        return ret, frame if ret else None
        
    def read_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Read a specific frame by index.
        
        Args:
            frame_idx: Frame index (0-based)
            
        Returns:
            Frame as numpy array or None if failed
        """
        if self.cap is None:
            raise RuntimeError("Video not opened. Call open() first.")
            
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        return frame if ret else None
        
    def read_at_time(self, time_sec: float) -> Optional[np.ndarray]:
        """
        Read frame at a specific timestamp.
        
        Args:
            time_sec: Time in seconds
            
        Returns:
            Frame as numpy array or None if failed
        """
        if self.cap is None:
            raise RuntimeError("Video not opened. Call open() first.")
            
        self.cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
        ret, frame = self.cap.read()
        return frame if ret else None
        
    def iter_frames(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        step: int = 1
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Iterate over frames.
        
        Args:
            start_frame: Starting frame index
            end_frame: Ending frame index (exclusive), None for all frames
            step: Frame step (e.g., 2 = every other frame)
            
        Yields:
            Tuple of (frame_index, frame_array)
        """
        if self.cap is None:
            raise RuntimeError("Video not opened. Call open() first.")
            
        if end_frame is None:
            end_frame = self.info.frame_count
            
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for frame_idx in range(start_frame, end_frame, step):
            if frame_idx != start_frame and step > 1:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                
            ret, frame = self.cap.read()
            if not ret:
                break
                
            yield frame_idx, frame
            
            # Skip frames if step > 1 and we're reading sequentially
            if step == 1:
                continue
                
    def time_to_frame(self, time_sec: float) -> int:
        """Convert timestamp to frame index."""
        return int(time_sec * self.info.fps)
        
    def frame_to_time(self, frame_idx: int) -> float:
        """Convert frame index to timestamp."""
        return frame_idx / self.info.fps


class VideoWriter:
    """Context manager for writing video frames."""
    
    def __init__(
        self,
        output_path: str | Path,
        fps: float,
        width: int,
        height: int,
        codec: str = "mp4v"
    ):
        """
        Initialize video writer.
        
        Args:
            output_path: Path for output video
            fps: Frames per second
            width: Frame width
            height: Frame height
            codec: FourCC codec string (default: mp4v)
        """
        self.output_path = Path(output_path)
        self.fps = fps
        self.width = width
        self.height = height
        self.codec = codec
        self.writer: Optional[cv2.VideoWriter] = None
        self.frame_count = 0
        
    def __enter__(self) -> "VideoWriter":
        self.open()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def open(self) -> None:
        """Open the video writer."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (self.width, self.height)
        )
        
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {self.output_path}")
            
    def close(self) -> None:
        """Release the video writer."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            
    def write(self, frame: np.ndarray) -> None:
        """
        Write a frame to the video.
        
        Args:
            frame: Frame as numpy array (BGR format)
        """
        if self.writer is None:
            raise RuntimeError("Video writer not opened. Call open() first.")
            
        # Resize if necessary
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
            
        self.writer.write(frame)
        self.frame_count += 1
        
    @classmethod
    def from_video_info(
        cls,
        output_path: str | Path,
        info: VideoInfo,
        codec: str = "mp4v"
    ) -> "VideoWriter":
        """Create a VideoWriter matching source video properties."""
        return cls(
            output_path=output_path,
            fps=info.fps,
            width=info.width,
            height=info.height,
            codec=codec
        )


def get_video_info(video_path: str | Path) -> VideoInfo:
    """
    Get video metadata without keeping the file open.
    
    Args:
        video_path: Path to video file
        
    Returns:
        VideoInfo dataclass with video metadata
    """
    with VideoReader(video_path) as reader:
        return reader.info





