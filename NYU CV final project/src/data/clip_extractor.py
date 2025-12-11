"""Extract short clips around ability events from gameplay videos."""

import csv
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class AbilityEvent:
    """Represents a single ability cast event."""
    video_id: str
    ability: str  # "W" or "R"
    cast_time_sec: float
    notes: str = ""
    
    @property
    def event_id(self) -> str:
        """Generate unique event ID."""
        return f"{self.video_id}_{self.ability}_{self.cast_time_sec:.1f}"


# Default clip windows (seconds before/after cast)
CLIP_WINDOWS = {
    "W": (-0.5, 2.0),   # Zap: 0.5s before to 2s after
    "R": (-0.5, 4.0),   # Rocket: 0.5s before to 4s after
}


class ClipExtractor:
    """Extract short video clips around ability events."""
    
    def __init__(
        self,
        clip_windows: Optional[Dict[str, tuple]] = None,
        output_format: str = "mp4"
    ):
        """
        Initialize clip extractor.
        
        Args:
            clip_windows: Dict mapping ability to (pre_sec, post_sec) tuple
            output_format: Output video format
        """
        self.clip_windows = clip_windows or CLIP_WINDOWS
        self.output_format = output_format
        
    def load_events(self, events_csv: str | Path) -> List[AbilityEvent]:
        """
        Load ability events from CSV file.
        
        Expected CSV format:
            video_id,ability,cast_time_sec,notes
            
        Args:
            events_csv: Path to events CSV file
            
        Returns:
            List of AbilityEvent objects
        """
        events = []
        csv_path = Path(events_csv)
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Events file not found: {csv_path}")
            
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                event = AbilityEvent(
                    video_id=row["video_id"],
                    ability=row["ability"].upper(),
                    cast_time_sec=float(row["cast_time_sec"]),
                    notes=row.get("notes", "")
                )
                events.append(event)
                
        return events
        
    def get_clip_times(self, event: AbilityEvent) -> tuple:
        """
        Get start and end times for a clip.
        
        Args:
            event: AbilityEvent object
            
        Returns:
            Tuple of (start_sec, end_sec)
        """
        pre_sec, post_sec = self.clip_windows.get(event.ability, (-1.0, 3.0))
        start = max(0, event.cast_time_sec + pre_sec)
        end = event.cast_time_sec + post_sec
        return start, end
        
    def extract_clip_ffmpeg(
        self,
        video_path: str | Path,
        output_path: str | Path,
        start_sec: float,
        end_sec: float,
        copy_codec: bool = True
    ) -> bool:
        """
        Extract a clip using FFmpeg (fast, no re-encoding if copy_codec=True).
        
        Args:
            video_path: Path to source video
            output_path: Path for output clip
            start_sec: Start time in seconds
            end_sec: End time in seconds
            copy_codec: If True, use stream copy (fast but less precise)
            
        Returns:
            True if successful
        """
        video_path = Path(video_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        duration = end_sec - start_sec
        
        if copy_codec:
            # Fast extraction with stream copy
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start_sec),
                "-i", str(video_path),
                "-t", str(duration),
                "-c", "copy",
                str(output_path)
            ]
        else:
            # Re-encode for precise cuts
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start_sec),
                "-i", str(video_path),
                "-t", str(duration),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "18",
                str(output_path)
            ]
            
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e.stderr}")
            return False
        except FileNotFoundError:
            print("FFmpeg not found. Please install FFmpeg.")
            return False
            
    def extract_clip_opencv(
        self,
        video_path: str | Path,
        output_path: str | Path,
        start_sec: float,
        end_sec: float
    ) -> bool:
        """
        Extract a clip using OpenCV (slower but no FFmpeg dependency).
        
        Args:
            video_path: Path to source video
            output_path: Path for output clip
            start_sec: Start time in seconds
            end_sec: End time in seconds
            
        Returns:
            True if successful
        """
        from ..utils.video_utils import VideoReader, VideoWriter
        
        video_path = Path(video_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with VideoReader(video_path) as reader:
                info = reader.info
                start_frame = reader.time_to_frame(start_sec)
                end_frame = reader.time_to_frame(end_sec)
                
                with VideoWriter.from_video_info(output_path, info) as writer:
                    for frame_idx, frame in reader.iter_frames(start_frame, end_frame):
                        writer.write(frame)
                        
            return True
        except Exception as e:
            print(f"OpenCV extraction error: {e}")
            return False
            
    def extract_all_clips(
        self,
        video_dir: str | Path,
        events: List[AbilityEvent],
        output_dir: str | Path,
        use_ffmpeg: bool = True,
        video_extension: str = ".mp4"
    ) -> List[Path]:
        """
        Extract clips for all events.
        
        Args:
            video_dir: Directory containing source videos
            events: List of AbilityEvent objects
            output_dir: Directory for output clips
            use_ffmpeg: Whether to use FFmpeg (faster) or OpenCV
            video_extension: Video file extension
            
        Returns:
            List of paths to extracted clips
        """
        video_dir = Path(video_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        extracted = []
        
        for i, event in enumerate(events):
            # Find source video
            video_path = video_dir / f"{event.video_id}{video_extension}"
            if not video_path.exists():
                # Try with spaces replaced
                video_path = video_dir / f"{event.video_id.replace('_', ' ')}{video_extension}"
                
            if not video_path.exists():
                print(f"Warning: Video not found for {event.video_id}")
                continue
                
            # Generate output filename
            start_sec, end_sec = self.get_clip_times(event)
            output_name = f"{event.video_id}_{event.ability}_{i:03d}.{self.output_format}"
            output_path = output_dir / output_name
            
            print(f"Extracting clip {i+1}/{len(events)}: {output_name}")
            
            # Extract clip
            if use_ffmpeg:
                success = self.extract_clip_ffmpeg(video_path, output_path, start_sec, end_sec)
            else:
                success = self.extract_clip_opencv(video_path, output_path, start_sec, end_sec)
                
            if success:
                extracted.append(output_path)
                
        return extracted


def load_events_csv(csv_path: str | Path) -> List[AbilityEvent]:
    """Convenience function to load events from CSV."""
    extractor = ClipExtractor()
    return extractor.load_events(csv_path)





