#!/usr/bin/env python3
"""
Interactive Video Labeling Tool for Jinx Ability Detection.

A keyboard-driven tool to quickly annotate W and R ability casts while watching gameplay.

Usage:
    python tools/label_video.py --video "data/raw_videos/video 1 (long).mp4"

Recommended Workflow:
    1. Watch at 1.5x speed (press '2')
    2. When you see an ability, slow down (press '3' for 0.5x)
    3. Find the exact frame of the ability cast
    4. Press 'W' or 'R' to log the cast
    5. Watch for the outcome
    6. Press 'H' for hit or 'M' for miss
    7. Speed back up (press '2') and continue

Controls:
    W/R      - Log W/R cast at current time
    H        - Mark last cast as HIT
    M        - Mark last cast as MISS
    U        - Undo last entry
    Space    - Pause/Resume playback
    
    ←/→      - Seek ±1 second
    ,/.      - Fine step (1 frame when paused, 0.1s when playing)
    
    2        - 1.5x speed (watching)
    3        - 0.5x speed (slow-mo)
    4        - 0.25x speed (very slow)
    1        - 1x speed (normal)
    -/+      - Adjust speed gradually
    
    S        - Save events to CSV
    Q        - Save and quit
    Esc      - Quit without saving
"""

import argparse
import csv
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import cv2
import numpy as np


@dataclass
class AbilityEvent:
    """Single ability event with multiple frames."""
    ability: str  # "W", "R", or "IDLE"
    cast_frames: List[int] = field(default_factory=list)  # Multiple frames
    cast_times: List[float] = field(default_factory=list)  # Multiple timestamps
    result: Optional[str] = None  # "hit", "miss", or None
    impact_time_sec: Optional[float] = None
    impact_frame: Optional[int] = None
    closed: bool = False  # Whether event is closed (H/M pressed)
    notes: str = ""
    
    def add_frame(self, frame: int, time_sec: float) -> None:
        """Add a frame to this event."""
        self.cast_frames.append(frame)
        self.cast_times.append(time_sec)
    
    @property
    def frame_count(self) -> int:
        return len(self.cast_frames)
    
    @property
    def first_time(self) -> float:
        return self.cast_times[0] if self.cast_times else 0.0
    
    def to_row(self, video_id: str) -> dict:
        return {
            "video_id": video_id,
            "ability": self.ability,
            "cast_frames": ";".join(str(f) for f in self.cast_frames),
            "cast_times": ";".join(f"{t:.2f}" for t in self.cast_times),
            "result": self.result or "",
            "impact_time_sec": f"{self.impact_time_sec:.2f}" if self.impact_time_sec else "",
            "impact_frame": self.impact_frame if self.impact_frame else "",
            "notes": self.notes,
        }


class VideoLabeler:
    """Interactive video labeling tool."""
    
    # Playback speed options
    SPEEDS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 8.0]
    
    def __init__(self, video_path: str, output_csv: str):
        self.video_path = Path(video_path)
        self.output_csv = Path(output_csv)
        self.video_id = self.video_path.stem
        
        # Video state
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps = 60.0
        self.total_frames = 0
        self.duration_sec = 0.0
        self.current_frame = 0
        self.paused = True  # Start paused
        self.speed_idx = 3  # 1.0x
        
        # Events
        self.events: List[AbilityEvent] = []
        self.unsaved_changes = False
        
        # Current cast being recorded (multi-frame)
        self.current_cast: Optional[AbilityEvent] = None
        
        # Target counts for auto-completion
        self.targets = {
            "IDLE": 100,
            "W": 100,
            "R": 100,
        }
        
        # Display settings
        self.window_name = "Jinx Ability Labeler"
        self.display_width = 1280
        self.display_height = 720
        
        
    def load_video(self) -> bool:
        """Load the video file."""
        if not self.video_path.exists():
            print(f"Error: Video not found: {self.video_path}")
            return False
            
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            print(f"Error: Could not open video: {self.video_path}")
            return False
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 60.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_sec = self.total_frames / self.fps
        
        print(f"Loaded: {self.video_path.name}")
        print(f"  Resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"  FPS: {self.fps:.1f}")
        print(f"  Duration: {self._format_time(self.duration_sec)}")
        print(f"  Total frames: {self.total_frames}")
        
        return True
        
    def load_existing_events(self) -> None:
        """Load existing events from CSV if it exists."""
        if not self.output_csv.exists():
            return
            
        try:
            with open(self.output_csv, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("video_id") == self.video_id:
                        # Parse frames and times (semicolon-separated)
                        cast_frames_str = row.get("cast_frames", "")
                        cast_times_str = row.get("cast_times", "")
                        
                        cast_frames = [int(f) for f in cast_frames_str.split(";") if f] if cast_frames_str else []
                        cast_times = [float(t) for t in cast_times_str.split(";") if t] if cast_times_str else []
                        
                        impact_time = float(row["impact_time_sec"]) if row.get("impact_time_sec") else None
                        impact_frame = int(row["impact_frame"]) if row.get("impact_frame") else None
                        
                        event = AbilityEvent(
                            ability=row["ability"],
                            cast_frames=cast_frames,
                            cast_times=cast_times,
                            result=row.get("result") or None,
                            impact_time_sec=impact_time,
                            impact_frame=impact_frame,
                            closed=True,  # Loaded events are already closed
                            notes=row.get("notes", ""),
                        )
                        self.events.append(event)
                        
            if self.events:
                print(f"Loaded {len(self.events)} existing events for this video")
        except Exception as e:
            print(f"Warning: Could not load existing events: {e}")
            
    def save_events(self) -> None:
        """Save events to CSV."""
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing events for other videos
        other_events = []
        if self.output_csv.exists():
            try:
                with open(self.output_csv, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get("video_id") != self.video_id:
                            other_events.append(row)
            except Exception:
                pass
                
        # Write all events
        fieldnames = ["video_id", "ability", "cast_frames", "cast_times", "result", "impact_time_sec", "impact_frame", "notes"]
        
        with open(self.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # Write other videos' events
            for row in other_events:
                # Ensure row has all fieldnames (for backwards compatibility)
                for field in fieldnames:
                    if field not in row:
                        row[field] = ""
                writer.writerow(row)
                
            # Write this video's CLOSED events only (sorted by first time)
            closed_events = [e for e in self.events if e.closed]
            for event in sorted(closed_events, key=lambda e: e.first_time):
                writer.writerow(event.to_row(self.video_id))
                
        self.unsaved_changes = False
        closed_count = len([e for e in self.events if e.closed])
        print(f"Saved {closed_count} events to: {self.output_csv}")
        
    def _format_time(self, seconds: float) -> str:
        """Format seconds as MM:SS.d"""
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins:02d}:{secs:05.2f}"
        
        
    def _get_current_time(self) -> float:
        """Get current playback time in seconds."""
        return self.current_frame / self.fps
        
    def _seek_to_time(self, time_sec: float) -> None:
        """Seek to a specific time."""
        time_sec = max(0, min(time_sec, self.duration_sec))
        self.current_frame = int(time_sec * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        
    def _seek_relative(self, delta_sec: float) -> None:
        """Seek relative to current position."""
        self._seek_to_time(self._get_current_time() + delta_sec)
        
    def _get_counts(self) -> Dict[str, int]:
        """Get count of CLOSED events by ability type."""
        counts = {"IDLE": 0, "W": 0, "R": 0}
        for event in self.events:
            if event.closed and event.ability in counts:
                counts[event.ability] += 1
        return counts
        
    def _check_targets_met(self) -> bool:
        """Check if all target counts have been met."""
        counts = self._get_counts()
        for ability, target in self.targets.items():
            if counts.get(ability, 0) < target:
                return False
        return True
        
    def _add_frame_to_cast(self, ability: str) -> None:
        """Add a frame to current cast or start a new cast."""
        current_time = self._get_current_time()
        current_frame = self.current_frame
        
        # Check if we have an active cast of the same type
        if self.current_cast is not None:
            if self.current_cast.ability == ability:
                # Add frame to existing cast
                self.current_cast.add_frame(current_frame, current_time)
                print(f"  + {ability} frame {self.current_cast.frame_count} @ {self._format_time(current_time)} (frame {current_frame})")
            else:
                # Different ability - close current cast as incomplete and start new one
                print(f"  ! Closing incomplete {self.current_cast.ability} cast (no H/M)")
                self.current_cast.closed = True
                self.current_cast.result = "incomplete"
                
                # Start new cast
                self.current_cast = AbilityEvent(ability=ability)
                self.current_cast.add_frame(current_frame, current_time)
                self.events.append(self.current_cast)
                print(f"  + NEW {ability} cast - frame 1 @ {self._format_time(current_time)} (frame {current_frame})")
        else:
            # Start new cast
            self.current_cast = AbilityEvent(ability=ability)
            self.current_cast.add_frame(current_frame, current_time)
            self.events.append(self.current_cast)
            print(f"  + NEW {ability} cast - frame 1 @ {self._format_time(current_time)} (frame {current_frame})")
        
        self.unsaved_changes = True
    
    def _close_current_cast(self, result: str) -> bool:
        """Close current cast with hit/miss result. Returns True if all targets met."""
        if self.current_cast is None:
            print("  ! No active cast to close")
            return False
        
        self.current_cast.result = result
        self.current_cast.closed = True
        
        # Capture impact frame for hits
        if result == "hit":
            self.current_cast.impact_time_sec = self._get_current_time()
            self.current_cast.impact_frame = self.current_frame
            print(f"  ✓ {self.current_cast.ability} HIT - {self.current_cast.frame_count} cast frames + impact @ frame {self.current_cast.impact_frame}")
        else:
            print(f"  ✗ {self.current_cast.ability} MISS - {self.current_cast.frame_count} cast frames")
        
        # Get updated counts
        counts = self._get_counts()
        ability = self.current_cast.ability
        target = self.targets.get(ability, 0)
        current = counts.get(ability, 0)
        print(f"    Progress: {ability} [{current}/{target}]")
        
        # Clear current cast
        self.current_cast = None
        self.unsaved_changes = True
        
        # Check if all targets met
        return self._check_targets_met()
        
        
    def _undo_last(self) -> None:
        """Undo last action - remove frame from current cast or remove last closed event."""
        if self.current_cast is not None:
            # Remove last frame from current cast
            if self.current_cast.frame_count > 1:
                removed_frame = self.current_cast.cast_frames.pop()
                removed_time = self.current_cast.cast_times.pop()
                print(f"  - Removed frame {removed_frame} from current {self.current_cast.ability} cast ({self.current_cast.frame_count} frames remain)")
            else:
                # Only one frame, cancel the whole cast
                print(f"  - Cancelled {self.current_cast.ability} cast")
                self.events.remove(self.current_cast)
                self.current_cast = None
        elif self.events:
            # Remove last closed event
            closed_events = [e for e in self.events if e.closed]
            if closed_events:
                last_closed = closed_events[-1]
                self.events.remove(last_closed)
                print(f"  - Removed {last_closed.ability} event ({last_closed.frame_count} frames)")
            else:
                print("  Nothing to undo")
        else:
            print("  Nothing to undo")
            
        self.unsaved_changes = True
        
    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw status overlay on frame."""
        # Resize frame for display
        h, w = frame.shape[:2]
        scale = min(self.display_width / w, self.display_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))
        
        # Create overlay area at bottom
        overlay_height = 130
        overlay = np.zeros((overlay_height, new_w, 3), dtype=np.uint8)
        overlay[:] = (40, 40, 40)  # Dark gray background
        
        # Current time and progress
        current_time = self._get_current_time()
        progress = current_time / max(self.duration_sec, 1)
        
        # Progress bar
        bar_y = 15
        bar_height = 8
        cv2.rectangle(overlay, (10, bar_y), (new_w - 10, bar_y + bar_height), (80, 80, 80), -1)
        progress_width = int((new_w - 20) * progress)
        cv2.rectangle(overlay, (10, bar_y), (10 + progress_width, bar_y + bar_height), (0, 200, 255), -1)
        
        # Draw event markers on progress bar
        for event in self.events:
            if not event.cast_times:
                continue
            event_progress = event.first_time / max(self.duration_sec, 1)
            marker_x = 10 + int((new_w - 20) * event_progress)
            color = (0, 255, 0) if event.result == "hit" else (0, 0, 255) if event.result == "miss" else (255, 255, 0)
            cv2.line(overlay, (marker_x, bar_y - 3), (marker_x, bar_y + bar_height + 3), color, 2)
        
        # Time display
        time_text = f"{self._format_time(current_time)} / {self._format_time(self.duration_sec)}"
        speed_text = f"{self.SPEEDS[self.speed_idx]:.2f}x"
        status_text = "PAUSED" if self.paused else "PLAYING"
        
        cv2.putText(overlay, time_text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        # Speed display - highlight if not 1x
        speed_color = (0, 255, 255) if self.SPEEDS[self.speed_idx] != 1.0 else (200, 200, 200)
        cv2.putText(overlay, speed_text, (200, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, speed_color, 1)
        cv2.putText(overlay, status_text, (300, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                    (0, 255, 255) if self.paused else (0, 255, 0), 1)
        
        # Progress tracker
        counts = self._get_counts()
        
        # Draw progress bars for each type
        progress_x = 420
        for ability, target in self.targets.items():
            current = counts.get(ability, 0)
            progress = min(current / target, 1.0)
            
            # Color: green if complete, yellow if in progress
            if current >= target:
                color = (0, 255, 0)  # Green - complete
                text_color = (0, 255, 0)
            else:
                color = (0, 200, 255)  # Yellow - in progress
                text_color = (255, 255, 255)
            
            # Draw label and count
            label = f"{ability}:{current}/{target}"
            cv2.putText(overlay, label, (progress_x, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)
            
            # Draw mini progress bar
            bar_x = progress_x
            bar_y = 48
            bar_w = 70
            bar_h = 6
            cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
            fill_w = int(bar_w * progress)
            if fill_w > 0:
                cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
            
            progress_x += 90
        
        # Unsaved indicator
        if self.unsaved_changes:
            cv2.putText(overlay, "*UNSAVED*", (new_w - 100, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Current cast status
        if self.current_cast is not None:
            cast_text = f"RECORDING: {self.current_cast.ability} [{self.current_cast.frame_count} frames] - Press H (hit) or M (miss) to close"
            cv2.putText(overlay, cast_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        else:
            # Show last closed event
            closed_events = [e for e in self.events if e.closed]
            if closed_events:
                last = closed_events[-1]
                result_str = f" ({last.result})" if last.result else ""
                last_text = f"Last: {last.ability}{result_str} - {last.frame_count} frames"
                cv2.putText(overlay, last_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        # Controls help
        controls1 = "[W/R/E] Log  [H]it [M]iss  [U]ndo  [Space] Pause  [Arrows] +/-1s  [,/.] Fine"
        controls2 = "[2] 1.5x  [3] 0.5x  [4] 0.25x  [1] 1x  [S]ave  [Q]uit"
        cv2.putText(overlay, controls1, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        cv2.putText(overlay, controls2, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        
        # Combine frame and overlay
        combined = np.vstack([frame, overlay])
        
        return combined
        
    def run(self) -> None:
        """Run the labeling tool."""
        if not self.load_video():
            return
            
        self.load_existing_events()
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.display_width, self.display_height + 130)
        
        counts = self._get_counts()
        
        print("\n" + "="*60)
        print("JINX ABILITY LABELER - Multi-Frame Mode")
        print("="*60)
        print("")
        print("TARGETS (auto-saves when all complete):")
        for ability, target in self.targets.items():
            current = counts.get(ability, 0)
            status = "✓ DONE" if current >= target else f"{current}/{target}"
            print(f"  {ability}: {status}")
        print("")
        print("WORKFLOW:")
        print("  1. Watch at 1.5x speed [press 2]")
        print("  2. See ability -> slow down [press 3]")
        print("  3. Press W/R multiple times to capture frames")
        print("  4. Watch for hit/miss outcome")
        print("  5. Press H (hit) or M (miss) to close the event")
        print("")
        print("CONTROLS:")
        print("  W        - Add frame to W cast (multiple times OK)")
        print("  R        - Add frame to R cast (multiple times OK)")
        print("  E        - Log IDLE frame (auto-closes)")
        print("  H        - Close cast as HIT (captures impact frame)")
        print("  M        - Close cast as MISS")
        print("  U        - Undo")
        print("  Space    - Pause/Resume")
        print("")
        print("  NAV:   Arrows=+/-1s  ,/.=fine step")
        print("  SPEED: 2=1.5x  3=0.5x  4=0.25x  1=1x")
        print("  S=Save  Q=Save+Quit  Esc=Quit")
        print("="*60 + "\n")
        
        # Start paused at beginning
        self.paused = True
        frame_delay = int(1000 / self.fps)  # Base delay for 1x speed
        current_frame_img = None  # Cache current frame when paused
        
        while True:
            # Determine delay
            if not self.paused:
                # Adjust delay for playback speed
                current_speed = self.SPEEDS[self.speed_idx]
                adjusted_delay = max(1, int(frame_delay / current_speed))
            else:
                adjusted_delay = 30  # Responsive when paused
            
            # Only read new frame if playing
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    # End of video, loop back
                    self._seek_to_time(0)
                    continue
                current_frame_img = frame.copy()
                self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            else:
                # When paused, use cached frame or read current position
                if current_frame_img is None:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                    ret, frame = self.cap.read()
                    if ret:
                        current_frame_img = frame.copy()
                        # Reset position so next read gets same frame
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                    else:
                        continue
                frame = current_frame_img
            
            # Draw overlay and display
            display_frame = self._draw_overlay(frame)
            cv2.imshow(self.window_name, display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(adjusted_delay) & 0xFF
            
            if key == 255:  # No key pressed
                continue
            elif key == ord('q'):  # Quit and save
                if self.unsaved_changes:
                    self.save_events()
                break
            elif key == 27:  # Esc - quit without saving
                if self.unsaved_changes:
                    print("Warning: Quitting without saving!")
                break
            elif key == ord('s'):  # Save
                self.save_events()
            elif key == ord(' '):  # Pause/Resume
                self.paused = not self.paused
                if self.paused:
                    # When pausing, make sure we're at the right frame
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                else:
                    # When resuming, clear cached frame
                    current_frame_img = None
            elif key == ord('w'):  # W cast frame
                self._add_frame_to_cast("W")
            elif key == ord('r'):  # R cast frame
                self._add_frame_to_cast("R")
            elif key == ord('e'):  # Idle/No ability (negative sample) - auto-closes
                self._add_frame_to_cast("IDLE")
                # IDLE events auto-close immediately (single frame)
                if self._close_current_cast("none"):
                    print("\n" + "="*60)
                    print("🎉 ALL TARGETS MET! Auto-saving and exiting...")
                    print("="*60)
                    self.save_events()
                    break
            elif key == ord('h'):  # Hit - closes current cast
                if self._close_current_cast("hit"):
                    print("\n" + "="*60)
                    print("🎉 ALL TARGETS MET! Auto-saving and exiting...")
                    print("="*60)
                    self.save_events()
                    break
            elif key == ord('m'):  # Miss - closes current cast
                if self._close_current_cast("miss"):
                    print("\n" + "="*60)
                    print("🎉 ALL TARGETS MET! Auto-saving and exiting...")
                    print("="*60)
                    self.save_events()
                    break
            elif key == ord('u'):  # Undo
                self._undo_last()
            elif key == 81 or key == 2:  # Left arrow - 1 second back
                self._seek_relative(-1.0)
                current_frame_img = None  # Clear cache to read new frame
            elif key == 83 or key == 3:  # Right arrow - 1 second forward
                self._seek_relative(1.0)
                current_frame_img = None  # Clear cache to read new frame
            elif key == ord(','):  # Fine step back (1 frame when paused, 0.1s when playing)
                step = -1/self.fps if self.paused else -0.1
                self._seek_relative(step)
                current_frame_img = None  # Clear cache to read new frame
            elif key == ord('.'):  # Fine step forward (1 frame when paused, 0.1s when playing)
                step = 1/self.fps if self.paused else 0.1
                self._seek_relative(step)
                current_frame_img = None  # Clear cache to read new frame
            elif key == ord('[') or key == ord('-'):  # Slower
                self.speed_idx = max(0, self.speed_idx - 1)
                print(f"  Speed: {self.SPEEDS[self.speed_idx]}x")
            elif key == ord(']') or key == ord('=') or key == ord('+'):  # Faster
                self.speed_idx = min(len(self.SPEEDS) - 1, self.speed_idx + 1)
                print(f"  Speed: {self.SPEEDS[self.speed_idx]}x")
            elif key == ord('1'):  # 1x speed (normal)
                self.speed_idx = self.SPEEDS.index(1.0) if 1.0 in self.SPEEDS else 3
                print(f"  Speed: {self.SPEEDS[self.speed_idx]}x")
            elif key == ord('2'):  # 1.5x speed (watching speed)
                self.speed_idx = self.SPEEDS.index(1.5) if 1.5 in self.SPEEDS else 5
                print(f"  Speed: {self.SPEEDS[self.speed_idx]}x")
            elif key == ord('3'):  # 0.5x speed (slow-mo for catching abilities)
                self.speed_idx = self.SPEEDS.index(0.5) if 0.5 in self.SPEEDS else 1
                print(f"  Speed: {self.SPEEDS[self.speed_idx]}x")
            elif key == ord('4'):  # 0.25x speed (very slow for precise frame)
                self.speed_idx = self.SPEEDS.index(0.25) if 0.25 in self.SPEEDS else 0
                print(f"  Speed: {self.SPEEDS[self.speed_idx]}x")
                
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("\nLabeling session ended.")
        print(f"Total events logged: {len(self.events)}")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive video labeling tool for Jinx ability detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Targets:
    The tool tracks progress toward labeling targets.
    Default: 100 IDLE, 100 W, 100 R
    Auto-saves and exits when all targets are met.

Controls:
    W/R/E    - Log W cast / R cast / IDLE (no ability)
    H/M      - Mark last cast as HIT / MISS
    U        - Undo last entry
    Space    - Pause/Resume
    ←/→      - Seek ±1 second
    ,/.      - Fine step (frame-by-frame when paused)
    2/3/4/1  - Speed: 1.5x / 0.5x / 0.25x / 1x
    S        - Save events
    Q        - Save and quit

Examples:
    python tools/label_video.py --video "data/raw_videos/video 1 (long).mp4"
    
    # Custom targets
    python tools/label_video.py --video gameplay.mp4 --idle 50 --w 50 --r 50
        """
    )
    
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to video file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/events/events.csv",
        help="Output CSV file (default: data/events/events.csv)"
    )
    
    parser.add_argument(
        "--idle",
        type=int,
        default=100,
        help="Target count for IDLE events (default: 100)"
    )
    
    parser.add_argument(
        "--w",
        type=int,
        default=100,
        help="Target count for W events (default: 100)"
    )
    
    parser.add_argument(
        "--r",
        type=int,
        default=100,
        help="Target count for R events (default: 100)"
    )
    
    args = parser.parse_args()
    
    labeler = VideoLabeler(args.video, args.output)
    labeler.targets = {
        "IDLE": args.idle,
        "W": args.w,
        "R": args.r,
    }
    labeler.run()


if __name__ == "__main__":
    main()

