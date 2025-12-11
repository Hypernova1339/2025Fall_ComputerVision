"""
Event aggregation - convert frame detections to ability cast events.

Uses a state machine to track ability lifecycle:
    IDLE -> CAST_DETECTED (projectile seen)
    CAST_DETECTED -> HIT (impact seen within window)
    CAST_DETECTED -> MISS (window expires without impact)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict
import json
from pathlib import Path


class AbilityState(Enum):
    """State of ability tracking."""
    IDLE = "idle"
    PROJECTILE_DETECTED = "projectile_detected"
    COMPLETED = "completed"


@dataclass
class AbilityEvent:
    """Detected ability cast event."""
    event_id: int
    ability: str  # "W" or "R"
    cast_frame: int
    cast_time_sec: float
    impact_frame: Optional[int] = None
    impact_time_sec: Optional[float] = None
    hit: Optional[bool] = None
    projectile_confidence: float = 0.0
    impact_confidence: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "ability": self.ability,
            "cast_frame": self.cast_frame,
            "cast_time": self.cast_time_sec,
            "impact_frame": self.impact_frame,
            "impact_time": self.impact_time_sec,
            "hit": self.hit,
            "confidence": max(self.projectile_confidence, self.impact_confidence),
            "projectile_confidence": self.projectile_confidence,
            "impact_confidence": self.impact_confidence,
        }
        
    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class AbilityTracker:
    """Track state for a single ability type."""
    ability: str  # "W" or "R"
    projectile_class_id: int
    impact_class_id: int
    
    # Time windows (seconds)
    min_impact_delay: float = 0.1  # Min time from cast to impact
    max_impact_delay: float = 2.0  # Max time to wait for impact
    
    # Cooldown (seconds) - minimum time between casts
    cooldown: float = 0.5
    
    # Internal state
    state: AbilityState = field(default=AbilityState.IDLE)
    cast_frame: int = 0
    cast_time: float = 0.0
    projectile_conf: float = 0.0
    last_completed_time: float = -999.0
    
    def reset(self) -> None:
        """Reset to idle state."""
        self.state = AbilityState.IDLE
        self.cast_frame = 0
        self.cast_time = 0.0
        self.projectile_conf = 0.0


class EventAggregator:
    """
    Aggregate frame-level detections into ability events.
    
    Tracks W and R abilities separately using state machines.
    """
    
    # Default time windows for each ability
    DEFAULT_CONFIG = {
        "W": {
            "projectile_class_id": 0,
            "impact_class_id": 1,
            "min_impact_delay": 0.1,
            "max_impact_delay": 1.5,  # W is fast
            "cooldown": 0.3,
        },
        "R": {
            "projectile_class_id": 2,
            "impact_class_id": 3,
            "min_impact_delay": 0.2,
            "max_impact_delay": 8.0,  # R can travel far
            "cooldown": 1.0,
        },
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize event aggregator.
        
        Args:
            config: Optional custom configuration for abilities
        """
        self.config = config or self.DEFAULT_CONFIG
        self.trackers: Dict[str, AbilityTracker] = {}
        self.events: List[AbilityEvent] = []
        self.event_counter = 0
        
        # Initialize trackers
        for ability, cfg in self.config.items():
            self.trackers[ability] = AbilityTracker(
                ability=ability,
                projectile_class_id=cfg["projectile_class_id"],
                impact_class_id=cfg["impact_class_id"],
                min_impact_delay=cfg["min_impact_delay"],
                max_impact_delay=cfg["max_impact_delay"],
                cooldown=cfg["cooldown"],
            )
            
    def reset(self) -> None:
        """Reset all trackers and events."""
        for tracker in self.trackers.values():
            tracker.reset()
        self.events = []
        self.event_counter = 0
        
    def process_frame(
        self,
        frame_idx: int,
        timestamp_sec: float,
        detections: List[dict]
    ) -> List[AbilityEvent]:
        """
        Process detections for a single frame.
        
        Args:
            frame_idx: Frame index
            timestamp_sec: Frame timestamp in seconds
            detections: List of detection dicts with 'class_id' and 'confidence'
            
        Returns:
            List of newly completed events (if any)
        """
        completed_events = []
        
        # Group detections by class
        det_by_class = {}
        for det in detections:
            class_id = det["class_id"]
            conf = det["confidence"]
            if class_id not in det_by_class or conf > det_by_class[class_id]:
                det_by_class[class_id] = conf
                
        # Process each ability tracker
        for ability, tracker in self.trackers.items():
            event = self._process_tracker(
                tracker,
                frame_idx,
                timestamp_sec,
                det_by_class
            )
            if event:
                completed_events.append(event)
                
        return completed_events
        
    def _process_tracker(
        self,
        tracker: AbilityTracker,
        frame_idx: int,
        timestamp: float,
        det_by_class: Dict[int, float]
    ) -> Optional[AbilityEvent]:
        """Process a single ability tracker."""
        
        has_projectile = tracker.projectile_class_id in det_by_class
        has_impact = tracker.impact_class_id in det_by_class
        proj_conf = det_by_class.get(tracker.projectile_class_id, 0.0)
        impact_conf = det_by_class.get(tracker.impact_class_id, 0.0)
        
        completed_event = None
        
        if tracker.state == AbilityState.IDLE:
            # Look for new projectile
            if has_projectile:
                # Check cooldown
                if timestamp - tracker.last_completed_time >= tracker.cooldown:
                    tracker.state = AbilityState.PROJECTILE_DETECTED
                    tracker.cast_frame = frame_idx
                    tracker.cast_time = timestamp
                    tracker.projectile_conf = proj_conf
                    
        elif tracker.state == AbilityState.PROJECTILE_DETECTED:
            time_since_cast = timestamp - tracker.cast_time
            
            # Check for impact within valid window
            if has_impact and time_since_cast >= tracker.min_impact_delay:
                # HIT detected
                self.event_counter += 1
                completed_event = AbilityEvent(
                    event_id=self.event_counter,
                    ability=tracker.ability,
                    cast_frame=tracker.cast_frame,
                    cast_time_sec=tracker.cast_time,
                    impact_frame=frame_idx,
                    impact_time_sec=timestamp,
                    hit=True,
                    projectile_confidence=tracker.projectile_conf,
                    impact_confidence=impact_conf,
                )
                self.events.append(completed_event)
                tracker.last_completed_time = timestamp
                tracker.reset()
                
            elif time_since_cast > tracker.max_impact_delay:
                # Timeout - MISS
                self.event_counter += 1
                completed_event = AbilityEvent(
                    event_id=self.event_counter,
                    ability=tracker.ability,
                    cast_frame=tracker.cast_frame,
                    cast_time_sec=tracker.cast_time,
                    impact_frame=None,
                    impact_time_sec=None,
                    hit=False,
                    projectile_confidence=tracker.projectile_conf,
                    impact_confidence=0.0,
                )
                self.events.append(completed_event)
                tracker.last_completed_time = timestamp
                tracker.reset()
                
            # Keep tracking projectile confidence (use max seen)
            if has_projectile and proj_conf > tracker.projectile_conf:
                tracker.projectile_conf = proj_conf
                
        return completed_event
        
    def finalize(self, final_timestamp: float) -> List[AbilityEvent]:
        """
        Finalize any pending events at end of video.
        
        Args:
            final_timestamp: Final frame timestamp
            
        Returns:
            List of events that were pending
        """
        finalized = []
        
        for tracker in self.trackers.values():
            if tracker.state == AbilityState.PROJECTILE_DETECTED:
                # Mark as miss (no impact seen)
                self.event_counter += 1
                event = AbilityEvent(
                    event_id=self.event_counter,
                    ability=tracker.ability,
                    cast_frame=tracker.cast_frame,
                    cast_time_sec=tracker.cast_time,
                    impact_frame=None,
                    impact_time_sec=None,
                    hit=False,
                    projectile_confidence=tracker.projectile_conf,
                    impact_confidence=0.0,
                )
                self.events.append(event)
                finalized.append(event)
                tracker.reset()
                
        return finalized
        
    def get_events(self) -> List[AbilityEvent]:
        """Get all detected events."""
        return self.events
        
    def get_summary(self) -> dict:
        """Get summary statistics."""
        w_events = [e for e in self.events if e.ability == "W"]
        r_events = [e for e in self.events if e.ability == "R"]
        
        return {
            "total_events": len(self.events),
            "W": {
                "total": len(w_events),
                "hits": sum(1 for e in w_events if e.hit),
                "misses": sum(1 for e in w_events if not e.hit),
                "hit_rate": sum(1 for e in w_events if e.hit) / max(len(w_events), 1),
            },
            "R": {
                "total": len(r_events),
                "hits": sum(1 for e in r_events if e.hit),
                "misses": sum(1 for e in r_events if not e.hit),
                "hit_rate": sum(1 for e in r_events if e.hit) / max(len(r_events), 1),
            },
        }
        
    def save_events(self, output_path: str | Path) -> None:
        """Save events to JSONL file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            for event in self.events:
                f.write(event.to_jsonl() + "\n")
                
        print(f"Saved {len(self.events)} events to: {output_path}")
        
    @staticmethod
    def load_events(jsonl_path: str | Path) -> List[AbilityEvent]:
        """Load events from JSONL file."""
        events = []
        
        with open(jsonl_path, "r") as f:
            for line in f:
                data = json.loads(line)
                event = AbilityEvent(
                    event_id=data["event_id"],
                    ability=data["ability"],
                    cast_frame=data["cast_frame"],
                    cast_time_sec=data["cast_time"],
                    impact_frame=data.get("impact_frame"),
                    impact_time_sec=data.get("impact_time"),
                    hit=data.get("hit"),
                    projectile_confidence=data.get("projectile_confidence", 0),
                    impact_confidence=data.get("impact_confidence", 0),
                )
                events.append(event)
                
        return events





