"""
Hit/miss classification logic for ability events.

This module provides additional heuristics and rules for determining
whether an ability hit or missed, beyond simple impact detection.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import numpy as np

from .aggregator import AbilityEvent


@dataclass
class HitAnalysis:
    """Detailed hit/miss analysis for an event."""
    event_id: int
    hit: bool
    confidence: float
    reasons: List[str]
    
    # Additional metrics
    projectile_duration_sec: Optional[float] = None
    impact_distance_from_center: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "hit": self.hit,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "projectile_duration_sec": self.projectile_duration_sec,
        }


class HitClassifier:
    """
    Advanced hit/miss classification for Jinx abilities.
    
    Uses multiple heuristics:
    1. Impact detection (primary)
    2. Projectile travel time
    3. Impact location relative to expected target area
    4. Confidence thresholds
    """
    
    # Expected travel times (seconds)
    EXPECTED_TRAVEL_TIME = {
        "W": (0.2, 1.2),  # Min, max for hit
        "R": (0.3, 6.0),  # R can travel far
    }
    
    # Confidence thresholds
    MIN_PROJECTILE_CONF = 0.3
    MIN_IMPACT_CONF = 0.4
    
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        use_travel_time: bool = True
    ):
        """
        Initialize hit classifier.
        
        Args:
            confidence_threshold: Minimum confidence for hit classification
            use_travel_time: Whether to use travel time heuristics
        """
        self.confidence_threshold = confidence_threshold
        self.use_travel_time = use_travel_time
        
    def classify_event(
        self,
        event: AbilityEvent,
        impact_location: Optional[Tuple[float, float]] = None
    ) -> HitAnalysis:
        """
        Classify a single event as hit or miss.
        
        Args:
            event: AbilityEvent to classify
            impact_location: Optional (x, y) normalized coordinates of impact
            
        Returns:
            HitAnalysis with classification result
        """
        reasons = []
        confidence_factors = []
        
        # Factor 1: Impact detection
        has_impact = event.impact_frame is not None
        
        if has_impact:
            reasons.append("Impact VFX detected")
            confidence_factors.append(event.impact_confidence)
        else:
            reasons.append("No impact VFX detected")
            confidence_factors.append(1.0 - event.projectile_confidence * 0.5)
            
        # Factor 2: Travel time analysis
        projectile_duration = None
        if has_impact and event.impact_time_sec and event.cast_time_sec:
            projectile_duration = event.impact_time_sec - event.cast_time_sec
            
            min_time, max_time = self.EXPECTED_TRAVEL_TIME.get(
                event.ability, (0.1, 5.0)
            )
            
            if self.use_travel_time:
                if min_time <= projectile_duration <= max_time:
                    reasons.append(f"Travel time ({projectile_duration:.2f}s) within expected range")
                    confidence_factors.append(0.9)
                elif projectile_duration < min_time:
                    reasons.append(f"Travel time ({projectile_duration:.2f}s) too short")
                    confidence_factors.append(0.6)
                else:
                    reasons.append(f"Travel time ({projectile_duration:.2f}s) longer than typical")
                    confidence_factors.append(0.7)
                    
        # Factor 3: Confidence scores
        if event.projectile_confidence < self.MIN_PROJECTILE_CONF:
            reasons.append(f"Low projectile confidence ({event.projectile_confidence:.2f})")
            confidence_factors.append(0.5)
            
        if has_impact and event.impact_confidence >= self.MIN_IMPACT_CONF:
            reasons.append(f"Strong impact confidence ({event.impact_confidence:.2f})")
            confidence_factors.append(event.impact_confidence)
            
        # Calculate final confidence
        if confidence_factors:
            final_confidence = np.mean(confidence_factors)
        else:
            final_confidence = 0.5
            
        # Make final decision
        hit = has_impact and final_confidence >= self.confidence_threshold
        
        return HitAnalysis(
            event_id=event.event_id,
            hit=hit,
            confidence=final_confidence,
            reasons=reasons,
            projectile_duration_sec=projectile_duration,
        )
        
    def classify_events(
        self,
        events: List[AbilityEvent]
    ) -> List[HitAnalysis]:
        """
        Classify multiple events.
        
        Args:
            events: List of AbilityEvent objects
            
        Returns:
            List of HitAnalysis results
        """
        return [self.classify_event(e) for e in events]
        
    def get_accuracy_stats(
        self,
        predictions: List[HitAnalysis],
        ground_truth: List[bool]
    ) -> dict:
        """
        Calculate accuracy statistics.
        
        Args:
            predictions: List of HitAnalysis predictions
            ground_truth: List of actual hit/miss labels
            
        Returns:
            Dict with accuracy metrics
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
            
        if not predictions:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }
            
        # Calculate metrics
        tp = sum(1 for p, gt in zip(predictions, ground_truth) if p.hit and gt)
        fp = sum(1 for p, gt in zip(predictions, ground_truth) if p.hit and not gt)
        fn = sum(1 for p, gt in zip(predictions, ground_truth) if not p.hit and gt)
        tn = sum(1 for p, gt in zip(predictions, ground_truth) if not p.hit and not gt)
        
        accuracy = (tp + tn) / len(predictions)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        }


def analyze_ability_usage(events: List[AbilityEvent]) -> dict:
    """
    Analyze ability usage patterns.
    
    Args:
        events: List of detected ability events
        
    Returns:
        Dict with usage statistics
    """
    if not events:
        return {"total_events": 0}
        
    w_events = [e for e in events if e.ability == "W"]
    r_events = [e for e in events if e.ability == "R"]
    
    def calc_stats(ability_events: List[AbilityEvent]) -> dict:
        if not ability_events:
            return {
                "count": 0,
                "hits": 0,
                "misses": 0,
                "hit_rate": 0.0,
            }
            
        hits = sum(1 for e in ability_events if e.hit)
        misses = len(ability_events) - hits
        
        # Calculate time between casts
        cast_times = sorted([e.cast_time_sec for e in ability_events])
        if len(cast_times) > 1:
            intervals = [cast_times[i+1] - cast_times[i] for i in range(len(cast_times)-1)]
            avg_interval = np.mean(intervals)
        else:
            avg_interval = None
            
        return {
            "count": len(ability_events),
            "hits": hits,
            "misses": misses,
            "hit_rate": hits / len(ability_events),
            "avg_cast_interval_sec": avg_interval,
        }
        
    total_duration = max(e.cast_time_sec for e in events) - min(e.cast_time_sec for e in events)
    
    return {
        "total_events": len(events),
        "duration_sec": total_duration,
        "events_per_minute": len(events) / max(total_duration / 60, 1),
        "W": calc_stats(w_events),
        "R": calc_stats(r_events),
    }





