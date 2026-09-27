"""solution.py - interface used by the organizers' harness (run_submission.py).

Part A: detector (YOLOv8) + tracker (ByteTrack) + hand-written rules on a
fixed set of zones drawn once on this camera (zones.json). Part B: a small
causal risk estimator using time-to-collision, sudden braking, and
pedestrian-near-vehicle signals from its own lightweight internal tracker.
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

random.seed(0)
np.random.seed(0)
try:
    import torch
    torch.manual_seed(0)
except Exception:
    pass

from src.rules_congestion import congestion_events
from src.rules_jaywalking import jaywalking_events
from src.rules_stopped import stopped_vehicle_events
from src.rules_yield import failure_to_yield_events
from src.risk import RiskEstimator
from src.tracking import get_tracks

CLASSES: list[str] = [
    "stopped_vehicle", "jaywalking", "failure_to_yield", "congestion",
]

ZONES = str(Path(__file__).parent / "zones.json")


def detect_events(video_path: str) -> list[list]:
    """Part A: return [[start_sec, end_sec, label], ...] for one video."""
    cap = cv2.VideoCapture(video_path)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 25.0)
    cap.release()

    tracks_csv = get_tracks(video_path)

    events = []
    try:
        events += stopped_vehicle_events(tracks_csv, ZONES, duration)
    except Exception:
        pass
    try:
        events += jaywalking_events(tracks_csv, ZONES, duration)
    except Exception:
        pass
    try:
        events += failure_to_yield_events(tracks_csv, ZONES, duration)
    except Exception:
        pass
    try:
        events += congestion_events(tracks_csv, ZONES, duration)
    except Exception:
        pass
    return events


__all__ = ["CLASSES", "detect_events", "RiskEstimator"]
