"""solution.py - interface used by the organizers' harness (run_submission.py)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.rules_stopped import stopped_vehicle_events
from src.tracking import get_tracks
from src.rules_jaywalking import jaywalking_events

CLASSES: list[str] = [
    "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
    "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
    "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke",
]

RISK_HORIZON_SEC = 5.0
ZONES = str(Path(__file__).parent / "zones.json")


def detect_events(video_path: str) -> list[list]:
    """Part A: return [[start_sec, end_sec, label], ...] for one video."""
    cap = cv2.VideoCapture(video_path)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 25.0)
    cap.release()

    tracks_csv = get_tracks(video_path)
    events = []
    events += stopped_vehicle_events(tracks_csv, ZONES, duration)
    events += jaywalking_events(tracks_csv, ZONES, duration)
    return events


class RiskEstimator:
    """Part B (not built yet): always returns zero risk."""

    def reset(self, meta: dict) -> None:
        self.meta = meta

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        return 0.0