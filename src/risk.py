"""Part B: RiskEstimator - causal, frame-by-frame accident risk score.

Runs a lightweight detector+tracker internally (this is NOT the cached Part-A
tracks: those are computed from the whole video, and Part B must be causal,
seeing only frames it has already received). To stay well inside the time
budget:
  - detection only runs every DETECT_STRIDE frames (skipping and reusing the
    last score in between is explicitly allowed by the task)
  - a hard wall-clock cutoff stops calling the detector for the rest of the
    video once Part B has used its share of the time budget, guaranteeing it
    can never push a video over the 3x-duration limit on its own, regardless
    of how fast/slow the grading hardware turns out to be
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent.parent
WEIGHTS = HERE / "weights" / "yolov8s.pt"
ZONES_PATH = HERE / "zones.json"

DETECT_STRIDE = 10          # run the detector every N frames (speed)
WIDTH = 800                  # detect on a small frame; risk only needs rough positions
IMG_SIZE = 480
CONF = 0.25
KEEP = [0, 1, 2, 3, 5, 7]    # person, bicycle, car, motorcycle, bus, truck
VEHICLES = {"car", "truck", "bus", "motorcycle"}

TTC_DANGER = 3.0             # seconds; below this + closing, treat as high risk
SMOOTH_ALPHA = 0.35          # exponential smoothing factor for the output score
TIME_BUDGET_FRACTION = 0.5   # Part B's own detector time is capped at this x video duration


def _load_zones():
    if not ZONES_PATH.exists():
        return []
    zones = json.load(open(ZONES_PATH))
    return [np.array(z["points"], np.int32) for z in zones if z["type"] == "carriageway"]


class RiskEstimator:
    """Part B (optional). Causal: step() sees frames in order and nothing else."""

    def reset(self, meta: dict) -> None:
        self.meta = meta
        fps = meta.get("fps") or 25.0
        n_frames = meta.get("n_frames") or 0
        duration = n_frames / fps if fps else 0
        self.time_budget = duration * TIME_BUDGET_FRACTION
        self.detect_elapsed = 0.0
        self.budget_exhausted = False

        self.road = _load_zones()
        self.model = None
        self.frame_idx = -1
        self.tracks = defaultdict(list)   # tid -> [(t, x, y), ...] recent history (resized-frame px)
        self.kind = {}
        self.smoothed = 0.0

    def _get_model(self):
        if self.model is None:
            from ultralytics import YOLO
            self.model = YOLO(str(WEIGHTS))
        return self.model

    def _on_road(self, x, y):
        if not self.road:
            return True
        return any(cv2.pointPolygonTest(p, (float(x), float(y)), False) >= 0 for p in self.road)

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        self.frame_idx += 1
        if self.budget_exhausted or self.frame_idx % DETECT_STRIDE != 0:
            return self.smoothed
        if self.detect_elapsed > self.time_budget:
            self.budget_exhausted = True    # stop detecting for the rest of this video
            return self.smoothed

        t0 = time.time()
        h, w = frame.shape[:2]