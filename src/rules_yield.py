"""failure_to_yield: a MOVING vehicle drives across a crosswalk polygon while a
pedestrian is on it (or entering it). A vehicle just queued/stopped near the
crossing (e.g. waiting at a red light while pedestrians cross) is not this event."""
import csv
import json
from collections import defaultdict
from math import hypot

import cv2
import numpy as np

VEHICLES = {"car", "truck", "bus", "motorcycle"}
CROSSWALK_MARGIN = 0
MIN_SEC = 0.5
MIN_VEHICLE_MOVEMENT_PX = 150


def _load_zones(zones_path):
    zones = json.load(open(zones_path))
    cross = [np.array(z["points"], np.int32) for z in zones if z["type"] == "crosswalk"]
    return cross


def failure_to_yield_events(csv_path, zones_path="zones.json", duration=None):
    cross = _load_zones(zones_path)

    def on_crosswalk(x, y):
        return any(cv2.pointPolygonTest(p, (float(x), float(y)), True) >= -CROSSWALK_MARGIN
                   for p in cross)

    by_track = defaultdict(dict)
    kind = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["class"] not in VEHICLES and row["class"] != "person":
                continue
            x1, y1, x2, y2 = (float(row[k]) for k in ("x1", "y1", "x2", "y2"))
            tid = int(row["track_id"])
            by_track[tid][round(float(row["t_sec"]), 1)] = ((x1 + x2) / 2, y2)
            kind[tid] = row["class"]

    ped_times = set()
    for tid, pts in by_track.items():
        if kind[tid] != "person":
            continue
        for t, (x, y) in pts.items():
            if on_crosswalk(x, y):
                ped_times.add(t)

    events = []
    for tid, pts in by_track.items():
        if kind[tid] not in VEHICLES:
            continue
        run = None
        for t in sorted(pts):
            x, y = pts[t]
            active = on_crosswalk(x, y) and t in ped_times
            if active:
                run = (run[0], t, run[2] + [(x, y)]) if run else (t, t, [(x, y)])
            elif run:
                _flush(run, duration, events)
                run = None
        if run:
            _flush(run, duration, events)

    events.sort()
    merged = []
    for s, e, label in events:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e, label])
    return merged


def _flush(run, duration, events):
    start, end, positions = run
    if end - start < MIN_SEC:
        return
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    moved = hypot(max(xs) - min(xs), max(ys) - min(ys))
    if moved < MIN_VEHICLE_MOVEMENT_PX:
        return
    e = min(end, duration) if duration else end
    events.append([start, e, "failure_to_yield"])
