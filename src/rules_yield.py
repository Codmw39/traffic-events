"""failure_to_yield: a vehicle is inside a crosswalk polygon while a pedestrian
is on it (or entering it) at the same time."""
import csv
import json
from collections import defaultdict

import cv2
import numpy as np

VEHICLES = {"car", "truck", "bus", "motorcycle"}
CROSSWALK_MARGIN = 20     # px: count as "on the crosswalk" if this close to it too
MIN_SEC = 0.5             # ignore single-frame flickers


def _load_zones(zones_path):
    zones = json.load(open(zones_path))
    cross = [np.array(z["points"], np.int32) for z in zones if z["type"] == "crosswalk"]
    return cross


def failure_to_yield_events(csv_path, zones_path="zones.json", duration=None):
    cross = _load_zones(zones_path)

    def on_crosswalk(x, y):
        return any(cv2.pointPolygonTest(p, (float(x), float(y)), True) >= -CROSSWALK_MARGIN
                   for p in cross)

    # positions per second, per track
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

    # seconds when at least one pedestrian is on a crosswalk
    ped_times = set()
    for tid, pts in by_track.items():
        if kind[tid] != "person":
            continue
        for t, (x, y) in pts.items():
            if on_crosswalk(x, y):
                ped_times.add(t)

    # vehicle stretches on a crosswalk while a pedestrian is also on it
    events = []
    for tid, pts in by_track.items():
        if kind[tid] not in VEHICLES:
            continue
        run = None
        for t in sorted(pts):
            x, y = pts[t]
            active = on_crosswalk(x, y) and t in ped_times
            if active:
                run = (run[0], t) if run else (t, t)
            elif run:
                if run[1] - run[0] >= MIN_SEC:
                    end = min(run[1], duration) if duration else run[1]
                    events.append([run[0], end, "failure_to_yield"])
                run = None
        if run and run[1] - run[0] >= MIN_SEC:
            end = min(run[1], duration) if duration else run[1]
            events.append([run[0], end, "failure_to_yield"])

    events.sort()
    merged = []
    for s, e, label in events:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e, label])
    return merged


if __name__ == "__main__":
    cap = cv2.VideoCapture("samples/C3905.MP4")
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    for s, e, label in failure_to_yield_events("tracks_full.csv", "zones.json", duration):
        m0, s0 = divmod(int(s), 60)
        m1, s1 = divmod(int(e), 60)
        print(f"{m0}:{s0:02d} - {m1}:{s1:02d}  {label}  ({e - s:.1f}s)")