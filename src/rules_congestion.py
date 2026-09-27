"""congestion: traffic at a standstill or crawling, across the tracked vehicles
on the road, for a sustained period. Reuses the same per-second 'is this
vehicle standing still' signal as stopped_vehicle, but aggregated across the
whole scene instead of per vehicle. A red-light queue with most vehicles
stopped at once legitimately matches this definition."""
import csv
import json
from collections import defaultdict
from math import hypot

import cv2
import numpy as np

VEHICLES = {"car", "truck", "motorcycle", "bus"}
STILL_RATIO = 0.25       # moved less than 25% of its own width in 1s = still
FRACTION_THRESH = 0.6    # this share of on-road vehicles must be still at once
MIN_VEHICLES = 3         # need at least this many vehicles present to call it congestion
MIN_SEC = 15             # sustained for at least this long
MAX_GAP = 2               # seconds; bridge short gaps in the congested state


def _median(values):
    values = sorted(values)
    return values[len(values) // 2]


def congestion_events(csv_path, zones_path="zones.json", duration=None):
    zones = json.load(open(zones_path))
    road = [np.array(z["points"], np.int32) for z in zones if z["type"] == "carriageway"]

    def on_road(x, y):
        return any(cv2.pointPolygonTest(p, (float(x), float(y)), False) >= 0 for p in road)

    raw = defaultdict(lambda: defaultdict(list))
    kind = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["class"] not in VEHICLES:
                continue
            x1, y1, x2, y2 = (float(row[k]) for k in ("x1", "y1", "x2", "y2"))
            tid = int(row["track_id"])
            raw[tid][int(float(row["t_sec"]))].append(((x1 + x2) / 2, y2, x2 - x1))
            kind[tid] = row["class"]

    pos = {}
    for tid, secs in raw.items():
        for s, items in secs.items():
            pos[(tid, s)] = tuple(_median([p[i] for p in items]) for i in range(3))

    still = {}
    for (tid, s), (x, y, w) in pos.items():
        nxt = pos.get((tid, s + 1))
        if nxt:
            still[(tid, s)] = hypot(nxt[0] - x, nxt[1] - y) < STILL_RATIO * w

    # per second: which on-road vehicles exist and how many are still
    by_second = defaultdict(list)   # second -> list of (tid, is_still)
    for (tid, s), is_still in still.items():
        x, y, _ = pos[(tid, s)]
        if on_road(x, y):
            by_second[s].append((tid, is_still))

    congested_secs = set()
    for s, entries in by_second.items():
        total = len(entries)
        if total < MIN_VEHICLES:
            continue
        stopped = sum(1 for _, st in entries if st)
        if stopped / total >= FRACTION_THRESH:
            congested_secs.add(s)

    if not congested_secs:
        return []

    ordered = sorted(congested_secs)
    runs = []
    start = prev = ordered[0]
    for s in ordered[1:]:
        if s - prev <= MAX_GAP:
            prev = s
        else:
            runs.append((start, prev))
            start = prev = s
    runs.append((start, prev))

    events = []
    for a, b in runs:
        if b - a >= MIN_SEC:
            end = min(b + 1, duration) if duration else b + 1
            events.append([float(a), float(end), "congestion"])
    return events


if __name__ == "__main__":
    cap = cv2.VideoCapture("samples/C3905.MP4")
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    for s, e, label in congestion_events("tracks_full.csv", "zones.json", duration):
        m0, s0 = divmod(int(s), 60)
        m1, s1 = divmod(int(e), 60)
        print(f"{m0}:{s0:02d} - {m1}:{s1:02d}  {label}  ({e - s:.0f}s)")
