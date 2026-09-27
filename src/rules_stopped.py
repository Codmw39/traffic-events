"""stopped_vehicle: a vehicle stationary on the carriageway for >=10s, not just
part of a signal queue. A stretch counts only if the vehicle spends enough of
it with no queue of other standing vehicles around it (a queued car waiting at
red is NOT this event; a car that stops on its own, away from a queue, is)."""
import csv
import json
from collections import defaultdict
from math import hypot

import cv2
import numpy as np

VEHICLES = {"car", "truck", "motorcycle", "bus"}
MIN_STILL_SEC = 10
STILL_RATIO = 0.25
NEIGHBOR_PX = 300
MIN_NEIGHBORS = 2
ALONE_MIN_SEC = 30
IGNORE_BOXES = [(0, 450, 250, 650)]   # known parked vehicle at the left curb (C3905 camera)


def _median(values):
    values = sorted(values)
    return values[len(values) // 2]


def stopped_vehicle_events(csv_path, zones_path="zones.json", duration=None):
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

    standing = defaultdict(list)
    for (tid, s), is_still in still.items():
        if is_still and kind[tid] != "bus":
            standing[s].append((tid, pos[(tid, s)][0], pos[(tid, s)][1]))

    events = []
    for tid, secs in raw.items():
        if kind[tid] == "bus":
            continue
        run, runs = None, []
        for s in sorted(secs):
            st = still.get((tid, s))
            if st is None:
                continue
            x, y, _ = pos[(tid, s)]
            if st and on_road(x, y):
                if run and s <= run[1] + 2:
                    run[1] = s + 1
                else:
                    if run:
                        runs.append(run)
                    run = [s, s + 1]
            elif run:
                runs.append(run)
                run = None
        if run:
            runs.append(run)

        for a, b in runs:
            if b - a < MIN_STILL_SEC:
                continue
            here = [pos[(tid, s)] for s in range(a, b) if (tid, s) in pos]
            mx, my = _median([p[0] for p in here]), _median([p[1] for p in here])
            if any(x1 <= mx <= x2 and y1 <= my <= y2 for x1, y1, x2, y2 in IGNORE_BOXES):
                continue
            alone = 0
            for s in range(a, b):
                me = pos.get((tid, s))
                if me is None:
                    continue
                near = sum(1 for (t2, x2, y2) in standing[s]
                           if t2 != tid and hypot(x2 - me[0], y2 - me[1]) < NEIGHBOR_PX)
                if near < MIN_NEIGHBORS:
                    alone += 1
            if alone >= ALONE_MIN_SEC:
                end = min(b, duration) if duration else b
                events.append([float(a), float(end), "stopped_vehicle"])

    events.sort()
    merged = []
    for s, e, label in events:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e, label])
    return merged
