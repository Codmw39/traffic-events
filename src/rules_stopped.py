import csv
import json
from collections import defaultdict
from math import hypot

import cv2
import numpy as np

VEHICLES = {"car", "truck", "motorcycle", "bus"}
MIN_STILL_SEC = 10                    # standing still at least this long
STILL_RATIO = 0.25                    # moved less than 25% of its width in 1 s = still
NEIGHBOR_PX = 300                     # "close by" distance (pixels, 1920-wide picture)
MIN_NEIGHBORS = 2                     # this many other standing vehicles nearby = queue
ALONE_MIN_SEC = 30
DEBUG = False                   # must be alone (no queue around) at least this long
IGNORE_BOXES = [(0, 450, 250, 650)]   # (x1, y1, x2, y2): parked van at the left curb


def _median(values):
    values = sorted(values)
    return values[len(values) // 2]


def stopped_vehicle_events(csv_path, zones_path="zones.json", duration=None):
    """Read a tracks csv and return [[start, end, "stopped_vehicle"], ...]."""
    zones = json.load(open(zones_path))
    road = [np.array(z["points"], np.int32) for z in zones if z["type"] == "carriageway"]

    def on_road(x, y):
        return any(cv2.pointPolygonTest(p, (float(x), float(y)), False) >= 0 for p in road)

    # 1. one position per vehicle per second
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

    pos = {}    # (id, second) -> (x, y, width)
    for tid, secs in raw.items():
        for s, items in secs.items():
            pos[(tid, s)] = tuple(_median([p[i] for p in items]) for i in range(3))

    # 2. standing still or moving in each second
    still = {}
    for (tid, s), (x, y, w) in pos.items():
        nxt = pos.get((tid, s + 1))
        if nxt:
            still[(tid, s)] = hypot(nxt[0] - x, nxt[1] - y) < STILL_RATIO * w

    standing = defaultdict(list)       # second -> vehicles standing still (no buses)
    for (tid, s), is_still in still.items():
        if is_still and kind[tid] != "bus":
            standing[s].append((tid, pos[(tid, s)][0], pos[(tid, s)][1]))

    # 3. long stretches of standing still that are not just a queue
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
            # longest stretch with no queue around the vehicle
            alone = 0
            for s in range(a, b):
                me = pos.get((tid, s))
                if me is None:
                    continue
                near = sum(1 for (t2, x2, y2) in standing[s]
                           if t2 != tid and hypot(x2 - me[0], y2 - me[1]) < NEIGHBOR_PX)
                if near < MIN_NEIGHBORS:
                    alone += 1
            if DEBUG:
                print(f"  id {tid}: {a}-{b} s, alone {alone} s")
            if alone >= ALONE_MIN_SEC:
                end = min(b, duration) if duration else b
                events.append([float(a), float(end), "stopped_vehicle"])

    # same class must not overlap: merge overlapping events into one segment
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
    for s, e, label in stopped_vehicle_events("tracks_full.csv", "zones.json", duration):
        print(f"{int(s) // 60}:{int(s) % 60:02d} - {int(e) // 60}:{int(e) % 60:02d}  {label}  ({e - s:.0f} s)")