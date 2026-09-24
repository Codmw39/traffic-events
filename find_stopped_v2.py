import csv
import json
import os
from collections import defaultdict
from math import hypot

import cv2
import numpy as np

CSV = "tracks_full.csv" if os.path.exists("tracks_full.csv") else "tracks.csv"
VEHICLES = {"car", "truck", "motorcycle", "bus"}
MIN_STILL_SEC = 10     # standing still at least this long
STILL_RATIO = 0.25     # moved less than 25% of its width in 1 s = still
NEIGHBOR_PX = 300      # "close by" distance (pixels)
MIN_NEIGHBORS = 2      # this many other standing vehicles nearby = a queue

zones = json.load(open("zones.json"))
road = [np.array(z["points"], np.int32) for z in zones if z["type"] == "carriageway"]


def on_road(x, y):
    return any(cv2.pointPolygonTest(p, (float(x), float(y)), False) >= 0 for p in road)


def mmss(t):
    return f"{int(t) // 60}:{int(t) % 60:02d}"


def median(values):
    values = sorted(values)
    return values[len(values) // 2]


# 1. one position per vehicle per second
raw = defaultdict(lambda: defaultdict(list))
kind = {}
with open(CSV) as f:
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
        pos[(tid, s)] = tuple(median([p[i] for p in items]) for i in range(3))

# 2. is the vehicle standing still in each second? (compare with the next second)
still = {}
for (tid, s), (x, y, w) in pos.items():
    nxt = pos.get((tid, s + 1))
    if nxt:
        still[(tid, s)] = hypot(nxt[0] - x, nxt[1] - y) < STILL_RATIO * w

# who is standing still at each second (buses are ignored: bus stops)
standing = defaultdict(list)
for (tid, s), is_still in still.items():
    if is_still and kind[tid] != "bus":
        standing[s].append((tid, pos[(tid, s)][0], pos[(tid, s)][1]))

# 3. find long stretches of standing still
found = []
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
        crowded = 0
        for s in range(a, b):
            me = pos.get((tid, s))
            if not me:
                continue
            near = sum(1 for (t2, x2, y2) in standing[s]
                       if t2 != tid and hypot(x2 - me[0], y2 - me[1]) < NEIGHBOR_PX)
            if near >= MIN_NEIGHBORS:
                crowded += 1
        share = crowded / len(here)
        moved_before = any(still.get((tid, s)) is False for s in secs if s < a)
        found.append((a, b, tid, kind[tid], median([p[0] for p in here]),
                      median([p[1] for p in here]), share, moved_before))

found.sort()
print("using", CSV)
print("start  end    dur  id     class       x     y    crowded  arrived  verdict")
for a, b, tid, cls, x, y, share, moved in found:
    verdict = "QUEUE" if share >= 0.5 else "ISOLATED"
    print(f"{mmss(a):<7}{mmss(b):<7}{b - a:>3}  {tid:<7}{cls:<11}{x:>5.0f} {y:>5.0f}"
          f"   {share:>5.0%}    {'yes' if moved else 'no':<7}  {verdict}")
print(len(found), "stretches")