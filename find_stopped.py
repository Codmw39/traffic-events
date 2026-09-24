import csv
import json
import os
from collections import defaultdict
from math import hypot

import cv2
import numpy as np

CSV = "tracks_full.csv" if os.path.exists("tracks_full.csv") else "tracks.csv"
VEHICLES = {"car", "bus", "truck", "motorcycle"}
MIN_STILL_SEC = 10     # how long it must stand still
WINDOW = 1.0           # compare positions 1 second apart
STILL_RATIO = 0.25     # moved less than 25% of its own width in 1 s = standing still
MAX_GAP = 1.0          # ignore short interruptions (detector flicker)

zones = json.load(open("zones.json"))
road = [np.array(z["points"], np.int32) for z in zones if z["type"] == "carriageway"]


def on_road(x, y):
    return any(cv2.pointPolygonTest(p, (float(x), float(y)), False) >= 0 for p in road)


def mmss(t):
    return f"{int(t // 60)}:{int(t % 60):02d}"


tracks = defaultdict(list)
with open(CSV) as f:
    for row in csv.DictReader(f):
        if row["class"] not in VEHICLES:
            continue
        x1, y1, x2, y2 = (float(row[k]) for k in ("x1", "y1", "x2", "y2"))
        tracks[int(row["track_id"])].append(
            (float(row["t_sec"]), (x1 + x2) / 2, y2, x2 - x1, row["class"]))

found = []
for tid, pts in tracks.items():
    if len(pts) < 10:
        continue
    still = []                      # (start_time, end_time) of each "still" moment
    j = 0
    for i in range(len(pts)):
        t, x, y, w, cls = pts[i]
        j = max(j, i)
        while j < len(pts) - 1 and pts[j][0] - t < WINDOW:
            j += 1
        if pts[j][0] - t < WINDOW * 0.8:
            break
        moved = hypot(pts[j][1] - x, pts[j][2] - y)
        if moved < STILL_RATIO * w:
            still.append((t, pts[j][0]))

    if not still:
        continue
    start, end = still[0]
    runs = []
    for s, e in still[1:]:
        if s <= end + MAX_GAP:
            end = max(end, e)
        else:
            runs.append((start, end))
            start, end = s, e
    runs.append((start, end))

    for s, e in runs:
        if e - s >= MIN_STILL_SEC:
            mid = [p for p in pts if s <= p[0] <= e]
            mx = sum(p[1] for p in mid) / len(mid)
            my = sum(p[2] for p in mid) / len(mid)
            found.append((s, e, tid, pts[0][4], mx, my))

found.sort()
print("using", CSV)
print("start  end    dur(s)  id    class       x     y    on_road")
for s, e, tid, cls, x, y in found:
    print(f"{mmss(s):<7}{mmss(e):<7}{e - s:>5.0f}  {tid:<6}{cls:<11}{x:>5.0f} {y:>5.0f}   {on_road(x, y)}")
print(len(found), "candidates")