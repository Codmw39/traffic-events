import csv
import json
from collections import defaultdict

import cv2
import numpy as np

CSV = "tracks_full.csv"
VIDEO = "samples/C3905.MP4"
BASE_W = 1920                # coordinates in the csv use this width
MIN_SEC = 1.5                # on the road (outside crosswalks) at least this long
MAX_GAP = 0.5                # ignore detector flicker shorter than this
CROSSWALK_MARGIN = 40        # feet this close to a crosswalk count as "on the crosswalk" (px)
MAX_PICTURES = 10            # save pictures for the longest candidates

zones = json.load(open("zones.json"))
road = [np.array(z["points"], np.int32) for z in zones if z["type"] == "carriageway"]
cross = [np.array(z["points"], np.int32) for z in zones if z["type"] == "crosswalk"]


def in_road(x, y):
    return any(cv2.pointPolygonTest(p, (float(x), float(y)), False) >= 0 for p in road)


def near_crosswalk(x, y):
    return any(cv2.pointPolygonTest(p, (float(x), float(y)), True) >= -CROSSWALK_MARGIN
               for p in cross)


def mmss(t):
    return f"{int(t) // 60}:{int(t) % 60:02d}"


# 1. foot positions of every pedestrian track
people = defaultdict(list)
with open(CSV) as f:
    for row in csv.DictReader(f):
        if row["class"] != "person":
            continue
        x1, y1, x2, y2 = (float(row[k]) for k in ("x1", "y1", "x2", "y2"))
        people[int(row["track_id"])].append((float(row["t_sec"]), (x1 + x2) / 2, y2))

# 2. stretches where a pedestrian stands on the road, outside every crosswalk
found = []


def close(tid, start, last, spots):
    if start is not None and last - start >= MIN_SEC:
        mx = sum(p[0] for p in spots) / len(spots)
        my = sum(p[1] for p in spots) / len(spots)
        found.append((last - start, start, last, tid, mx, my))


for tid, pts in people.items():
    if len(pts) < 8:
        continue
    start = last = None
    spots = []
    for t, x, y in pts:
        if in_road(x, y) and not near_crosswalk(x, y):
            if start is not None and t - last > MAX_GAP:
                close(tid, start, last, spots)
                start, spots = None, []
            if start is None:
                start, spots = t, []
            last = t
            spots.append((x, y))
        elif start is not None:
            close(tid, start, last, spots)
            start, spots = None, []
    if start is not None:
        close(tid, start, last, spots)

found.sort(reverse=True)
print("length  start  end    id     x     y")
for length, start, last, tid, x, y in found:
    print(f"{length:>5.1f}s  {mmss(start):<6} {mmss(last):<6} {tid:<6}{x:>5.0f} {y:>5.0f}")
print(len(found), "candidates")


# 3. pictures: zoomed crops at the start, middle and end of each candidate
def position_at(tid, t):
    best = min(people[tid], key=lambda p: abs(p[0] - t))
    return best[1], best[2]


cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
scale = cap.get(cv2.CAP_PROP_FRAME_WIDTH) / BASE_W
PANEL = (720, 480)
for n, (length, start, last, tid, _, _) in enumerate(found[:MAX_PICTURES], 1):
    panels = []
    for t in (start, (start + last) / 2, last):
        x, y = position_at(tid, t)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok:
            continue
        H, W = frame.shape[:2]
        cw, ch = int(480 * scale), int(320 * scale)
        cx, cy = int(x * scale), int((y - 40) * scale)      # a bit above the feet
        x0 = max(0, min(cx - cw // 2, W - cw))
        y0 = max(0, min(cy - ch // 2, H - ch))
        crop = cv2.resize(frame[y0:y0 + ch, x0:x0 + cw].copy(), PANEL)
        px = int((cx - x0) * PANEL[0] / cw)
        py = int((cy - y0) * PANEL[1] / ch)
        cv2.circle(crop, (px, py), 50, (0, 255, 255), 3)
        cv2.putText(crop, f"#{n} id{tid}  t={mmss(t)}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
        panels.append(crop)
    if panels:
        cv2.imwrite(f"jw_{n:02d}_id{tid}.jpg", np.hstack(panels))
cap.release()
print("saved pictures jw_XX_idYYY.jpg for the", min(len(found), MAX_PICTURES), "longest")