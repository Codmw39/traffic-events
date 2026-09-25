"""jaywalking rule: pedestrian on the road, outside every crosswalk, for a while,
and actually moving (not a static false detection like a pole or a sign)."""
import csv
import json
from collections import defaultdict
from math import hypot

import cv2
import numpy as np

MIN_SEC = 1.5              # on the road, outside crosswalks, at least this long
MAX_GAP = 1.0               # ignore detector flicker shorter than this
CROSSWALK_MARGIN = 40       # px: this close to a crosswalk counts as "on the crosswalk"
MIN_MOVEMENT_PX = 60  
ROAD_MARGIN = 40            # must be at least this far inside the road, not just past the edge      # must move at least this many px during the stretch
                            # (a real person crossing moves; a false detection on a
                            # static object like a pole or sign does not)


def _load_zones(zones_path):
    zones = json.load(open(zones_path))
    road = [np.array(z["points"], np.int32) for z in zones if z["type"] == "carriageway"]
    cross = [np.array(z["points"], np.int32) for z in zones if z["type"] == "crosswalk"]
    return road, cross


def jaywalking_events(csv_path, zones_path="zones.json", duration=None):
    road, cross = _load_zones(zones_path)

    def in_road(x, y):
        return any(cv2.pointPolygonTest(p, (float(x), float(y)), True) >= ROAD_MARGIN for p in road)

    def near_crosswalk(x, y):
        return any(cv2.pointPolygonTest(p, (float(x), float(y)), True) >= -CROSSWALK_MARGIN
                   for p in cross)

    people = defaultdict(list)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["class"] != "person":
                continue
            x1, y1, x2, y2 = (float(row[k]) for k in ("x1", "y1", "x2", "y2"))
            people[int(row["track_id"])].append((float(row["t_sec"]), (x1 + x2) / 2, y2))

    events = []
    for tid, pts in people.items():
        pts.sort()
        run, runs = None, []
        for t, x, y in pts:
            ok = in_road(x, y) and not near_crosswalk(x, y)
            if ok:
                if run and t - run[-1][0] > MAX_GAP:
                    runs.append(run)
                    run = None
                run = (run or []) + [(t, x, y)]
            elif run:
                runs.append(run)
                run = None
        if run:
            runs.append(run)

        for r in runs:
            t0, t1 = r[0][0], r[-1][0]
            if t1 - t0 < MIN_SEC:
                continue
            xs = [p[1] for p in r]
            ys = [p[2] for p in r]
            moved = hypot(max(xs) - min(xs), max(ys) - min(ys))
            if moved < MIN_MOVEMENT_PX:
                continue          # likely a static false detection, not a real pedestrian
            end = min(t1, duration) if duration else t1
            events.append([round(t0, 2), round(end, 2), "jaywalking"])

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
    for s, e, label in jaywalking_events("tracks_full.csv", "zones.json", duration):
        m0, s0 = divmod(int(s), 60)
        m1, s1 = divmod(int(e), 60)
        print(f"{m0}:{s0:02d} - {m1}:{s1:02d}  {label}  ({e - s:.1f}s)")