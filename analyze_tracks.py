import csv
from collections import defaultdict

VEHICLES = {"car", "bus", "truck", "motorcycle"}

tracks = defaultdict(list)
with open("tracks.csv") as f:
    for row in csv.DictReader(f):
        if row["class"] not in VEHICLES:
            continue
        x = (float(row["x1"]) + float(row["x2"])) / 2
        y = float(row["y2"])      # bottom edge = where the vehicle touches the road
        tracks[int(row["track_id"])].append((float(row["t_sec"]), x, y, row["class"]))

rows = []
for tid, pts in tracks.items():
    if len(pts) < 5:              # ignore very short-lived tracks
        continue
    t0, x0, y0, cls = pts[0]
    t1, x1, y1, _ = pts[-1]
    duration = t1 - t0
    moved = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    speed = moved / duration if duration > 0 else 0
    rows.append((speed, tid, cls, duration, moved))

rows.sort()
print("id   class      seen(s)   moved(px)   speed(px/s)")
for speed, tid, cls, duration, moved in rows:
    print(f"{tid:<5}{cls:<11}{duration:>7.1f}{moved:>12.0f}{speed:>13.1f}")