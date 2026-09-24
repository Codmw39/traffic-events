import csv
import json
import cv2
import numpy as np

WIDTH = 1920

cap = cv2.VideoCapture("samples/C3905.MP4")
ok, frame = cap.read()
cap.release()
h, w = frame.shape[:2]
frame = cv2.resize(frame, (WIDTH, int(WIDTH * h / w)))

zones = json.load(open("zones.json"))
road = [np.array(z["points"], np.int32) for z in zones if z["type"] == "carriageway"]

for pts in road:
    cv2.polylines(frame, [pts], True, (0, 255, 0), 3)

with open("tracks.csv") as f:
    for row in csv.DictReader(f):
        if row["frame"] != "0":
            continue
        if row["class"] not in ("car", "bus", "truck", "motorcycle"):
            continue
        x = (float(row["x1"]) + float(row["x2"])) / 2
        y = float(row["y2"])
        inside = any(cv2.pointPolygonTest(p, (x, y), False) >= 0 for p in road)
        color = (0, 255, 0) if inside else (0, 0, 255)
        cv2.circle(frame, (int(x), int(y)), 10, color, -1)
        cv2.putText(frame, row["track_id"], (int(x) + 12, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

cv2.imwrite("zones_check.jpg", frame)
print("saved zones_check.jpg")