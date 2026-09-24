import json

import cv2
import numpy as np

COLORS = {"carriageway": (0, 255, 0), "crosswalk": (255, 255, 0), "stop_line": (0, 0, 255),
          "signal": (255, 0, 255), "road_tight": (0, 165, 255)}

zones = json.load(open("zones.json"))
print("zones in zones.json:", len(zones))
for i, z in enumerate(zones, 1):
    print(f"  {i}. {z['type']}  ({len(z['points'])} points)")

cap = cv2.VideoCapture("samples/C3905.MP4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
ok, frame = cap.read()
cap.release()
h, w = frame.shape[:2]
frame = cv2.resize(frame, (1920, int(1920 * h / w)))

for z in zones:
    pts = np.array(z["points"], np.int32)
    color = COLORS.get(z["type"], (255, 255, 255))
    cv2.polylines(frame, [pts], z["type"] != "stop_line", color, 3)
    cv2.putText(frame, z["type"], (int(pts[0][0]), int(pts[0][1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

cv2.imwrite("zones_view.jpg", frame)
print("saved zones_view.jpg")