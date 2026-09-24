import json
import os

import cv2
import numpy as np

VIDEO = "samples/C3905.MP4"
WIDTH = 1920        # same size as tracks csv, so coordinates match
SHOW = 0.6          # display scale so it fits your screen

TYPES = {"1": "carriageway", "2": "crosswalk", "3": "stop_line", "4": "signal",
         "5": "road_tight"}
COLORS = {"carriageway": (0, 255, 0), "crosswalk": (255, 255, 0), "stop_line": (0, 0, 255),
          "signal": (255, 0, 255), "road_tight": (0, 165, 255)}

cap = cv2.VideoCapture(VIDEO)
cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
ok, frame = cap.read()
cap.release()
h, w = frame.shape[:2]
frame = cv2.resize(frame, (WIDTH, int(WIDTH * h / w)))

# keep the zones you already saved, new ones are added to them
zones = json.load(open("zones.json")) if os.path.exists("zones.json") else []
current = []
current_type = "1"


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        current.append([int(x / SHOW), int(y / SHOW)])


def draw_shape(img, points, kind, closed):
    pts = (np.array(points) * SHOW).astype(np.int32)
    cv2.polylines(img, [pts], closed, COLORS[kind], 2)
    for p in pts:
        cv2.circle(img, tuple(int(v) for v in p), 4, COLORS[kind], -1)


cv2.namedWindow("zones")
cv2.setMouseCallback("zones", on_mouse)

while True:
    img = cv2.resize(frame, None, fx=SHOW, fy=SHOW)
    for z in zones:
        draw_shape(img, z["points"], z["type"], z["type"] != "stop_line")
    kind = TYPES[current_type]
    if current:
        draw_shape(img, current, kind, False)
    cv2.putText(img, f"drawing: {kind}  zones saved: {len(zones)}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(img, "1-5 type | c close shape | u undo point | d delete last shape | s save | q quit",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow("zones", img)

    key = cv2.waitKey(30) & 0xFF
    if chr(key) in TYPES:
        current_type = chr(key)
    elif key == ord("u") and current:
        current.pop()
    elif key == ord("d") and zones and not current:
        zones.pop()
    elif key == ord("c") and len(current) >= 2:
        zones.append({"type": TYPES[current_type], "points": current.copy()})
        current.clear()
    elif key == ord("s"):
        with open("zones.json", "w") as f:
            json.dump(zones, f, indent=1)
        print("saved zones.json with", len(zones), "zones")
        break
    elif key == ord("q"):
        break

cv2.destroyAllWindows()