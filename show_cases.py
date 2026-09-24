import cv2
import numpy as np

VIDEO = "samples/C3905.MP4"
BASE_W = 1920                 # coordinates in the tables use this width
PANEL = (720, 480)            # size of each picture in the strip

# (name, start_sec, end_sec, x, y) -- from the table
CASES = [
    ("id859", 37, 72, 697, 223),
    ("id10", 5, 85, 841, 258),
    ("id2", 1, 15, 1525, 555),
    ("id5", 9, 30, 154, 106),
    ("id818", 30, 49, 1252, 577),
    ("id1971", 56, 74, 1753, 796),
    ("id2121", 57, 68, 610, 573),
    ("id2038", 65, 78, 1450, 616),
    ("id3418", 87, 98, 1539, 566),
]

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
scale = cap.get(cv2.CAP_PROP_FRAME_WIDTH) / BASE_W     # 4K video: scale = 2

for name, t0, t1, x, y in CASES:
    panels = []
    for t in (t0 + 1, (t0 + t1) / 2, t1 - 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok:
            continue
        H, W = frame.shape[:2]
        cw, ch = int(480 * scale), int(320 * scale)         # crop window
        cx, cy = int(x * scale), int(y * scale)
        x0 = max(0, min(cx - cw // 2, W - cw))
        y0 = max(0, min(cy - ch // 2, H - ch))
        crop = frame[y0:y0 + ch, x0:x0 + cw].copy()
        crop = cv2.resize(crop, PANEL)
        px = int((cx - x0) * PANEL[0] / cw)
        py = int((cy - y0) * PANEL[1] / ch)
        cv2.circle(crop, (px, py), 40, (0, 255, 255), 3)
        cv2.putText(crop, f"{name}  t={int(t // 60)}:{int(t % 60):02d}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
        panels.append(crop)
    if panels:
        cv2.imwrite(f"case_{name}.jpg", np.hstack(panels))
        print("saved", f"case_{name}.jpg")
cap.release()