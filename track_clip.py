import csv
import time
import cv2
from ultralytics import YOLO

VIDEO = "samples/C3905.MP4"
START_SEC = 0        # where to start in the video
LENGTH_SEC = 130      # how many seconds to process
STRIDE = 3           # use every 3rd frame (faster)
WIDTH = 1920         # shrink the 4K frames to this width
# COCO classes: person, bicycle, car, motorcycle, bus, truck, traffic light
KEEP = [0, 1, 2, 3, 5, 7, 9]

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
w0 = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h0 = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
height = int(WIDTH * h0 / w0)

first = int(START_SEC * fps)
last = int((START_SEC + LENGTH_SEC) * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, first)

model = YOLO("yolov8s.pt")
out = cv2.VideoWriter("tracks_full_demo.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                      fps / STRIDE, (WIDTH, height))

csv_file = open("tracks_full.csv", "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["frame", "t_sec", "track_id", "class", "x1", "y1", "x2", "y2"])

start_time = time.time()
idx = first
done = 0
while idx < last:
    if (idx - first) % STRIDE != 0:
        cap.grab()          # skip this frame quickly
        idx += 1
        continue

    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.resize(frame, (WIDTH, height))

    r = model.track(frame, imgsz=1280, conf=0.25, classes=KEEP,
                    persist=True, tracker="bytetrack.yaml", verbose=False)[0]

    if r.boxes.id is not None:
        for box, tid, cls in zip(r.boxes.xyxy.tolist(),
                                 r.boxes.id.tolist(),
                                 r.boxes.cls.tolist()):
            writer.writerow([idx, round(idx / fps, 3), int(tid),
                             r.names[int(cls)], *[round(v, 1) for v in box]])

    out.write(r.plot())
    done += 1
    if done % 10 == 0:
        print("processed", done, "frames, elapsed", round(time.time() - start_time), "s")
    idx += 1

out.release()
csv_file.close()
cap.release()
print("done in", round(time.time() - start_time), "seconds")
print("saved tracks_demo.mp4 and tracks.csv")