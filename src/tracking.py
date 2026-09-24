"""Video -> tracks. Runs the detector + tracker once and caches the result as a csv."""
import csv
import os
import tempfile
import time
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent.parent
WEIGHTS = HERE / "weights" / "yolov8s.pt"
CACHE_DIR = HERE / "cache"
STRIDE = 3                      # process every 3rd frame
WIDTH = 1920                    # shrink frames to this width before detection
IMG_SIZE = 1280                 # detector input size
CONF = 0.25
KEEP = [0, 1, 2, 3, 5, 7, 9]    # person, bicycle, car, motorcycle, bus, truck, traffic light
TIME_LIMIT_FACTOR = 1.0         # stop tracking after this many x the video duration
HEADER = ["frame", "t_sec", "track_id", "class", "x1", "y1", "x2", "y2"]


def get_tracks(video_path):
    """Return the path of a tracks csv for this video (computed once, then cached)."""
    video_path = Path(video_path)
    cache_dir = CACHE_DIR
    try:
        cache_dir.mkdir(exist_ok=True)
    except OSError:
        cache_dir = Path(tempfile.gettempdir())
    out = cache_dir / f"{video_path.stem}_{video_path.stat().st_size}.csv"
    if out.exists():
        return str(out)
    part = out.with_suffix(".part")
    finished = _extract(video_path, part)
    if finished:                # a run that hit the time limit is not cached as complete
        os.replace(part, out)
        return str(out)
    return str(part)


def _extract(video_path, part_path):
    from ultralytics import YOLO      # imported here so the cached path works without it

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
    height = int(WIDTH * cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    model = YOLO(str(WEIGHTS))
    started = time.time()
    finished = True

    with open(part_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        idx = 0
        while True:
            if idx % STRIDE != 0:           # skip this frame quickly
                if not cap.grab():
                    break
                idx += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            if time.time() - started > TIME_LIMIT_FACTOR * duration:
                finished = False            # out of time: keep what we have
                break
            frame = cv2.resize(frame, (WIDTH, height))
            r = model.track(frame, imgsz=IMG_SIZE, conf=CONF, classes=KEEP, persist=True,
                            tracker="bytetrack.yaml", verbose=False)[0]
            if r.boxes.id is not None:
                for box, tid, cls in zip(r.boxes.xyxy.tolist(), r.boxes.id.tolist(),
                                         r.boxes.cls.tolist()):
                    writer.writerow([idx, round(idx / fps, 3), int(tid), r.names[int(cls)],
                                     *[round(v, 1) for v in box]])
            idx += 1
    cap.release()
    return finished