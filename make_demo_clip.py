import cv2

SRC = "samples/C3905.MP4"
OUT = "demo_clip.mp4"
SECONDS = 90       # short clip, plenty for a demo
WIDTH = 960          # much smaller resolution = much smaller file

cap = cv2.VideoCapture(SRC)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(WIDTH * h / w)

out = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (WIDTH, height))
n = 0
max_frames = int(SECONDS * fps)
while n < max_frames:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.resize(frame, (WIDTH, height))
    out.write(frame)
    n += 1

cap.release()
out.release()
print(f"wrote {OUT}, {n} frames (~{n/fps:.1f}s)")