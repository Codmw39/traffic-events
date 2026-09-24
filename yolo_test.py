import cv2
from ultralytics import YOLO

# 1. read one frame from the video
video = cv2.VideoCapture("samples/C3905.MP4")
video.set(cv2.CAP_PROP_POS_FRAMES, 300)   # jump to frame 300 (about 10 seconds in)
success, frame = video.read()
video.release()

# 2. shrink it: 4K is too big, 1920 wide is enough
h, w = frame.shape[:2]
frame = cv2.resize(frame, (1920, int(1920 * h / w)))

# 3. run the detector
model = YOLO("yolov8s.pt")   # downloads about 22 MB the first time
results = model(frame, imgsz=1280, conf=0.25)

# 4. print what it found
r = results[0]
names = r.names
counts = {}
for cls_id in r.boxes.cls.tolist():
    name = names[int(cls_id)]
    counts[name] = counts.get(name, 0) + 1
print(counts)

# 5. save the picture with boxes drawn
cv2.imwrite("yolo_frame.jpg", r.plot())
print("saved yolo_frame.jpg")