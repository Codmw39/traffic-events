import cv2

VIDEO = "samples/C3905.MP4"
WIDTH = 1920

# (name, time in seconds, x, y)
CASES = [
    ("id21_left_edge", 60, 79, 545),
    ("id10_far_lane", 40, 841, 257),
    ("id859_far_side", 60, 698, 223),
    ("id2121_bottom_right", 80, 1253, 915),
    ("id736_queue_example", 26, 483, 458),
]

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)

for name, t, x, y in CASES:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, frame = cap.read()
    if not ok:
        print("cannot read", name)
        continue
    h, w = frame.shape[:2]
    frame = cv2.resize(frame, (WIDTH, int(WIDTH * h / w)))
    cv2.circle(frame, (x, y), 45, (0, 0, 255), 4)
    cv2.putText(frame, f"{name}  t={t}s", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3)
    cv2.imwrite(f"case_{name}.jpg", frame)
    print("saved", f"case_{name}.jpg")
cap.release()