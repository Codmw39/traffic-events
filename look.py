import cv2

video = cv2.VideoCapture("samples/C3905.MP4")

fps = video.get(cv2.CAP_PROP_FPS)
frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

print("fps:", fps)
print("duration (s):", frames / fps)
print("size:", width, "x", height)

success, frame = video.read()
small = cv2.resize(frame, (1280, int(1280 * height / width)))
cv2.imwrite("first_frame.jpg", small)