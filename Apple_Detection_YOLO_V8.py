import cv2
from ultralytics import YOLO
import argparse
import time
import cvzone

ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', required=True, help='Path to input video file')
args = ap.parse_args()

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(args.video)
prev_frame_time = 0
new_frame_time = 0

while (cap.isOpened()):
    ret, frame = cap.read()

    if not ret:
        break 

    font = cv2.FONT_HERSHEY_SIMPLEX
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)
    cvzone.putTextRect(frame, fps, (100, 50), scale = 3, thickness=5, offset=20, colorR=(0, 0, 255))
    results = model.predict(source = frame, show = True, save = False)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
