import cv2
from ultralytics import YOLO
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', required=True, help='Path to input video file')
args = ap.parse_args()

model = YOLO("yolov8n.pt")

video_source = cv2.VideoCapture(args.video)

while True:
    ret, frame = video_source.read()

    if not ret:
        break 

    results = model.predict(source=args.video, show=True)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_source.release()
cv2.destroyAllWindows()
