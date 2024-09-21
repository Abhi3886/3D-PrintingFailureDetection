import torch
from ultralytics import YOLO
import cv2
import math

width = 640
height = 480

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Model
model = YOLO("3D-PrintFailureDetection/weights/best.pt")

# Object classes
classNames = ["fault", "success"]

while True:
    success, img = cap.read()
    if not success:
        break
    
    results = model(img)
    
    # Coordinates
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Confidence
            confidence = math.ceil((box.conf[0].cpu().numpy() * 100)) / 100
            print("Confidence --->", confidence)

            # Class name
            cls = int(box.cls[0].cpu().numpy())
            print("Class name -->", classNames[cls])

            # Object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
