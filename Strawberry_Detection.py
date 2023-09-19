import cv2
import numpy as np

cap = cv2.VideoCapture("/home/c1ph3r/Machine_Vision_Projects/Strawberry_detection/strawberry_farm.mp4")

while True:    
    ret, frame = cap.read()
    
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    low_red = np.array([0, 100, 100])  
    high_red = np.array([10, 255, 255])  
    
    mask = cv2.inRange(hsv_frame, low_red, high_red)
    
    low_red2 = np.array([160, 100, 100])  
    high_red2 = np.array([179, 255, 255])  
    mask2 = cv2.inRange(hsv_frame, low_red2, high_red2)
    
    mask = cv2.bitwise_or(mask, mask2)
    
    red = cv2.bitwise_and(frame, frame, mask=mask)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 500:  
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            text = f"strawberry ({x}, {y})"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    
    cv2.imshow("frame", frame)
    cv2.imshow("red_mask", red)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
