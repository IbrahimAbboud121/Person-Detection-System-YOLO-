from ultralytics import YOLO
import cv2

model = YOLO("C:/Users/USER/Downloads/PersonProject.pt")

cap = cv2.VideoCapture(0)  # 1 = DroidCam, 0 = laptop camera

while True:
    ret, frame = cap.read()
    frame=cv2.flip(frame,1)
    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow("Person Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()