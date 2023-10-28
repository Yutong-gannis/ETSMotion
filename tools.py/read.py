import cv2

cap = cv2.VideoCapture("lane.mp4")

while True:
    ret, frame = cap.read()
    print(frame)
    cv2.imshow("frame", frame)
    cv2.waitKey(1)