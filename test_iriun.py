import cv2

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Cannot open camera at index 1")
    exit()

print("Streaming from index 1. Press 'q' in the window to close.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    cv2.imshow("Index 1 Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()