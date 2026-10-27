from ultralytics import YOLO
import cv2

print("🧠 Loading YOLO Sky-AI Brain...")
# IMPORTANT: Point this to the exact path where your best.pt file was saved!
model = YOLO("runs/detect/train2/weights/best.pt") 
print("✅ Brain Loaded!")

cap = cv2.VideoCapture(0)
print("🚀 Starting Live YOLO Scanner... Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)

    # 1. Ask the YOLO AI to analyze the entire frame instantly
    # We set conf=0.15 so it shows us even slightly uncertain guesses
    results = model(frame, stream=True, conf=0.15) 

    # 2. Draw the boxes and labels on the screen
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get the exact coordinates of the blemish
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 

            # Get the confidence score
            confidence = int(box.conf[0] * 100)
            
            # Get the name of the blemish (e.g., 'dark spot')
            class_index = int(box.cls[0])
            class_name = model.names[class_index]

            # Draw the box and the label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{class_name} {confidence}%"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Sky-AI YOLO Scanner", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()