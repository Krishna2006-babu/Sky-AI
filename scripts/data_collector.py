import cv2
import os
import uuid

# 1. Setup Directories
base_dir = "dataset/train"
acne_dir = os.path.join(base_dir, "acne")
clear_dir = os.path.join(base_dir, "clear_skin")

os.makedirs(acne_dir, exist_ok=True)
os.makedirs(clear_dir, exist_ok=True)

# 2. Camera Setup
cap = cv2.VideoCapture(0)
box_size = 64  # The CNN will look at 64x64 pixel patches

print("🚀 DATA COLLECTOR STARTED")
print("Align your face in the GREEN box.")
print("Press 'A' to save an ACNE crop.")
print("Press 'C' to save a CLEAR SKIN crop.")
print("Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    
    # Calculate box coordinates (Center of screen)
    start_x = width // 2 - box_size // 2
    start_y = height // 2 - box_size // 2
    end_x = start_x + box_size
    end_y = start_y + box_size

    # Draw the targeting box
    display_frame = frame.copy()
    cv2.rectangle(display_frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
    
    cv2.imshow("Data Collector", display_frame)
    
    # Keypress Logic
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('a'):
        # Crop the exact region inside the box
        crop = frame[start_y:end_y, start_x:end_x]
        filename = os.path.join(acne_dir, f"acne_{uuid.uuid4().hex[:8]}.jpg")
        cv2.imwrite(filename, crop)
        print(f"🔴 Saved ACNE image! Total: {len(os.listdir(acne_dir))}")
        
    elif key == ord('c'):
        # Crop the exact region inside the box
        crop = frame[start_y:end_y, start_x:end_x]
        filename = os.path.join(clear_dir, f"clear_{uuid.uuid4().hex[:8]}.jpg")
        cv2.imwrite(filename, crop)
        print(f"🟢 Saved CLEAR SKIN image! Total: {len(os.listdir(clear_dir))}")

cap.release()
cv2.destroyAllWindows()