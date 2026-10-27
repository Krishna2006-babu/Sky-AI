import cv2
import numpy as np
import tensorflow as tf

print("🧠 Loading Sky-AI Neural Network...")
model = tf.keras.models.load_model("skin_model.h5")

# Load OpenCV's built-in, lightweight face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
print("✅ Brain and Face Tracker loaded successfully!")

cap = cv2.VideoCapture(0)

# The size of the patch the AI studies
WINDOW_SIZE = 64
# How far the box moves each step. 32 pixels means a 50% overlap between scans
STEP_SIZE = 32 

print("🚀 Starting Full Face Scanner... Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)
    
    # The face detector requires grayscale to work fast
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 1. Find the face(s) in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (fx, fy, fw, fh) in faces:
        # Draw a blue box around the whole face so we know the scanning zone
        cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 150, 0), 2)
        cv2.putText(frame, "ACTIVE SCAN ZONE", (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 0), 2)

        # 2. Slide the 64x64 window across the face!
        # We use a nested loop to go row by row, column by column
        for y in range(fy, fy + fh - WINDOW_SIZE, STEP_SIZE):
            for x in range(fx, fx + fw - WINDOW_SIZE, STEP_SIZE):
                
                # Extract the 64x64 patch
                skin_patch = frame[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE]
                
                # Convert color (BGR to RGB) and prep for the AI
                ai_feed = cv2.cvtColor(skin_patch, cv2.COLOR_BGR2RGB)
                ai_feed = np.expand_dims(ai_feed, axis=0)
                
                # 3. Ask the AI to predict
                prediction = model.predict(ai_feed, verbose=0)[0][0]
                
                # 4. If Acne is detected, flag it!
                # We use 0.3 instead of 0.5 to make it strict (only highly confident pimples)
                if prediction < 0.3: 
                    # Draw a red box precisely where the window found the blemish
                    cv2.rectangle(frame, (x, y), (x+WINDOW_SIZE, y+WINDOW_SIZE), (0, 0, 255), 2)
                    
    cv2.imshow("Sky-AI Sliding Window Scanner", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()