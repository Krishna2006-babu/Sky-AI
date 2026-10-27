import cv2
import numpy as np
import tensorflow as tf

print("🧠 Loading Neural Network...")
# 1. Load your trained brain
model = tf.keras.models.load_model("skin_model.h5")
print("✅ Brain loaded successfully!")

# 2. Camera Setup
cap = cv2.VideoCapture(0)

# 🚨 THE FIX: Shrink the box so it only captures a tiny patch of skin!
box_size = 80 
AI_input_size = 64

print("🚀 Starting Live Scanner... Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Flip like a mirror
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    
    # Calculate targeting box coordinates (Center of screen)
    start_x = width // 2 - box_size // 2
    start_y = height // 2 - box_size // 2
    end_x = start_x + box_size
    end_y = start_y + box_size

    # 3. Slice out the skin patch and prep it for the AI
    # 3. Slice out the skin patch and prep it for the AI
    skin_crop = frame[start_y:end_y, start_x:end_x]
    
    # 🚨 THE FIX: Convert OpenCV's BGR color to TensorFlow's RGB color!
    ai_feed = cv2.cvtColor(skin_crop, cv2.COLOR_BGR2RGB)
    
    # Resize to the 64x64 format our AI expects
    ai_feed = cv2.resize(ai_feed, (AI_input_size, AI_input_size))
    
    # Keras expects a "batch" of images, so we wrap our 1 image in an extra array
    ai_feed = np.expand_dims(ai_feed, axis=0)
    
    # 4. Ask the AI to predict!
    prediction = model.predict(ai_feed, verbose=0)[0][0] 
    
    # 5. Display the Results
    if prediction > 0.5:
        label = "CLEAR SKIN"
        color = (0, 255, 0) # Green
        confidence = prediction * 100
    else:
        label = "ACNE DETECTED"
        color = (0, 0, 255) # Red
        confidence = (1 - prediction) * 100
        
    # Draw the targeting box and the AI's diagnosis
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)
    
    # UI Instructions for the user
    cv2.putText(frame, "Move close! Put ONE blemish inside the box.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Center the text above the smaller box
    text = f"{label} ({confidence:.1f}%)"
    cv2.putText(frame, text, (start_x - 40, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    cv2.imshow("Sky-AI Live Scanner", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()