import cv2
import mediapipe as mp

# --- STEP 1: DEFINE OUR LISTS (The "Shopping List") ---
# Instead of "Oily" or "Dry", we store the specific ID numbers for the forehead.
# These numbers come from the official MediaPipe map.
# These numbers are the vertical line down the MIDDLE of your forehead
FOREHEAD_INDICES = [10, 151, 9, 8, 168, 6, 197, 195, 5, 4]

# --- SETUP (Standard AI Boilerplate) ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

print("✅ Targeting System Active.")
print(f"🎯 Tracking {len(FOREHEAD_INDICES)} specific forehead points.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    
    # Get the image dimensions (Height and Width) so we know where to draw
    h, w, _ = image.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            
            # --- STEP 2: USE THE LIST (The Application) ---
            # We Loop through our "Shopping List" of IDs
            for id in FOREHEAD_INDICES:
                
                # 1. Get the specific landmark from the AI's big list
                # (This is exactly like doing skin_types[2])
                landmark = face_landmarks.landmark[id]
                
                # 2. Convert to pixel coordinates (Math!)
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                
                # 3. Draw a bright purple dot on that specific spot
                cv2.circle(image, (cx, cy), 3, (255, 0, 255), -1)

    cv2.imshow('Day 3 - List Targeting', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()