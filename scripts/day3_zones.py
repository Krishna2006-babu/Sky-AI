import cv2              # OpenCV: The "Eyes" (Handles camera & images)
import mediapipe as mp  # MediaPipe: The "Brain" (Google's AI model)

# ==========================================
# STEP 1: DEFINE THE ZONES (The "Shopping List")
# ==========================================
# We don't want to analyze the whole face (lips, eyes, etc.).
# We only want the skin. So, we create Lists of specific ID numbers.
# These numbers correspond to points on the MediaPipe Face Mesh map.

# 1. FOREHEAD ZONE (Purple Dots)
# We chose a "Diamond" shape in the very center of the forehead.
# Why? This is the "T-Zone" area, which is best for detecting oiliness.
FOREHEAD_ZONE = [10, 338, 297, 9, 8, 168, 6, 197, 195, 5]

# 2. LEFT CHEEK ZONE (Yellow Dots)
# We chose 4 points under the eye.
# Why? This area is prone to acne but usually free of facial hair.
LEFT_CHEEK = [330, 347, 450, 425]

# 3. RIGHT CHEEK ZONE (Yellow Dots)
# The symmetrical mirror of the left cheek.
RIGHT_CHEEK = [101, 118, 230, 205]


# ==========================================
# STEP 2: SETUP THE AI (The "Engine")
# ==========================================
# Load the Face Mesh tool from MediaPipe
mp_face_mesh = mp.solutions.face_mesh

# Configure the AI:
# max_num_faces=1: Only look for one person.
# refine_landmarks=True: Use the high-quality model (adds iris tracking, etc.)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Turn on the Webcam (0 is usually the default laptop camera)
cap = cv2.VideoCapture(0)

print("✅ Day 3 Complete: Target Zones Locked.")
print("ℹ️  Press 'q' to quit the window.")


# ==========================================
# STEP 3: THE MAIN LOOP (The "Heartbeat")
# ==========================================
# This loop runs endlessly, processing one video frame at a time.
while cap.isOpened():
    
    # A. Read a frame from the camera
    success, image = cap.read()
    
    # If the camera fails to send a picture, skip this loop
    if not success: 
        continue

    # B. Flip the image horizontally (mirror effect)
    # 1 = Flip horizontally, 0 = Flip vertically
    image = cv2.flip(image, 1)

    # C. Color Conversion
    # OpenCV uses Blue-Green-Red (BGR). MediaPipe needs Red-Green-Blue (RGB).
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # D. Run the AI Processing
    # This line sends the image to Google's model and gets back the coordinates.
    results = face_mesh.process(rgb_image)

    # E. Get Image Dimensions
    # We need the height (h) and width (w) to calculate pixel positions later.
    h, w, _ = image.shape

    # ==========================================
    # STEP 4: DRAW THE DOTS (The "Painting")
    # ==========================================
    # Check: Did the AI actually find a face?
    if results.multi_face_landmarks:
        # Loop through found faces (usually just 1)
        for face_landmarks in results.multi_face_landmarks:
            
            # --- DRAW FOREHEAD (Purple) ---
            # Loop through our specific Forehead ID numbers
            for id in FOREHEAD_ZONE:
                # Get the raw coordinate (0.0 to 1.0)
                landmark = face_landmarks.landmark[id]
                
                # Convert math coordinates to pixels (Math * Width)
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                
                # Draw the dot: (Image, Center, Radius, Color, Fill)
                # Color (255, 0, 255) is Purple in BGR
                cv2.circle(image, (cx, cy), 2, (255, 0, 255), -1) 

            # --- DRAW CHEEKS (Yellow) ---
            # We combine LEFT and RIGHT lists into one big loop
            for id in LEFT_CHEEK + RIGHT_CHEEK:
                landmark = face_landmarks.landmark[id]
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                
                # Color (0, 255, 255) is Yellow in BGR (Blue=0, Green=255, Red=255)
                cv2.circle(image, (cx, cy), 2, (0, 255, 255), -1)

    # F. Show the final image in a window
    cv2.imshow('Skincare AI - Target Locked', image)

    # G. Check for Quit Key
    # If user presses 'q', break the loop
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Cleanup: Release the camera and close windows
cap.release()
cv2.destroyAllWindows()


# ==========================================
# 📘 DAY 3 SUMMARY & PROCESS REPORT
# ==========================================
"""
WHAT WE DID:
In Day 3, we moved from "Seeing a Face" (the wireframe mesh) to "Targeting Skin" (specific zones).
We accomplished this using Python Lists and MediaPipe Indexing.

THE PROCESS:
1.  **Variable Storage (Lists):** We learned that MediaPipe generates 468 landmarks. Instead of using all of them, 
    we created Python Lists (`FOREHEAD_ZONE`, `LEFT_CHEEK`) to store only the specific IDs 
    we care about. This is a form of "Data Filtering."

2.  **Coordinate Mapping:**
    The AI gives us coordinates as percentages (0.0 to 1.0). We wrote code to multiply 
    these by the image width/height (`landmark.x * w`) to find the exact pixel on the screen.

3.  **Visualization:**
    We used `cv2.circle` to draw colored dots on these coordinates.
    -   Purple for Forehead (T-Zone for oil analysis).
    -   Yellow for Cheeks (Cheekbones for acne analysis).

4.  **Debugging & Iteration:**
    We initially tried a "Rectangle" list, but it included chin points (bad data). 
    We iterated and switched to a "Diamond" list, proving that changing the Data (List) 
    changes the AI's behavior without changing the Code (Logic).

NEXT STEP (DAY 4):
Now that we know WHERE the skin is, we will use Python Slicing to "Crop" these 
squares out of the video feed so we can analyze the texture.
"""