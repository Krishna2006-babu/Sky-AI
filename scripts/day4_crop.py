import cv2              # OpenCV: For camera access and image saving
import mediapipe as mp  # MediaPipe: For AI-powered facial landmark detection
import os 
import numpy as np  # Needed for math              # OS: To handle folder creation on your computer
from scripts.day6_brain import generate_report # This 'plugs in' your AI brain
# ==========================================
# STEP 1: CONFIGURATION & SETUP
# ==========================================

# 1. Create a folder named 'skin_samples' to store our cropped images
# os.path.exists checks if it's already there so we don't get an error
if not os.path.exists('skin_samples'):
    os.makedirs('skin_samples')
    print("✅ Created folder: 'skin_samples'")

# --- THE ZONES (Data Lists) ---
# We use specific MediaPipe ID numbers to target clean skin patches.
# Forehead: Targets high center skin (10=Hairline, 151=Mid-Forehead)
FOREHEAD_ZONE = [10, 151, 104, 333]

# --- CORRECTED VERTICAL CHEEKS (No Overlap) ---
# We moved the bottom points OUTWARD to the jawline (away from the mouth/chin).

# LEFT CHEEK
# LEFT CHEEK (Screen Left / Your Right)
# 266 = Top (Under glasses)
# 435 = Bottom (Lower outer jaw)
# 425 = Outer Edge (Side of face)
LEFT_CHEEK = [266, 435, 425]

# RIGHT CHEEK (Screen Right / Your Left)
# 36 = Top (Under glasses)
# 215 = Bottom (Lower outer jaw)
# 205 = Outer Edge (Side of face)
RIGHT_CHEEK = [36, 215, 205]

# --- HELPER FUNCTION: Get Box Coordinates ---
# This function takes the AI landmarks and turns them into a square (box).
def get_bounding_box(landmarks, indices, w, h):
    # a. Convert the AI's percentage coordinates (0.0 - 1.0) into actual pixels
    x_points = [landmarks[i].x * w for i in indices]
    y_points = [landmarks[i].y * h for i in indices]
    
    # b. Find the edges: min is the top/left, max is the bottom/right
    x_min, x_max = int(min(x_points)), int(max(x_points))
    y_min, y_max = int(min(y_points)), int(max(y_points))
    
    # c. Add 10 pixels of 'padding' around the box so it's not too tight
    return x_min - 10, y_min - 10, x_max + 10, y_max + 10

# --- AI SETUP ---
# Initialize the MediaPipe Face Mesh model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Connect to your webcam (0 is usually the default camera)
# --- IMPROVED CAMERA SETUP ---
def find_iriun_index():
    # Search the first 5 slots for an active camera
    for i in range(5):
        temp_cap = cv2.VideoCapture(i)
        if temp_cap.isOpened():
            success, _ = temp_cap.read()
            if success:
                print(f"✅ Found active camera at Index {i}")
                temp_cap.release()
                return i
            temp_cap.release()
    return 0 # Default to 0 if nothing found

# Automatically find and connect
correct_index = find_iriun_index()
cap = cv2.VideoCapture(correct_index)

print("📸 PRESS 's' TO SAVE YOUR SKIN SAMPLES!")
print("ℹ️  Press 'q' to quit.")

def run_real_analysis():
    # 1. Load the sample we just saved
    img = cv2.imread("skin_samples/left_cheek.jpg")
    if img is None: 
        print("❌ Analysis Failed: Image not found.")
        return 0, 0, 0
    
    # 2. COLOR (Redness Math)
    avg_color = np.mean(img, axis=(0, 1))
    # Red - Green = Inflammation Score
    r_score = int(avg_color[2] - avg_color[1]) 
    
    # 3. OIL (HSV Shine Math)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    # Using your calibrated 180 threshold
    shiny_mask = cv2.inRange(v, 180, 255) 
    oil_perc = (cv2.countNonZero(shiny_mask) / v.size) * 100
    
    # 4. TEXTURE (Laplacian Smoothness)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tex_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    return r_score, oil_perc, tex_score
# ==========================================
# STEP 2: MAIN PROCESSING LOOP
# ==========================================
while cap.isOpened():
    # Read a single frame from the camera
    success, image = cap.read()
    if not success: 
        continue

    # Mirror the image so it feels natural (left is left)
    image = cv2.flip(image, 1)
    
    # MediaPipe requires RGB color format, but OpenCV uses BGR
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run the AI to find the 468 points on your face
    results = face_mesh.process(rgb_image)
    
    # Get the height (h) and width (w) of the current camera frame
    h, w, _ = image.shape

    # Create a COPY of the image to draw boxes on. 
    # This keeps the original 'image' clean for saving later!
    display_image = image.copy()

    # If the AI successfully finds a face
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # --- 1. CALCULATE BOXES ---
            # Call our helper function for each zone to get the X/Y coordinates
            fx1, fy1, fx2, fy2 = get_bounding_box(landmarks, FOREHEAD_ZONE, w, h)
            lx1, ly1, lx2, ly2 = get_bounding_box(landmarks, LEFT_CHEEK, w, h)
            rx1, ry1, rx2, ry2 = get_bounding_box(landmarks, RIGHT_CHEEK, w, h)

            # --- 2. DRAW BOXES (For Display Only) ---
            # Draw rectangles: cv2.rectangle(image, top_left, bottom_right, color, thickness)
            cv2.rectangle(display_image, (fx1, fy1), (fx2, fy2), (255, 0, 255), 2) # Purple
            cv2.rectangle(display_image, (lx1, ly1), (lx2, ly2), (0, 255, 255), 2) # Yellow
            cv2.rectangle(display_image, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2) # Yellow

            # --- 3. SAVE LOGIC (Triggered by 's' key) ---
            key = cv2.waitKey(5) & 0xFF
            if key == ord('s'):
                forehead_crop = image[fy1:fy2, fx1:fx2]
                left_crop = image[ly1:ly2, lx1:lx2]
                right_crop = image[ry1:ry2, rx1:rx2]
                # A. Save the images
                cv2.imwrite("skin_samples/forehead.jpg", forehead_crop)
                cv2.imwrite("skin_samples/left_cheek.jpg", left_crop)
                cv2.imwrite("skin_samples/right_cheek.jpg", right_crop)
                
                # Visual Feedback (Flash)
                display_image[:] = 255 
                print("\n✅ SAMPLES SAVED. STARTING AI DIAGNOSIS...")

                # B. RUN REAL ANALYSIS (Day 5 Math)
                real_r, real_o, real_t = run_real_analysis()

                # C. FEED TO BRAIN (Day 6 Logic)
                issues, ingredients = generate_report(real_r, real_o, real_t) 
                
                # D. PRINT THE FINAL LIVE REPORT
                print(f"🏥 --- SKY-AI LIVE REPORT ---")
                print(f"📊 SCORES | Redness: {real_r} | Oil: {real_o:.1f}% | Texture: {real_t:.1f}")
                print(f"🔎 ISSUES: {issues}")
                print(f"🧪 SOLUTIONS: {ingredients}")
                print("-" * 30)
    # Show the video window with the boxes
    cv2.imshow('Day 4 - The Crop', display_image)
    
    # If 'q' is pressed, exit the loop
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Stop the camera and close all windows
cap.release()
cv2.destroyAllWindows()

# ==========================================
# 📘 DAY 4 SUMMARY & PROCESS REPORT
# ==========================================
"""
WHAT WE HAVE DONE:
Today, we transitioned from just "looking" at the face to "extracting" data from it. 
This is a critical step in any AI pipeline called "Data Preparation."

THE PROCESS:
1.  Targeting boundaries: We used the min() and max() functions to find the 
    extremes of our landmark points. This allowed us to define a "Bounding Box" 
    around the skin zones we targeted on Day 3.

2.  Image Slicing (The "Crop"): We utilized NumPy array slicing, which allows 
    us to treat an image like a 2D grid. By specifying [y_start:y_end, x_start:x_end], 
    we "cut" the skin patches out of the high-resolution camera frame.

3.  Original vs. Display: We learned to keep two versions of the image. 
    The 'display_image' has the purple and yellow boxes drawn on it for us to see, 
    but we save the 'image' (clean version) so the AI doesn't get confused by the box lines.

4.  Automation: We used the 'os' library to automatically create folders and 
    'cv2.imwrite' to save our data. This creates a repeatable dataset that we 
    can now use for Phase 2: Analysis.
"""