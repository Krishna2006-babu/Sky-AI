import cv2              # OpenCV: The "Eye" (Handles camera & images)
import mediapipe as mp  # MediaPipe: The "Brain" (Google's AI model)

# --- CONFIGURATION (The Setup) ---
# We are preparing the "Face Mesh" tool. 
# Think of this like loading a video game level before playing.
mp_face_mesh = mp.solutions.face_mesh 

# Configure the AI model settings:
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,           # LIMIT: We only look for 1 face. If 2 people are in frame, ignore the second.
    refine_landmarks=True,     # PRECISION: This turns on "High Def" mode for eyes and lips.
    min_detection_confidence=0.5, # SENSITIVITY: Needs 50% confidence to say "Yes, that's a face."
    min_tracking_confidence=0.5   # STABILITY: Needs 50% confidence to keep tracking it as you move.
)

# Create a "pen" to draw the lines on the screen.
# color=(0, 255, 0) is RGB code for Green.
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

# --- MAIN PROGRAM (The Action Loop) ---
def main():
    # 1. OPEN THE CAMERA
    # '0' tells the computer to use the default webcam. 
    # If you have an external USB cam, you might change this to '1'.
    cap = cv2.VideoCapture(0)

    # SAFETY CHECK: Did the camera actually open?
    if not cap.isOpened():
        print("❌ Error: Camera not found. Try changing 0 to 1 in the code.")
        return

    print("✅ System Active: Face Mesh running.")
    print("ℹ️  Press 'q' to quit.")

    # 2. START THE VIDEO LOOP
    # This 'while' loop runs 30 times per second (30 FPS).
    # Every time it loops, it processes ONE single frame of video.
    while cap.isOpened():
        
        # Read one image frame from the camera
        # 'success' is True/False (did we get a picture?)
        # 'image' is the actual photo data (array of pixels)
        success, image = cap.read()
        
        # If the camera blinks or fails, skip this loop turn
        if not success:
            continue

        # 3. PREPARE THE IMAGE
        # Flip it horizontally so it feels like looking in a mirror
        image = cv2.flip(image, 1)

        # COLOR CORRECTION: 
        # OpenCV records images in BGR (Blue-Green-Red) format.
        # MediaPipe (the AI) expects RGB (Red-Green-Blue).
        # We must convert it, or the AI will be confused (blue skin!).
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 4. AI PROCESSING (The Heavy Lifting)
        # We send the RGB image to the AI. It returns 'results'.
        # 'results' contains the list of 468 X,Y coordinates for your face.
        results = face_mesh.process(rgb_image)

        # 5. DRAW THE MESH (If a face was found)
        if results.multi_face_landmarks:
            # Loop through every face found (we set limit to 1, but it's still a list)
            for face_landmarks in results.multi_face_landmarks:
                
                # DRAW LAYER 1: The "Tesselation" (The triangular net)
                # This connects the dots to form the 3D shape structure.
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_spec # Use our Green pen
                )
                
                # DRAW LAYER 2: The "Contours" (The outlines)
                # This draws clean lines around eyes, lips, and face shape.
                # We define a WHITE pen (255, 255, 255) right here inside the function.
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
                )

        # 6. DISPLAY THE RESULT
        # Show the processed image in a window titled "Skincare AI..."
        cv2.imshow('Skincare AI - Face Mapper', image)

        # 7. QUIT CONTROLLER
        # cv2.waitKey(5) waits 5ms for a key press.
        # ord('q') checks if that key was 'q'.
        # If yes, we 'break' the loop (stop the program).
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # CLEANUP: Turn off the camera and close the window properly.
    cap.release()
    cv2.destroyAllWindows()

# STANDARD PYTHON STARTUP
# This line checks: "Are we running this file directly?"
# If yes, start the 'main()' function.
if __name__ == "__main__":
    main()