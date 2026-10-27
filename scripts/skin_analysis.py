import os  # <--- ADD THIS LINE
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import mediapipe as mp
from recommendation_engine import SkinRecommender
class SkinAnalyzer:
    def __init__(self):
        # 1. Initialize Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        # --- NEW: LOAD THE DEEP LEARNING BRAIN ---
        self.cnn_model = None
        if os.path.exists("skin_model.h5"):
            print("🧠 Deep Learning Model Found! Loading the Brain...")
            self.cnn_model = load_model("skin_model.h5")
        else:
            print("⚠️ Warning: skin_model.h5 not found! Falling back to basic OpenCV rules.")
        # 2. Initialize the Database
        self.recommender = SkinRecommender()
        
        # 3. Define Exclusion Zones (Eyes & Lips)
        self.LEFT_EYE_ZONE = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
        self.RIGHT_EYE_ZONE = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
        self.LIP_ZONE = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

    def calculate_health_score(self, oil, texture, acne_count):
        """
        WEEK 4 MATH: WEIGHTED SCORING
        Start at 100. Subtract points based on severity weights.
        """
        score = 100
        
        # 1. Acne Penalty (Weight: 1.5)
        # Example: 10 pimples * 1.5 = 15 points off
        acne_penalty = min(50, acne_count * 1.5)
        score -= acne_penalty
        
        # 2. Texture Penalty (Weight: 0.5)
        # Only penalize if texture is rough (>20)
        texture_penalty = max(0, (texture - 20) * 0.5)
        score -= min(30, texture_penalty)
        
        # 3. Oil Penalty (Weight: 0.5)
        # Only penalize if very oily (>50%)
        oil_penalty = max(0, (oil - 50) * 0.5)
        score -= min(20, oil_penalty)
        
        return int(max(0, score)) # Ensure score is not negative

    def analyze_image(self, frame):
        height, width, _ = frame.shape
        annotated_frame = frame.copy()
        
        # --- 1. MASKING ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        skin_mask = np.zeros((height, width), dtype=np.uint8)

        acne_count = 0
        oil_score = 0
        texture_score = 0

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Full Face
                all_pts = np.array([[int(l.x * width), int(l.y * height)] for l in face_landmarks.landmark])
                hull = cv2.convexHull(all_pts) 
                cv2.fillPoly(skin_mask, [hull], 255)

                # Exclude Eyes/Lips
                for zone in [self.LEFT_EYE_ZONE, self.RIGHT_EYE_ZONE, self.LIP_ZONE]:
                    pts = np.array([[int(face_landmarks.landmark[idx].x * width), int(face_landmarks.landmark[idx].y * height)] for idx in zone])
                    cv2.fillPoly(skin_mask, [pts], 0)

                # --- 2. METRICS ---
                # A. Oil
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                v_channel = hsv[:, :, 2]
                roi_oil = v_channel[height//2-50:height//2+50, width//2-50:width//2+50] 
                if roi_oil.size > 0:
                    oil_score = min(100, (np.mean(roi_oil) / 255) * 100)

                # B. Texture
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                texture_score = min(100, cv2.Laplacian(gray, cv2.CV_64F).var() / 5)

                # C. Acne (Redness)
                lower_red1, upper_red1 = np.array([0, 60, 50]), np.array([10, 255, 255])
                lower_red2, upper_red2 = np.array([170, 60, 50]), np.array([180, 255, 255])
                red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
                final_acne_mask = cv2.bitwise_and(red_mask, red_mask, mask=skin_mask)

                # --- ACNE DETECTION (HYBRID CNN APPROACH) ---
            # --- ACNE DETECTION (HYBRID CNN APPROACH) ---
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # (Keep your existing lower_red / upper_red thresholding here)
            mask1 = cv2.inRange(hsv, np.array([0, 60, 50]), np.array([10, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([170, 60, 50]), np.array([180, 255, 255]))
            mask = mask1 + mask2
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            verified_acne_count = 0

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 3 < area < 1000:
                    # 1. THE SCOUT: Get the coordinates of the suspicious red spot
                    x, y, w, h = cv2.boundingRect(cnt)
                    
                    # 2. Crop a square box around the spot (adding 15px padding so the CNN sees the edges)
                    pad = 15
                    y1 = max(0, y - pad)
                    y2 = min(frame.shape[0], y + h + pad) 
                    x1 = max(0, x - pad)
                    x2 = min(frame.shape[1], x + w + pad) 
                    
                    spot_crop = frame[y1:y2, x1:x2]
                    
                    # 3. THE DOCTOR: Ask the CNN to verify it
                    is_acne = True # Default to True if CNN fails to load
                    
                    if self.cnn_model is not None and spot_crop.size > 0:
                        # Preprocess the crop to exactly match how we trained it
                        spot_resized = cv2.resize(spot_crop, (64, 64))
                        spot_rgb = cv2.cvtColor(spot_resized, cv2.COLOR_BGR2RGB)
                        spot_array = np.expand_dims(spot_rgb, axis=0) # Convert to 4D Tensor
                        
                        # Get the probability!
                        prediction = self.cnn_model.predict(spot_array, verbose=0)[0][0]
                        
                        # Remember: 0 is Acne, 1 is Clear Skin
                        if prediction > 0.5:
                            is_acne = False # The CNN overruled OpenCV! It says it's clear skin/noise.

                    # 4. Only draw the circle if the CNN confirms it
                    if is_acne:
                        verified_acne_count += 1
                        cv2.circle(annotated_frame, (int(x+w/2), int(y+h/2)), int(w/2)+2, (0, 255, 0), 1)
                        
            # Pass the verified count to your scoring math
            acne_count = verified_acne_count

        # --- 3. SCORING (NEW!) ---
        overall_score = self.calculate_health_score(oil_score, texture_score, acne_count)
        
        # --- 4. RECOMMENDATIONS ---
        if oil_score > 60: skin_type = "Oily"
        elif oil_score < 30: skin_type = "Dry"
        else: skin_type = "Normal"

        issues = []
        products = []
        
        if acne_count > 5:
            issues.append(f"🔴 Acne Detected ({acne_count} spots)")
            products.extend(self.recommender.get_advice(skin_type, "Acne"))
        
        if texture_score > 35:
            issues.append(f"🧩 High Texture ({int(texture_score)})")
            products.extend(self.recommender.get_advice(skin_type, "Texture"))
            
        if oil_score > 60:
            issues.append(f"💧 High Sebum ({int(oil_score)}%)")
            products.extend(self.recommender.get_advice(skin_type, "Oiliness"))

        if not issues:
            issues.append("✅ Skin looks healthy!")
            products.append("Maintain with Vitamin C and SPF.")

        return {
            "oil_score": round(oil_score, 1),
            "texture_score": round(texture_score, 1),
            "acne_count": acne_count,
            "overall_score": overall_score, # <--- Sending this to UI
            "annotated_image": annotated_frame,
            "issues": issues,
            "products": list(set(products))
        }