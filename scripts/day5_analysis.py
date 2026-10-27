import cv2
import numpy as np

# --- STEP 1: LOAD THE DATA ---
# We load the image we saved yesterday
image_path = "skin_samples/forehead.jpg"
image = cv2.imread(image_path)

# Safety Check: Did it load?
if image is None:
    print("❌ Error: Could not find the image. Make sure you ran Day 4 first!")
else:
    print(f"✅ Loaded Image: {image_path}")
    
    # --- STEP 2: CALCULATE AVERAGE COLOR ---
    # We use NumPy's mean() function to average all pixels
    # axis=(0, 1) means "Average the Rows and Columns, but keep the 3 Colors separate"
    average_color = np.mean(image, axis=(0, 1))
    
    # OpenCV uses BGR order (Blue, Green, Red)
    blue = int(average_color[0])
    green = int(average_color[1])
    red = int(average_color[2])

    print("\n--- 🎨 COLOR ANALYSIS ---")
    print(f"🔹 Blue Score:  {blue}")
    print(f"Mf Green Score: {green}")
    print(f"🔺 Red Score:   {red}")

    # --- STEP 3: SIMPLE DIAGNOSIS ---
    # A simple logic test for "Redness"
    # Usually, skin has more Red than Green. But if Red is WAY higher, it's inflamed.
    
    print("\n--- 🩺 DIAGNOSIS ---")
    if red > green + 20:
        print("⚠️  Result: High Redness Detected (Possible Inflammation)")
    else:
        print("✅  Result: Normal Skin Tone Balance")