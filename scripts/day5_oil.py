import cv2
import numpy as np
import os

def analyze_skin(filename, zone_name):
    path = f"skin_samples/{filename}"
    if not os.path.exists(path):
        print(f"❌ File not found for {zone_name}")
        return
    
    # 1. LOAD IMAGE
    image = cv2.imread(path)
    
    # 2. COLOR ANALYSIS (REDNESS)
    # Average color across the whole patch
    avg_color = np.mean(image, axis=(0, 1))
    blue, green, red = int(avg_color[0]), int(avg_color[1]), int(avg_color[2])
    redness_score = red - green

    # 3. OIL ANALYSIS (SHINE)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    # Using your calibrated threshold of 180
    shiny_mask = cv2.inRange(v, 180, 255)
    shiny_pixels = cv2.countNonZero(shiny_mask)
    oil_percentage = (shiny_pixels / v.size) * 100

    # 4. TEXTURE ANALYSIS (SMOOTHNESS)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detects sharp edges. Higher = Rougher/More Pores.
    texture_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    # --- PRINT THE FULL REPORT ---
    print(f"\n{'='*30}")
    print(f"📍 ZONE: {zone_name.upper()}")
    print(f"{'='*30}")
    
    # Redness Diagnosis
    print(f"🎨 Color: R:{red} G:{green} B:{blue} (Score: {redness_score})")
    if redness_score > 20:
        print("   🔴 Result: High Redness (Inflammation)")
    else:
        print("   🟢 Result: Calm/Normal Tone")

    # Oil Diagnosis
    print(f"✨ Shine: {oil_percentage:.1f}% Area")
    if oil_percentage > 10:
        print("   💧 Result: Oily Surface")
    elif oil_percentage > 2:
        print("   ✅ Result: Healthy Glow")
    else:
        print("   🌵 Result: Matte/Dry Surface")

    # Texture Diagnosis
    print(f"🧩 Texture: {texture_score:.1f}")
    if texture_score > 150:
        print("   ⚠️  Result: Visible Texture/Pores")
    else:
        print("   ✅ Result: Smooth Surface")

# --- EXECUTION ---
print("🚀 STARTING FINAL SKIN DIAGNOSTIC...")
analyze_skin("forehead.jpg", "Forehead")
analyze_skin("left_cheek.jpg", "Left Cheek")
analyze_skin("right_cheek.jpg", "Right Cheek")