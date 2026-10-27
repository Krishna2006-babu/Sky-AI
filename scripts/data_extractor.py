import os
import cv2
import glob
import random

# 1. Define Paths 
ROBOFLOW_DIR = "roboflow_data/train" 
IMG_DIR = os.path.join(ROBOFLOW_DIR, "images")
LABEL_DIR = os.path.join(ROBOFLOW_DIR, "labels")

OUT_ACNE = "dataset/acne"
OUT_CLEAR = "dataset/clear_skin"

# 2. Settings
CROP_SIZE = 64
acne_count = 0
clear_count = 0
missing_labels = 0
math_errors = 0

def overlaps(x1, y1, x2, y2, boxes):
    for (bx1, by1, bx2, by2) in boxes:
        if not (x2 < bx1 or x1 > bx2 or y2 < by1 or y1 > by2):
            return True
    return False

print("⚙️ Starting Diagnostic Extraction Pipeline...")

# 3. Fix the Windows Double-Counting Bug (Uses a Set for unique files only)
image_paths = set(glob.glob(os.path.join(IMG_DIR, "*.*")))
image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"📂 Found {len(image_paths)} UNIQUE images. Slicing now...")

# 4. Process every image
for img_path in image_paths:
    img = cv2.imread(img_path)
    if img is None: 
        continue
    h, w, _ = img.shape
    
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(LABEL_DIR, base_name + ".txt")
    
    # DIAGNOSTIC CHECK 1: Is the label file missing?
    if not os.path.exists(label_path):
        missing_labels += 1
        continue
        
    boxes = []
    
    with open(label_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    x_c, y_c, bw, bh = map(float, parts[1:5])
                    
                    x_center, y_center = int(x_c * w), int(y_c * h)
                    box_w, box_h = int(bw * w), int(bh * h)
                    
                    pad = 5
                    x1 = max(0, x_center - box_w // 2 - pad)
                    y1 = max(0, y_center - box_h // 2 - pad)
                    x2 = min(w, x_center + box_w // 2 + pad)
                    y2 = min(h, y_center + box_h // 2 + pad)
                    
                    boxes.append((x1, y1, x2, y2))
                    
                    acne_crop = img[y1:y2, x1:x2]
                    if acne_crop.size > 0:
                        acne_crop_resized = cv2.resize(acne_crop, (CROP_SIZE, CROP_SIZE))
                        cv2.imwrite(os.path.join(OUT_ACNE, f"acne_{acne_count}.jpg"), acne_crop_resized)
                        acne_count += 1
                except Exception:
                    math_errors += 1

    # 5. Generate "Clear Skin" Samples
    attempts = 0
    while attempts < 10 and len(boxes) > 0: 
        rx = random.randint(0, max(0, w - CROP_SIZE - 1))
        ry = random.randint(0, max(0, h - CROP_SIZE - 1))
        
        if not overlaps(rx, ry, rx + CROP_SIZE, ry + CROP_SIZE, boxes):
            clear_crop = img[ry:ry+CROP_SIZE, rx:rx+CROP_SIZE]
            if clear_crop.shape == (CROP_SIZE, CROP_SIZE, 3):
                cv2.imwrite(os.path.join(OUT_CLEAR, f"clear_{clear_count}.jpg"), clear_crop)
                clear_count += 1
                break 
        attempts += 1

print("✅ EXTRACTION COMPLETE!")
print(f"🔴 Total Acne Patches Extracted: {acne_count}")
print(f"🟢 Total Clear Skin Patches Generated: {clear_count}")
print("--- DIAGNOSTICS ---")
print(f"⚠️ Images skipped (No label file): {missing_labels}")
print(f"⚠️ Math/Cropping errors: {math_errors}")