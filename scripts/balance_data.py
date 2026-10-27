import os
import random

# 1. Define Paths (Looking inside the current Skincare_AI folder)
ACNE_DIR = "dataset/acne"
CLEAR_DIR = "dataset/clear_skin"

print("⚖️ Starting Dataset Balancer...")

# 2. Count the minority class (Clear Skin)
clear_images = os.listdir(CLEAR_DIR)
target_count = len(clear_images) 

print(f"🟢 Clear Skin Count: {target_count}")

# 3. Count the majority class (Acne)
acne_images = os.listdir(ACNE_DIR)
print(f"🔴 Initial Acne Count: {len(acne_images)}")

# 4. Balance the scales by deleting excess Acne images
if len(acne_images) > target_count:
    # Shuffle the acne images to keep a random, diverse variety
    random.shuffle(acne_images)
    
    # Identify the extra images we don't need
    images_to_delete = acne_images[target_count:]
    
    # Delete them from the hard drive
    for img in images_to_delete:
        os.remove(os.path.join(ACNE_DIR, img))
        
    print(f"🗑️ Deleted {len(images_to_delete)} excess acne images to prevent AI bias.")
    print(f"✅ DATASET PERFECTLY BALANCED: {target_count} Acne vs {target_count} Clear Skin!")
else:
    print("Dataset is already balanced or clear skin is the majority!")