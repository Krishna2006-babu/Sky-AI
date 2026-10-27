import mediapipe
import os

print("\n" + "="*50)
print("🕵️ DETECTIVE REPORT")
print("="*50)
print(f"1. I am importing 'mediapipe' from here:\n   👉 {mediapipe.__file__}")
print("="*50 + "\n")

if "site-packages" in str(mediapipe.__file__):
    print("✅ GOOD: This looks like the real library.")
else:
    print("❌ BAD: This is a local file! DELETE IT.")
    print("   (It is likely named 'mediapipe.py' or in '__pycache__')")
    quit