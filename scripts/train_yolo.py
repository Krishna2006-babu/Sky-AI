from ultralytics import YOLO

print("🧠 Downloading Blank YOLOv8 Nano Brain...")
# Load a blank, ultra-fast YOLOv8 model 
model = YOLO("yolov8n.pt") 

print("🚀 Starting YOLO Training Sequence...")
# Train the model on your dataset
results = model.train(
    data="roboflow_data/data.yaml", # Point it to your newly fixed map!
    epochs=15,                      # Train for 15 cycles
    imgsz=640,                      # Compress images to 640x640 for speed
    batch=16,                       # Process 16 images at a time
    device="cpu"                    # Run on your CPU
)

print("✅ YOLO TRAINING COMPLETE!")