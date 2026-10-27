import tensorflow as tf
from tensorflow.keras import layers, models
import os

# 1. Define Paths and Constants
DATASET_DIR = "dataset" # Looking inside the current Skincare_AI folder
IMG_HEIGHT = 64
IMG_WIDTH = 64
BATCH_SIZE = 32
EPOCHS = 15 # Training for 15 full cycles

print("📊 Loading Balanced Production Dataset...")

# 2. Load the Training Data (80% of your 1914 images)
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# 3. Load the Validation Data (20% reserved for the "Final Exam")
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

class_names = train_dataset.class_names
print(f"✅ Found classes: {class_names}")

# Optimize pipeline speed so your CPU doesn't bottleneck
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# 4. Build the "Production" CNN Architecture
print("🧠 Building Neural Network Architecture...")

# Data Augmentation: Digitally alters images so the AI never sees the exact same image twice
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
  layers.RandomZoom(0.1),
])

model = models.Sequential([
    data_augmentation,
    layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    
    # Deeper Convolutional Layers to spot microscopic textures
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Flatten(),
    layers.Dropout(0.5), # Drops 50% of the neurons randomly to prevent memorization
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid') # Binary Output: 0 (Acne) or 1 (Clear Skin)
])

# 5. Compile the AI
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 6. Train the AI
print("🚀 Starting Training Sequence...")
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)

# 7. Save the Brain
model.save("skin_model.h5")
print("💾 SUCCESS! New Production Model saved as 'skin_model.h5'")