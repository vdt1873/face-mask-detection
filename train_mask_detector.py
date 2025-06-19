import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Step 1: Load Dataset
data_dir = "dataset"
categories = ["with_mask", "without_mask"]
data = []
labels = []

img_size = 224

print("[INFO] Loading images...")

for category in categories:
    path = os.path.join(data_dir, category)
    class_num = categories.index(category)
    
    for img in os.listdir(path):
        try:
            image = load_img(os.path.join(path, img), target_size=(img_size, img_size))
            image = img_to_array(image)
            image = image / 255.0  # normalize
            data.append(image)
            labels.append(class_num)
        except Exception as e:
            print(f"[ERROR] Skipping image: {img} due to {e}")

# Convert to numpy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Step 2: Train/Test Split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Step 3: Build CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

# Step 4: Train the Model
print("[INFO] Training model...")
model.fit(x_train, y_train, batch_size=32, epochs=25, validation_data=(x_test, y_test))

# Step 5: Save the Model
model.save("mask_detector.h5")
print("[INFO] Model saved as mask_detector.h5")
