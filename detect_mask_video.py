import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("mask_detector.h5")

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Labels and colors
labels_dict = {0: 'No Mask', 1: 'Mask'}
color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}

# Start webcam
cap = cv2.VideoCapture(0)

print("[INFO] Starting webcam. Press 'q' to quit.")

while True:
    ret, img = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        resized = cv2.resize(face_img, (224, 224))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 224, 224, 3))
        
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]

        # Draw rectangle and label
        cv2.rectangle(img, (x, y), (x+w, y+h), color_dict[label], 2)
        cv2.putText(img, labels_dict[label], (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_dict[label], 2)

    # Show frame
    cv2.imshow('Face Mask Detection', img)

    # Break on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
