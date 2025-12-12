# utils.py

import cv2
import numpy as np
from PIL import Image

# Emotion labels used across the project
EMOTION_LABELS = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}

# Load Haar Cascade classifier (face detection)
haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_path)

def detect_face_and_crop(image_np):
    """
    Detect face using OpenCV Haar cascade and return cropped face region.
    If no face found, return None.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    cropped = image_np[y:y+h, x:x+w]
    return cropped

def preprocess_image_pil(img_pil, target_size=(48,48)):
    """
    Convert PIL image to numpy array, resize, normalize (0-1)
    """
    img = img_pil.convert("RGB").resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    return arr

def predict_emotion(model, pil_image):
    """
    Takes a PIL image -> preprocess -> model.predict -> return (label, confidence)
    """
    arr = preprocess_image_pil(pil_image)
    arr = np.expand_dims(arr, axis=0)  # add batch dimension

    preds = model.predict(arr)[0]
    idx = int(np.argmax(preds))

    return EMOTION_LABELS[idx], float(preds[idx])
