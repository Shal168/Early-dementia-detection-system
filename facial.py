# facial_analysis.py

import tensorflow as tf
import numpy as np
import cv2
import os
import time
import matplotlib
import matplotlib.pyplot as plt

# Use default GUI backend (so plt.show() works)
# matplotlib.use('TkAgg')  # Uncomment if needed

# ==========================
# LABELS
# ==========================
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# ==========================
# PATHS
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "facial_model.h5")

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# ==========================
# FACE DETECTOR
# ==========================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ==========================
# PREPROCESS FACE
# ==========================
def preprocess_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=2, minSize=(20, 20)
    )
    if len(faces) == 0:
        return None, None
    (x, y, w, h) = faces[0]
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (48, 48))
    face = face / 255.0
    face = face.reshape(1, 48, 48, 1)
    return face, (x, y, w, h)

# ==========================
# EMOTION GRAPH
# ==========================
import io
import base64

def plot_graph(pred):
    plt.figure(figsize=(6,3))
    plt.bar(emotion_labels, pred)
    plt.xticks(rotation=30)
    plt.title("Emotion Probabilities")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    return base64.b64encode(buf.read()).decode('utf-8')

# ==========================
# CLINICAL DECISION
# ==========================
def clinical_decision(emotion):
    if emotion in ["Sad", "Neutral"]:
        return 0.6, "⚠️ Possible cognitive concern"
    elif emotion in ["Fear", "Disgust"]:
        return 0.5, "⚠️ Emotional instability"
    else:
        return 0.2, "✅ Normal"

# ==========================
# PREDICT FACE
# ==========================
def predict_face(image_path):
    start = time.time()
    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"Image not found: {image_path}"}

    face, box = preprocess_face(img)
    if face is None:
        return {
            "emotion": "Unknown",
            "confidence": 0.0,
            "risk": 1.0,
            "remark": "No face detected",
            "probabilities": [0]*7,
            "labels": emotion_labels
        }

    pred = model.predict(face)[0]
    emotion = emotion_labels[np.argmax(pred)]
    confidence = float(np.max(pred))
    risk, remark = clinical_decision(emotion)

    # Show graph
    graph_img = plot_graph(pred)

    end = time.time()
    return {
    "emotion": emotion,
    "confidence": confidence,
    "risk": risk,
    "remark": remark,
    "probabilities": pred.tolist(),
    "labels": emotion_labels,
    "graph": graph_img,   # ✅ ADD THIS
    "time": round(end-start, 3)
}

# ==========================
# MAIN EXECUTION
# ==========================
if __name__ == "__main__":
    print("\n===== FACIAL ANALYSIS SYSTEM =====")
    print("Using dataset image for testing...")

    # Directly set your dataset image path here
    image_path = "C:/Users/shali/Downloads/happy.jpg"

    if not os.path.exists(image_path):
        print("❌ File not found:", image_path)
        exit()

    result = predict_face(image_path)
    print("\n===== PREDICTION RESULT =====")
    print(result)