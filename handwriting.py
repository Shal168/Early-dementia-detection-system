# ===== HANDWRITING ANALYSIS SYSTEM (FINAL VERSION) =====

import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# ===============================
# PARAMETERS
# ===============================
IMG_SIZE = 128
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "spiral_wave_model.keras")

model = load_model(MODEL_PATH)

# ===============================
# PREPROCESS
# ===============================
def preprocess(path):
    img = cv2.imread(path)

    if img is None:
        raise ValueError(f"❌ Image not found: {path}")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0

    return np.expand_dims(img, axis=0)

# ===============================
# VISUALIZATION (DISPLAY)
# ===============================
def visualize(path, name):

    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"❌ Cannot read image: {path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 80, 200)

    contour_img = img.copy()
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = [c for c in contours if cv2.contourArea(c) > 50]

    cv2.drawContours(contour_img, filtered, -1, (0, 255, 0), 1)

    # DISPLAY
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(thresh, cmap="gray")
    plt.title("Threshold")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(edges, cmap="gray")
    plt.title("Edges")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
    plt.title("Contours")
    plt.axis("off")

    plt.suptitle(name)
    plt.tight_layout()
    plt.show()

    return thresh, edges

# ===============================
# FEATURE EXTRACTION
# ===============================
def analyze_features(thresh, edges):

    stroke_density = np.sum(thresh == 255) / thresh.size
    edge_density = np.sum(edges > 0) / edges.size

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    tremor = min(len(contours) / 200, 1)

    smoothness = 1 - edge_density

    return {
        "stroke_density": stroke_density,
        "edge_density": edge_density,
        "tremor": tremor,
        "smoothness": smoothness
    }

# ===============================
# FEATURE SCORE
# ===============================
def compute_score(features, test_type):

    score = 0

    if features["stroke_density"] < 0.02:
        score += 0.25

    if features["tremor"] > 0.3:
        score += 0.25

    if test_type == "spiral" and features["edge_density"] > 0.07:
        score += 0.25

    if test_type == "wave" and features["edge_density"] > 0.1:
        score += 0.25

    return min(score, 1)

# ===============================
# STABLE AI PREDICTION
# ===============================
def stable_prediction(spiral_img, wave_img):
    probs = []
    for _ in range(3):
        p = model.predict([spiral_img, wave_img])[0][0]
        probs.append(p)
    return np.mean(probs)

# ===============================
# MAIN PREDICTION
# ===============================
def predict_combined(spiral_path, wave_path):

    if not os.path.exists(spiral_path):
        raise ValueError("❌ Spiral image not found")

    if not os.path.exists(wave_path):
        raise ValueError("❌ Wave image not found")

    # Preprocess
    spiral_img = preprocess(spiral_path)
    wave_img = preprocess(wave_path)

    # AI Score
    ai_score = stable_prediction(spiral_img, wave_img)

    # DISPLAY IMAGES
    spiral_thresh, spiral_edges = visualize(spiral_path, "Spiral Analysis")
    wave_thresh, wave_edges = visualize(wave_path, "Wave Analysis")

    # Features
    spiral_features = analyze_features(spiral_thresh, spiral_edges)
    wave_features = analyze_features(wave_thresh, wave_edges)

    # Feature Score
    spiral_score = compute_score(spiral_features, "spiral")
    wave_score = compute_score(wave_features, "wave")
    feature_score = (0.6 * spiral_score) + (0.4 * wave_score)

    # Final Score
    final_score = (0.7 * ai_score) + (0.3 * feature_score)

    # Prediction
    if final_score >= 0.6:
        prediction = "Dementia"
    elif final_score <= 0.4:
        prediction = "Healthy"
    else:
        prediction = "Uncertain"

    return {
        "prediction": prediction,
        "confidence": float(final_score),
        "ai_score": float(ai_score),
        "feature_score": float(feature_score),
        "spiral_features": spiral_features,
        "wave_features": wave_features
    }

# ===============================
# FINAL REPORT
# ===============================
def print_full_report(result, spiral_path, wave_path):

    print("\n===== PATIENT INPUT =====")
    print(f"Spiral Image : {spiral_path}")
    print(f"Wave Image   : {wave_path}")

    print("\n===== FINAL RESULT =====")
    print(f"Prediction   : {result['prediction']}")
    print(f"Confidence   : {result['confidence']:.2f}")

    print("\n===== AI vs FEATURE =====")
    print(f"AI Score       : {result['ai_score']}")
    print(f"Feature Score  : {result['feature_score']}")

    print("\n===== SPIRAL FEATURES =====")
    for k, v in result["spiral_features"].items():
        print(f"{k} : {v}")

    print("\n===== WAVE FEATURES =====")
    for k, v in result["wave_features"].items():
        print(f"{k} : {v}")

    print("\n===== CLINICAL INTERPRETATION =====")

    if result['confidence'] < 0.4:
        print("✔ No significant motor abnormalities detected.")
    elif result['confidence'] < 0.7:
        print("⚠ Mild irregularities detected. Further monitoring recommended.")
    else:
        print("❗ Strong signs of motor impairment. Clinical evaluation advised.")

    print("\n===== REPORT SUMMARY =====")
    print(f"Final Risk Score : {result['confidence'] * 100:.2f}%")
    print("Decision Support : AI + Motor Feature Analysis")

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":

    print("\n===== HANDWRITING ANALYSIS SYSTEM =====")

    spiral_path = "C:/Users/shali/Downloads/handwriting/sprial/dementia/aug_0_V01PE03.png"
    wave_path = "C:/Users/shali/Downloads/handwriting/wave/dementia/aug_3_V07PO03.png"

    result = predict_combined(spiral_path, wave_path)

    print_full_report(result, spiral_path, wave_path)