from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import matplotlib
matplotlib.use('Agg')  # ✅ NO GUI (VERY IMPORTANT)

import matplotlib.pyplot as plt
from scipy.signal import welch
import base64
import io
import os
import uuid

from analysis.speech import predict
import tempfile
from analysis.handwriting import predict_combined
from analysis.facial import predict_face
from analysis.cog import analyze_cognitive as analyze_cognitive_model

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===============================
# HOME
# ===============================
@app.route("/")
def home():
    return "Backend is running successfully 🚀"

# ===============================
# TEST
# ===============================
@app.route("/api/test")
def test():
    return jsonify({"message": "Backend connected successfully!"})

# ===============================
# VOICE ANALYSIS
# ===============================
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import matplotlib
matplotlib.use('Agg')  # ✅ NO GUI (VERY IMPORTANT)

import matplotlib.pyplot as plt
from scipy.signal import welch
import base64
import io
import os
import uuid

from analysis.speech import predict
import tempfile
from analysis.handwriting import predict_combined
from analysis.facial import predict_face
from analysis.cog import analyze_cognitive as analyze_cognitive_model

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===============================
# HOME
# ===============================
@app.route("/")
def home():
    return "Backend is running successfully 🚀"

# ===============================
# TEST
# ===============================
@app.route("/api/test")
def test():
    return jsonify({"message": "Backend connected successfully!"})

# ===============================
# VOICE ANALYSIS
# ===============================
@app.route("/analyze/voice", methods=["POST"])
def analyze_voice():
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            file.save(tmp.name)
            response = predict(file_path=tmp.name)

        return response

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===============================
# EEG ANALYSIS
# ===============================
@app.route("/analyze/eeg", methods=["POST"])
def analyze_eeg():
    file = request.files.get("eeg")

    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        eeg_data = np.load(file)
    except:
        eeg_data = np.loadtxt(file, delimiter=",")

    if eeg_data.ndim == 1:
        eeg_data = eeg_data.reshape(-1, 1)

    signal = eeg_data[:, 0]

    freqs, psd = welch(signal, fs=256)

    plt.figure(figsize=(6, 3))
    plt.plot(freqs, psd)
    plt.title("EEG PSD")
    plt.xlim(0, 50)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    image_base64 = base64.b64encode(buf.read()).decode("utf-8")

    return jsonify({
        "result": "Normal",
        "confidence": 0.7,
        "eegImage": f"data:image/png;base64,{image_base64}"
    })


# ===============================
# HANDWRITING ANALYSIS
# ===============================
@app.route("/analyze/handwriting", methods=["POST"])
def analyze_handwriting():
    try:
        spiral_file = request.files.get("spiral")
        wave_file = request.files.get("wave")

        if not spiral_file or not wave_file:
            return jsonify({"error": "Both files required"}), 400

        # Save files
        spiral_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_spiral.png")
        wave_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_wave.png")

        spiral_file.save(spiral_path)
        wave_file.save(wave_path)

        # Run model
        result = predict_combined(spiral_path, wave_path)

        # Cleanup
        try:
            os.remove(spiral_path)
            os.remove(wave_path)
        except:
            pass

        # ===============================
        # ✅ LABEL LOGIC (CORRECT PLACE)
        # ===============================
        spiral_tremor = result["spiral_features"]["tremor"]
        wave_tremor = result["wave_features"]["tremor"]

        def get_label(t):
            if t > 0.6:
                return "Severe Tremor"
            elif t > 0.3:
                return "Mild Tremor"
            else:
                return "Normal"

        # ===============================
        # ✅ RETURN RESPONSE
        # ===============================
        return jsonify({
            "final_result": result["prediction"],
            "confidence": round(result["confidence"] * 100, 2),

            "ai_score": round(result["ai_score"] * 100, 2),
            "feature_score": round(result["feature_score"] * 100, 2),

            "details": {
                "spiral": {
                    "result": get_label(spiral_tremor),
                    "confidence": round(spiral_tremor * 100, 2),
                    "features": result["spiral_features"]
                },
                "wave": {
                    "result": get_label(wave_tremor),
                    "confidence": round(wave_tremor * 100, 2),
                    "features": result["wave_features"]
                }
            }
        })

    except Exception as e:
        print("❌ HANDWRITING ERROR:", e)
        return jsonify({"error": str(e)}), 500
    
# ===============================
# FACE ANALYSIS
# ===============================
@app.route("/analyze/face", methods=["POST"])
def analyze_face():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        result = predict_face(path)

        os.remove(path)

        if result is None:
            return jsonify({"error": "Face not detected"}), 400

        faceScore = round((1 - result["risk"]) * 100, 2)

        return jsonify({
            "faceScore": faceScore,
            "emotion": result["emotion"],
            "confidence": result["confidence"],  # ✅ keep 0–1
            "probabilities": result.get("probabilities", []),  # ✅ ADD
            "labels": ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]  # ✅ ADD
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===============================
# COGNITIVE
# ===============================
@app.route("/analyze/cognitive", methods=["POST"])
def analyze_cognitive():
    data = request.json
    result = analyze_cognitive_model(data)
    return jsonify(result)


# ===============================
# FINAL RESULT
# ===============================
@app.route("/final_result", methods=["POST"])
def final_result():
    return jsonify({"final_result": "Low Dementia Risk"})


# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    print("\n📌 ACTIVE ROUTES:")
    for rule in app.url_map.iter_rules():
        print(rule)

    app.run(debug=True)

# ===============================
# EEG ANALYSIS
# ===============================
@app.route("/analyze/eeg", methods=["POST"])
def analyze_eeg():
    file = request.files.get("eeg")

    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        eeg_data = np.load(file)
    except:
        eeg_data = np.loadtxt(file, delimiter=",")

    if eeg_data.ndim == 1:
        eeg_data = eeg_data.reshape(-1, 1)

    signal = eeg_data[:, 0]

    freqs, psd = welch(signal, fs=256)

    plt.figure(figsize=(6, 3))
    plt.plot(freqs, psd)
    plt.title("EEG PSD")
    plt.xlim(0, 50)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    image_base64 = base64.b64encode(buf.read()).decode("utf-8")

    return jsonify({
        "result": "Normal",
        "confidence": 0.7,
        "eegImage": f"data:image/png;base64,{image_base64}"
    })


# ===============================
# HANDWRITING ANALYSIS
# ===============================
@app.route("/analyze/handwriting", methods=["POST"])
def analyze_handwriting():
    try:
        spiral_file = request.files.get("spiral")
        wave_file = request.files.get("wave")

        if not spiral_file or not wave_file:
            return jsonify({"error": "Both files required"}), 400

        # Save files
        spiral_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_spiral.png")
        wave_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_wave.png")

        spiral_file.save(spiral_path)
        wave_file.save(wave_path)

        # Run model
        result = predict_combined(spiral_path, wave_path)

        # Cleanup
        try:
            os.remove(spiral_path)
            os.remove(wave_path)
        except:
            pass

        # ===============================
        # ✅ LABEL LOGIC (CORRECT PLACE)
        # ===============================
        spiral_tremor = result["spiral_features"]["tremor"]
        wave_tremor = result["wave_features"]["tremor"]

        def get_label(t):
            if t > 0.6:
                return "Severe Tremor"
            elif t > 0.3:
                return "Mild Tremor"
            else:
                return "Normal"

        # ===============================
        # ✅ RETURN RESPONSE
        # ===============================
        return jsonify({
            "final_result": result["prediction"],
            "confidence": round(result["confidence"] * 100, 2),

            "ai_score": round(result["ai_score"] * 100, 2),
            "feature_score": round(result["feature_score"] * 100, 2),

            "details": {
                "spiral": {
                    "result": get_label(spiral_tremor),
                    "confidence": round(spiral_tremor * 100, 2),
                    "features": result["spiral_features"]
                },
                "wave": {
                    "result": get_label(wave_tremor),
                    "confidence": round(wave_tremor * 100, 2),
                    "features": result["wave_features"]
                }
            }
        })

    except Exception as e:
        print("❌ HANDWRITING ERROR:", e)
        return jsonify({"error": str(e)}), 500
    
# ===============================
# FACE ANALYSIS
# ===============================
@app.route("/analyze/face", methods=["POST"])
def analyze_face():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        result = predict_face(path)

        os.remove(path)

        if result is None:
            return jsonify({"error": "Face not detected"}), 400

        faceScore = round((1 - result["risk"]) * 100, 2)

        return jsonify({
            "faceScore": faceScore,
            "emotion": result["emotion"],
            "confidence": result["confidence"],  # ✅ keep 0–1
            "probabilities": result.get("probabilities", []),  # ✅ ADD
            "labels": ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]  # ✅ ADD
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===============================
# COGNITIVE
# ===============================
@app.route("/analyze/cognitive", methods=["POST"])
def analyze_cognitive():
    data = request.json
    result = analyze_cognitive_model(data)
    return jsonify(result)


# ===============================
# FINAL RESULT
# ===============================
@app.route("/final_result", methods=["POST"])
def final_result():
    return jsonify({"final_result": "Low Dementia Risk"})


# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    print("\n📌 ACTIVE ROUTES:")
    for rule in app.url_map.iter_rules():
        print(rule)

    app.run(debug=True)