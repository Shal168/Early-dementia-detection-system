# speech.py

import os
import time
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import joblib
import parselmouth

# ==========================
# PARAMETERS
# ==========================
SAMPLE_RATE = 16000
DURATION = 8
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "speech_model.pkl")
OUTPUT_AUDIO = os.path.join(BASE_DIR, "outputs", "patient_audio.wav")

# ==========================
# LOAD MODEL
# ==========================
model = joblib.load(MODEL_PATH)

# ==========================
# AUDIO HELPERS
# ==========================
def make_fixed_length(audio, sr, duration):
    target_length = int(sr * duration)
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]
    return audio


def preprocess_audio(audio):
    audio = librosa.util.normalize(audio)
    audio, _ = librosa.effects.trim(audio)
    return audio


# ==========================
# FEATURE EXTRACTION
# ==========================
def extract_features(audio):

    audio = preprocess_audio(audio)
    audio = make_fixed_length(audio, SAMPLE_RATE, DURATION)

    # -------- MFCC (MODEL INPUT) --------
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    # -------- SPECTRAL --------
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=SAMPLE_RATE))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=SAMPLE_RATE))
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    rms = np.mean(librosa.feature.rms(y=audio))

    # -------- PARSELMOUTH --------
    pitch_mean = jitter_local = shimmer_local = hnr_value = 0
    f1 = f2 = f3 = 0

    try:
        snd = parselmouth.Sound(audio, SAMPLE_RATE)

        # Pitch
        pitch = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=300)
        pitch_mean = parselmouth.praat.call(pitch, "Get mean", 0, 0, "Hertz")

        # Jitter & Shimmer
        try:
            point_process = parselmouth.praat.call(
                snd, "To PointProcess (periodic, cc)", 75, 300
            )

            jitter_local = parselmouth.praat.call(
                point_process, "Get jitter (local)",
                0, 0, 0.0001, 0.02, 1.3
            )

            shimmer_local = parselmouth.praat.call(
                [snd, point_process], "Get shimmer (local)",
                0, 0, 0.0001, 0.02, 1.3, 1.6
            )
        except:
            pass

        # HNR
        try:
            hnr = parselmouth.praat.call(
                snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0
            )
            hnr_value = parselmouth.praat.call(hnr, "Get mean", 0, 0)
        except:
            pass

        # 🔥 FORMANTS FIX (FINAL)
        try:
            snd_pre = snd.copy()
            snd_pre.pre_emphasize()

            formant = snd_pre.to_formant_burg(
                time_step=0.01,
                max_number_of_formants=5,
                maximum_formant=5000,
                window_length=0.025
            )

            duration = snd.get_total_duration()

            times = np.linspace(0.1, duration - 0.1, 5)

            f1_vals, f2_vals, f3_vals = [], [], []

            for t in times:
                f1_val = formant.get_value_at_time(1, t)
                f2_val = formant.get_value_at_time(2, t)
                f3_val = formant.get_value_at_time(3, t)

                if f1_val and not np.isnan(f1_val):
                    f1_vals.append(f1_val)
                if f2_val and not np.isnan(f2_val):
                    f2_vals.append(f2_val)
                if f3_val and not np.isnan(f3_val):
                    f3_vals.append(f3_val)

            f1 = np.mean(f1_vals) if f1_vals else 0
            f2 = np.mean(f2_vals) if f2_vals else 0
            f3 = np.mean(f3_vals) if f3_vals else 0

        except Exception as e:
            print("Formant warning:", e)

    except Exception as e:
        print("Parselmouth warning:", e)

    # -------- OUTPUT --------
    extra_features = {
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "spectral_rolloff": spectral_rolloff,
        "zcr": zcr,
        "rms": rms,
        "pitch_mean": pitch_mean,
        "jitter_local": jitter_local,
        "shimmer_local": shimmer_local,
        "hnr_value": hnr_value,
        "formants": (f1, f2, f3)
    }

    return mfcc_mean, mfcc, extra_features


# ==========================
# RECORD AUDIO
# ==========================
def record_audio(duration=DURATION, sr=SAMPLE_RATE, filename=OUTPUT_AUDIO):
    print(f"\n🎤 Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    audio = audio.flatten()

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    sf.write(filename, audio, sr)

    print("✅ Recording complete")
    return audio


# ==========================
# VISUALIZATION
# ==========================
def plot_waveform(audio):
    plt.figure(figsize=(10, 3))
    plt.plot(audio)
    plt.title("Voice Waveform")
    plt.show()


def plot_mfcc(mfcc):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title("MFCC")
    plt.show()


# ==========================
# STABLE PREDICTION
# ==========================
def stable_prediction(features):
    probs = []
    for _ in range(3):
        prob = model.predict_proba(features)[0][1]
        probs.append(prob)
    return np.mean(probs)


# ==========================
# PREDICT
# ==========================
from flask import jsonify

def predict(audio_array=None, file_path=None):

    start = time.time()

    if file_path:
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    else:
        audio = audio_array

    mfcc_mean, mfcc, extra_features = extract_features(audio)

    features_input = mfcc_mean.reshape(1, -1)
    prob = stable_prediction(features_input)

    # Clinical decision
    if prob >= 0.6:
        prediction = "Dementia"
    elif prob <= 0.4:
        prediction = "Healthy"
    else:
        prediction = "Uncertain"

    end = time.time()

    return jsonify({
        "prediction": prediction,
        "probability": float(prob),
        "time": round(end - start, 2),
        "features": {
            "spectral_centroid": float(extra_features["spectral_centroid"]),
            "spectral_bandwidth": float(extra_features["spectral_bandwidth"]),
            "spectral_rolloff": float(extra_features["spectral_rolloff"]),
            "zcr": float(extra_features["zcr"]),
            "rms": float(extra_features["rms"]),
            "pitch_mean": float(extra_features["pitch_mean"]),
            "jitter_local": float(extra_features["jitter_local"]),
            "shimmer_local": float(extra_features["shimmer_local"]),
            "hnr_value": float(extra_features["hnr_value"]),
            "formants": [
                float(extra_features["formants"][0]),
                float(extra_features["formants"][1]),
                float(extra_features["formants"][2])
            ]
        },
        "mfcc": mfcc_mean.tolist()
    })

# ==========================
# MAIN
# ==========================
if __name__ == "__main__":

    print("\n===== SPEECH ANALYSIS SYSTEM =====")
    print("1. Record Audio")
    print("2. Use Audio File")

    choice = input("Enter choice: ")

    if choice == "1":
        print("\n👉 Speak clearly (try 'aaaaa' for best formants)")
        audio = record_audio()
        result = predict(audio_array=audio)

    elif choice == "2":
        file_path = "D:/dataset/nodementia/BillWyman_3.wav"

        if not os.path.exists(file_path):
            print("❌ File not found")
            exit()

        result = predict(file_path=file_path)

    else:
        print("❌ Invalid choice")
        exit()

    # ==========================
    # OUTPUT
    # ==========================
    print("\n===== FINAL RESULT =====")
    print(f"Prediction   : {result['prediction']}")
    print(f"Probability  : {result['probability']:.2f}")
    print(f"Test Time    : {result['time']} seconds")

    if result["prediction"] == "Uncertain":
        print("⚠️ Borderline case. Recommend further evaluation.")

    print("\n===== FEATURES =====")
    for k, v in result["features"].items():
        print(f"{k}: {v}")

    # Helpful notes
    if result["features"]["formants"] == (0, 0, 0):
        print("\nNote: Formants not detected (requires clear vowel sound)")

    if result["features"]["jitter_local"] == 0:
        print("Note: Jitter/Shimmer require sustained voice (e.g., 'aaaa')")