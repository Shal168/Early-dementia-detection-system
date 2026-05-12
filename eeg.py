import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import welch

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

SFREQ = 256


# ================= LOAD EEG =================
def load_eeg(file_path):
    try:
        eeg_data = np.load(file_path)
    except:
        eeg_data = np.loadtxt(file_path, delimiter=",")

    if eeg_data.ndim == 1:
        eeg_data = eeg_data.reshape(-1, 1)

    return eeg_data


# ================= PSD FEATURES =================
def extract_psd_features(signal):
    nperseg = min(len(signal), SFREQ * 2)
    freqs, psd = welch(signal, fs=SFREQ, nperseg=nperseg)

    bands = [
        (0.5, 4),   # Delta
        (4, 8),     # Theta
        (8, 12),    # Alpha
        (12, 30),   # Beta
        (30, 50)    # Gamma
    ]

    features = []
    for low, high in bands:
        idx = np.logical_and(freqs >= low, freqs <= high)
        features.append(np.mean(psd[idx]))

    return np.array(features)


# ================= FEATURE EXTRACTION =================
def extract_features(eeg_data):
    features_all = []

    for ch in range(eeg_data.shape[1]):
        signal = eeg_data[:, ch]

        # PSD features
        psd_feat = extract_psd_features(signal)

        # Statistical features
        stats = [
    np.mean(signal),
    np.std(signal),
    np.max(signal),
    np.min(signal),
    np.median(signal),
    np.var(signal)
]

        combined = np.concatenate([psd_feat, stats])
        features_all.append(combined)

    features_all = np.array(features_all)

    # 🔥 KEY STEP: Reduce dimension (IMPORTANT)
    mean_feat = features_all.mean(axis=0)
    std_feat = features_all.std(axis=0)

    return np.concatenate([mean_feat, std_feat])


# ================= LOAD DATASET =================
def load_dataset(base_path):
    X = []
    y = []

    class_map = {
        "CN": 0,
        "FTD": 1
    }

    for folder, label in class_map.items():
        folder_path = os.path.join(base_path, folder)

        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} not found")
            continue

        for file in os.listdir(folder_path):
            if file.endswith(".npy") or file.endswith(".csv.npy"):
                file_path = os.path.join(folder_path, file)

                eeg_data = load_eeg(file_path)
                features = extract_features(eeg_data)

                X.append(features)
                y.append(label)

    return np.array(X), np.array(y)


# ================= TRAIN MODEL =================
def train_model(X, y):
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier

    if len(np.unique(y)) < 2:
        print("ERROR: Need at least 2 classes")
        return None

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        class_weight='balanced',
        random_state=42
    )

    # 🔥 Cross-validated predictions
    y_pred = cross_val_predict(model, X, y, cv=5)

    # ================= METRICS =================
    print("\n===== FINAL EEG METRICS (CN vs FTD) =====")

    # Accuracy
    acc = accuracy_score(y, y_pred)
    print(f"Accuracy  : {acc:.2f}")

    # Classification report (best for viva 🔥)
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=["CN", "FTD"]))

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["CN", "FTD"]
    )
    disp.plot()
    plt.title("Confusion Matrix - EEG (CN vs FTD)")
    plt.show()

    return model

# ================= MAIN =================
if __name__ == "__main__":

    BASE_PATH = r"C:/Users/shali/Downloads/eeg/dataset"

    print("Loading dataset...")
    X, y = load_dataset(BASE_PATH)

    print("Feature shape:", X.shape)
    print("Labels shape:", y.shape)

    print("Running final model...")
    model = train_model(X, y)