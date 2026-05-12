# =============================================
# FINAL EEG CN vs FTD (SMALL DATA OPTIMIZED)
# =============================================

import os
import numpy as np
from scipy.signal import welch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# CONFIG
# -----------------------------
DATASET_PATH = r"C:/Users/shali/Downloads/eeg/dataset"

SFREQ = 256
WINDOW_SIZE = 512      # 🔥 Increased
STEP_SIZE = 256
N_SPLITS = 5
RANDOM_STATE = 42

label_mapping = {'CN': 0, 'FTD': 1}

# =============================================
# FEATURE EXTRACTION (SIMPLIFIED + STRONG)
# =============================================
def extract_features(segment):

    features = []

    for ch in segment:

        ch = np.nan_to_num(ch)

        # PSD
        f, Pxx = welch(ch, fs=SFREQ, nperseg=256)

        theta = np.mean(Pxx[(f>=4) & (f<8)])
        alpha = np.mean(Pxx[(f>=8) & (f<12)])

        # Key features only 🔥
        ratio = theta / (alpha + 1e-6)
        std = np.std(ch)

        features.extend([theta, alpha, ratio, std])

    return np.array(features)


# =============================================
# LOAD SUBJECTS
# =============================================
subjects = []

for label in ['CN', 'FTD']:
    folder = os.path.join(DATASET_PATH, label)

    for file in os.listdir(folder):
        if file.endswith('.npy'):
            subjects.append((os.path.join(folder, file), label_mapping[label]))

subject_paths = [s[0] for s in subjects]
subject_labels = np.array([s[1] for s in subjects])


# =============================================
# CROSS VALIDATION (SUBJECT LEVEL)
# =============================================
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

subject_true = []
subject_pred = []

for train_idx, test_idx in skf.split(subject_paths, subject_labels):

    X_train, y_train = [], []

    # -------- TRAIN --------
    for i in train_idx:
        file_path, label = subjects[i]

        eeg = np.load(file_path)

        if eeg.shape[0] > eeg.shape[1]:
            eeg = eeg.T

        # 🔥 Use only 8 channels
        eeg = eeg[:8, :]

        eeg = (eeg - np.mean(eeg)) / (np.std(eeg) + 1e-6)

        for start in range(0, eeg.shape[1]-WINDOW_SIZE, STEP_SIZE):
            segment = eeg[:, start:start+WINDOW_SIZE]

            X_train.append(extract_features(segment))
            y_train.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # 🔥 BEST MODEL FOR SMALL DATA
    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        random_state=42
    )

    clf.fit(X_train, y_train)

    # -------- TEST (SUBJECT LEVEL) --------
    for i in test_idx:
        file_path, label = subjects[i]

        eeg = np.load(file_path)

        if eeg.shape[0] > eeg.shape[1]:
            eeg = eeg.T

        eeg = eeg[:8, :]
        eeg = (eeg - np.mean(eeg)) / (np.std(eeg) + 1e-6)

        probs = []

        for start in range(0, eeg.shape[1]-WINDOW_SIZE, STEP_SIZE):
            segment = eeg[:, start:start+WINDOW_SIZE]

            feat = extract_features(segment)
            feat = scaler.transform([feat])

            prob = clf.predict_proba(feat)[0][1]
            probs.append(prob)

        # 🔥 FINAL SUBJECT DECISION
        final_prob = np.median(probs)
        pred = 1 if final_prob > 0.5 else 0

        subject_true.append(label)
        subject_pred.append(pred)


# =============================================
# RESULTS
# =============================================
print("\nConfusion Matrix:")
print(confusion_matrix(subject_true, subject_pred))

print("\nClassification Report:")
print(classification_report(subject_true, subject_pred, target_names=['CN','FTD']))