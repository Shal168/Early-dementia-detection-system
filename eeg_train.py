import os
import numpy as np
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib

# =============================================
# CONFIGURATION
# =============================================

DATASET_PATH = r"C:/Users/shali/Downloads/eeg/dataset"

SFREQ = 256
WINDOW_SIZE = 256
STEP_SIZE = 64
N_CHANNELS = 32

label_mapping = {'CN':0,'FTD':1}

MODEL_DIR = "../models"

os.makedirs(MODEL_DIR, exist_ok=True)

# =============================================
# FEATURE EXTRACTION
# =============================================

def extract_psd_features(segment):

    features=[]

    for ch_data in segment:

        ch_data=np.nan_to_num(ch_data)

        f,Pxx = welch(ch_data,fs=SFREQ,nperseg=len(ch_data))

        delta=Pxx[(f>=0.5)&(f<4)].mean()
        theta=Pxx[(f>=4)&(f<8)].mean()
        alpha=Pxx[(f>=8)&(f<12)].mean()
        beta=Pxx[(f>=12)&(f<30)].mean()

        total=delta+theta+alpha+beta+1e-6

        features.extend([
            delta/total,
            theta/total,
            alpha/total,
            beta/total
        ])

    return np.array(features)

# =============================================
# LOAD DATASET
# =============================================

X=[]
y=[]

for label in ['CN','FTD']:

    folder=os.path.join(DATASET_PATH,label)

    files=[f for f in os.listdir(folder) if f.endswith(".npy")]

    print(label,"files:",len(files))

    for file in files:

        file_path=os.path.join(folder,file)

        eeg=np.load(file_path)

        eeg=np.nan_to_num(eeg)

        if eeg.shape[0] >= eeg.shape[1]:
            eeg=eeg.T

        eeg=eeg[:N_CHANNELS,:]

        total_samples=eeg.shape[1]

        for start in range(0,total_samples-WINDOW_SIZE+1,STEP_SIZE):

            segment=eeg[:,start:start+WINDOW_SIZE]

            features=extract_psd_features(segment)

            X.append(features)

            y.append(label_mapping[label])

X=np.array(X)
y=np.array(y)

print("Total training samples:",len(X))

# =============================================
# FEATURE SCALING
# =============================================

scaler=StandardScaler()

X=scaler.fit_transform(X)

# =============================================
# TRAIN MODEL
# =============================================

clf=XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric='logloss'
)

clf.fit(X,y)

print("Model training completed")

# =============================================
# SAVE MODEL
# =============================================

joblib.dump(clf, os.path.join(MODEL_DIR,"eeg_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR,"eeg_scaler.pkl"))

print("Model saved in models folder")