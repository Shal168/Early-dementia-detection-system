import os
import numpy as np
import librosa
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

DATASET_PATH = r"D:/dataset"
SAMPLE_RATE = 16000
DURATION = 1.5

labels = ["nodementia","dementia"]

def make_fixed_length(audio, sr, duration):

    target_length = int(sr*duration)

    if len(audio) < target_length:
        audio = np.pad(audio,(0,target_length-len(audio)))
    else:
        audio = audio[:target_length]

    return audio


def extract_features(audio):

    audio = make_fixed_length(audio,SAMPLE_RATE,DURATION)

    mfcc = librosa.feature.mfcc(y=audio,sr=SAMPLE_RATE,n_mfcc=13)

    return np.mean(mfcc.T,axis=0)


X=[]
y=[]

for label in labels:

    folder=os.path.join(DATASET_PATH,label)

    for file in os.listdir(folder):

        path=os.path.join(folder,file)

        audio,_=librosa.load(path,sr=SAMPLE_RATE)

        features=extract_features(audio)

        X.append(features)

        y.append(labels.index(label))


X=np.array(X)
y=np.array(y)

print("Dataset loaded:",len(X))


X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,stratify=y,random_state=42
)


model=RandomForestClassifier(n_estimators=300)

model.fit(X_train,y_train)

print("Model trained")

os.makedirs("models",exist_ok=True)

joblib.dump(model,"models/speech_model.pkl")

print("Model saved in models/speech_model.pkl")