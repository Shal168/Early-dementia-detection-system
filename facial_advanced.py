# ============================================================
# ADVANCED FACIAL EMOTION + DEMENTIA ANALYSIS SYSTEM
# USING TRANSFER LEARNING (MobileNetV2)
# ============================================================

import os
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# ============================================================
# SETUP
# ============================================================

os.makedirs("models", exist_ok=True)

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# ============================================================
# LOAD DATA
# ============================================================

file_path = r"D:/facial.csv.xlsx"

if file_path.endswith(".csv"):
    data = pd.read_csv(file_path)
else:
    data = pd.read_excel(file_path)

# ============================================================
# PREPROCESS DATA
# ============================================================

pixels = data['pixels'].tolist()
X = []

for pixel in pixels:
    face = np.array(pixel.split(), dtype='float32').reshape(48,48)

    # Convert to 224x224 RGB
    face = tf.image.resize(face[..., np.newaxis], (224,224))
    face = np.repeat(face, 3, axis=-1)

    X.append(face)

X = np.array(X) / 255.0
y = to_categorical(data['emotion'], num_classes=7)

# ============================================================
# TRAIN-VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ============================================================
# DATA AUGMENTATION
# ============================================================

datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# ============================================================
# CLASS WEIGHTS (for imbalanced classes)
# ============================================================

y_labels = np.argmax(y_train, axis=1)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_labels),
    y=y_labels
)
class_weights = dict(enumerate(class_weights))

# ============================================================
# BUILD FACIAL NETWORK (TRANSFER LEARNING)
# ============================================================

base_model = MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)

# Freeze base initially
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)

output = Dense(7, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ============================================================
# CALLBACKS
# ============================================================

early_stop = EarlyStopping(patience=6, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(patience=3, factor=0.3)

# ============================================================
# TRAIN PHASE 1 (Frozen Base)
# ============================================================

model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=5,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr]
)

# ============================================================
# FINE-TUNE LAST LAYERS
# ============================================================

for layer in base_model.layers[-50:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=10,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr]
)

# ============================================================
# SAVE MODEL
# ============================================================

model.save("models/advanced_facial_model.h5")

# ============================================================
# PREDICTIONS
# ============================================================

predictions = model.predict(X)
predicted_classes = np.argmax(predictions, axis=1)
predicted_emotions = [emotion_labels[i] for i in predicted_classes]
predicted_confidence = np.max(predictions, axis=1)

# ============================================================
# BEHAVIORAL ANALYSIS
# ============================================================

emotion_counts = pd.Series(predicted_emotions).value_counts()
dominant_emotion = emotion_counts.idxmax()
dominant_percentage = (emotion_counts.max() / len(predicted_emotions)) * 100
emotion_changes = np.sum(predicted_classes[1:] != predicted_classes[:-1])
reaction_score = emotion_changes / len(predicted_classes)
emotion_variance = np.var(predicted_classes)
stability_score = 1 / (1 + emotion_variance)
neutral_index = emotion_labels.index("Neutral")
neutral_frames = np.sum(predicted_classes == neutral_index)
neutral_percentage = (neutral_frames / len(predicted_classes)) * 100
flat_status = "High Flat Emotion (Risk)" if neutral_percentage > 50 else "Normal Variation"

risk = neutral_percentage*0.4
if reaction_score < 0.05: risk += 20
if stability_score < 0.2: risk += 20
risk += dominant_percentage*0.2
final_risk = min(risk, 100)

# ============================================================
# SAVE RESULTS
# ============================================================

data['Predicted_Emotion'] = predicted_emotions
data['Confidence'] = (predicted_confidence*100).round(2)
data['Dominant_Emotion'] = dominant_emotion
data['Dominant_%'] = round(dominant_percentage,2)
data['Reaction_Score'] = round(reaction_score,4)
data['Emotional_Stability'] = round(stability_score,4)
data['Neutral_%'] = round(neutral_percentage,2)
data['Flat_Status'] = flat_status
data['Final_Behavioral_Risk_%'] = round(final_risk,2)

output_path = r"C:\Users\shali\OneDrive\Desktop\web designing series\FINAL_DEMENTIA_ANALYSIS.xlsx"
data.to_excel(output_path, index=False)

# ============================================================
# EVALUATION
# ============================================================

y_true = data['emotion'].values
y_pred = predicted_classes

print("Overall Accuracy:", round(accuracy_score(y_true,y_pred)*100,2), "%")
print(classification_report(y_true, y_pred, target_names=emotion_labels))
print(confusion_matrix(y_true, y_pred))

print("\nDominant Emotion:", dominant_emotion)
print("Dominant Percentage:", round(dominant_percentage,2), "%")
print("Reaction Score:", round(reaction_score,4))
print("Emotional Stability:", round(stability_score,4))
print("Neutral Percentage:", round(neutral_percentage,2), "%")
print("Flat Emotion Status:", flat_status)
print("Final Dementia Behavioral Risk:", round(final_risk,2), "%")