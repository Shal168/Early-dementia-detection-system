import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split

IMG_SIZE = 128

# ===============================
# LOAD DATA FUNCTION
# ===============================
def load_images(folder):
    data = []
    labels = []

    for label_type in ["healthy", "dementia"]:
        path = os.path.join(folder, label_type)

        for file in os.listdir(path):
            img_path = os.path.join(path, file)

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0

            data.append(img)
            labels.append(0 if label_type == "healthy" else 1)

    return np.array(data), np.array(labels)

# Load spiral & wave
spiral_X, y = load_images("C:/Users/shali/Downloads/handwriting/sprial")
wave_X, _ = load_images("C:/Users/shali/Downloads/handwriting/wave")

# Split
X_train_s, X_test_s, y_train, y_test = train_test_split(
    spiral_X, y, test_size=0.2, random_state=42
)

X_train_w, X_test_w, _, _ = train_test_split(
    wave_X, y, test_size=0.2, random_state=42
)

# ===============================
# BUILD MODEL
# ===============================
from tensorflow.keras.models import clone_model

def build_branch(prefix):

    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )

    base.trainable = False

    # 🔥 CLONE MODEL WITH UNIQUE NAMES
    def clone_fn(layer):
        config = layer.get_config()
        config["name"] = prefix + "_" + config["name"]
        return layer.__class__.from_config(config)

    base = clone_model(base, clone_function=clone_fn)

    x = base.output
    x = layers.GlobalAveragePooling2D(name=prefix + "_gap")(x)
    x = layers.Dense(64, activation='relu', name=prefix + "_dense")(x)

    return models.Model(inputs=base.input, outputs=x, name=prefix)

spiral_branch = build_branch("spiral_branch")
wave_branch = build_branch("wave_branch")

# Combine
combined = layers.concatenate([
    spiral_branch.output,
    wave_branch.output
])

x = layers.Dense(128, activation='relu')(combined)
x = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(
    inputs=[spiral_branch.input, wave_branch.input],
    outputs=output
)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ===============================
# TRAIN
# ===============================
model.fit(
    [X_train_s, X_train_w],
    y_train,
    validation_data=([X_test_s, X_test_w], y_test),
    epochs=15,
    batch_size=16
)

# SAVE
model.save("spiral_wave_model.keras")
print("Model saved!")