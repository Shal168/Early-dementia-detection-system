import tensorflow as tf
import pandas as pd
import numpy as np
import os

os.makedirs("models", exist_ok=True)

data = pd.read_excel("D:/facial.csv.xlsx")

X = []
for pixel in data["pixels"]:
    img = np.array(pixel.split(), dtype="float32").reshape(48,48)
    X.append(img)

X = np.array(X).reshape(-1,48,48,1) / 255.0
y = tf.keras.utils.to_categorical(data["emotion"],7)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation="relu",input_shape=(48,48,1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation="relu"),
    tf.keras.layers.Dense(7,activation="softmax")
])

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

model.fit(X,y,epochs=10)

model.save("models/facial_model.h5")

print("Facial model saved")