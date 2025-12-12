import tensorflow as tf
from tensorflow.keras import layers, models

IMG = (48,48,3)
CLASSES = 7

def build_model():
    inputs = layers.Input(shape=IMG)
    x = layers.Resizing(96,96)(inputs)

    base = tf.keras.applications.MobileNetV2(
        input_shape=(96,96,3), include_top=False, weights="imagenet")
    base.trainable = False

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(CLASSES, activation="softmax")(x)

    model = models.Model(inputs, out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def load_model(path="models/emotion_model.h5"):
    return tf.keras.models.load_model(path)
