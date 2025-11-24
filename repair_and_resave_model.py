# src/models/repair_and_resave_model.py
"""
Run this once if loading the .h5 raises "Unknown layer: 'TrueDivide'".
It attempts to register a fallback object, load the H5 model, and re-save
in native Keras format (.keras) which is more robust.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Lambda
from tensorflow.keras.utils import get_custom_objects

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
h5_path = os.path.join(BASE_DIR, "src", "models", "emotion_cnn_final.h5")
out_path = os.path.join(BASE_DIR, "src", "models", "emotion_cnn_final.keras")

def _true_divide(x):
    # fallback matching common MobileNetV2 preprocess; tweak if your model used different scaling
    return tf.math.divide(x, 127.5)

get_custom_objects()["TrueDivide"] = Lambda(_true_divide)

print("Loading H5 model from:", h5_path)
model = tf.keras.models.load_model(h5_path, compile=False)
print("Model loaded. Saving as native Keras format:", out_path)
model.save(out_path)
print("Saved. You can now load the .keras file without HDF5 custom object issues.")
