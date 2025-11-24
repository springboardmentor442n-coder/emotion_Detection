# src/models/rebuild_and_save_weights.py
import os
from pathlib import Path
import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parents[2]  # moodmate/
H5_PATH = BASE_DIR / "src" / "models" / "emotion_cnn_final.h5"
OUT_KERAS = BASE_DIR / "src" / "models" / "emotion_cnn_final_recovered.keras"

print("H5 path:", H5_PATH)
if not H5_PATH.exists():
    raise SystemExit("H5 model not found at: " + str(H5_PATH))

IMG_SIZE = (224, 224)
NUM_CLASSES = 7

def build_model():
    # must match the training architecture exactly
    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.layers.Input(shape=IMG_SIZE + (3,))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model

print("Building model architecture (MobileNetV2 base + head)...")
model = build_model()
model.summary()

# Try loading weights using different approaches:
loaded = False
try:
    print("Attempt 1: load_weights(h5_path) (exact format)...")
    model.load_weights(str(H5_PATH))
    loaded = True
    print("Loaded weights with model.load_weights(h5_path).")
except Exception as e1:
    print("Attempt1 failed:", repr(e1))

if not loaded:
    try:
        print("Attempt 2: load_weights(h5_path, by_name=True) (match by layer names)...")
        model.load_weights(str(H5_PATH), by_name=True)
        loaded = True
        print("Loaded weights by name from H5.")
    except Exception as e2:
        print("Attempt2 failed:", repr(e2))

if not loaded:
    # As a last resort try to extract weights directly from the HDF5 file using h5py
    try:
        import h5py
        print("Attempt 3: trying to read weights from HDF5 'model_weights' group and set by name.")
        with h5py.File(str(H5_PATH), 'r') as f:
            if 'model_weights' in f:
                weights_group = f['model_weights']
                # iterate model layers and set weights where available
                set_count = 0
                for layer in model.layers:
                    if layer.name in weights_group:
                        g = weights_group[layer.name]
                        # collect weights arrays
                        w = []
                        for k in g.keys():
                            # k could be kernel, bias etc. But order is what keras expects.
                            # We'll attempt to load all datasets in the subgroup.
                            ds = g[k]
                            w.append(np.array(ds))
                        try:
                            layer.set_weights(w)
                            set_count += 1
                        except Exception:
                            # skipping if shapes mismatch
                            pass
                print(f"Attempt3: set weights for {set_count} layers (may be partial).")
                loaded = set_count > 0
            else:
                print("H5 file does not have 'model_weights' group.")
    except Exception as e3:
        print("Attempt3 failed or h5py not available:", repr(e3))

if not loaded:
    raise SystemExit("Failed to load weights from H5 (all attempts failed). At this point you can share the H5 traceback and I'll adapt further.")

# If loaded (fully or partially), save the recovered model in native format:
print("Saving recovered model to:", OUT_KERAS)
model.save(str(OUT_KERAS))
print("Saved recovered model. You can now load this file with tf.keras.models.load_model() without HDF5 custom-object issues.")
