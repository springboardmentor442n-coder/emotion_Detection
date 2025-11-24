# src/api/main.py

import os
import io
import base64
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import tensorflow as tf
from recommender.recommender import Recommender, emotion_to_tags, emotion_to_name

reco = Recommender(csv_path="data/music_clean.csv")


# Text sentiment model (VADER)
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
except Exception:
    analyzer = None
    print("⚠ Warning: VADER sentiment not available.")

# Local recommender
from src.recommender.recommender import Recommender, emotion_to_tags, emotion_to_name

# --------------------------------------------------------------------
# GLOBAL PATHS
# --------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

CSV_PATH = os.path.join(BASE_DIR, "data", "music_clean.csv")
if not os.path.exists(CSV_PATH):
    CSV_PATH = os.path.join(BASE_DIR, "data", "music.csv")

reco = Recommender(csv_path=CSV_PATH)

# --------------------------------------------------------------------
# CLEAN CNN MODEL LOADER (FINAL FIX)
# --------------------------------------------------------------------

H5_PATH = os.path.join(BASE_DIR, "src", "models", "emotion_cnn_final.h5")
KERAS_RECOVERED = os.path.join(BASE_DIR, "src", "models", "emotion_cnn_final_clean.keras")

cnn_model = None


def build_model():
    """Rebuild same CNN architecture used during training."""
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(7, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs)


def load_clean_model():
    global cnn_model

    # 1) If clean model already exists, load it
    if os.path.exists(KERAS_RECOVERED):
        cnn_model = tf.keras.models.load_model(KERAS_RECOVERED, compile=False)
        cnn_model.predict(np.zeros((1, 224, 224, 3)))
        print("✅ Loaded cleaned CNN model:", KERAS_RECOVERED)
        return

    # 2) Build fresh architecture
    model = build_model()

    # 3) Load weights from corrupted .h5 (skip mismatches)
    try:
        model.load_weights(H5_PATH, by_name=True, skip_mismatch=True)
        print("⚠ Partially loaded weights from H5 with skip_mismatch=True")
    except Exception as e:
        print("❌ Failed to load partial weights:", e)

    # 4) Save clean model
    model.save(KERAS_RECOVERED)
    print("💾 Saved clean model:", KERAS_RECOVERED)

    cnn_model = model


# Load on import
load_clean_model()

# --------------------------------------------------------------------
# FASTAPI SETUP
# --------------------------------------------------------------------

app = FastAPI(title="MoodMate API")

# Allow Streamlit access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------

def preprocess_image_bytes(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img).astype("float32")
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def make_thumbnail_b64(image_bytes, size=(128, 128)):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img.thumbnail(size)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None

# --------------------------------------------------------------------
# API ENDPOINTS
# --------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "MoodMate API running",
        "note": "CNN model loaded using clean architecture recovery."
    }


# ----------------------------- TEXT SENTIMENT -----------------------------

@app.post("/predict_text/")
async def predict_text(text: str = Form(...)):
    try:
        if analyzer is None:
            label = 6
            emotion_name = emotion_to_name[label]
            tags = emotion_to_tags[label]
            recs = reco.recommend_for_tags(tags)
            return {"success": True, "emotion_label": label,
                    "emotion_name": emotion_name, "vader": None,
                    "tags": tags, "recommendations": recs}

        vs = analyzer.polarity_scores(text)
        comp = vs["compound"]

        if comp >= 0.5:
            label = 3
        elif comp >= 0.05:
            label = 6
        elif comp <= -0.5:
            label = 0
        elif comp <= -0.05:
            label = 4
        else:
            label = 6

        emotion_name = emotion_to_name[label]
        tags = emotion_to_tags[label]
        recs = reco.recommend_for_tags(tags)

        return {
            "success": True,
            "emotion_label": label,
            "emotion_name": emotion_name,
            "vader": vs,
            "tags": tags,
            "recommendations": recs
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

# ----------------------------- IMAGE EMOTION CNN -----------------------------

@app.post("/predict_image/")
async def predict_image(file: UploadFile = File(...)):
    try:
        if cnn_model is None:
            return {"success": False, "error": "CNN model not loaded."}

        contents = await file.read()
        x = preprocess_image_bytes(contents)

        probs = cnn_model.predict(x)[0]
        label = int(np.argmax(probs))
        conf = float(np.max(probs))

        emotion_name = emotion_to_name[label]
        tags = emotion_to_tags[label]
        recs = reco.recommend_for_tags(tags)
        thumb = make_thumbnail_b64(contents)

        return {
            "success": True,
            "emotion_label": label,
            "emotion_name": emotion_name,
            "confidence": conf,
            "tags": tags,
            "recommendations": recs,
            "thumbnail": thumb
        }

    except Exception as e:
        import traceback
        return {"success": False, "error": str(e), "trace": traceback.format_exc()}
