import streamlit as st
import numpy as np
from PIL import Image
import time
import tensorflow as tf
from mtcnn import MTCNN
from model import load_model
from recommender import ContentRecommender, EMOTION_TO_TAGS
import difflib

st.set_page_config(page_title="MoodMate", layout="wide")

# Load resources
@st.cache_resource
def load_resources():
    emotion_model = load_model("models/emotion_model.h5")
    recommender = ContentRecommender.load("models/recommender.joblib")
    detector = MTCNN()
    return emotion_model, recommender, detector

emotion_model, recommender, detector = load_resources()

# Extract face using MTCNN (safe convert to RGB)
def extract_face(image):
    image = image.convert("RGB")
    img_array = np.array(image)
    results = detector.detect_faces(img_array)
    if not results:
        return None  # No face detected
    x, y, w, h = results[0]["box"]
    # ensure positive coordinates & bounds
    x, y = max(0, x), max(0, y)
    h = max(1, h); w = max(1, w)
    face = img_array[y:y+h, x:x+w]
    return Image.fromarray(face)

# Predict emotion (RGB -> 48x48)
def predict_emotion(image):
    img = image.convert("RGB").resize((48, 48))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 48, 48, 3)
    preds = emotion_model.predict(img_array)[0]
    label_index = int(np.argmax(preds))
    conf = float(preds[label_index])
    EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    return EMOTION_LABELS[label_index], conf

# ---------------------------
# Robust text mood detection
# ---------------------------
# Mapping of emotion -> trigger keywords (includes synonyms)
TEXT_KEYWORDS = {
    "happy": ["happy","happiness","joy","joyful","joyous","elated","glee","cheer","cheerful","delight","content","pleased","glad","smile","smiling","jolly"],
    "sad":   ["sad","sadness","sorrow","depressed","down","downcast","blue","tear","cry","lonely","heartbroken","melancholy","gloom"],
    "angry": ["angry","anger","mad","furious","annoyed","irritated","rage","resentment","hate","annoying"],
    "fear":  ["fear","scared","afraid","anxious","anxiety","panic","terrified","nervous","worried"],
    "surprise":["surprise","surprised","astonished","shocked","amazed","startled","wow"],
    "neutral":["neutral","okay","fine","meh","so-so","normal","alright","calm","okayish","okay"],
    "disgust":["disgust","disgusted","repulsed","gross","nausea","offended"],
    "love": ["love","loving","in love","adore","adored","cherish","affection","darling","sweetheart"]
}

# Flatten all keywords for fuzzy matching
ALL_KEYWORDS = []
for klist in TEXT_KEYWORDS.values():
    ALL_KEYWORDS.extend(klist)

def detect_mood_from_text(user_text):
    """
    Robust detection:
     1. normalize text and split into words
     2. check exact keyword membership
     3. if no exact match, use difflib.get_close_matches on each token (fuzzy)
     4. if still nothing, try simple polarity via TextBlob if available (optional)
     5. fallback to 'neutral'
    Returns emotion label that matches recommender expectation (happy,sad,angry,fear,surprise,neutral,disgust,love)
    """
    text = user_text.lower().strip()
    if not text:
        return "neutral"

    # simple normalization: replace punctuation with spaces
    import re
    text_clean = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text_clean.split() if t]

    # 1) exact token match
    for token in tokens:
        for emo, keys in TEXT_KEYWORDS.items():
            if token in keys:
                return emo

    # 2) fuzzy match each token to ALL_KEYWORDS
    # get_close_matches returns best matches with a cutoff similarity
    for token in tokens:
        matches = difflib.get_close_matches(token, ALL_KEYWORDS, n=2, cutoff=0.78)
        if matches:
            matched_word = matches[0]
            # map matched_word back to emotion
            for emo, keys in TEXT_KEYWORDS.items():
                if matched_word in keys:
                    return emo

    # 3) fuzzy phrase-level matching: try to match whole text to keywords
    matches = difflib.get_close_matches(text_clean, ALL_KEYWORDS, n=1, cutoff=0.75)
    if matches:
        matched = matches[0]
        for emo, keys in TEXT_KEYWORDS.items():
            if matched in keys:
                return emo

    # 4) optional: fallback to polarity using TextBlob if installed
    try:
        from textblob import TextBlob
        polarity = TextBlob(user_text).sentiment.polarity
        if polarity > 0.3:
            return "happy"
        elif polarity < -0.3:
            return "sad"
    except Exception:
        # textblob not installed or corpora missing — ignore
        pass

    # final fallback
    return "neutral"

# ---------------------------
# UI: Image + Text Tabs
# ---------------------------
st.markdown("<h1 style='text-align:center;'>🌟 MoodMate – Emotion Based Mood Recommender</h1>", unsafe_allow_html=True)
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📷 Image Mood Detection", "✍🏼 Text Mood Input"])

# TAB 1 — Image
with tab1:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", width=350)

        with st.spinner("Detecting face & emotion..."):
            face = extract_face(img)
            if face is None:
                st.warning("No clear face found — model will use entire image instead.")
                # fallback: use entire image (no crop)
                label, conf = predict_emotion(img)   # no crop
            else:
                label, conf = predict_emotion(face)
            time.sleep(0.6)

        st.success(f"🎭 Emotion Detected: **{label.upper()}** | Confidence: **{conf:.2f}**")
        st.markdown("---")
        st.subheader("🎧 Recommended Songs For Your Mood")
        # recommender expects label or tag; our recommender has recommend(emotion, n)
        # some recommenders expect tags; adjust accordingly.
        try:
            results = recommender.recommend(label, n=5)
        except Exception:
            # older recommender uses recommend_by_tags(tag_query)
            tag_query = EMOTION_TO_TAGS.get(label, label)
            results = recommender.recommend_by_tags(tag_query, topn=5)

        for i, row in results.iterrows():
            st.markdown(f"**🎵 {row.get('title','Unknown')}**  \n👤 {row.get('artist','Unknown')}  \n🔖 Tags: `{row.get('tags','')}`  \n")

# TAB 2 — Text
with tab2:
    text_mood = st.text_input("Describe your mood (try words like 'joyful', 'anxious', 'angry', 'lonely' etc.)")

    if text_mood:
        with st.spinner("Understanding mood from text..."):
            label = detect_mood_from_text(text_mood)
            time.sleep(0.5)

        st.success(f"📝 Mood Detected from Text: **{label.upper()}**")
        st.subheader("🎧 Recommended Songs For Your Mood")

        try:
            results = recommender.recommend(label, n=5)
        except Exception:
            tag_query = EMOTION_TO_TAGS.get(label, label)
            results = recommender.recommend_by_tags(tag_query, topn=5)

        for i, row in results.iterrows():
            st.markdown(f"**🎵 {row.get('title','Unknown')}**  \n👤 {row.get('artist','Unknown')}  \n🔖 Tags: `{row.get('tags','')}`  \n")
