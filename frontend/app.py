# src/frontend/app.py
import streamlit as st
import requests
import base64
from io import BytesIO
from PIL import Image

API_URL = "http://127.0.0.1:8000"
# project spec PDF (local path you uploaded)
PROJECT_PDF = "/mnt/data/AI MoodMate-1.pdf"

st.set_page_config(page_title="AI MoodMate", page_icon="🎧", layout="wide")

# Simple CSS
st.markdown(
    """
    <style>
    .card { background:#fff; padding:14px; border-radius:10px; box-shadow:0 6px 18px rgba(0,0,0,0.06); }
    .muted { color:#6b7280; font-size:13px }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
col1, col2 = st.columns([6,2])
with col1:
    st.markdown("## 🎧 AI MoodMate")
    st.markdown("Detect emotion from text or a selfie and get music recommendations.")
with col2:
    st.markdown(f"[Project spec (PDF)]({PROJECT_PDF})")

st.write("")

left, right = st.columns([1.2, 1])

# TEXT PANEL
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ✍️ Text Emotion")
    txt = st.text_area("Write how you're feeling...", height=120)
    if st.button("Analyze text"):
        if not txt.strip():
            st.warning("Type something first.")
        else:
            with st.spinner("Analyzing..."):
                try:
                    resp = requests.post(f"{API_URL}/predict_text/", data={"text": txt}, timeout=15)
                    data = resp.json()
                except Exception as e:
                    st.error("Request failed: " + str(e))
                    data = None
            if data:
                if data.get("success"):
                    st.markdown(f"**Emotion:** {data['emotion_name']}  —  **Label:** {data['emotion_label']}")
                    if data.get("vader") is not None:
                        st.write("VADER scores:", data["vader"])
                    st.markdown("**Recommendations**")
                    for r in data.get("recommendations", []):
                        st.write(f"- **{r['title']}** — {r['artist']}  ·  _{r.get('tags','')}_")
                else:
                    st.error("Error: " + str(data.get("error", "unknown")))
    st.markdown('</div>', unsafe_allow_html=True)

# IMAGE PANEL
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📸 Image Emotion (upload a selfie)")
    uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
    if uploaded:
        st.image(uploaded.getvalue(), width=260, caption="preview")
    if st.button("Analyze image"):
        if not uploaded:
            st.warning("Please upload an image first.")
        else:
            with st.spinner("Sending to API..."):
                try:
                    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                    resp = requests.post(f"{API_URL}/predict_image/", files=files, timeout=30)
                    data = resp.json()
                except Exception as e:
                    st.error("Request failed: " + str(e))
                    data = None

            if data:
                if data.get("success"):
                    st.markdown(f"**Emotion:** {data['emotion_name']}  —  **Label:** {data['emotion_label']}")
                    st.markdown(f"**Confidence:** {round(data.get('confidence',0)*100,1)}%")
                    thumb = data.get("thumb_b64") or data.get("thumbnail")
                    if thumb:
                        try:
                            b = base64.b64decode(thumb)
                            st.image(b, width=160, caption="Detected face")
                        except Exception:
                            pass
                    st.markdown("**Recommendations**")
                    for r in data.get("recommendations", []):
                        st.write(f"- **{r['title']}** — {r['artist']}  ·  _{r.get('tags','')}_")
                else:
                    st.error("Error: " + str(data.get("error", "unknown")))

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("Built with FastAPI + Streamlit. Model: MobileNetV2 transfer + custom head.")
