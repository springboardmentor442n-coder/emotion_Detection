import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# SIMPLE emotion keyword extraction
EMOTION_KEYWORDS = {
    "happy": ["happy", "joy", "smile", "bright", "cheer", "delight"],
    "sad": ["sad", "cry", "tears", "lonely", "broken", "blue"],
    "angry": ["anger", "rage", "hate", "fire", "furious"],
    "fear": ["fear", "scared", "nightmare", "dark", "ghost"],
    "surprise": ["shock", "surprise", "wow", "unexpected"],
    "love": ["love", "heart", "kiss", "sweet", "darling"],
    "calm": ["calm", "peace", "soft", "quiet", "still"]
}

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_tags(lyrics):
    lyrics = clean_text(lyrics)
    tags = []

    for emotion, words in EMOTION_KEYWORDS.items():
        if any(w in lyrics for w in words):
            tags.append(emotion)

    if not tags:
        tags.append("neutral")

    return " ".join(tags)

def guess_genre(lyrics):
    lyrics = lyrics.lower()
    if "love" in lyrics or "heart" in lyrics:
        return "romantic"
    if "cry" in lyrics or "sad" in lyrics:
        return "sad"
    if "fire" in lyrics or "anger" in lyrics:
        return "rock"
    return "unknown"

# --- MAIN ---
def build_dataset(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    # Normalizing column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Expected columns: artist, song, link, text
    df["tags"] = df["text"].apply(extract_tags)
    df["genre"] = df["text"].apply(guess_genre)

    final_df = pd.DataFrame({
        "title": df["song"],
        "artist": df["artist"],
        "tags": df["tags"],
        "genre": df["genre"],
        "preview_url": df["link"]
    })

    final_df.to_csv(output_csv, index=False)
    print(f"\nDataset created successfully → {output_csv}")
    print(final_df.head(10))

if __name__ == "__main__":
    build_dataset("data/spotify_millsongdata.csv", "data/songs_metadata.csv")
