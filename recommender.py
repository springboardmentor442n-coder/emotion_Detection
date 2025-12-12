import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

EMOTION_TO_TAGS = {
    "happy": "happy joy upbeat bright",
    "sad": "sad lonely heartbreak emotional",
    "angry": "angry rage fire intense",
    "fear": "fear dark cold mysterious",
    "surprise": "surprise shock wow unexpected",
    "neutral": "calm relaxed soft chill",
    "love": "love romantic heart sweet"
}

class ContentRecommender:

    def __init__(self, csv_path="data/songs_metadata.csv", model_path="models/recommender.joblib"):
        df = pd.read_csv(csv_path)
        self.df = df

        df["combined"] = df["tags"] + " " + df["genre"]

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(df["combined"])

        joblib.dump((self.vectorizer, self.tfidf_matrix, df), model_path)
        print(f"Recommender saved → {model_path}")

    @staticmethod
    def load(model_path="models/recommender.joblib"):
        vectorizer, tfidf_matrix, df = joblib.load(model_path)

        rec = ContentRecommender.__new__(ContentRecommender)
        rec.vectorizer = vectorizer
        rec.tfidf_matrix = tfidf_matrix
        rec.df = df
        return rec

    def recommend(self, emotion, n=5):
        if emotion not in EMOTION_TO_TAGS:
            emotion = "neutral"

        query = EMOTION_TO_TAGS[emotion]

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        top_idx = scores.argsort()[-n:][::-1]
        return self.df.iloc[top_idx]
