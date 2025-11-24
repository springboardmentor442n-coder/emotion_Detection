import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

emotion_to_tags = {
    0: ["sad", "melancholy", "slow", "soft"],               
    1: ["disgust", "dark", "intense"],                      
    2: ["fear", "dark", "ambient"],                         
    3: ["happy", "energetic", "upbeat", "dance"],           
    4: ["neutral", "chill", "calm", "ambient"],             
    5: ["sad", "soft", "acoustic", "calm"],                 
    6: ["surprise", "electronic", "fast", "excited"],       
}

emotion_to_name = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise"
}

class Recommender:

    def __init__(self, csv_path="data/music_clean.csv"):
        self.tracks = pd.read_csv(csv_path)

        # Make sure tags column exists
        self.tracks["tags"] = self.tracks["tags"].fillna("")

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.tracks["tags"])

    def recommend(self, emotion_label, top_n=10):
        tags = emotion_to_tags[emotion_label]
        query = " ".join(tags)

        query_vec = self.vectorizer.transform([query])
        similarity = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        idx = similarity.argsort()[::-1][:top_n]

        return self.tracks.iloc[idx][["title", "artist", "tags"]].to_dict(orient="records")
