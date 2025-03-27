from flask import Flask, request, jsonify
import joblib
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import os

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")

# Clean text
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text.lower())
    return ' '.join([w for w in text.split() if w not in stop_words])

# Jaccard
def jaccard_sim(a, b):
    a, b = set(a.split()), set(b.split())
    return len(a & b) / len(a | b) if a | b else 0.0

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text1 = clean_text(data['text1'])
    text2 = clean_text(data['text2'])

    vec1 = tfidf.transform([text1])
    vec2 = tfidf.transform([text2])

    cosine = cosine_similarity(vec1, vec2)[0][0]
    jaccard = jaccard_sim(text1, text2)
    len_diff = abs(len(text1.split()) - len(text2.split()))

    features = np.array([[cosine, jaccard, len_diff]])
    score = model.predict(features)[0]

    return jsonify({"similarity score": round(float(score), 4)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
