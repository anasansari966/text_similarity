from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
import nltk
import os
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ✅ Use existing nltk_data folder (no download or creation)
nltk_data_path = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path.append(nltk_data_path)

# ✅ Load saved model and vectorizer
model = joblib.load("similarity_model.pkl")
scaler = joblib.load("similarity_scaler.pkl")
word2vec = KeyedVectors.load("small_word2vec.kv")

app = Flask(__name__)

# ✅ Health check endpoint
@app.route("/test", methods=["GET"])
def test_route():
    return jsonify({"status": "success", "message": "API is running ✅"})

# ✅ Text preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    tokens = [WordNetLemmatizer().lemmatize(w) for w in tokens]
    return tokens

# ✅ Embedding and similarity features
def embed(tokens):
    vectors = [word2vec[w] for w in tokens if w in word2vec]
    return np.mean(vectors, axis=0) if vectors else np.zeros(word2vec.vector_size)

def jaccard_similarity(a, b):
    a_set, b_set = set(a), set(b)
    return len(a_set & b_set) / len(a_set | b_set) if a_set | b_set else 0

# ✅ Main similarity route
@app.route("/similarity", methods=["POST"])
def get_similarity():
    data = request.get_json()
    text1 = data.get("text1", "")
    text2 = data.get("text2", "")

    tokens1 = clean_text(text1)
    tokens2 = clean_text(text2)

    emb1 = embed(tokens1)
    emb2 = embed(tokens2)

    cosine_sim = float(cosine_similarity([emb1], [emb2])[0][0])
    jaccard_sim = jaccard_similarity(tokens1, tokens2)
    len_diff = abs(len(tokens1) - len(tokens2))
    len_ratio = min(len(tokens1), len(tokens2)) / max(len(tokens1), len(tokens2)) if max(len(tokens1), len(tokens2)) else 0

    features = np.array([[cosine_sim, jaccard_sim, len_diff, len_ratio]])
    similarity_score = float(model.predict(features)[0])
    similarity_score = max(0, min(1, similarity_score))

    return jsonify({"similarity score": round(similarity_score, 3)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
