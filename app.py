# app.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load saved model and scaler
model = joblib.load("similarity_model.pkl")
scaler = joblib.load("similarity_scaler.pkl")

# Load word2vec model
from gensim.models import KeyedVectors
# word2vec = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
word2vec = KeyedVectors.load("small_word2vec.kv")

# Minimal preprocessing
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# API setup
app = FastAPI()

class TextPair(BaseModel):
    text1: str
    text2: str

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [w for w in tokens if w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return tokens

def embed(tokens):
    vectors = [word2vec[w] for w in tokens if w in word2vec]
    return np.mean(vectors, axis=0) if vectors else np.zeros(word2vec.vector_size)

def jaccard_similarity(a, b):
    a_set, b_set = set(a), set(b)
    return len(a_set & b_set) / len(a_set | b_set) if a_set | b_set else 0

@app.post("/similarity")
def get_similarity(data: TextPair):
    tokens1 = clean_text(data.text1)
    tokens2 = clean_text(data.text2)

    emb1 = embed(tokens1)
    emb2 = embed(tokens2)

    cosine_sim = float(cosine_similarity([emb1], [emb2])[0][0])
    jaccard_sim = jaccard_similarity(tokens1, tokens2)
    len_diff = abs(len(tokens1) - len(tokens2))
    len_ratio = min(len(tokens1), len(tokens2)) / max(len(tokens1), len(tokens2)) if max(len(tokens1), len(tokens2)) else 0

    features = np.array([[cosine_sim, jaccard_sim, len_diff, len_ratio]])
    similarity_score = float(model.predict(features)[0])
    similarity_score = max(0, min(1, similarity_score))

    return {"similarity score": round(similarity_score, 3)}
