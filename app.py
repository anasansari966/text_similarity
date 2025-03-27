# app.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Load models
model = joblib.load("similarity_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
scaler = joblib.load("similarity_scaler.pkl")

app = FastAPI()

class TextPair(BaseModel):
    text1: str
    text2: str

def clean(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    tokens = [WordNetLemmatizer().lemmatize(t) for t in tokens]
    return " ".join(tokens)

def jaccard_sim(a, b):
    set1, set2 = set(a.split()), set(b.split())
    return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0

@app.post("/similarity")
def predict_similarity(data: TextPair):
    t1 = clean(data.text1)
    t2 = clean(data.text2)

    vec1 = tfidf.transform([t1])
    vec2 = tfidf.transform([t2])
    cos_sim = float((vec1 @ vec2.T).A[0][0])
    jaccard = jaccard_sim(t1, t2)
    len_diff = abs(len(t1.split()) - len(t2.split()))
    len_ratio = min(len(t1.split()), len(t2.split())) / max(len(t1.split()), len(t2.split())) if max(len(t1.split()), len(t2.split())) else 0

    features = np.array([[cos_sim, jaccard, len_diff, len_ratio]])
    score = float(model.predict(features)[0])
    return {"similarity score": round(score, 3)}
