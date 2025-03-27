# lstm_app.py

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.utils import register_keras_serializable

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Custom cosine sim function (must match training)
@register_keras_serializable()
def cosine_sim(vects):
    x, y = vects
    x = tf.math.l2_normalize(x, axis=1)
    y = tf.math.l2_normalize(y, axis=1)
    return tf.reduce_sum(x * y, axis=1, keepdims=True)

# Load model & tokenizer
model = load_model("siamese_lstm_model.h5", compile=False, custom_objects={"cosine_sim": cosine_sim})
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

maxlen = 50

app = FastAPI()

class TextPair(BaseModel):
    text1: str
    text2: str

def clean(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)

@app.post("/similarity")
def predict_similarity(data: TextPair):
    t1 = clean(data.text1)
    t2 = clean(data.text2)

    X1 = pad_sequences(tokenizer.texts_to_sequences([t1]), maxlen=maxlen)
    X2 = pad_sequences(tokenizer.texts_to_sequences([t2]), maxlen=maxlen)

    pred = model.predict([X1, X2])[0][0]
    pred = max(0, min(1, float(pred)))

    return {"similarity score": round(pred, 3)}
