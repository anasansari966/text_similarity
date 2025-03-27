import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
import tensorflow_hub as hub
import tensorflow as tf
from numpy import dot
from numpy.linalg import norm

# Load USE model once at startup
model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# FastAPI app
app = FastAPI()

# Request schema
class TextPair(BaseModel):
    text1: str
    text2: str

# Function to embed & compute cosine similarity
def get_similarity(text1, text2):
    embeddings = model([text1, text2])
    vectors = tf.make_ndarray(tf.make_tensor_proto(embeddings))
    cos_sim = dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))
    normalized_score = (cos_sim + 1) / 2  # to range [0, 1]
    return round(float(normalized_score), 3)

# API Endpoint
@app.post("/")
async def compute_similarity(data: TextPair):
    score = get_similarity(data.text1, data.text2)
    return {"similarity score": score}

# Run with: uvicorn app:app --host 0.0.0.0 --port 8000
