# part_a_similarity_model_word2vec.py

import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import KeyedVectors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import joblib

# Download NLTK assets
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load data
df = pd.read_csv("DataNeuron_Text_Similarity.csv")
df.columns = ['text1', 'text2']

# Text cleaner
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    tokens = [WordNetLemmatizer().lemmatize(t) for t in tokens]
    return tokens

df["text1_tokens"] = df["text1"].apply(clean_text)
df["text2_tokens"] = df["text2"].apply(clean_text)

# Load pretrained word2vec (e.g., Google News)
word2vec = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

# Sentence embedding: average of word vectors
def embed(tokens):
    vectors = [word2vec[w] for w in tokens if w in word2vec]
    return np.mean(vectors, axis=0) if vectors else np.zeros(word2vec.vector_size)

# Create embeddings
embeddings1 = df["text1_tokens"].apply(embed)
embeddings2 = df["text2_tokens"].apply(embed)

# Cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = [cosine_similarity([a], [b])[0][0] for a, b in zip(embeddings1, embeddings2)]

# Jaccard similarity
def jaccard_similarity(a, b):
    a_set, b_set = set(a), set(b)
    return len(a_set & b_set) / len(a_set | b_set) if a_set | b_set else 0

jaccard_sim = [jaccard_similarity(a, b) for a, b in zip(df["text1_tokens"], df["text2_tokens"])]

# Length features
len_diff = [abs(len(a) - len(b)) for a, b in zip(df["text1_tokens"], df["text2_tokens"])]
len_ratio = [min(len(a), len(b)) / max(len(a), len(b)) if max(len(a), len(b)) != 0 else 0 for a, b in zip(df["text1_tokens"], df["text2_tokens"])]

# Final feature matrix
X = pd.DataFrame({
    "cosine_sim": cosine_sim,
    "jaccard_sim": jaccard_sim,
    "len_diff": len_diff,
    "len_ratio": len_ratio
})

# Use cosine similarity as weak label
y = np.array(cosine_sim)
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Save artifacts
joblib.dump(model, "similarity_model.pkl")
joblib.dump(scaler, "similarity_scaler.pkl")
