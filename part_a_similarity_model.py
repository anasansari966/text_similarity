# part_a_similarity_model.py

import pandas as pd
import numpy as np
import re
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK assets
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Load data
df = pd.read_csv("DataNeuron_Text_Similarity.csv")

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    tokens = [WordNetLemmatizer().lemmatize(t) for t in tokens]
    return " ".join(tokens)

df["text1_clean"] = df["text1"].apply(clean_text)
df["text2_clean"] = df["text2"].apply(clean_text)

# TF-IDF cosine similarity
tfidf = TfidfVectorizer()
tfidf.fit(pd.concat([df["text1_clean"], df["text2_clean"]]))
vec1 = tfidf.transform(df["text1_clean"])
vec2 = tfidf.transform(df["text2_clean"])
cos_sim = [cosine_similarity(vec1[i], vec2[i])[0][0] for i in range(len(df))]

# Jaccard similarity
def jaccard_sim(a, b):
    set1 = set(a.split())
    set2 = set(b.split())
    return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0

jaccard = [jaccard_sim(a, b) for a, b in zip(df["text1_clean"], df["text2_clean"])]

# Text length features
len_diff = [abs(len(a.split()) - len(b.split())) for a, b in zip(df["text1_clean"], df["text2_clean"])]
len_ratio = [min(len(a.split()), len(b.split())) / max(len(a.split()), len(b.split())) for a, b in zip(df["text1_clean"], df["text2_clean"])]

# Feature matrix
X = pd.DataFrame({
    "cosine_sim": cos_sim,
    "jaccard_sim": jaccard,
    "len_diff": len_diff,
    "len_ratio": len_ratio,
})

# Weak label using cosine similarity
y = np.array(cos_sim)
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("✅ RMSE:", rmse)

# Save everything
joblib.dump(model, "similarity_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
joblib.dump(scaler, "similarity_scaler.pkl")
