import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import nltk
import re
import joblib
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load dataset
df = pd.read_csv("DataNeuron_Text_Similarity.csv").dropna()

# Clean text
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text.lower())
    return ' '.join([w for w in text.split() if w not in stop_words])

df['text1_clean'] = df['text1'].apply(clean_text)
df['text2_clean'] = df['text2'].apply(clean_text)

# TF-IDF features
tfidf = TfidfVectorizer()
all_text = df['text1_clean'].tolist() + df['text2_clean'].tolist()
tfidf_matrix = tfidf.fit_transform(all_text)
vec1 = tfidf_matrix[:len(df)]
vec2 = tfidf_matrix[len(df):]
cosine = [cosine_similarity(vec1[i], vec2[i])[0][0] for i in range(len(df))]

# Jaccard similarity
def jaccard_sim(a, b):
    a, b = set(a.split()), set(b.split())
    return len(a & b) / len(a | b) if a | b else 0.0

jaccard = [jaccard_sim(a, b) for a, b in zip(df['text1_clean'], df['text2_clean'])]

# Length difference
len_diff = [abs(len(a.split()) - len(b.split())) for a, b in zip(df['text1_clean'], df['text2_clean'])]

# Features
X = pd.DataFrame({
    'cosine': cosine,
    'jaccard': jaccard,
    'len_diff': len_diff
})

# Generate pseudo-labels from cosine similarity for unsupervised setting
scaler = MinMaxScaler()
y = scaler.fit_transform(np.array(cosine).reshape(-1, 1)).flatten()

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
rmse = mean_squared_error(y_test, model.predict(X_test)) ** 0.5
print(f"RMSE: {rmse:.4f}")

# Save model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(tfidf, "tfidf.pkl")
