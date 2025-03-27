# model.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Preprocessing function
def preprocess(text):
    text = str(text).lower()
    text = ''.join([c for c in text if c.isalnum() or c.isspace()])
    return text

# Load data
df = pd.read_csv('DataNeuron_Text_Similarity.csv')

# Preprocess text
df['text1_clean'] = df['text1'].apply(preprocess)
df['text2_clean'] = df['text2'].apply(preprocess)

# Create TF-IDF vectorizer
all_texts = pd.concat([df['text1_clean'], df['text2_clean']]).tolist()
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
vectorizer.fit(all_texts)

# Save vectorizer for API
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Calculate similarity scores
def calculate_similarity(row):
    vec1 = vectorizer.transform([row['text1_clean']])
    vec2 = vectorizer.transform([row['text2_clean']])
    return cosine_similarity(vec1, vec2)[0][0]

df['similarity_score'] = df.apply(calculate_similarity, axis=1)
df.to_csv('results.csv', index=False)