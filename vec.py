# shrink_word2vec_model.py

import pandas as pd
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import KeyedVectors
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load full model
full_model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

# Load dataset
df = pd.read_csv("DataNeuron_Text_Similarity.csv")

# Combine all text
texts = df['text1'].tolist() + df['text2'].tolist()

# Preprocess and collect vocab
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return tokens

vocab = set()
for text in texts:
    vocab.update(clean(text))

# Filter word vectors
filtered_model = KeyedVectors(vector_size=full_model.vector_size)
filtered_words = [word for word in vocab if word in full_model]
filtered_model.add_vectors(filtered_words, [full_model[word] for word in filtered_words])

# Save small model in fast format
filtered_model.save("small_word2vec.kv")
