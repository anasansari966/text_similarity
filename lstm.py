# siamese_lstm_train.py

import pandas as pd
import numpy as np
import re
import string
import nltk
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import register_keras_serializable
from nltk.corpus import stopwords

nltk.download("stopwords")

# Clean text
def clean(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = " ".join([w for w in text.split() if w not in stopwords.words("english")])
    return text

# Load and preprocess
df = pd.read_csv("DataNeuron_Text_Similarity.csv")
df["text1"] = df["text1"].apply(clean)
df["text2"] = df["text2"].apply(clean)

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(pd.concat([df["text1"], df["text2"]]))
vocab_size = len(tokenizer.word_index) + 1
maxlen = 50

X1 = pad_sequences(tokenizer.texts_to_sequences(df["text1"]), maxlen=maxlen)
X2 = pad_sequences(tokenizer.texts_to_sequences(df["text2"]), maxlen=maxlen)

# Weak labels using cosine TF-IDF similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfidf = TfidfVectorizer()
vectors = tfidf.fit_transform(df["text1"].tolist() + df["text2"].tolist())
cosine_scores = [cosine_similarity(vectors[i], vectors[i + len(df)])[0][0] for i in range(len(df))]
y = np.array(cosine_scores)

# Custom cosine sim for Lambda
@register_keras_serializable()
def cosine_sim(vects):
    x, y = vects
    x = tf.math.l2_normalize(x, axis=1)
    y = tf.math.l2_normalize(y, axis=1)
    return tf.reduce_sum(x * y, axis=1, keepdims=True)

# Siamese LSTM model
def build_model():
    input = Input(shape=(maxlen,))
    x = Embedding(vocab_size, 128)(input)
    x = LSTM(64)(x)
    return Model(inputs=input, outputs=x)

input1 = Input(shape=(maxlen,))
input2 = Input(shape=(maxlen,))
base = build_model()

out1 = base(input1)
out2 = base(input2)

sim = Lambda(cosine_sim)([out1, out2])
output = Dense(1, activation="linear")(sim)

model = Model(inputs=[input1, input2], outputs=output)
model.compile(optimizer="adam", loss="mse")
model.summary()

# Train
model.fit([X1, X2], y, batch_size=32, epochs=10, validation_split=0.1)

# Save model and tokenizer
model.save("siamese_lstm_model.h5")
import pickle
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
