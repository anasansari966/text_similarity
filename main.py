import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
from numpy import dot
from numpy.linalg import norm

# Load Universal Sentence Encoder model
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)

def embed(input):
    return model(input)

# Read dataset
Data = pd.read_csv("DataNeuron_Text_Similarity.csv")

# Compute cosine similarity
similarity_scores = []
for i in range(len(Data)):
    sentences = [Data['text1'][i], Data['text2'][i]]
    embeddings = embed(sentences)
    vectors = tf.make_ndarray(tf.make_tensor_proto(embeddings))
    cos_sim = dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))
    similarity_scores.append(cos_sim)

# Add similarity scores to dataframe
Data['Similarity_Score'] = similarity_scores

# Normalize scores to range [0, 1]
Data['Similarity_Score'] = Data['Similarity_Score'] + 1  # from [-1, 1] to [0, 2]
Data['Similarity_Score'] = Data['Similarity_Score'] / Data['Similarity_Score'].abs().max()

# Create submission with auto-generated Unique_ID
Data['Unique_ID'] = range(len(Data))
Submission = Data[['Unique_ID', 'Similarity_Score']]
Submission.set_index("Unique_ID", inplace=True)
Submission.to_csv("Submission.csv")
