# app.py
from flask import Flask, request, jsonify
import joblib
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
vectorizer = joblib.load('tfidf_vectorizer.pkl')


def preprocess(text):
    text = str(text).lower()
    text = ''.join([c for c in text if c.isalnum() or c.isspace()])
    return text


@app.route('/similarity', methods=['POST'])
def get_similarity():
    data = request.get_json()
    text1 = preprocess(data['text1'])
    text2 = preprocess(data['text2'])

    vec1 = vectorizer.transform([text1])
    vec2 = vectorizer.transform([text2])

    score = cosine_similarity(vec1, vec2)[0][0]
    return jsonify({'similarity score': round(score, 2)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)