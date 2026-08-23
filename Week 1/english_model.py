import pickle
from preprocessing_pipeline import preprocess_english

with open("English_model_weights.pkl", "rb") as f:
    _saved = pickle.load(f)

vectorizer = _saved["vectorizer"]
model = _saved["model"]


def predict_sentiment(text):
    clean = preprocess_english(text)
    vec = vectorizer.transform([clean])
    return model.predict(vec)[0]
