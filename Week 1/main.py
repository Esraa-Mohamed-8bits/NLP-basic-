import pickle
from preprocessing_pipeline import clean_for_language_detection
import arabic_model
import english_model

with open("Language_classifier_weights.pkl", "rb") as f:
    _lang = pickle.load(f)

lang_vectorizer = _lang["vectorizer"]
lang_model = _lang["model"]


def detect_language(text):
    clean = clean_for_language_detection(text)
    vec = lang_vectorizer.transform([clean])
    label = lang_model.predict(vec)[0]
    return "Arabic" if label == "ar" else "English"


def run_pipeline(text):
    language = detect_language(text)
    if language == "Arabic":
        sentiment = arabic_model.predict_sentiment(text)
    else:
        sentiment = english_model.predict_sentiment(text)

    return {
        "User Text": text,
        "Language": language,
        "Sentiment Classification": sentiment,
    }


if __name__ == "__main__":
    user_text = input("Enter your text: ")
    result = run_pipeline(user_text)
    for key, value in result.items():
        print(f"{key}: {value}")
