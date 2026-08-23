# Neurova NLP Pipeline

1. Download both Kaggle datasets and put them in a `data/` folder:
   - `data/arabic_customer_reviews.csv`
   - `data/imdb_reviews.csv`
2. Run the three notebooks in this order:
   - `Training_language_classifier.ipynb`
   - `Training_arabic_model.ipynb`
   - `Training_english_model.ipynb`
   Each one saves its own `.pkl` weights file in this folder.
3. Run `main.py` to try the full pipeline on your own text.

Everything here is classical NLP (TF-IDF/n-grams + Naive Bayes or Logistic Regression) — no transformers, per the project brief.
