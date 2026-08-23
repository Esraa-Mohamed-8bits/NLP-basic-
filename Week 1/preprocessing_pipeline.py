import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.isri import ISRIStemmer

# grab what we need once, quiet=True so it doesn't spam the console
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    nltk.download(pkg, quiet=True)

en_stopwords = set(stopwords.words("english"))
ar_stopwords = set(stopwords.words("arabic"))
lemmatizer = WordNetLemmatizer()
ar_stemmer = ISRIStemmer()

arabic_diacritics = re.compile(r"[\u0617-\u061A\u064B-\u0652]")


def normalize_arabic(text):
    text = arabic_diacritics.sub("", text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = re.sub(r"[^ء-ي\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_for_language_detection(text):
    # simple and cheap on purpose, we just need enough signal to tell the
    # two languages apart, not a full linguistic cleanup
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zء-ي\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess_english(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in en_stopwords and len(t) > 1]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def preprocess_arabic(text):
    text = normalize_arabic(str(text))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in ar_stopwords and len(t) > 1]
    tokens = [ar_stemmer.stem(t) for t in tokens]
    return " ".join(tokens)
