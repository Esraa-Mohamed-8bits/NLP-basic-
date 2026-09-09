import torch

import config
from utils import clean_text, tokenize
from preprocess import load_vocab, encode_tokens
from model import LSTMClassifier

_model = None
_vocab = None


def get_model_and_vocab():
    """Load the model + vocab once, reuse for every prediction."""
    global _model, _vocab
    if _model is None:
        _vocab = load_vocab()
        _model = LSTMClassifier(vocab_size=len(_vocab))
        _model.load_state_dict(torch.load(config.MODEL_PATH, map_location="cpu"))
        _model.eval()
    return _model, _vocab


def predict(text):
    """Returns a dict like {'toxic': 0.91, 'insult': 0.74, ...}."""
    model, vocab = get_model_and_vocab()

    tokens = tokenize(clean_text(text))
    ids = encode_tokens(tokens, vocab)
    x = torch.tensor([ids], dtype=torch.long)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).squeeze(0).tolist()

    return {label: round(p, 4) for label, p in zip(config.LABELS, probs)}


def is_toxic(scores, threshold=config.CLASSIFICATION_THRESHOLD):
    return any(score >= threshold for score in scores.values())
