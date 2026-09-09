"""
Settings for the app. Point MODEL_PATH / VOCAB_PATH to the files you trained
in the toxic_comment_lstm_project (train.py) and copy them into models/.
"""

import os

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.pt")
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.json")

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

MAX_SEQ_LEN = 150
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

EMBED_DIM = 128
HIDDEN_DIM = 128
NUM_LSTM_LAYERS = 1
DROPOUT = 0.3
BIDIRECTIONAL = True

CLASSIFICATION_THRESHOLD = 0.5

# whisper model size: tiny / base / small / medium / large
WHISPER_MODEL_SIZE = "base"
