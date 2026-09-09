"""
All the settings for this project live here so we don't have
magic numbers scattered across every script.
"""

import os

# paths
DATA_PATH = os.path.join("data", "train.csv")
OUTPUT_DIR = "outputs"
VOCAB_PATH = os.path.join(OUTPUT_DIR, "vocab.json")
MODEL_PATH = os.path.join(OUTPUT_DIR, "lstm_model.pt")
HISTORY_PLOT_PATH = os.path.join(OUTPUT_DIR, "training_history.png")
METRICS_PLOT_PATH = os.path.join(OUTPUT_DIR, "label_metrics.png")

# the six labels we're predicting, in a fixed order
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# text / vocab
MAX_VOCAB_SIZE = 20000
MAX_SEQ_LEN = 150
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

# model
EMBED_DIM = 128
HIDDEN_DIM = 128
NUM_LSTM_LAYERS = 1
DROPOUT = 0.3
BIDIRECTIONAL = True

# training
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
RANDOM_SEED = 42
CLASSIFICATION_THRESHOLD = 0.5
