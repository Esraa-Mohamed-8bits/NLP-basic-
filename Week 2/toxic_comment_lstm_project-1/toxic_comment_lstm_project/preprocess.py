import json
import pandas as pd
from collections import Counter

import config
from utils import clean_text, tokenize


def load_data(path=config.DATA_PATH):
    df = pd.read_csv(path)
    df["comment_text"] = df["comment_text"].fillna("")
    df["clean_text"] = df["comment_text"].apply(clean_text)
    df["tokens"] = df["clean_text"].apply(tokenize)
    return df


def build_vocab(token_lists, max_size=config.MAX_VOCAB_SIZE):
    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)

    most_common = counter.most_common(max_size - 2)  # leave room for pad/unk
    vocab = {config.PAD_TOKEN: 0, config.UNK_TOKEN: 1}
    for word, _ in most_common:
        vocab[word] = len(vocab)
    return vocab


def save_vocab(vocab, path=config.VOCAB_PATH):
    with open(path, "w") as f:
        json.dump(vocab, f)


def load_vocab(path=config.VOCAB_PATH):
    with open(path) as f:
        return json.load(f)


def encode_tokens(tokens, vocab, max_len=config.MAX_SEQ_LEN):
    unk_id = vocab[config.UNK_TOKEN]
    ids = [vocab.get(tok, unk_id) for tok in tokens[:max_len]]
    if len(ids) < max_len:
        ids = ids + [vocab[config.PAD_TOKEN]] * (max_len - len(ids))
    return ids
