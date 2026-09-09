import json

import config


def load_vocab(path=config.VOCAB_PATH):
    with open(path) as f:
        return json.load(f)


def encode_tokens(tokens, vocab, max_len=config.MAX_SEQ_LEN):
    unk_id = vocab[config.UNK_TOKEN]
    ids = [vocab.get(tok, unk_id) for tok in tokens[:max_len]]
    if len(ids) < max_len:
        ids = ids + [vocab[config.PAD_TOKEN]] * (max_len - len(ids))
    return ids
