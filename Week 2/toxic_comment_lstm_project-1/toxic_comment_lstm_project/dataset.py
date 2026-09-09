import torch
from torch.utils.data import Dataset

import config
from preprocess import encode_tokens


class ToxicDataset(Dataset):
    def __init__(self, df, vocab):
        self.tokens = df["tokens"].tolist()
        self.labels = df[config.LABELS].values.astype("float32")
        self.vocab = vocab

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        ids = encode_tokens(self.tokens[idx], self.vocab)
        x = torch.tensor(ids, dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y
