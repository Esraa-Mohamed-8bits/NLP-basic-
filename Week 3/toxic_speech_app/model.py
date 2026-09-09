import torch
import torch.nn as nn

import config


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, num_labels=len(config.LABELS)):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.EMBED_DIM, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=config.EMBED_DIM,
            hidden_size=config.HIDDEN_DIM,
            num_layers=config.NUM_LSTM_LAYERS,
            batch_first=True,
            bidirectional=config.BIDIRECTIONAL,
            dropout=config.DROPOUT if config.NUM_LSTM_LAYERS > 1 else 0.0,
        )
        lstm_out_dim = config.HIDDEN_DIM * (2 if config.BIDIRECTIONAL else 1)
        self.dropout = nn.Dropout(config.DROPOUT)
        self.fc = nn.Linear(lstm_out_dim, num_labels)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)

        if config.BIDIRECTIONAL:
            final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            final_hidden = hidden[-1]

        out = self.dropout(final_hidden)
        return self.fc(out)
