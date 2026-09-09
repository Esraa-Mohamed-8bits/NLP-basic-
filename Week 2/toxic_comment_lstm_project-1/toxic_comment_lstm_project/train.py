import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import config
from utils import set_seed
from preprocess import load_data, build_vocab, save_vocab
from dataset import ToxicDataset
from model import LSTMClassifier


def get_pos_weights(df):
    """Weight for BCEWithLogitsLoss so rare labels (like 'threat') aren't ignored."""
    pos_counts = df[config.LABELS].sum().values
    total = len(df)
    neg_counts = total - pos_counts
    # avoid divide-by-zero for labels with (almost) no positives
    weights = neg_counts / np.clip(pos_counts, 1, None)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


def main():
    set_seed(config.RANDOM_SEED)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    print("loading and cleaning data...")
    df = load_data()

    # train / val / test split
    train_df, temp_df = train_test_split(df, test_size=config.VAL_SPLIT + config.TEST_SPLIT,
                                          random_state=config.RANDOM_SEED)
    rel_test_size = config.TEST_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT)
    val_df, test_df = train_test_split(temp_df, test_size=rel_test_size, random_state=config.RANDOM_SEED)
    print(f"train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    # vocab is built only from the training split to avoid leaking info
    vocab = build_vocab(train_df["tokens"])
    save_vocab(vocab)
    print(f"vocab size: {len(vocab)}")

    train_ds = ToxicDataset(train_df, vocab)
    val_ds = ToxicDataset(val_df, vocab)
    test_ds = ToxicDataset(test_df, vocab)
    test_df.to_csv(os.path.join(config.OUTPUT_DIR, "test_split.csv"), index=False)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE)

    model = LSTMClassifier(vocab_size=len(vocab)).to(device)

    pos_weight = get_pos_weights(train_df).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    for epoch in range(1, config.EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"epoch {epoch}/{config.EPOCHS} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.MODEL_PATH)
            print("  -> new best model saved")

    plot_history(history)
    print("training done. best val loss:", best_val_loss)


def plot_history(history):
    plt.figure(figsize=(6, 4))
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["val_loss"], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("BCE loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.HISTORY_PLOT_PATH)
    plt.close()


if __name__ == "__main__":
    main()
