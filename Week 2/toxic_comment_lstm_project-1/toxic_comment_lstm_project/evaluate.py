import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

import config
from preprocess import load_vocab
from dataset import ToxicDataset
from model import LSTMClassifier


def get_predictions(model, loader, device):
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(y.numpy())
    return np.vstack(all_probs), np.vstack(all_targets)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = load_vocab()
    test_df = pd.read_csv(config.OUTPUT_DIR + "/test_split.csv")
    # tokens were saved as a stringified list, so re-tokenize the clean text instead
    from utils import tokenize
    test_df["clean_text"] = test_df["clean_text"].fillna("")
    test_df["tokens"] = test_df["clean_text"].apply(tokenize)

    test_ds = ToxicDataset(test_df, vocab)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE)

    model = LSTMClassifier(vocab_size=len(vocab)).to(device)
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))

    probs, targets = get_predictions(model, test_loader, device)
    preds = (probs >= config.CLASSIFICATION_THRESHOLD).astype(int)

    rows = []
    for i, label in enumerate(config.LABELS):
        y_true, y_pred, y_prob = targets[:, i], preds[:, i], probs[:, i]
        row = {
            "label": label,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }
        # AUC metrics need both classes present in the sample
        if len(np.unique(y_true)) > 1:
            row["roc_auc"] = roc_auc_score(y_true, y_prob)
            row["pr_auc"] = average_precision_score(y_true, y_prob)
        else:
            row["roc_auc"] = np.nan
            row["pr_auc"] = np.nan
        rows.append(row)

    results = pd.DataFrame(rows)
    print(results.round(3).to_string(index=False))
    results.to_csv(config.OUTPUT_DIR + "/test_metrics.csv", index=False)

    plot_metrics(results)


def plot_metrics(results):
    plt.figure(figsize=(8, 4))
    x = np.arange(len(results))
    width = 0.25
    plt.bar(x - width, results["precision"], width, label="precision")
    plt.bar(x, results["recall"], width, label="recall")
    plt.bar(x + width, results["f1"], width, label="f1")
    plt.xticks(x, results["label"], rotation=30, ha="right")
    plt.ylabel("score")
    plt.title("Per-label test metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.METRICS_PLOT_PATH)
    plt.close()


if __name__ == "__main__":
    main()
