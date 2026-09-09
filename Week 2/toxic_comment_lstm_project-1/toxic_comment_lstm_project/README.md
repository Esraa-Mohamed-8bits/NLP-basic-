# Toxic Comment Classification (LSTM, multi-label)

Simple Embedding + LSTM model for the Jigsaw Toxic Comment dataset.
Each comment gets 6 independent probabilities (toxic, severe_toxic,
obscene, threat, insult, identity_hate) using a sigmoid output + BCE loss.

## Setup

```
pip install -r requirements.txt
```

Download `train.csv` from the Kaggle link below and put it in `data/train.csv`:
https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge

## Files

- `config.py` — paths and hyperparameters, edit here first
- `utils.py` — seeding + text cleaning
- `preprocess.py` — loads csv, cleans text, builds vocab, turns tokens into padded id sequences
- `dataset.py` — PyTorch `Dataset` wrapper
- `model.py` — Embedding + BiLSTM + Linear classifier
- `train.py` — splits data (train/val/test), trains the model, saves best checkpoint + loss plot
- `evaluate.py` — loads the saved model, reports precision/recall/f1/ROC-AUC/PR-AUC per label

## Run

```
python train.py
python evaluate.py
```

Outputs (model checkpoint, vocab, plots, metrics csv) are written to `outputs/`.

## Notes / limitations

- Vocabulary is built only from the training split to avoid leakage.
- Labels are imbalanced (e.g. `threat` has very few positive examples), so
  `BCEWithLogitsLoss` is weighted per-label using `pos_weight` computed from
  the training set instead of plain unweighted BCE.
- LSTM reads comments as a single pass over word ids — it captures word order
  and short-range context well, but for very long comments or subtle context
  a Transformer-based model (e.g. fine-tuned BERT) would likely do better.
- Threshold for turning probabilities into predictions is fixed at 0.5
  (`config.CLASSIFICATION_THRESHOLD`) — for very imbalanced labels like
  `threat`, tuning this per label could improve recall.
