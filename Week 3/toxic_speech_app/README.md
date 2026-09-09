# Speech Toxicity Checker (Whisper + LSTM)

Records or uploads audio, transcribes it with OpenAI Whisper, then runs the
transcript through the LSTM multi-label classifier from `toxic_comment_lstm_project`
to check for toxic content.

## Repo structure

```
toxic_speech_app/
├── app.py              # Streamlit UI — run this
├── transcriber.py       # Whisper wrapper (speech -> text)
├── classifier.py        # loads the LSTM model + vocab, text -> toxicity scores
├── model.py             # LSTM architecture (must match the trained checkpoint)
├── preprocess.py        # vocab loading + text-to-id encoding
├── utils.py             # text cleaning / tokenizing
├── config.py            # paths, labels, thresholds
├── requirements.txt
├── models/              # put lstm_model.pt and vocab.json here
└── README.md
```

## Setup

```
pip install -r requirements.txt
```

You also need `ffmpeg` installed on your system for Whisper to read audio files
(`sudo apt install ffmpeg` on Linux, `brew install ffmpeg` on Mac).

Copy the two files produced by `train.py` in `toxic_comment_lstm_project` into
`models/`:

```
cp ../toxic_comment_lstm_project/outputs/lstm_model.pt models/
cp ../toxic_comment_lstm_project/outputs/vocab.json models/
```

## Run

```
streamlit run app.py
```

Then open the local URL Streamlit prints, upload or record a clip, and you'll
get the transcript plus a per-label toxicity score (toxic, severe_toxic,
obscene, threat, insult, identity_hate).

## Notes

- The first run downloads the Whisper model (`base` by default, set in
  `config.py` — use `tiny` for a faster/lighter option, `small`/`medium` for
  better accuracy).
- `model.py` here must stay identical to the one used for training, otherwise
  the saved weights won't load correctly.
- This reuses the exact same text cleaning/tokenizing/encoding logic as
  training, so predictions match what the model actually learned.
