import whisper

import config

_model = None


def get_model():
    """Load whisper once and reuse it, loading it every call is slow."""
    global _model
    if _model is None:
        _model = whisper.load_model(config.WHISPER_MODEL_SIZE)
    return _model


def transcribe(audio_path):
    model = get_model()
    result = model.transcribe(audio_path)
    return result["text"].strip()
