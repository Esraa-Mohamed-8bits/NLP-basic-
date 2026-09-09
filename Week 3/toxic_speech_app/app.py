import os
import tempfile

import streamlit as st

import config
from transcriber import transcribe
from classifier import predict, is_toxic

st.set_page_config(page_title="Speech Toxicity Checker", page_icon="🎙️")

st.title("🎙️ Speech Toxicity Checker")
st.write("Upload or record audio, it gets transcribed with Whisper, then checked for toxicity with our LSTM model.")

audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "ogg"])
mic_audio = st.audio_input("...or record directly")

audio_source = audio_file or mic_audio

if audio_source is not None:
    st.audio(audio_source)

    with st.spinner("Transcribing audio..."):
        suffix = os.path.splitext(getattr(audio_source, "name", "recording.wav"))[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_source.read())
            tmp_path = tmp.name

        try:
            text = transcribe(tmp_path)
        finally:
            os.remove(tmp_path)

    st.subheader("Transcript")
    st.write(text if text else "_(nothing was transcribed)_")

    if text:
        with st.spinner("Checking for toxicity..."):
            scores = predict(text)

        st.subheader("Toxicity scores")
        st.bar_chart(scores)

        if is_toxic(scores):
            st.error("This message looks toxic.")
        else:
            st.success("This message looks clean.")

        with st.expander("Raw scores"):
            st.json(scores)

st.caption(f"Threshold: {config.CLASSIFICATION_THRESHOLD} — model + vocab loaded from `{config.MODEL_DIR}/`")
