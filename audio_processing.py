import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write as write_wav
import whisper


def record_audio(duration: int = 5, sample_rate: int = 16000) -> str:
    """Record audio from the microphone."""
    st.info("Recording... Speak your question.")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    st.info("Recording finished.")

    audio_file = "temp_audio.wav"
    write_wav(audio_file, sample_rate, recording)
    return audio_file


@st.cache_resource
def load_whisper_model():
    """Load the Whisper model for speech recognition."""
    return whisper.load_model("base")


def transcribe_audio(audio_file: str) -> str:
    """Transcribe the given audio file using the Whisper model."""
    model = load_whisper_model()
    result = model.transcribe(audio_file)
    return result["text"]
