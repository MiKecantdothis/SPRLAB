import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import numpy as np
from faster_whisper import WhisperModel
import av
import os

# -------------------------------
# Load environment variables (HF token)
# -------------------------------
from dotenv import load_dotenv
load_dotenv()

# Only login if token exists, and prevent repeated prompts
HF_TOKEN = os.getenv("hf_token")
if HF_TOKEN and not st.session_state.get("hf_logged_in", False):
    from huggingface_hub import login
    login(token=HF_TOKEN)
    st.session_state["hf_logged_in"] = True  # prevent repeated login

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🎤 Real-Time Speech Transcription (Whisper + Streamlit)")
st.markdown("Speak into your microphone and see the transcribed text below!")

# -------------------------------
# Load Whisper model (cached)
# -------------------------------
@st.cache_resource
def load_model():
    return WhisperModel("small.en", device="cpu", compute_type="int8")

model = load_model()

# -------------------------------
# Audio Processor
# -------------------------------
class WhisperAudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.sample_rate = 16000
        self.buffer = np.array([], dtype=np.float32)
        self.text_result = ""

    def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray().mean(axis=0).astype(np.float32)
        print(f"Received audio frame with {len(audio)} samples")
        self.buffer = np.concatenate([self.buffer, audio])

        if len(self.buffer) > 5 * self.sample_rate:
            chunk = self.buffer[: 5 * self.sample_rate]
            self.buffer = self.buffer[5 * self.sample_rate :]
            segments, _ = model.transcribe(chunk, beam_size=1, language="en")
            text_out = " ".join([seg.text.strip() for seg in segments])
            if text_out:
                self.text_result += " " + text_out
                st.session_state["transcribed_text"] = self.text_result

        return frame

# -------------------------------
# Initialize session state
# -------------------------------
if "transcribed_text" not in st.session_state:
    st.session_state["transcribed_text"] = ""

# -------------------------------
# Start WebRTC audio stream
# -------------------------------
ctx = webrtc_streamer(
    key="speech-transcriber",
    mode=WebRtcMode.RECVONLY,
    audio_processor_factory=WhisperAudioProcessor,
    media_stream_constraints={
    "audio": {"echoCancellation": True, "noiseSuppression": True, "sampleRate": 16000},
    "video": False
}

)

# -------------------------------
# Display transcription
# -------------------------------
st.subheader("🗣️ Transcribed Text")
st.text_area(
    "Live Transcription",
    value=st.session_state["transcribed_text"],
    height=250,
)

st.markdown("---")
st.caption("Powered by Faster-Whisper + Streamlit-WebRTC")
