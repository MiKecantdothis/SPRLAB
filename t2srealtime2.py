from faster_whisper import WhisperModel
import sounddevice as sd 
import numpy as np
import queue
import threading
import os 
from dotenv import load_dotenv
load_dotenv()



chunk_duraton = 5
block_duration = 0.5
sampling_rate = 16000

frames_per_block = sampling_rate * block_duration
frames_per_chunk = sampling_rate * chunk_duraton

audio_queue = queue.Queue()
audio_buffer = []

model = WhisperModel("small.en", device="cpu", compute_type="int8")

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

def recorder():
    with sd.InputStream(samplerate = sampling_rate, channels = 1, callback = audio_callback, blocksize = int(frames_per_block)):
        print("Recording... control C to stop")
        while True:
            sd.sleep(100)

def transcriber():
    global audio_buffer
    while True:
        block = audio_queue.get()
        audio_buffer.append(block)

        total_frames = sum(len(b) for b in audio_buffer)
        if total_frames >= frames_per_chunk:
            # Combine blocks and clear buffer
            audio_data = np.concatenate(audio_buffer, axis=0)[:frames_per_chunk]
            audio_buffer = []

            # flatten and normalize to [-1, 1]
            audio_data = audio_data.flatten().astype(np.float32)
            audio_data = np.clip(audio_data, -1.0, 1.0)

            # check energy to avoid transcribing silence
            energy = np.mean(np.abs(audio_data))
            if energy < 0.01:
                print("🤫 Skipping silent chunk...")
                continue

            print("🔍 Transcribing...")
            try:
                segments, info = model.transcribe(audio_data, beam_size=1, language="en")
                for segment in segments:
                    text = segment.text.strip()
                    if text:
                        print(f"🗣️ {text}")
            except Exception as e:
                print("⚠️ Error during transcription:", e)


if __name__ == "__main__":
 threading.Thread(target=recorder, daemon=True).start()
 transcriber()