"""
Streaming: get the audio chunk by chunk, so playback starts after the first sentence instead of
after the whole text.

Run:
    uv run python examples/stream.py
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.append(".")
from src.blue_onnx import BlueTTS

onnx_dir = os.environ.get("ONNX_DIR", "bluev3_onnx/onnx")
Path("examples/out").mkdir(parents=True, exist_ok=True)
tts = BlueTTS(onnx_dir=onnx_dir, style_json="voices/noa.json")

text = (
    "Streaming returns each sentence as soon as it is ready. "
    "You can send every chunk straight to the sound card or over a socket. "
    "The rest of the text keeps rendering while the first sentence plays."
)
start = time.perf_counter()
chunks = []
for i, chunk in enumerate(tts.stream(text, lang="en")):
    # chunk: mono float32 at tts.sample_rate -- play it here (e.g. sounddevice.OutputStream.write)
    print(f"chunk {i}: {len(chunk) / tts.sample_rate:.2f} s of audio, ready at {time.perf_counter() - start:.2f} s")
    chunks.append(chunk)
sf.write("examples/out/stream.wav", np.concatenate(chunks), tts.sample_rate)
print("Saved examples/out/stream.wav")
