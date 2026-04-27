import os
import sys
from pathlib import Path

import soundfile as sf

sys.path.append(".")
from src.blue_onnx import load_text_to_speech, load_voice_style

Path("examples/out").mkdir(parents=True, exist_ok=True)

onnx_dir = os.environ.get("ONNX_DIR", "onnx_models")
tts = load_text_to_speech(onnx_dir=onnx_dir)
style = load_voice_style(["voices/female1.json"])
audio, _ = tts(
    "Hello, this is a short test.",
    lang="en",
    style=style,
    total_step=5,
    cfg_scale=3.0,
)
if audio.ndim == 2:
    audio = audio[0]
out = "examples/out/basic.wav"
sf.write(out, audio, tts.sample_rate)
print("Saved", out)
