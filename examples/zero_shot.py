"""
Voice cloning from a reference WAV, ONNX only (no PyTorch).

With the BlueV3 bundle (default, ``./bluev3_onnx``) the bundle's ``onnx/reference_encoder.onnx``
turns the clip into a voice. Any sample rate and channel count works (it is mixed to mono and
resampled to 44.1 kHz); use 8-10 s of one clean speaker. Only about the first 9 s are used.

Prepare a reference clip:
    wget https://github.com/thewh1teagle/phonikud-chatterbox/releases/download/asset-files-v1/male1.wav -O ref.wav

Run:
    uv run python examples/zero_shot.py

The cloned voice is also saved as ``examples/out/my_voice.json`` (load it like any file in
``voices/``).
"""

import json
import os
import sys
from pathlib import Path

import soundfile as sf

sys.path.append(".")
from src.blue_onnx import BlueTTS
from src.blue_onnx.style import export_voice_style

Path("examples/out").mkdir(parents=True, exist_ok=True)

onnx_dir = os.environ.get("ONNX_DIR", "bluev3_onnx/onnx")
ref_wav = os.environ.get("REF_WAV", "ref.wav")
if not os.path.exists(ref_wav):
    raise SystemExit(f"Reference clip not found: {ref_wav} (see the header of this file).")

payload = export_voice_style(ref_wav, onnx_dir=onnx_dir)
with open("examples/out/my_voice.json", "w", encoding="utf-8") as f:
    json.dump(payload, f)

tts = BlueTTS(onnx_dir=onnx_dir, style_json="examples/out/my_voice.json")
samples, sr = tts.synthesize(
    "שימו לב נוסעים יקרים, הרכבת תיכנס לתחנת תל אביב מרכז בעוד מספר דקות.",
    lang="he",
)
out = "examples/out/zero_shot.wav"
sf.write(out, samples, sr)
print("Saved", out, "and examples/out/my_voice.json")
