"""
Voice cloning with BlueV3: speak in the voice of a reference recording.

Use 3-10 s of clean speech from one speaker (WAV or FLAC, any sample rate). The bundle's own
``reference_encoder.onnx`` turns the clip into a voice; no PyTorch needed.

Run:
    uv run python examples/clone.py path/to/reference.wav
"""

import os
import sys
from pathlib import Path

import soundfile as sf

sys.path.append(".")
from src.blue_onnx import BlueTTS

ref_wav = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("REF_WAV", "ref.wav")
onnx_dir = os.environ.get("ONNX_DIR", "bluev3_onnx/onnx")
if not os.path.exists(ref_wav):
    raise SystemExit(f"Reference clip not found: {ref_wav}")
Path("examples/out").mkdir(parents=True, exist_ok=True)

tts = BlueTTS(onnx_dir=onnx_dir, ref_wav=ref_wav)

# Keep the voice: the JSON loads later with BlueTTS(style_json="examples/out/my_voice.json").
tts.voice_from_wav(ref_wav, save_json="examples/out/my_voice.json")

samples, sr = tts.synthesize("Hello, this is my cloned voice speaking.", lang="en")
sf.write("examples/out/clone_en.wav", samples, sr)
samples, sr = tts.synthesize("שלום, זה הקול המשוכפל שלי.", lang="he")
sf.write("examples/out/clone_he.wav", samples, sr)
print("Saved examples/out/clone_en.wav, clone_he.wav and my_voice.json")
