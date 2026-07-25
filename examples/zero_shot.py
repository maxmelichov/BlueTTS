"""
Zero-shot voice style from a reference WAV using ONNX-only style extraction.

Prepare the reference clip:
    wget https://github.com/thewh1teagle/phonikud-chatterbox/releases/download/asset-files-v1/male1.wav -O ref.wav

Hebrew G2P (RenikudPlus) downloads its own weights on first use.

Run:
    uv run python examples/zero_shot.py
"""

import os
import sys
from pathlib import Path

import soundfile as sf

sys.path.append(".")
from src.blue_onnx import limit_peak, load_text_to_speech
from src.blue_onnx.style import style_from_wav

Path("examples/out").mkdir(parents=True, exist_ok=True)

onnx_dir = os.environ.get("ONNX_DIR", "onnx_models")
ref_wav = os.environ.get("REF_WAV", "ref.wav")

missing = [
    g
    for g in ("codec_encoder", "style_encoder", "duration_style_encoder")
    if not os.path.exists(os.path.join(onnx_dir, f"{g}.onnx"))
]
if missing:
    raise SystemExit(
        f"{onnx_dir} is missing the zero-shot graphs: {', '.join(missing)}. "
        "Download the full bundle (see README > Models)."
    )
if not os.path.exists(ref_wav):
    raise SystemExit(f"Reference clip not found: {ref_wav} (see the header of this file).")

style = style_from_wav(ref_wav, onnx_dir=onnx_dir, config=f"{onnx_dir}/tts.json")
tts = load_text_to_speech(onnx_dir=onnx_dir)

audio, _ = tts(
    "שימו לב נוסעים יקרים, הרכבת תיכנס לתחנת תל אביב מרכז בעוד מספר דקות.",
    lang="he",
    style=style,
    total_step=5,
    cfg_scale=4.0,
    text_is_phonemes=False,
)
if audio.ndim == 2:
    audio = audio[0]
audio = limit_peak(audio)  # keep the WAV from clipping when the vocoder overshoots

out = "examples/out/zero_shot.wav"
sf.write(out, audio, tts.sample_rate)
print("Saved", out)
