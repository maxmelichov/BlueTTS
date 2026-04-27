"""
Zero-shot voice style from a reference WAV using ONNX-only style extraction.

Prepare the reference clip:
    wget https://github.com/thewh1teagle/phonikud-chatterbox/releases/download/asset-files-v1/male1.wav -O ref.wav

Prepare Hebrew G2P:
    wget https://huggingface.co/thewh1teagle/renikud/resolve/main/model.onnx -O model.onnx

Run:
    uv run python examples/zero_shot.py
"""

from pathlib import Path

import soundfile as sf

from blue_onnx import load_text_to_speech
from blue_onnx.style import style_from_wav

Path("examples/out").mkdir(parents=True, exist_ok=True)

onnx_dir = "onnx_models-int8"
ref_wav = "ref.wav"
renikud = "model.onnx"

style = style_from_wav(ref_wav, onnx_dir=onnx_dir, config=f"{onnx_dir}/tts.json")
tts = load_text_to_speech(onnx_dir=onnx_dir)
if tts.g2p is not None:
    tts.g2p = type(tts.g2p)(renikud)

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

out = "examples/out/zero_shot.wav"
sf.write(out, audio, tts.sample_rate)
print("Saved", out)
