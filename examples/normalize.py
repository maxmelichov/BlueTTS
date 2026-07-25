"""Synthesize messy real-world text: numbers, dates, times, codes, emails.

`normalize_text=True` rewrites anything G2P cannot read aloud into words before
phonemization, and reads spelled codes / phone numbers / dates / times more
slowly so digit groups stay intelligible.

Run:
    uv run python examples/normalize.py
"""

import os
import sys
from pathlib import Path

import soundfile as sf

sys.path.append(".")
from src.blue_onnx import (
    limit_peak,
    load_text_to_speech,
    load_voice_style,
    prepare_text_for_synthesis,
    split_slow_segments,
)

Path("examples/out").mkdir(parents=True, exist_ok=True)

onnx_dir = os.environ.get("ONNX_DIR", "onnx_models")
tts = load_text_to_speech(onnx_dir=onnx_dir)
style = load_voice_style(["voices/female1.json"])

SAMPLES = [
    ("he", 'ההזמנה IL-4829-7361-05 תגיע ב 12/05/2024 בשעה 08:15, מחיר 1,500 ש"ח (50% הנחה).'),
    ("en", "Invoice INV-77120 for 1,500 dollars is due 12/05/2024 at 08:15 (30% off)."),
]

for lang, raw in SAMPLES:
    prepared = prepare_text_for_synthesis(raw, lang=lang)
    print(f"\n[{lang}] raw:      {raw}")
    print(f"[{lang}] prepared: {prepared}")
    for segment, is_slow in split_slow_segments(prepared):
        print(f"    {'slow' if is_slow else 'norm'}: {segment}")

    audio, _ = tts(
        raw,
        lang=lang,
        style=style,
        total_step=8,
        cfg_scale=4.0,
        normalize_text=True,
    )
    if audio.ndim == 2:
        audio = audio[0]
    audio = limit_peak(audio)
    out = f"examples/out/normalize_{lang}.wav"
    sf.write(out, audio, tts.sample_rate)
    print(f"[{lang}] saved {out} ({len(audio) / tts.sample_rate:.2f}s)")
