#!/usr/bin/env python3
"""
CLI demo: chunk long text, phonemize per chunk, synthesize with ONNX.

Run from the repo root (paths default to ``onnx_models/``, ``config/tts.json``,
``voices/female1.json``, ``model.onnx`` under the repo)::

    uv run python examples/app.py --text "שלום עולם" --lang he

Override paths if needed::

    uv run python examples/app.py --text "Hello" --lang en --style_json path/to/voice.json

For mixed-language text, use base ``--lang`` (usually ``he``) and inline tags
``<en>...</en>``, ``<es>...</es>``, ``<it>...</it>``, ``<ge>...</ge>``, ``<he>...</he>``.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_repo_import_path() -> None:
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))


def _chunk_text(text: str, max_len: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]
    pattern = r"(?<=[.!?])\s+"
    chunks: list[str] = []
    for para in paragraphs:
        sentences = re.split(pattern, para)
        current = ""
        for s in sentences:
            if len(current) + len(s) + 1 <= max_len:
                current += (" " if current else "") + s
            else:
                if current:
                    chunks.append(current.strip())
                current = s
        if current:
            chunks.append(current.strip())
    return chunks or [text]


def main() -> int:
    _ensure_repo_import_path()
    from src.blue_onnx import BlueTTS

    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_dir", default=str(_REPO_ROOT / "onnx_models"))
    parser.add_argument(
        "--config",
        default=str(_REPO_ROOT / "config" / "tts.json")
        if (_REPO_ROOT / "config" / "tts.json").is_file()
        else "tts.json",
    )
    parser.add_argument("--renikud_path", default=str(_REPO_ROOT / "model.onnx"))
    parser.add_argument("--style_json", default=str(_REPO_ROOT / "voices" / "female1.json"))
    parser.add_argument("--out", default="audio.wav")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk_len", type=int, default=1024)
    parser.add_argument("--text", type=str, default="שלום עולם")
    parser.add_argument(
        "--lang",
        type=str,
        default="he",
        help="Phonemizer language: he, en, es, it, ge (German). For mixed text, use base lang + <en>...</en> tags.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.onnx_dir):
        print(f"Missing ONNX directory: {args.onnx_dir}\nSee README: hf download notmax123/Blue …")
        return 1
    if not os.path.isfile(args.style_json):
        print(f"Missing voice JSON: {args.style_json}")
        return 1

    renikud = args.renikud_path if os.path.isfile(args.renikud_path) else None

    tts = BlueTTS(
        onnx_dir=args.onnx_dir,
        config_path=args.config,
        style_json=args.style_json,
        steps=args.steps,
        cfg_scale=args.cfg,
        speed=args.speed,
        seed=args.seed,
        use_gpu=not args.cpu,
        renikud_path=renikud,
    )

    chunks = _chunk_text(args.text, max_len=args.chunk_len)
    silence = np.zeros(int(0.15 * tts.sample_rate), dtype=np.float32)

    t0 = time.time()
    parts: list[np.ndarray] = []
    for i, chunk in enumerate(chunks):
        wav, _ = tts.synthesize(chunk, lang=args.lang)
        parts.append(wav)
        if i < len(chunks) - 1:
            parts.append(silence)

    wav = np.concatenate(parts) if parts else np.array([], dtype=np.float32)
    t1 = time.time()

    dur = len(wav) / float(tts.sample_rate) if len(wav) > 0 else 0.0
    if dur > 0:
        print(f"Generated {dur:.2f}s in {t1 - t0:.2f}s (RTF: {(t1 - t0) / dur:.3f})")
    sf.write(args.out, wav, tts.sample_rate)
    print(f"Saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
