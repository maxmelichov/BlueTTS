#!/usr/bin/env python3
"""
Synthesize one WAV per language (he, en, es, it, ge) plus a mixed tagged utterance.

Run from the repo root (paths resolve relative to the repository)::

    uv run python examples/basic.py

Needs ``onnx_models/``, ``voices/*.json``, ``config/tts.json`` (or cwd ``tts.json``),
and ``model.onnx`` for Hebrew G2P — see the main README.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_repo_import_path() -> None:
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    _ensure_repo_import_path()
    from src.blue_onnx import LightBlueTTS

    ap = argparse.ArgumentParser(description="Write per-language and mixed sample WAVs.")
    ap.add_argument("--onnx_dir", type=Path, default=_REPO_ROOT / "onnx_models")
    ap.add_argument("--style_json", type=Path, default=_REPO_ROOT / "voices" / "female1.json")
    ap.add_argument("--renikud_path", type=Path, default=_REPO_ROOT / "model.onnx")
    ap.add_argument("--out_dir", type=Path, default=_REPO_ROOT / "examples" / "out")
    ap.add_argument("--cpu", action="store_true", help="Force ONNX Runtime CPU providers.")
    args = ap.parse_args()

    cfg = _REPO_ROOT / "config" / "tts.json"
    config_path = str(cfg) if cfg.is_file() else "tts.json"

    if not args.onnx_dir.is_dir():
        print(f"Missing ONNX directory: {args.onnx_dir}\nSee README: hf download notmax123/Blue …")
        return 1
    if not args.style_json.is_file():
        print(f"Missing voice JSON: {args.style_json}")
        return 1

    renikud = str(args.renikud_path) if args.renikud_path.is_file() else None
    if renikud is None:
        print("[WARN] renikud model.onnx not found; Hebrew lines may be wrong or fail.")

    tts = LightBlueTTS(
        onnx_dir=str(args.onnx_dir),
        config_path=config_path,
        style_json=str(args.style_json),
        renikud_path=renikud,
        use_gpu=not args.cpu,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_lang: list[tuple[str, str]] = [
        ("he", "שלום, זהו מודל דיבור בעברית."),
        ("en", "Hello, this is a short English test."),
        ("es", "Hola, esta es una prueba breve en español."),
        ("it", "Ciao, questa è una prova breve in italiano."),
        ("ge", "Hallo, das ist ein kurzer deutscher Test."),
    ]

    for code, text in per_lang:
        audio, sr = tts.synthesize(text, lang=code)
        out_path = args.out_dir / f"sample_{code}.wav"
        sf.write(str(out_path), audio, sr)
        print(f"Wrote {out_path} ({len(audio) / sr:.2f}s)")

    mixed = (
        "שלום לכולם, <en>welcome to the demo</en>, <es>gracias por escuchar</es>, "
        "<it>buona giornata</it>, <ge>auf Wiederhören</ge>."
    )
    audio, sr = tts.synthesize(mixed, lang="he")
    mixed_path = args.out_dir / "sample_mixed.wav"
    sf.write(str(mixed_path), audio, sr)
    print(f"Wrote {mixed_path} ({len(audio) / sr:.2f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
