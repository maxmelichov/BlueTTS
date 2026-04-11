#!/usr/bin/env python3
"""
Synthesize every registered vocabulary language plus one mixed tagged utterance,
then write WAVs and a small JSON manifest under ``examples/out/``.

Run from the repo root::

    uv run python examples/all_langs_and_mix.py
    uv sync --extra tensorrt   # once, before:
    uv run python examples/all_langs_and_mix.py --tensorrt   # needs trt_engines/ + torch/tensorrt (rebuild engines if mixed clip fails — text axis must allow long sequences)

Requires ``onnx_models/``, ``voices/*.json``, ``config/tts.json``, and ``model.onnx``
(Renikud G2P — see README). ``renikud-onnx`` is installed with ``uv sync``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import soundfile as sf

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_repo_import_path() -> None:
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))


# One short phrase per canonical LANG_ID key (see src/_blue_vocab.py).
_PHRASES: dict[str, str] = {
    "he": "שלום, זהו מודל דיבור בעברית.",
    "en": "Hello, this is a short English test.",
    "es": "Hola, esta es una prueba breve en español.",
    "it": "Ciao, questa è una prova breve in italiano.",
    "de": "Hallo, das ist ein kurzer deutscher Test.",
}

# Mixed clip: base Hebrew + every other language via tags (includes <ge> alias → de).
_MIXED = (
    "פתיחה בעברית, <en>then English in the middle</en>, "
    "<es>un poco de español</es>, <it>un po' di italiano</it>, "
    "<ge>und ein bisschen Deutsch</ge>, "
    "<he>וסיום בעברית</he>."
)


def main() -> int:
    _ensure_repo_import_path()
    from src._blue_vocab import LANG_ID

    ap = argparse.ArgumentParser(description="Write WAVs for all langs + mixed; save under examples/out/.")
    ap.add_argument("--out_dir", type=Path, default=_REPO_ROOT / "examples" / "out")
    ap.add_argument("--prefix", type=str, default="langs_", help="Output filename prefix.")
    ap.add_argument("--onnx_dir", type=Path, default=_REPO_ROOT / "onnx_models")
    ap.add_argument("--trt_dir", type=Path, default=_REPO_ROOT / "trt_engines")
    ap.add_argument("--style_json", type=Path, default=_REPO_ROOT / "voices" / "female1.json")
    ap.add_argument("--renikud_path", type=Path, default=_REPO_ROOT / "model.onnx")
    ap.add_argument("--cpu", action="store_true", help="ONNX only: CPU providers.")
    ap.add_argument(
        "--tensorrt",
        action="store_true",
        help="Use BlueTRT (CUDA). Requires built engines in --trt_dir and uv sync --extra tensorrt.",
    )
    args = ap.parse_args()

    cfg = _REPO_ROOT / "config" / "tts.json"
    config_path = str(cfg) if cfg.is_file() else "tts.json"

    if not args.style_json.is_file():
        print(f"Missing voice JSON: {args.style_json}")
        return 1
    style_path = str(args.style_json)

    if args.tensorrt:
        if not args.trt_dir.is_dir():
            print(f"Missing TRT engine directory: {args.trt_dir}\nBuild with scripts/create_tensorrt.py")
            return 1
        try:
            from src.blue_trt import BlueTRT
        except ModuleNotFoundError as e:
            name = getattr(e, "name", "") or ""
            if name in ("torch", "tensorrt"):
                print(
                    "TensorRT inference needs PyTorch and TensorRT (not part of the default uv sync).\n"
                    "  uv sync --extra tensorrt\n"
                    "Then re-run this script."
                )
                return 1
            raise

        renikud = str(args.renikud_path) if args.renikud_path.is_file() else None
        tts = BlueTRT(
            trt_dir=str(args.trt_dir),
            config_path=config_path,
            style_json=style_path,
            renikud_path=renikud,
        )
    else:
        if not args.onnx_dir.is_dir():
            print(f"Missing ONNX directory: {args.onnx_dir}")
            return 1
        from src.blue_onnx import BlueTTS

        renikud = str(args.renikud_path) if args.renikud_path.is_file() else None
        if renikud is None:
            print("[WARN] model.onnx not found; Hebrew quality may suffer.")
        tts = BlueTTS(
            onnx_dir=str(args.onnx_dir),
            config_path=config_path,
            style_json=style_path,
            renikud_path=renikud,
            use_gpu=not args.cpu,
        )

    codes = sorted(LANG_ID.keys())
    missing = [c for c in codes if c not in _PHRASES]
    if missing:
        print(f"Add phrases in _PHRASES for: {missing}")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "backend": "tensorrt" if args.tensorrt else "onnx",
        "sample_rate": None,
        "files": [],
    }

    for code in codes:
        text = _PHRASES[code]
        audio, sr = tts.synthesize(text, lang=code)
        out_path = args.out_dir / f"{args.prefix}{code}.wav"
        sf.write(str(out_path), audio, sr)
        sec = round(len(audio) / sr, 3)
        manifest["sample_rate"] = int(sr)
        manifest["files"].append({"kind": "monolingual", "lang": code, "path": str(out_path), "seconds": sec})
        print(f"OK {code}: {out_path.name} ({sec:.2f}s)")

    audio, sr = tts.synthesize(_MIXED, lang="he")
    mixed_path = args.out_dir / f"{args.prefix}mixed.wav"
    sf.write(str(mixed_path), audio, sr)
    sec = round(len(audio) / sr, 3)
    manifest["files"].append(
        {
            "kind": "mixed",
            "lang": "he",
            "path": str(mixed_path),
            "seconds": sec,
            "text_note": "Includes <en>, <es>, <it>, <ge>, <he> segments.",
        }
    )
    print(f"OK mixed: {mixed_path.name} ({sec:.2f}s)")

    man_path = args.out_dir / f"{args.prefix}manifest.json"
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"Wrote {man_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
