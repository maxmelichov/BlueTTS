"""
Usage:
    wget https://huggingface.co/thewh1teagle/renikud/resolve/main/model.onnx
    wget https://github.com/maxmelichov/Light-BlueTTS/releases/download/model-files-v1.0/onnx_models.tar.gz
    wget https://github.com/maxmelichov/Light-BlueTTS/releases/download/model-files-v1.0/voices.tar.gz
    tar -xf onnx_models.tar.gz
    tar -xf voices.tar.gz
    uv pip install renikud-onnx
    uv run examples/app.py --text "שלום עולם" --style_json voices/female1.json
"""
import os
import re
import time
import argparse

import numpy as np
import soundfile as sf

from lightblue_onnx import LightBlueTTS

try:
    from renikud_onnx import G2P
except ImportError:
    G2P = None


def _phonemize(text: str, renikud: object) -> str:
    is_hebrew = any('\u0590' <= c <= '\u05ff' for c in text)

    if not is_hebrew:
        return text
    else:
        if renikud is None:
            raise RuntimeError("renikud not available for Hebrew text")
        return renikud.phonemize(text)


def _chunk_text(text: str, max_len: int) -> list:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]
    pattern = r"(?<=[.!?])\s+"
    chunks = []
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


def _try_load_z_ref(path: str) -> np.ndarray:
    if path.endswith(".pt"):
        try:
            import torch
        except Exception as e:
            raise RuntimeError(f"torch is required to load {path}: {e}")
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            for key in ("z_ref_raw", "z_ref_norm"):
                if key in payload:
                    z = payload[key]
                    break
            else:
                z = next((v for v in payload.values() if torch.is_tensor(v)), None)
                if z is None:
                    raise ValueError(f"Unsupported .pt keys: {list(payload.keys())}")
        else:
            z = payload
        if torch.is_tensor(z):
            z = z.detach().cpu().numpy()
        z = np.array(z, dtype=np.float32)
    else:
        z = np.load(path).astype(np.float32)
    if z.ndim == 2:
        z = z[None, :, :]
    return z


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_dir", default="onnx_models")
    parser.add_argument("--config", default="tts.json")
    parser.add_argument("--renikud_path", default="model.onnx")
    parser.add_argument("--z_ref", default=None)
    parser.add_argument("--style_json", default="voices/female1.json")
    parser.add_argument("--out", default="audio.wav")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk_len", type=int, default=1024)
    parser.add_argument("--text", type=str, default="שלום עולם")
    args = parser.parse_args()

    renikud = None
    if G2P is not None and os.path.exists(args.renikud_path):
        renikud = G2P(args.renikud_path)

    tts = LightBlueTTS(
        onnx_dir=args.onnx_dir,
        config_path=args.config,
        style_json=args.style_json if os.path.exists(args.style_json) else None,
        steps=args.steps,
        cfg_scale=args.cfg,
        speed=args.speed,
        seed=args.seed,
        use_gpu=not args.cpu,
    )

    z_ref = _try_load_z_ref(args.z_ref) if args.z_ref else None

    chunks = _chunk_text(args.text, max_len=args.chunk_len)
    silence = np.zeros(int(0.15 * tts.sample_rate), dtype=np.float32)

    t0 = time.time()
    parts = []
    for i, chunk in enumerate(chunks):
        phonemes = _phonemize(chunk, renikud)
        wav, _ = tts.create(phonemes)
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


if __name__ == "__main__":
    main()
