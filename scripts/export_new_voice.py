#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a *voice style* JSON for Light-BlueTTS from one reference WAV.

The JSON is consumed by ONNX inference (``src.blue_onnx``): it expects
``style_ttl`` (reference-encoder outputs) and optionally ``style_dp``
(duration-predictor style tokens). We do **not** embed ``style_keys``; the
runtime uses ``style_ttl`` for both value and key conditioning when keys are
absent.

**Pipeline (high level)**

1. Load ``tts.json``-style config and per-channel mean/std stats (same
   normalization as training: compressed latents are scaled by
   ``normalizer_scale`` after ``(z - mean) / std``).
2. Run the Blue codec **encoder** on the waveform: mel → latent → **compress**
   adjacent frames by ``chunk_compress_factor`` → ``z_ref_raw`` / ``z_ref_norm``.
3. Optionally run the TTL **ReferenceEncoder** on ``z_ref_norm`` to get
   ``style_ttl`` (shape ``[1, n_style, d_model]``).
4. Optionally run the duration model’s **ref_encoder** on the same trimmed
   latents to get ``style_dp`` (shape ``[1, n_tokens, dim]``).
5. Write JSON plus optional ``.pt`` with raw compressed latents for PyTorch
   tooling.

**Weights**

Latent normalization uses ``stats_multilingual.pt`` from Hugging Face
``notmax123/Blue``. The codec and style heads need
``blue_codec.safetensors``, ``vf_estimator.safetensors``, and
``duration_predictor.safetensors`` from the same repo (or any matching paths
on disk). To refresh stats only::

    hf download notmax123/Blue stats_multilingual.pt --local-dir ./pt_weights

Run this script from the repo root and pass paths, e.g.::

    PYTHONPATH=training uv run python scripts/export_new_voice.py \\
        --ref_wav /path/to/ref.wav \\
        --out voices/mine.json \\
        --config config/tts.json \\
        --ae_ckpt pt_weights/blue_codec.safetensors \\
        --ttl_ckpt pt_weights/vf_estimator.safetensors \\
        --dp_ckpt pt_weights/duration_predictor.safetensors \\
        --stats pt_weights/stats_multilingual.pt \\
        --verify_hf_sizes
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch

# ``models.*`` lives under ``training/`` (see scripts/README.md).
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_TRAINING = os.path.join(_ROOT, "training")
if _TRAINING not in sys.path:
    sys.path.insert(0, _TRAINING)

from bluecodec.autoencoder.latent_encoder import LatentEncoder
from models.utils import LinearMelSpectrogram, compress_latents, load_ttl_config

# Expected file sizes (bytes) for ``notmax123/Blue`` main branch — sanity check only.
HF_REPO_ID = "notmax123/Blue"
HF_WEIGHT_SIZES: dict[str, int] = {
    "blue_codec.safetensors": 245_114_104,
    "duration_predictor.safetensors": 2_040_512,
    "stats_multilingual.pt": 3_133,
    "vf_estimator.safetensors": 179_313_224,
}


def load_torch_or_safetensors(path: str, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a flat PyTorch state dict from ``.safetensors`` or pickled ``.pt``."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path, device=str(map_location))
    return torch.load(path, map_location=map_location, weights_only=False)


def state_dict_with_prefix(flat_sd: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Keep keys ``prefix.*`` and strip the prefix (HF exports use flat names)."""
    p = prefix + "."
    out = {k[len(p) :]: v for k, v in flat_sd.items() if k.startswith(p)}
    if not out:
        raise KeyError(f"No keys with prefix {p!r} in checkpoint")
    return out


def load_ae_encoder_state(raw: dict[str, Any]) -> dict[str, Any]:
    """
    ``blue_codec.safetensors`` may store weights as:

    - nested ``{"encoder": {...}}`` (training checkpoints), or
    - flat ``encoder.*`` tensors (HF), or
    - a bare encoder state dict.
    """
    enc = raw.get("encoder")
    if isinstance(enc, dict):
        return enc
    if any(k.startswith("encoder.") for k in raw):
        return state_dict_with_prefix(raw, "encoder")
    return raw


def verify_hf_file_sizes(paths: list[str]) -> None:
    """Fail fast if a known HF filename has an unexpected size (corrupt/partial download)."""
    for p in paths:
        if not os.path.isfile(p):
            continue
        base = os.path.basename(p)
        expected = HF_WEIGHT_SIZES.get(base)
        if expected is None:
            continue
        got = os.path.getsize(p)
        if got != expected:
            raise ValueError(
                f"Size mismatch for {base}: got {got} bytes, expected {expected} for {HF_REPO_ID}."
            )
        print(f"[OK] Size check {base}: {got} bytes")


def load_stats(device: str, preferred: str, fallback: str = "stats.pt") -> tuple[torch.Tensor, torch.Tensor, str]:
    stats_path = preferred if os.path.exists(preferred) else fallback
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Missing stats file: tried {preferred} and {fallback}")
    stats = torch.load(stats_path, map_location=device, weights_only=False)
    mean = stats["mean"].to(device).view(1, -1, 1)
    std = stats["std"].to(device).view(1, -1, 1)
    return mean, std, stats_path


def read_wav_mono(path: str, target_sr: int = 44100) -> tuple[np.ndarray, int]:
    """Load WAV, downmix to mono, resample to ``target_sr`` if needed."""
    wav, sr = sf.read(path)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return wav, sr


def resolve_ttl_checkpoint(path: str) -> str:
    """
    ``path`` may be a single ``vf_estimator.safetensors`` file or a training
    directory containing ``ckpt_step_*.pt`` (we take the highest step).
    """
    if not os.path.isdir(path):
        return path
    ckpt_files = glob.glob(os.path.join(path, "ckpt_step_*.pt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No ckpt_step_*.pt under {path}")

    def step_key(p: str) -> int:
        m = re.search(r"ckpt_step_(\d+)", p)
        return int(m.group(1)) if m else -1

    ckpt_files.sort(key=step_key)
    return ckpt_files[-1]


def trim_reference_latents(z: torch.Tensor, label: str, *, max_frames: int = 150) -> torch.Tensor:
    """
    Match inference preprocessing: drop ~5% of trailing frames (transient tail),
    then cap length so reference encoders see the same span as at runtime.
    """
    z = z.clone()
    t = z.shape[2]
    tail = max(2, int(t * 0.05))
    t_trim = max(1, t - tail)
    if t_trim < t:
        print(f"[INFO] {label}: trim tail {t - t_trim} frames ({t} -> {t_trim})")
        z = z[:, :, :t_trim]
    if z.shape[2] > max_frames:
        print(f"[INFO] {label}: cap {z.shape[2]} -> {max_frames} frames")
        z = z[:, :, :max_frames]
    return z


def main() -> None:
    ap = argparse.ArgumentParser(description="Export style_ttl / style_dp JSON from a reference WAV.")
    ap.add_argument("--ref_wav", type=str, required=True, help="Reference WAV (ideally 44.1 kHz; resampled if not).")
    ap.add_argument("--out", type=str, default="voice.json", help="Output JSON path.")
    ap.add_argument(
        "--out_pt",
        type=str,
        default=None,
        help="If set, save compressed pre-norm latents for PyTorch ``--z_ref`` tooling.",
    )
    ap.add_argument("--ae_ckpt", type=str, default="blue_codec.safetensors", help="Codec checkpoint (HF name or .pt).")
    ap.add_argument("--stats", type=str, default="stats_multilingual.pt", help="Stats .pt (fallback: stats.pt).")
    ap.add_argument(
        "--ttl_ckpt",
        type=str,
        default="vf_estimator.safetensors",
        help="VF bundle .safetensors or directory of ckpt_step_*.pt.",
    )
    ap.add_argument("--dp_ckpt", type=str, default="duration_predictor.safetensors", help="Duration model weights.")
    ap.add_argument("--verify_hf_sizes", action="store_true", help="Check sizes for known HF filenames.")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--config", type=str, default="configs/tts.json", help="tts.json (dims, mel, normalizer).")
    args = ap.parse_args()

    device = args.device

    if args.verify_hf_sizes:
        verify_hf_file_sizes([args.ae_ckpt, args.ttl_ckpt, args.dp_ckpt, args.stats])

    cfg: dict[str, Any] | None = None
    if os.path.isfile(args.config):
        cfg = load_ttl_config(args.config)
        ver = cfg["full_config"].get("tts_version", "?")
        print(f"[INFO] config {args.config} (v{ver})")
    else:
        print(f"[WARN] missing {args.config}, using built-in defaults")

    chunk_compress_factor = cfg["chunk_compress_factor"] if cfg else 6
    compressed_channels = cfg["compressed_channels"] if cfg else 144
    normalizer_scale = cfg["normalizer_scale"] if cfg else 1.0

    mean, std, stats_path = load_stats(device, preferred=args.stats)
    print(f"[INFO] stats {stats_path}")

    if not os.path.isfile(args.ae_ckpt):
        raise FileNotFoundError(f"Missing AE checkpoint: {args.ae_ckpt}")

    ae_raw = load_torch_or_safetensors(args.ae_ckpt, map_location="cpu")
    enc_sd = load_ae_encoder_state(ae_raw)

    ae_enc_cfg = cfg["ae_enc_cfg"] if cfg else {
        "ksz": 7,
        "hdim": 512,
        "intermediate_dim": 2048,
        "dilation_lst": [1] * 10,
        "odim": 24,
        "idim": 228,
    }
    sr = cfg["ae_sample_rate"] if cfg else 44100
    n_fft = cfg["ae_n_fft"] if cfg else 2048
    hop = cfg["ae_hop_length"] if cfg else 512
    n_mels = cfg["ae_n_mels"] if cfg else 228

    mel_spec = LinearMelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels).to(device)
    ae_encoder = LatentEncoder(cfg=ae_enc_cfg).to(device).eval()
    ae_encoder.load_state_dict(enc_sd, strict=True)

    wav, _ = read_wav_mono(args.ref_wav, target_sr=sr)
    wav_t = torch.from_numpy(wav).to(device).unsqueeze(0).unsqueeze(0)

    with torch.inference_mode():
        mel = mel_spec(wav_t.squeeze(1))
        tm = mel.shape[-1]
        aligned = (tm // chunk_compress_factor) * chunk_compress_factor
        if aligned < tm:
            mel = mel[..., :aligned]

        z = ae_encoder(mel)
        zc = compress_latents(z, factor=chunk_compress_factor)
        z_ref_raw = zc.clone()
        z_ref_norm = ((zc - mean) / std) * normalizer_scale

        if z_ref_norm.shape[1] != compressed_channels:
            raise ValueError(f"latent C={z_ref_norm.shape[1]}, expected {compressed_channels}")
        if mean.shape[1] != compressed_channels or std.shape[1] != compressed_channels:
            raise ValueError(f"stats C mismatch: mean {mean.shape}, std {std.shape}")
        if not torch.isfinite(z_ref_norm).all():
            raise RuntimeError("non-finite latents (bad audio or stats mismatch)")

    if args.out_pt:
        out_pt = args.out_pt if args.out_pt.endswith(".pt") else args.out_pt + ".pt"
        os.makedirs(os.path.dirname(out_pt) or ".", exist_ok=True)
        torch.save(
            {
                "z_ref_raw": z_ref_raw.cpu(),
                "is_normalized": False,
                "metadata": {
                    "ref_wav": os.path.abspath(args.ref_wav),
                    "stats_path": stats_path,
                    "sr": sr,
                    "z_ref_raw_shape": list(z_ref_raw.shape),
                },
            },
            out_pt,
        )
        print(f"[OK] z_ref_raw -> {out_pt}")

    # --- style_ttl (TTL reference encoder) ---
    style_ttl: np.ndarray | None = None
    try:
        from models.text2latent.reference_encoder import ReferenceEncoder

        ttl_path = resolve_ttl_checkpoint(args.ttl_ckpt)
        if not os.path.exists(ttl_path):
            print(f"[WARN] no TTL checkpoint at {ttl_path}; skip style_ttl")
        else:
            ref_enc = ReferenceEncoder(
                in_channels=compressed_channels,
                d_model=cfg["se_d_model"] if cfg else 256,
                hidden_dim=cfg["se_hidden_dim"] if cfg else 1024,
                num_blocks=cfg["se_num_blocks"] if cfg else 6,
                num_tokens=cfg["se_n_style"] if cfg else 50,
                num_heads=cfg["se_n_heads"] if cfg else 2,
            ).to(device).eval()

            ckpt = load_torch_or_safetensors(ttl_path, map_location=device)
            if "reference_encoder" in ckpt:
                ref_enc.load_state_dict(ckpt["reference_encoder"], strict=True)
            elif any(k.startswith("reference_encoder.") for k in ckpt):
                ref_enc.load_state_dict(state_dict_with_prefix(ckpt, "reference_encoder"), strict=True)
            else:
                raise KeyError(f"no reference_encoder weights in {ttl_path}")

            z_tr = trim_reference_latents(z_ref_norm, "RefEnc")
            mask = torch.ones(1, 1, z_tr.shape[2], device=device, dtype=torch.float32)
            with torch.inference_mode():
                style_ttl = ref_enc(z_tr, mask=mask).cpu().numpy().astype(np.float32)
            print(f"[OK] style_ttl <- {ttl_path}")
    except Exception as e:
        print(f"[WARN] style_ttl: {e}")

    # --- style_dp (duration model reference path) ---
    style_dp: np.ndarray | None = None
    try:
        if args.dp_ckpt and os.path.isfile(args.dp_ckpt):
            from models.text2latent.dp_network import DPNetwork

            tok = cfg["dp_style_tokens"] if cfg else 8
            dim = cfg["dp_style_dim"] if cfg else 16
            state = load_torch_or_safetensors(args.dp_ckpt, map_location=device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]

            emb = "sentence_encoder.text_embedder.char_embedder.weight"
            vs = cfg["dp_vocab_size"] if cfg else 37
            if isinstance(state, dict) and emb in state:
                vs = int(state[emb].shape[0])

            dp = DPNetwork(vocab_size=vs, style_tokens=tok, style_dim=dim).to(device).eval()
            dp.load_state_dict(state, strict=True)

            z_dp = trim_reference_latents(z_ref_norm, "DP")
            mask = torch.ones(1, 1, z_dp.shape[2], device=device, dtype=torch.float32)
            with torch.inference_mode():
                flat = dp.ref_encoder(z_dp, mask=mask)
                shaped = flat.reshape(flat.shape[0], tok, dim)
            style_dp = shaped.cpu().numpy().astype(np.float32)
            print(f"[OK] style_dp <- {args.dp_ckpt}")
        else:
            print(f"[INFO] skip style_dp ({args.dp_ckpt})")
    except Exception as e:
        print(f"[WARN] style_dp: {e}")

    payload: dict[str, Any] = {}
    if style_ttl is not None:
        payload["style_ttl"] = {"data": style_ttl.tolist(), "dims": list(style_ttl.shape)}
    if style_dp is not None:
        payload["style_dp"] = {"data": style_dp.tolist(), "dims": list(style_dp.shape)}
    payload["metadata"] = {
        "sr": sr,
        "ref_wav": os.path.abspath(args.ref_wav),
        "stats_path": stats_path,
        "z_ref_norm_shape": list(z_ref_norm.shape),
    }

    out_json = args.out if args.out.endswith(".json") else args.out + ".json"
    with open(out_json, "w") as f:
        json.dump(payload, f)
    print(f"[OK] wrote {out_json}")


if __name__ == "__main__":
    main()
