#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a *voice style* JSON for Blue (BlueTTS) from one reference WAV.

See repo README for usage. Requires the BlueTTS training codebase on
``PYTHONPATH`` and the PyTorch checkpoints (``blue_codec.safetensors``,
``vf_estimetor.safetensors``, ``duration_predictor_final.safetensors``,
``stats_multilingual.safetensors``).

    PYTHONPATH=training uv run python exports/export_new_voice.py \
        --ref_wav /path/to/ref.wav \
        --out voices/mine.json \
        --config config/tts.json \
        --ae_ckpt pt_weights/blue_codec.safetensors \
        --ttl_ckpt pt_weights/vf_estimetor.safetensors \
        --dp_ckpt pt_weights/duration_predictor_final.safetensors \
        --stats pt_weights/stats_multilingual.safetensors
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_TRAINING = os.path.join(_ROOT, "training")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _TRAINING not in sys.path:
    sys.path.insert(0, _TRAINING)

from bluecodec.autoencoder.latent_encoder import LatentEncoder  # noqa: E402
from training.utils import LinearMelSpectrogram, compress_latents, load_ttl_config  # noqa: E402

HF_REPO_ID = "notmax123/blue-v2"
HF_WEIGHT_SIZES: dict[str, int] = {
    "blue_codec.safetensors": 245_114_104,
    "duration_predictor_final.safetensors": 2_040_744,
    "stats_multilingual.safetensors": 1_416,
    "vf_estimetor.safetensors": 174_487_392,
}


def load_torch_or_safetensors(path: str, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path, device=str(map_location))
    return torch.load(path, map_location=map_location, weights_only=False)


def state_dict_with_prefix(flat_sd: dict[str, Any], prefix: str) -> dict[str, Any]:
    p = prefix + "."
    out = {k[len(p) :]: v for k, v in flat_sd.items() if k.startswith(p)}
    if not out:
        raise KeyError(f"No keys with prefix {p!r} in checkpoint")
    return out


def load_ae_encoder_state(raw: dict[str, Any]) -> dict[str, Any]:
    enc = raw.get("encoder")
    if isinstance(enc, dict):
        return enc
    if any(k.startswith("encoder.") for k in raw):
        return state_dict_with_prefix(raw, "encoder")
    return raw


def verify_hf_file_sizes(paths: list[str]) -> None:
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


def load_stats(
    device: str | torch.device,
    preferred: str,
    fallback: str = "stats.pt",
) -> tuple[torch.Tensor, torch.Tensor, str]:
    stats_path = preferred if os.path.exists(preferred) else fallback
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Missing stats file: tried {preferred} and {fallback}")
    stats = load_torch_or_safetensors(stats_path, map_location=device)
    mean = stats["mean"].to(device).view(1, -1, 1)
    std = stats["std"].to(device).view(1, -1, 1)
    return mean, std, stats_path


def ensure_sr(
    wav: torch.Tensor,
    sr_in: int,
    sr_out: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """High-quality kaiser-sinc resampler. Accepts [T], [1, T] or [B, 1, T]."""
    if device is None:
        device = wav.device
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if sr_in != sr_out:
        wav = torchaudio.functional.resample(
            wav,
            sr_in,
            sr_out,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method="sinc_interp_kaiser",
            beta=14.769656459379492,
        )
    return wav.to(device)


def read_wav_mono(path: str, target_sr: int = 44100) -> tuple[np.ndarray, int]:
    """Load any WAV as mono float32 at ``target_sr`` (44.1 kHz by default).

    Downmixes >1 channels to mono and resamples via the high-quality
    torchaudio kaiser-sinc filter in :func:`ensure_sr`.
    """
    try:
        wav_t, sr = torchaudio.load(path)  # [C, T] float32 in [-1, 1]
    except Exception:
        wav_np, sr = sf.read(path, always_2d=True)
        wav_t = torch.from_numpy(wav_np.T.astype(np.float32))

    if wav_t.shape[0] > 1:
        wav_t = wav_t.mean(dim=0, keepdim=True)

    wav_t = ensure_sr(wav_t, int(sr), int(target_sr), device=torch.device("cpu"))

    wav = wav_t.squeeze(0).contiguous().to(torch.float32).numpy()
    return wav, int(target_sr)


def resolve_ttl_checkpoint(path: str) -> str:
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


def export_voice_style(
    ref_wav: str,
    *,
    config: str = "config/tts.json",
    ae_ckpt: str = "blue_codec.safetensors",
    ttl_ckpt: str = "vf_estimetor.safetensors",
    dp_ckpt: str = "duration_predictor_final.safetensors",
    stats: str = "stats_multilingual.safetensors",
    device: str = "cpu",
    out_pt: str | None = None,
    verify_hf_sizes_flag: bool = False,
) -> dict[str, Any]:
    """Return the complete ONNX voice style payload for a reference WAV.

    This is the API the Gradio app should call. It returns the same structure as
    the voice JSON files:
      {"style_ttl": {"data": ..., "dims": ...}, "style_dp": {...}, "metadata": {...}}

    No fallback/skip behavior: missing weights or architecture mismatches raise.
    """
    if verify_hf_sizes_flag:
        verify_hf_file_sizes([ae_ckpt, ttl_ckpt, dp_ckpt, stats])

    if os.path.isfile(config):
        cfg = load_ttl_config(config)
        ver = cfg["full_config"].get("tts_version", "?")
        print(f"[INFO] config {config} (v{ver})")
    else:
        raise FileNotFoundError(f"Missing config: {config}")

    chunk_compress_factor = cfg["chunk_compress_factor"]
    compressed_channels = cfg["compressed_channels"]
    normalizer_scale = cfg["normalizer_scale"]

    mean, std, stats_path = load_stats(device, preferred=stats)
    print(f"[INFO] stats {stats_path}")

    if not os.path.isfile(ae_ckpt):
        raise FileNotFoundError(f"Missing AE checkpoint: {ae_ckpt}")

    ae_raw = load_torch_or_safetensors(ae_ckpt, map_location="cpu")
    enc_sd = load_ae_encoder_state(ae_raw)

    sr = cfg["ae_sample_rate"]
    n_fft = cfg["ae_n_fft"]
    hop = cfg["ae_hop_length"]
    n_mels = cfg["ae_n_mels"]

    mel_spec = LinearMelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels).to(device)
    ae_encoder = LatentEncoder(cfg=cfg["ae_enc_cfg"]).to(device).eval()
    ae_encoder.load_state_dict(enc_sd, strict=True)

    wav, _ = read_wav_mono(ref_wav, target_sr=sr)
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

    if out_pt:
        out_pt = out_pt if out_pt.endswith(".pt") else out_pt + ".pt"
        os.makedirs(os.path.dirname(out_pt) or ".", exist_ok=True)
        torch.save(
            {
                "z_ref_raw": z_ref_raw.cpu(),
                "is_normalized": False,
                "metadata": {
                    "ref_wav": os.path.abspath(ref_wav),
                    "stats_path": stats_path,
                    "sr": sr,
                    "z_ref_raw_shape": list(z_ref_raw.shape),
                },
            },
            out_pt,
        )
        print(f"[OK] z_ref_raw -> {out_pt}")

    from training.t2l.models.reference_encoder import ReferenceEncoder

    ttl_path = resolve_ttl_checkpoint(ttl_ckpt)
    if not os.path.exists(ttl_path):
        raise FileNotFoundError(f"Missing TTL checkpoint: {ttl_path}")
    ref_enc = ReferenceEncoder(
        in_channels=compressed_channels,
        d_model=cfg["se_d_model"],
        hidden_dim=cfg["se_hidden_dim"],
        num_blocks=cfg["se_num_blocks"],
        num_tokens=cfg["se_n_style"],
        num_heads=cfg["se_n_heads"],
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

    from training.dp.models.dp_network import DPNetwork

    dp_path = resolve_ttl_checkpoint(dp_ckpt)
    if not os.path.exists(dp_path):
        raise FileNotFoundError(f"Missing DP checkpoint: {dp_path}")
    ckpt = load_torch_or_safetensors(dp_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        ckpt = ckpt["state_dict"]

    emb_w = ckpt.get("sentence_encoder.text_embedder.char_embedder.weight")
    dp_vocab_size = int(emb_w.shape[0]) if emb_w is not None else cfg["dp_vocab_size"]
    dp = cfg["dp"]
    dp_net = DPNetwork(
        vocab_size=dp_vocab_size,
        sentence_encoder_cfg=dp.get("sentence_encoder"),
        style_encoder_cfg=dp.get("style_encoder"),
        predictor_cfg=dp.get("predictor"),
    ).to(device).eval()
    dp_net.load_state_dict(ckpt, strict=True)

    z_tr = trim_reference_latents(z_ref_norm, "DPRefEnc")
    mask = torch.ones(1, 1, z_tr.shape[2], device=device, dtype=torch.float32)
    with torch.inference_mode():
        sdp = dp_net.ref_encoder(z_tr, mask=mask)
        n_q = dp_net.ref_encoder.num_queries
        q_dim = dp_net.ref_encoder.query_dim
        style_dp = sdp.reshape(1, n_q, q_dim).cpu().numpy().astype(np.float32)
    print(f"[OK] style_dp <- {dp_path}")

    style_ttl_stats = {
        "mean": float(style_ttl.mean()),
        "std": float(style_ttl.std()),
        "min": float(style_ttl.min()),
        "max": float(style_ttl.max()),
    }
    style_dp_stats = {
        "mean": float(style_dp.mean()),
        "std": float(style_dp.std()),
        "min": float(style_dp.min()),
        "max": float(style_dp.max()),
    }
    print(f"[INFO] style_ttl stats: {style_ttl_stats}")
    print(f"[INFO] style_dp stats: {style_dp_stats}")

    return {
        "style_ttl": {"data": style_ttl.tolist(), "dims": list(style_ttl.shape)},
        "style_dp": {"data": style_dp.tolist(), "dims": list(style_dp.shape)},
        "metadata": {
            "sr": sr,
            "ref_wav": os.path.abspath(ref_wav),
            "stats_path": stats_path,
            "z_ref_norm_shape": list(z_ref_norm.shape),
            "normalizer_scale": normalizer_scale,
            "style_ttl_stats": style_ttl_stats,
            "style_dp_stats": style_dp_stats,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Export style_ttl / style_dp JSON from a reference WAV.")
    ap.add_argument("--ref_wav", type=str, required=True)
    ap.add_argument("--out", type=str, default="voice.json")
    ap.add_argument("--out_pt", type=str, default=None)
    ap.add_argument("--ae_ckpt", type=str, default="blue_codec.safetensors")
    ap.add_argument("--stats", type=str, default="stats_multilingual.safetensors")
    ap.add_argument("--ttl_ckpt", type=str, default="vf_estimetor.safetensors")
    ap.add_argument("--dp_ckpt", type=str, default="duration_predictor_final.safetensors")
    ap.add_argument("--verify_hf_sizes", action="store_true")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--config", type=str, default="config/tts.json")
    args = ap.parse_args()

    payload = export_voice_style(
        args.ref_wav,
        config=args.config,
        ae_ckpt=args.ae_ckpt,
        ttl_ckpt=args.ttl_ckpt,
        dp_ckpt=args.dp_ckpt,
        stats=args.stats,
        device=args.device,
        out_pt=args.out_pt,
        verify_hf_sizes_flag=args.verify_hf_sizes,
    )

    out_json = args.out if args.out.endswith(".json") else args.out + ".json"
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(payload, f)
    print(f"[OK] wrote {out_json}")


if __name__ == "__main__":
    main()
