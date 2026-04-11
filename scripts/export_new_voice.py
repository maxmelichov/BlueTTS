#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export voice style JSON from a reference WAV using weights from Hugging Face
``notmax123/Blue`` (see https://huggingface.co/notmax123/Blue).

Download weights into the repo (from the **repository root**, not inside ``pt_weights``)::

    hf download notmax123/Blue --local-dir ./pt_weights

Run the script from the **same repository root** so ``scripts/export_new_voice.py``
resolves. Pass weight paths if they are not in the current directory, e.g.::

    PYTHONPATH=training uv run python scripts/export_new_voice.py \\
        --ref_wav /path/to/ref.wav \\
        --out voice.json \\
        --config config/tts.json \\
        --ae_ckpt pt_weights/blue_codec.safetensors \\
        --ttl_ckpt pt_weights/vf_estimator.safetensors \\
        --dp_ckpt pt_weights/duration_predictor.safetensors \\
        --stats pt_weights/stats_multilingual.pt \\
        --verify_hf_sizes

(If you ``cd pt_weights`` first, you must use absolute paths to the script and
``PYTHONPATH`` must still point at this repo's ``training`` directory.)
"""

import os
import sys
import argparse
import numpy as np
import torch
import soundfile as sf
import librosa

# Repo layout: ``training/`` holds ``models.*`` (same as scripts/README.md).
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_TRAINING = os.path.join(_ROOT, "training")
if _TRAINING not in sys.path:
    sys.path.insert(0, _TRAINING)

from bluecodec.autoencoder.latent_encoder import LatentEncoder
from models.utils import LinearMelSpectrogram, compress_latents, load_ttl_config

# Hugging Face model repo (byte sizes from repo tree API, notmax123/Blue main).
HF_REPO_ID = "notmax123/Blue"
HF_WEIGHT_SIZES = {
    "blue_codec.safetensors": 245_114_104,
    "duration_predictor.safetensors": 2_040_512,
    "stats_multilingual.pt": 3_133,
    "vf_estimator.safetensors": 179_313_224,
}


def load_torch_or_safetensors(path: str, map_location="cpu"):
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path, device=str(map_location))
    return torch.load(path, map_location=map_location, weights_only=False)


def state_dict_with_prefix(flat_sd: dict, prefix: str) -> dict:
    """Extract ``prefix`` + '.' keys into a sub state_dict (strip prefix)."""
    p = prefix + "."
    out = {k[len(p) :]: v for k, v in flat_sd.items() if k.startswith(p)}
    if not out:
        raise KeyError(f"No keys with prefix {p!r} in checkpoint")
    return out


def load_ae_encoder_state(raw: dict) -> dict:
    if "encoder" in raw and isinstance(raw["encoder"], dict):
        return raw["encoder"]
    enc_keys = [k for k in raw if k.startswith("encoder.") or k == "encoder"]
    if enc_keys:
        return state_dict_with_prefix(raw, "encoder")
    return raw


def verify_hf_file_sizes(paths: list[str]) -> None:
    for p in paths:
        if not p or not os.path.isfile(p):
            continue
        base = os.path.basename(p)
        expected = HF_WEIGHT_SIZES.get(base)
        if expected is None:
            continue
        got = os.path.getsize(p)
        if got != expected:
            raise ValueError(
                f"Size mismatch for {base}: local file is {got} bytes, "
                f"expected {expected} for {HF_REPO_ID} (re-download if needed)."
            )
        print(f"[OK] Size check {base}: {got} bytes")


def load_stats(device: str, preferred="stats_multilingual.pt", fallback="stats.pt"):
    stats_path = preferred if os.path.exists(preferred) else fallback
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Missing stats file: tried {preferred} and {fallback}")
    stats = torch.load(stats_path, map_location=device)
    mean = stats["mean"].to(device).view(1, -1, 1)
    std = stats["std"].to(device).view(1, -1, 1)
    return mean, std, stats_path


def read_wav_mono(path: str, target_sr: int = 44100):
    wav, sr = sf.read(path)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)

    if sr != target_sr:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return wav, sr


import json
import re

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_wav", type=str, required=True, help="Reference WAV path (44.1kHz mono/stereo)")
    ap.add_argument("--out", type=str, default="yoav.json", help="Output style .json file (default: male1.json)")
    ap.add_argument(
        "--out_pt",
        type=str,
        default=None,
        help="Optional output .pt path for z_ref_raw (compressed AE latents before normalization). "
             "Compatible with inference_tts.py --z_ref.",
    )
    ap.add_argument(
        "--ae_ckpt",
        type=str,
        default="blue_codec.safetensors",
        help="Audio codec checkpoint (.safetensors or .pt with 'encoder' key). Default matches notmax123/Blue.",
    )
    ap.add_argument("--stats", type=str, default="stats_multilingual.pt", help="Preferred stats file (fallback to stats.pt)")
    ap.add_argument(
        "--ttl_ckpt",
        type=str,
        default="vf_estimator.safetensors",
        help="Text2latent bundle: vf_estimator.safetensors from HF, or a dir with ckpt_step_*.pt for training ckpts.",
    )
    ap.add_argument(
        "--dp_ckpt",
        type=str,
        default="duration_predictor.safetensors",
        help="Duration predictor (.safetensors or .pt). Default matches notmax123/Blue.",
    )
    ap.add_argument(
        "--verify_hf_sizes",
        action="store_true",
        help=f"Assert local weight files match expected byte sizes for {HF_REPO_ID} (when filenames match).",
    )
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--config", type=str, default="configs/tts.json", help="Path to tts.json config")
    args = ap.parse_args()

    device = args.device

    if args.verify_hf_sizes:
        verify_hf_file_sizes([args.ae_ckpt, args.ttl_ckpt, args.dp_ckpt, args.stats])

    # Load config
    cfg = None
    if os.path.exists(args.config):
        cfg = load_ttl_config(args.config)
        print(f"[INFO] Loaded config: {args.config} (v{cfg['full_config'].get('tts_version', '?')})")
    else:
        print(f"[WARN] Config {args.config} not found, using hardcoded defaults.")

    # Config-derived dimensions
    chunk_compress_factor = cfg["chunk_compress_factor"] if cfg else 6
    compressed_channels = cfg["compressed_channels"] if cfg else 144
    normalizer_scale = cfg["normalizer_scale"] if cfg else 1.0

    mean, std, stats_path = load_stats(device, preferred=args.stats)
    print(f"[INFO] Loaded stats: {stats_path}")

    if not os.path.exists(args.ae_ckpt):
        raise FileNotFoundError(f"Missing AE checkpoint: {args.ae_ckpt}")
    ae_raw = load_torch_or_safetensors(args.ae_ckpt, map_location="cpu")
    enc_sd = load_ae_encoder_state(ae_raw)

    # AE encoder config from tts.json
    ae_enc_cfg = cfg["ae_enc_cfg"] if cfg else {
        "ksz": 7, "hdim": 512, "intermediate_dim": 2048,
        "dilation_lst": [1] * 10, "odim": 24, "idim": 228,
    }
    ae_sample_rate = cfg["ae_sample_rate"] if cfg else 44100
    ae_n_fft = cfg["ae_n_fft"] if cfg else 2048
    ae_hop_length = cfg["ae_hop_length"] if cfg else 512
    ae_n_mels = cfg["ae_n_mels"] if cfg else 228

    mel_spec = LinearMelSpectrogram(sample_rate=ae_sample_rate, n_fft=ae_n_fft, hop_length=ae_hop_length, n_mels=ae_n_mels).to(device)
    ae_encoder = LatentEncoder(cfg=ae_enc_cfg).to(device).eval()
    ae_encoder.load_state_dict(enc_sd, strict=True)

    wav, _ = read_wav_mono(args.ref_wav, target_sr=ae_sample_rate)
    wav_t = torch.from_numpy(wav).to(device)[None, None, :]  # [1,1,T]

    with torch.inference_mode():
        mel = mel_spec(wav_t.squeeze(1))             # [1, n_mels, Tm]
        
        # Crop to multiple of chunk_compress_factor for compression
        Tm = mel.shape[-1]
        Tm_aligned = (Tm // chunk_compress_factor) * chunk_compress_factor
        if Tm_aligned != Tm:
            mel = mel[..., :Tm_aligned]

        z = ae_encoder(mel)                          # [1, latent_dim, Tm_aligned]
        zc = compress_latents(z, factor=chunk_compress_factor)  # [1, compressed_channels, Tm_aligned/ccf]
        
        z_ref_raw = zc.clone()

        # Normalization (matches training: z_1 = ((z_enc - mean) / std) * normalizer_scale)
        z_ref_norm = ((zc - mean) / std) * normalizer_scale

        # Sanity Checks
        assert z_ref_norm.ndim == 3 and z_ref_norm.shape[1] == compressed_channels, \
            f"Bad zc shape: {z_ref_norm.shape}, expected C={compressed_channels}"
        assert mean.shape[1] == compressed_channels and std.shape[1] == compressed_channels, \
            f"Bad stats shape: mean {mean.shape}, std {std.shape}"
        assert torch.isfinite(z_ref_norm).all(), "zc has NaN/Inf (stats mismatch or bad audio)"

    # Optional: Save z_ref_raw to .pt for inference_tts.py --z_ref
    if args.out_pt:
        out_pt = args.out_pt if args.out_pt.endswith(".pt") else (args.out_pt + ".pt")
        os.makedirs(os.path.dirname(out_pt) or ".", exist_ok=True)
        torch.save(
            {
                "z_ref_raw": z_ref_raw.detach().cpu(),
                "is_normalized": False,
                "metadata": {
                    "ref_wav": os.path.abspath(args.ref_wav),
                    "stats_path": stats_path,
                    "sr": ae_sample_rate,
                    "z_ref_raw_shape": list(z_ref_raw.shape),
                },
            },
            out_pt,
        )
        print(f"[OK] Saved z_ref_raw: {out_pt}")

    # Save as JSON (voice style JSON: style_ttl + style_dp; ONNX uses style_ttl only
    # and sets ref_keys = style_ttl when style_keys is omitted.)
    json_path = args.out if args.out.endswith(".json") else args.out + ".json"

    # 1) style_ttl via PyTorch ReferenceEncoder
    style_ttl = None
    try:
        from models.text2latent.reference_encoder import ReferenceEncoder
        
        # Find TTL checkpoint
        ttl_ckpt_path = args.ttl_ckpt
        if os.path.isdir(ttl_ckpt_path):
            # Find latest ckpt_step_*.pt in the directory
            import glob
            ckpt_files = glob.glob(os.path.join(ttl_ckpt_path, "ckpt_step_*.pt"))
            if ckpt_files:
                # Sort by step number and get the latest
                def _ckpt_step(path: str) -> int:
                    m = re.search(r"ckpt_step_(\d+)", path)
                    return int(m.group(1)) if m else -1

                ckpt_files.sort(key=_ckpt_step)
                ttl_ckpt_path = ckpt_files[-1]
            else:
                raise FileNotFoundError(f"No ckpt_step_*.pt files found in {ttl_ckpt_path}")
        
        if os.path.exists(ttl_ckpt_path):
            # Load ReferenceEncoder from config
            reference_encoder = ReferenceEncoder(
                in_channels=compressed_channels,
                d_model=cfg["se_d_model"] if cfg else 256,
                hidden_dim=cfg["se_hidden_dim"] if cfg else 1024,
                num_blocks=cfg["se_num_blocks"] if cfg else 6,
                num_tokens=cfg["se_n_style"] if cfg else 50,
                num_heads=cfg["se_n_heads"] if cfg else 2,
            ).to(device).eval()

            ttl_ckpt = load_torch_or_safetensors(ttl_ckpt_path, map_location=device)
            if "reference_encoder" in ttl_ckpt:
                reference_encoder.load_state_dict(ttl_ckpt["reference_encoder"], strict=True)
            elif any(k.startswith("reference_encoder.") for k in ttl_ckpt):
                reference_encoder.load_state_dict(
                    state_dict_with_prefix(ttl_ckpt, "reference_encoder"), strict=True
                )
            else:
                raise KeyError(
                    f"TTL checkpoint missing 'reference_encoder' or 'reference_encoder.*' keys: {ttl_ckpt_path}"
                )
            
            # Trim tail + cap to match build_reference_runtime in inference_tts.py
            z_ref_trimmed = z_ref_norm.clone()
            T_ref = z_ref_trimmed.shape[2]
            tail_trim = max(2, int(T_ref * 0.05))
            T_trimmed = max(1, T_ref - tail_trim)
            if T_trimmed < T_ref:
                print(f"[INFO] RefEnc: Trimming {T_ref - T_trimmed} tail frames ({T_ref} -> {T_trimmed})")
                z_ref_trimmed = z_ref_trimmed[:, :, :T_trimmed]
            target_frames = 150
            if z_ref_trimmed.shape[2] > target_frames:
                print(f"[INFO] RefEnc: Capping reference from {z_ref_trimmed.shape[2]} to {target_frames} frames")
                z_ref_trimmed = z_ref_trimmed[:, :, :target_frames]

            # Run inference
            ref_mask_t = torch.ones(1, 1, z_ref_trimmed.shape[2], device=device, dtype=torch.float32)
            with torch.inference_mode():
                ref_values = reference_encoder(z_ref_trimmed, mask=ref_mask_t)

            style_ttl = ref_values.detach().cpu().numpy().astype(np.float32)
            print(f"[OK] Extracted style_ttl from {ttl_ckpt_path}")
        else:
            print(f"[WARN] Missing TTL checkpoint at {ttl_ckpt_path}; skipping style_ttl export")
    except Exception as e:
        print(f"[WARN] Failed to export style_ttl via PyTorch ReferenceEncoder: {e}")

    # 2) style_dp tokens via DP checkpoint (optional)
    #    IMPORTANT: Apply the same preprocessing as inference_tts.py's
    #    build_reference_runtime (trim tail 5%, cap to target_frames) so the
    #    style_dp tokens match what the DP ref_encoder sees at inference time.
    style_dp = None
    try:
        if args.dp_ckpt and os.path.exists(args.dp_ckpt):
            from models.text2latent.dp_network import DPNetwork

            dp_style_tokens = cfg["dp_style_tokens"] if cfg else 8
            dp_style_dim = cfg["dp_style_dim"] if cfg else 16
            state = load_torch_or_safetensors(args.dp_ckpt, map_location=device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            dp_emb_key = "sentence_encoder.text_embedder.char_embedder.weight"
            dp_vocab_size = cfg["dp_vocab_size"] if cfg else 37
            if isinstance(state, dict) and dp_emb_key in state:
                dp_vocab_size = int(state[dp_emb_key].shape[0])
            dp = DPNetwork(
                vocab_size=dp_vocab_size,
                style_tokens=dp_style_tokens,
                style_dim=dp_style_dim,
            ).to(device).eval()
            dp.load_state_dict(state, strict=True)

            # Apply same trimming as build_reference_runtime in inference_tts.py
            z_dp = z_ref_norm.clone()
            T_dp = z_dp.shape[2]
            tail_trim = max(2, int(T_dp * 0.05))
            T_trimmed = max(1, T_dp - tail_trim)
            if T_trimmed < T_dp:
                print(f"[INFO] DP: Trimming {T_dp - T_trimmed} tail frames ({T_dp} -> {T_trimmed})")
                z_dp = z_dp[:, :, :T_trimmed]
            target_frames = 150
            if z_dp.shape[2] > target_frames:
                print(f"[INFO] DP: Capping reference from {z_dp.shape[2]} to {target_frames} frames")
                z_dp = z_dp[:, :, :target_frames]

            ref_mask_t = torch.ones(1, 1, z_dp.shape[2], device=device, dtype=torch.float32)
            with torch.inference_mode():
                # DPReferenceEncoder returns [B, n*d] where n*d = style_tokens * style_dim
                # dp is a DPNetwork(TTSDurationModel), so access ref_encoder directly
                style_dp_flat = dp.ref_encoder(z_dp, mask=ref_mask_t)  # [B, n*d]
                style_dp_t = style_dp_flat.reshape(style_dp_flat.shape[0], dp_style_tokens, dp_style_dim)
            style_dp = style_dp_t.detach().cpu().numpy().astype(np.float32)
            print(f"[OK] Extracted style_dp tokens from {args.dp_ckpt}")
        else:
            print(f"[INFO] DP checkpoint missing or not provided; skipping style_dp export: {args.dp_ckpt}")
    except Exception as e:
        print(f"[WARN] Failed to export style_dp tokens: {e}")

    # 3) Build JSON payload (style_ttl + optional style_dp; no style_keys — runtime uses style_ttl for keys too.)
    json_payload = {}

    if style_ttl is not None:
        json_payload["style_ttl"] = {"data": style_ttl.tolist(), "dims": list(style_ttl.shape)}
    if style_dp is not None:
        json_payload["style_dp"] = {"data": style_dp.tolist(), "dims": list(style_dp.shape)}

    # Always include metadata and (optionally) the normalized z_ref for debugging/fallback DP
    json_payload["metadata"] = {
        "sr": ae_sample_rate,
        "ref_wav": os.path.abspath(args.ref_wav),
        "stats_path": stats_path,
        "z_ref_norm_shape": list(z_ref_norm.shape),
    }
    
    with open(json_path, 'w') as f:
        json.dump(json_payload, f)
    print(f"[OK] Saved {json_path}")



if __name__ == "__main__":
    main()
