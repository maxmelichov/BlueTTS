#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
import torch
import soundfile as sf
import librosa
from data.audio_utils import ensure_sr

# Add project root
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from models.utils import MelSpectrogram, LinearMelSpectrogram, compress_latents, load_ttl_config
from models.autoencoder.latent_encoder import LatentEncoder

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
        wav_t = torch.from_numpy(wav)
        wav_t = ensure_sr(wav_t, sr, target_sr)
        wav = wav_t.squeeze().cpu().numpy()
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
    ap.add_argument("--ae_ckpt", type=str, default="checkpoints/ae/ae_latest.pt")
    ap.add_argument("--stats", type=str, default="stats_multilingual.pt", help="Preferred stats file (fallback to stats.pt)")
    ap.add_argument(
        "--ttl_ckpt",
        type=str,
        default="checkpoints/text2latent",
        help="TTL checkpoint dir or file (for style_ttl/style_keys export). Uses latest ckpt_step_*.pt if dir.",
    )
    ap.add_argument(
        "--dp_ckpt",
        type=str,
        default="checkpoints/duration_predictor/duration_predictor_final.pt",
        help="DP checkpoint to extract style_dp tokens (optional). If missing, style_dp won't be exported.",
    )
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--config", type=str, default="configs/tts.json", help="Path to tts.json config")
    args = ap.parse_args()

    device = args.device

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
    ae_ckpt = torch.load(args.ae_ckpt, map_location="cpu")
    if "encoder" not in ae_ckpt:
        raise KeyError("AE checkpoint missing key 'encoder'")

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
    ae_encoder.load_state_dict(ae_ckpt["encoder"], strict=True)

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

    # Save as JSON (voice style JSON: style_ttl + style_dp, optionally style_keys)
    json_path = args.out if args.out.endswith(".json") else args.out + ".json"
        
    # 1) style_ttl/style_keys via PyTorch ReferenceEncoder
    style_ttl = None
    style_keys = None
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
                ckpt_files.sort(key=lambda x: int(re.search(r"ckpt_step_(\d+)", x).group(1)))
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
            
            # Load state dict from TTL checkpoint
            ttl_ckpt = torch.load(ttl_ckpt_path, map_location=device)
            if "reference_encoder" in ttl_ckpt:
                reference_encoder.load_state_dict(ttl_ckpt["reference_encoder"])
            else:
                raise KeyError(f"TTL checkpoint missing 'reference_encoder' key: {ttl_ckpt_path}")
            
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
                ref_values, ref_keys = reference_encoder(z_ref_trimmed, mask=ref_mask_t)
                # ref_values: [B, num_tokens, d_model] = [1, 50, 256]
                # ref_keys: [B, num_tokens, d_model] = [1, 50, 256]
            
            style_ttl = ref_values.detach().cpu().numpy().astype(np.float32)
            style_keys = ref_keys.detach().cpu().numpy().astype(np.float32)
            print(f"[OK] Extracted style_ttl/style_keys from {ttl_ckpt_path}")
        else:
            print(f"[WARN] Missing TTL checkpoint at {ttl_ckpt_path}; skipping style_ttl/style_keys export")
    except Exception as e:
        print(f"[WARN] Failed to export style_ttl/style_keys via PyTorch ReferenceEncoder: {e}")

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
            dp = DPNetwork(
                vocab_size=cfg["dp_vocab_size"] if cfg else 37,
                style_tokens=dp_style_tokens,
                style_dim=dp_style_dim,
            ).to(device).eval()
            state = torch.load(args.dp_ckpt, map_location=device)
            # Support either raw state_dict or checkpoint dict
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            dp.load_state_dict(state)

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

    # 3) Build JSON payload compatible with both helper.py (style_ttl/style_dp)
    #    and hebrew_inference_helper.py (style_ttl/style_keys, optional style_dp).
    json_payload = {}
    
    # Removed z_ref from JSON as requested
    # json_payload["z_ref"] = {"data": z_ref_raw.detach().cpu().numpy().tolist(), "dims": list(z_ref_raw.shape)}

    if style_ttl is not None:
        json_payload["style_ttl"] = {"data": style_ttl.tolist(), "dims": list(style_ttl.shape)}
    if style_keys is not None:
        json_payload["style_keys"] = {"data": style_keys.tolist(), "dims": list(style_keys.shape)}
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
