import torch
import numpy as np
import os
import argparse

from models.utils import load_ttl_config

def main():
    parser = argparse.ArgumentParser(description="Convert .pt stats to .npz for ONNX inference")
    parser.add_argument("--pt", default="stats_multilingual.pt", help="Input torch stats file")
    parser.add_argument("--out", default="onnx_models/stats.npz", help="Output npz file")
    parser.add_argument("--config", type=str, default="configs/tts.json", help="Path to tts.json config")
    args = parser.parse_args()

    # Load config – REQUIRED so that normalizer_scale and compressed_channels
    # always come from tts.json (never silently fall back to wrong defaults).
    if not os.path.exists(args.config):
        print(f"[ERROR] Config not found: {args.config}")
        print("       The tts.json config is required. Pass --config <path> or place it at configs/tts.json")
        return

    cfg = load_ttl_config(args.config)
    normalizer_scale = cfg["normalizer_scale"]
    compressed_channels = cfg["compressed_channels"]
    print(f"[INFO] Loaded config: {args.config}")
    print(f"[INFO] normalizer_scale = {normalizer_scale}")
    print(f"[INFO] compressed_channels = {compressed_channels}")

    if not os.path.exists(args.pt):
        print(f"[ERROR] Stats file not found: {args.pt}")
        return

    print(f"Loading {args.pt}...")
    stats = torch.load(args.pt, map_location="cpu")
    
    mean = stats["mean"].numpy().astype(np.float32)
    std = stats["std"].numpy().astype(np.float32)
    
    # Ensure correct shape [1, C, 1] for broadcasting in inference
    if mean.ndim == 1:
        mean = mean.reshape(1, -1, 1)
    if std.ndim == 1:
        std = std.reshape(1, -1, 1)

    # Also handle 2D [C, 1] or [1, C] shapes
    if mean.ndim == 2:
        if mean.shape[0] == 1:
            mean = mean.reshape(1, -1, 1)
        else:
            mean = mean.reshape(1, mean.shape[0], 1)
    if std.ndim == 2:
        if std.shape[0] == 1:
            std = std.reshape(1, -1, 1)
        else:
            std = std.reshape(1, std.shape[0], 1)

    # Validate against config
    if mean.shape[1] != compressed_channels:
        print(f"[WARN] Stats channel dim {mean.shape[1]} != config compressed_channels {compressed_channels}")
    
    # Safety: clamp near-zero std to avoid division-by-zero in inference
    min_std = np.min(np.abs(std))
    near_zero = np.sum(np.abs(std) < 1e-6)
    if near_zero > 0:
        print(f"[WARN] {near_zero} channels have near-zero std (min={min_std:.2e}). Clamping to 1e-6.")
        std = np.clip(std, 1e-6, None)
    
    print(f"  Mean shape: {mean.shape}, range: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"  Std shape:  {std.shape},  range: [{std.min():.4f}, {std.max():.4f}]")
    
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)

    # Save stats along with normalizer_scale so ONNX inference can be self-contained
    np.savez(
        args.out,
        mean=mean,
        std=std,
        normalizer_scale=np.array([normalizer_scale], dtype=np.float32),
    )
    print(f"[OK] Saved {args.out}")
    
    # ---- Verification: roundtrip check ----
    print("\nVerification (roundtrip):")
    loaded = np.load(args.out)
    mean_rt = loaded["mean"]
    std_rt = loaded["std"]
    
    assert np.allclose(mean, mean_rt, atol=1e-7), "Mean roundtrip mismatch!"
    assert np.allclose(std, std_rt, atol=1e-7), "Std roundtrip mismatch!"
    
    if "normalizer_scale" in loaded:
        ns_rt = float(loaded["normalizer_scale"][0])
        print(f"  normalizer_scale: {ns_rt}")
        assert abs(ns_rt - normalizer_scale) < 1e-6, "normalizer_scale roundtrip mismatch!"
    
    # Quick sanity: simulate normalize/denormalize roundtrip
    dummy = np.random.randn(1, mean.shape[1], 10).astype(np.float32)
    normalized = ((dummy - mean) / std) * normalizer_scale
    denormalized = (normalized / normalizer_scale) * std + mean
    rt_error = np.max(np.abs(dummy - denormalized))
    print(f"  Normalization roundtrip max error: {rt_error:.2e}")
    assert rt_error < 1e-5, f"Normalization roundtrip error too large: {rt_error}"
    
    print("[OK] Stats conversion verified successfully.")

if __name__ == "__main__":
    main()
