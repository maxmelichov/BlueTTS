import os
import json

def _load_dp_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        full = json.load(f)
    dp_cfg = full.get("dp")
    if isinstance(dp_cfg, dict) and dp_cfg:
        return {"_full": full, **dp_cfg}
    ttl_cfg = full.get("ttl", {}) if isinstance(full.get("ttl"), dict) else {}
    return {
        "_full": full,
        "latent_dim": ttl_cfg.get("latent_dim", 24),
        "chunk_compress_factor": ttl_cfg.get("chunk_compress_factor", 6),
        "normalizer": {"scale": ttl_cfg.get("normalizer", {}).get("scale", 1.0)},
        "style_encoder": {"style_token_layer": {"n_style": 8, "style_value_dim": 16}},
    }
