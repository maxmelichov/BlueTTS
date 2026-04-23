"""Blue PT TTS — flat single-file PyTorch inference module (mirrors blue_onnx layout)."""

import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
import torch

from ..blue_onnx import (
    BLUE_SYNTH_MAX_CHUNK_LEN,
    TextProcessor,
    chunk_text,
    text_to_indices,
)

# Resolve training models on sys.path (imports below depend on it).
_HERE = os.path.dirname(os.path.abspath(__file__))
_TRAINING = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "training")
if _TRAINING not in sys.path:
    sys.path.insert(0, _TRAINING)

from models.text2latent.text_encoder import TextEncoder
from models.text2latent.vf_estimator import VectorFieldEstimator
from models.text2latent.dp_network import DPNetwork
from models.utils import load_ttl_config
from bluecodec import LatentDecoder1D
from bluecodec.utils import decompress_latents


@dataclass
class Style:
    ttl: Any
    dp: Optional[Any] = None


# ─── Loaders ──────────────────────────────────────────────────────────────────

def _load_sd(path: str, *candidates: str) -> dict:
    """Load ``.pt`` or flat ``.safetensors``; pick nested key or ``prefix.*`` strip (HF codec exports)."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        raw = load_file(path, device="cpu")
    else:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        other = [k for k in raw if k not in ("state_dict", "optimizer", "scheduler", "global_step")]
        if not other:
            raw = raw["state_dict"]
    if not isinstance(raw, dict):
        raise TypeError(f"{path}: expected a dict checkpoint, got {type(raw)}")
    for k in candidates:
        if k in raw and isinstance(raw[k], dict):
            return raw[k]
        p = f"{k}."
        sub = {x[len(p) :]: v for x, v in raw.items() if isinstance(x, str) and x.startswith(p)}
        if sub:
            return sub
    return raw


def load_cfgs(config_path: str) -> dict:
    if config_path and os.path.exists(config_path):
        return load_ttl_config(config_path)
    return {}


def load_stats(weights_dir: str, device: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    for name in ("stats_multilingual.pt", "stats.pt", "stats_real_data.pt", "stats_mixed.pt"):
        path = os.path.join(weights_dir, name)
        if not os.path.exists(path):
            continue
        stats = torch.load(path, map_location="cpu", weights_only=False)
        mean, std = stats["mean"], stats["std"]
        if mean.ndim == 1:
            mean = mean.view(1, -1, 1)
            std = std.view(1, -1, 1)
        return mean.to(device), std.to(device)
    return None, None


def load_voice_style(path: str, device: str = "cpu") -> Style:
    with open(path) as f:
        d = json.load(f)

    def _t(entry):
        arr = np.array(entry["data"], dtype=np.float32).reshape(entry["dims"])
        return torch.from_numpy(arr).to(device)

    ttl = _t(d["style_ttl"]) if "style_ttl" in d else None
    dp = _t(d["style_dp"]) if "style_dp" in d else None
    return Style(ttl=ttl, dp=dp)


def load_pt_models(
    weights_dir: str,
    cfg: dict,
    device: str,
    text2latent_ckpt: Optional[str] = None,
    ae_ckpt: Optional[str] = None,
    dp_ckpt: Optional[str] = None,
) -> Tuple[TextEncoder, VectorFieldEstimator, DPNetwork, LatentDecoder1D, Optional[torch.Tensor], Optional[torch.Tensor]]:
    vocab_size = cfg.get("vocab_size", 256)
    se_n_style = cfg.get("se_n_style", 50)
    latent_dim = int(cfg.get("latent_dim", 24))
    ccf = int(cfg.get("chunk_compress_factor", 6))
    compressed = latent_dim * ccf

    # Text encoder + VF + uncond params.
    u_text, u_ref = None, None
    if text2latent_ckpt:
        combined = torch.load(text2latent_ckpt, map_location="cpu", weights_only=False)
        te_sd = combined["text_encoder"]
        vf_sd = combined["vf_estimator"]
        u_text = combined.get("u_text")
        u_ref = combined.get("u_ref")
        if u_text is not None:
            u_text = u_text.to(device)
        if u_ref is not None:
            u_ref = u_ref.to(device)
    else:
        te_sd = _load_sd(os.path.join(weights_dir, "text_encoder.pt"), "text_encoder")
        vf_sd = _load_sd(os.path.join(weights_dir, "vector_estimator.pt"), "vf_estimator")

    emb_key = "text_embedder.char_embedder.weight"
    if emb_key in te_sd and te_sd[emb_key].shape[0] != vocab_size:
        vocab_size = te_sd[emb_key].shape[0]

    text_encoder = TextEncoder(
        vocab_size=vocab_size,
        d_model=cfg.get("te_d_model", 256),
        n_conv_layers=cfg.get("te_convnext_layers", 6),
        n_attn_layers=cfg.get("te_attn_n_layers", 4),
        expansion_factor=cfg.get("te_expansion_factor", 4),
        p_dropout=0.0,
    ).to(device).eval()
    text_encoder.load_state_dict(te_sd, strict=False)

    vf_estimator = VectorFieldEstimator(
        in_channels=compressed,
        out_channels=compressed,
        hidden_channels=cfg.get("vf_hidden", 512),
        text_dim=cfg.get("vf_text_dim", 256),
        style_dim=cfg.get("vf_style_dim", 256),
        num_style_tokens=se_n_style,
        num_superblocks=cfg.get("vf_n_blocks", 4),
        time_embed_dim=cfg.get("vf_time_dim", 64),
        rope_gamma=cfg.get("vf_rotary_scale", 10.0),
    ).to(device).eval()
    vf_estimator.load_state_dict(vf_sd, strict=False)

    # Duration predictor.
    dp_path = dp_ckpt or os.path.join(weights_dir, "duration_predictor.pt")
    dp_sd = _load_sd(dp_path, "state_dict")
    dp_vocab_size = vocab_size
    dp_emb_key = "sentence_encoder.text_embedder.char_embedder.weight"
    if dp_emb_key in dp_sd and dp_sd[dp_emb_key].shape[0] != dp_vocab_size:
        dp_vocab_size = dp_sd[dp_emb_key].shape[0]
    dp_model = DPNetwork(
        vocab_size=dp_vocab_size,
        latent_channels=compressed,
        style_dp=cfg.get("dp_style_tokens", 8),
        style_dim=cfg.get("dp_style_dim", 16),
    ).to(device).eval()
    dp_model.load_state_dict(dp_sd, strict=False)

    # Vocoder.
    voc_path = ae_ckpt or os.path.join(weights_dir, "vocoder.pt")
    voc_sd = _load_sd(voc_path, "decoder", "state_dict")
    vocoder = LatentDecoder1D(cfg=cfg.get("ae_dec_cfg", {})).to(device).eval()
    vocoder.load_state_dict(voc_sd, strict=False)

    return text_encoder, vf_estimator, dp_model, vocoder, u_text, u_ref


# ─── Helpers ──────────────────────────────────────────────────────────────────

@contextmanager
def timer(name: str):
    import time
    t0 = time.time()
    print(f"{name}...")
    yield
    print(f"  -> {name} completed in {time.time() - t0:.2f} sec")


# ─── BlueTTS ──────────────────────────────────────────────────────────────────

class BlueTTS:
    def __init__(
        self,
        weights_dir: str,
        config_path: str = "tts.json",
        style_json: Optional[str] = None,
        steps: int = 5,
        cfg_scale: float = 3.0,
        speed: float = 1.0,
        seed: int = 42,
        device: str = "cpu",
        chunk_len: int = BLUE_SYNTH_MAX_CHUNK_LEN,
        silence_sec: float = 0.15,
        fade_duration: float = 0.02,
        text2latent_ckpt: Optional[str] = None,
        ae_ckpt: Optional[str] = None,
        dp_ckpt: Optional[str] = None,
        renikud_path: Optional[str] = None,
    ):
        self.steps = steps
        self.cfg_scale = cfg_scale
        self.speed = speed
        self.seed = seed
        self.device = device
        self.chunk_len = min(max(1, chunk_len), BLUE_SYNTH_MAX_CHUNK_LEN)
        self.silence_sec = silence_sec
        self.fade_duration = fade_duration

        if renikud_path is None:
            for cand in ("model.onnx", os.path.join(weights_dir, "model.onnx")):
                if os.path.exists(cand):
                    renikud_path = cand
                    break

        cfgs = load_cfgs(config_path)
        self.normalizer_scale = float(cfgs.get("normalizer_scale", 1.0))
        self.latent_dim = int(cfgs.get("latent_dim", 24))
        self.chunk_compress_factor = int(cfgs.get("chunk_compress_factor", 6))
        self.hop_length = int(cfgs.get("ae_hop_length", 512))
        self.sample_rate = int(cfgs.get("ae_sample_rate", 44100))
        self.compressed_channels = self.latent_dim * self.chunk_compress_factor

        self._te, self._vf, self._dp, self._vocoder, self._u_text, self._u_ref = load_pt_models(
            weights_dir, cfgs, device, text2latent_ckpt, ae_ckpt, dp_ckpt
        )
        self.mean, self.std = load_stats(weights_dir, device)
        self._style = load_voice_style(style_json, device) if style_json else None
        self._text_proc = TextProcessor(renikud_path)

    # ── Public API ──────────────────────────────────────────────────────────

    def synthesize(self, text: str, lang: str = "he", cfg_scale: Optional[float] = None) -> Tuple[np.ndarray, int]:
        return self.create(self._text_proc.phonemize(text, lang=lang), cfg_scale=cfg_scale)

    def create(self, phonemes: str, cfg_scale: Optional[float] = None) -> Tuple[np.ndarray, int]:
        cfg = self.cfg_scale if cfg_scale is None else float(cfg_scale)
        chunks = chunk_text(phonemes, self.chunk_len)
        silence = np.zeros(int(self.silence_sec * self.sample_rate), dtype=np.float32)
        parts: List[np.ndarray] = []
        for i, chunk in enumerate(chunks):
            parts.append(self._infer_chunk(chunk, cfg_scale=cfg))
            if i < len(chunks) - 1:
                parts.append(silence)
        wav = np.concatenate(parts) if parts else np.array([], dtype=np.float32)
        return wav, self.sample_rate

    # ── Internals ───────────────────────────────────────────────────────────

    @torch.inference_mode()
    def _infer_chunk(self, phonemes: str, cfg_scale: float) -> np.ndarray:
        if self._style is None or self._style.ttl is None:
            raise ValueError("style_json is required (must contain style_ttl).")
        if self.mean is None or self.std is None:
            raise ValueError("Latent stats not loaded.")

        dev = self.device
        style_ttl = self._style.ttl
        style_dp = self._style.dp

        text_ids = torch.tensor([text_to_indices(phonemes)], dtype=torch.long, device=dev)
        text_mask = torch.ones(1, 1, text_ids.shape[1], device=dev)

        # Duration.
        T_lat = self._predict_duration(text_ids, text_mask, style_dp)

        # Text encoding.
        h_text = self._te(text_ids, style_ttl, text_mask=text_mask)

        # Flow matching (with CFG).
        x = self._flow_matching(h_text, style_ttl, text_mask, T_lat, cfg_scale)

        # Decode.
        return self._decode(x)

    def _predict_duration(self, text_ids, text_mask, style_dp) -> int:
        T_lat: Optional[int] = None
        if style_dp is not None:
            log_dur = self._dp(text_ids, text_mask=text_mask, style_dp=style_dp, return_log=True)
            val = torch.exp(log_dur) / max(self.speed, 1e-6)
            T_lat = int(val.round().item())
        if T_lat is None:
            T_lat = int(text_ids.shape[1] * 1.3)
        txt_len = int(text_ids.shape[1])
        T_cap = max(20, min(txt_len * 3 + 20, 600))
        return max(10, min(max(T_lat, 10), T_cap, 800))

    def _flow_matching(self, h_text, style_ttl, text_mask, T_lat, cfg_scale) -> torch.Tensor:
        dev = self.device
        rng = torch.Generator(device=dev)
        rng.manual_seed(self.seed)
        x = torch.randn(1, self.compressed_channels, T_lat, device=dev, generator=rng)
        latent_mask = torch.ones(1, 1, T_lat, device=dev)
        dt = 1.0 / self.steps

        use_cfg = cfg_scale != 1.0 and self._u_text is not None and self._u_ref is not None
        if use_cfg:
            h_text_null = self._u_text.expand(1, -1, h_text.shape[2])
            h_ref_null = self._u_ref.expand(1, -1, -1)
            u_mask = torch.ones(1, 1, h_text.shape[2], device=dev)

        for i in range(self.steps):
            t = torch.full((1,), i / self.steps, device=dev)
            x_in = x * latent_mask

            v_cond = self._vf(
                noisy_latent=x_in, text_emb=h_text, style_ttl=style_ttl,
                latent_mask=latent_mask, text_mask=text_mask, current_step=t,
            )
            if use_cfg:
                v_uncond = self._vf(
                    noisy_latent=x_in, text_emb=h_text_null, style_ttl=h_ref_null,
                    latent_mask=latent_mask, text_mask=u_mask, current_step=t,
                )
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = v_cond

            x = (x + v * dt) * latent_mask
        return x

    def _decode(self, x: torch.Tensor) -> np.ndarray:
        ns = float(self.normalizer_scale)
        z_pred = (x / ns) * self.std + self.mean if ns not in (0.0, 1.0) else x * self.std + self.mean
        z_pred = decompress_latents(z_pred, factor=self.chunk_compress_factor, target_channels=self.latent_dim)
        wav = self._vocoder(z_pred)

        frame_len = self.hop_length * self.chunk_compress_factor
        if wav.shape[-1] > 2 * frame_len:
            wav = wav[..., frame_len:-frame_len]

        wav_np = wav.squeeze().cpu().numpy().astype(np.float32)

        fs = int(self.fade_duration * self.sample_rate)
        if fs and len(wav_np) >= 2 * fs:
            wav_np = wav_np.copy()
            wav_np[:fs] *= np.linspace(0.0, 1.0, fs, dtype=np.float32)
            wav_np[-fs:] *= np.linspace(1.0, 0.0, fs, dtype=np.float32)
        return wav_np
