"""Blue PT TTS — PyTorch inference module mirroring ``blue_onnx`` layout."""

import json
import os
import re
import sys
import time
from contextlib import contextmanager
from typing import Optional, Tuple, Union

import numpy as np
import torch

from ..blue_onnx import (
    AVAILABLE_LANGS,
    DEFAULT_MIXED_PACE_BLEND,
    DEFAULT_PACE_BLEND,
    DEFAULT_SPEED,
    DURATION_PACE_DPT_REF,
    TextProcessor,
    UnicodeProcessor,
    blend_duration_pace,
    chunk_text,
    load_text_processor as _load_text_processor_onnx,
    strip_lang_tags_from_phoneme_string,
    trim_chunk_wav,
)

from training.t2l.models.text_encoder import TextEncoder  # noqa: E402
from training.t2l.models.vf_estimator import VectorFieldEstimator  # noqa: E402
from training.t2l.models.reference_encoder import ReferenceEncoder  # noqa: E402
from training.dp.models.dp_network import DPNetwork  # noqa: E402
from training.t2l.sampling import build_reference_only  # noqa: E402
from training.data.audio_utils import ensure_sr  # noqa: E402
from training.utils import LinearMelSpectrogram, compress_latents, load_ttl_config  # noqa: E402
from bluecodec import LatentDecoder1D, LatentEncoder  # noqa: E402
from bluecodec.utils import decompress_latents  # noqa: E402

_INLINE_LANG_PAIR = re.compile(r"<(\w+)>(.*?)(?:</\1>|<\1>)", re.DOTALL)


class Style:
    def __init__(self, style_ttl: torch.Tensor, style_dp: Optional[torch.Tensor]):
        self.ttl = style_ttl
        self.dp = style_dp


class TextToSpeech:
    def __init__(
        self,
        cfgs: dict,
        text_processor: UnicodeProcessor,
        dp_model: DPNetwork,
        text_encoder: TextEncoder,
        vector_estimator: VectorFieldEstimator,
        vocoder: LatentDecoder1D,
        device: str = "cpu",
        g2p: Optional[TextProcessor] = None,
        u_text: Optional[torch.Tensor] = None,
        u_ref: Optional[torch.Tensor] = None,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        seed: int = 42,
        dp_ort=None,
        vocoder_ort=None,
        denorm_before_vocoder: bool = False,
    ):
        self.cfgs = cfgs
        self.text_processor = text_processor
        self.g2p = g2p
        self.dp_model = dp_model
        self.text_encoder = text_encoder
        self.vector_estimator = vector_estimator
        self.vocoder = vocoder
        self.device = device
        self.seed = seed
        self._dp_ort = dp_ort
        self._vocoder_ort = vocoder_ort
        self.denorm_before_vocoder = bool(denorm_before_vocoder)

        self.sample_rate = int(cfgs.get("ae_sample_rate", cfgs.get("ae", {}).get("sample_rate", 44100)))
        self.base_chunk_size = int(cfgs.get("ae_hop_length", cfgs.get("ae", {}).get("base_chunk_size", 512)))
        self.chunk_compress_factor = int(
            cfgs.get("chunk_compress_factor", cfgs.get("ttl", {}).get("chunk_compress_factor", 6))
        )
        self.ldim = int(cfgs.get("latent_dim", cfgs.get("ttl", {}).get("latent_dim", 24)))
        self.normalizer_scale = float(cfgs.get("normalizer_scale", 1.0))
        self.compressed_channels = self.ldim * self.chunk_compress_factor

        self._u_text = u_text
        self._u_ref = u_ref
        self.mean = mean
        self.std = std

    # ── Latent shape + noise ────────────────────────────────────────────────

    def sample_noisy_latent(
        self, duration: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = duration.shape[0]
        wav_lengths = (duration * self.sample_rate).to(torch.long)
        wav_len_max = int(wav_lengths.max().item())
        chunk_size = self.base_chunk_size * self.chunk_compress_factor
        latent_len = (wav_len_max + chunk_size - 1) // chunk_size
        latent_dim = self.compressed_channels
        gen = torch.Generator(device=self.device).manual_seed(self.seed)
        noisy = torch.randn(bsz, latent_dim, latent_len, generator=gen, device=self.device)
        latent_lengths = (wav_lengths + chunk_size - 1) // chunk_size
        latent_mask = _length_to_mask(latent_lengths, max_len=latent_len).to(self.device)
        return noisy * latent_mask, latent_mask

    # ── Core inference ──────────────────────────────────────────────────────

    @torch.inference_mode()
    def _infer(
        self,
        text_list: list[str],
        lang_list: list[str],
        style: Style,
        total_step: int,
        speed: float = DEFAULT_SPEED,
        cfg_scale: float = 3.0,
        pace_blend: float = 0.0,
        pace_dpt_ref: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert (
            len(text_list) == style.ttl.shape[0]
        ), "Number of texts must match number of style vectors"
        bsz = len(text_list)

        text_ids_np, text_mask_np = self.text_processor(text_list, lang_list)
        text_ids = torch.from_numpy(text_ids_np).long().to(self.device)
        text_mask = torch.from_numpy(text_mask_np).float().to(self.device)

        # Duration predictor → seconds.
        # Prefer BlueV3-onnx DP: Hub PT DP is a different (broken/mismatched) checkpoint.
        if style.dp is None:
            raise ValueError("Style must contain style_dp for the PT duration predictor.")
        if self._dp_ort is not None:
            dur_np = np.asarray(
                self._dp_ort.run(
                    None,
                    {
                        "text_ids": text_ids_np.astype(np.int64),
                        "style_dp": style.dp.detach().cpu().numpy().astype(np.float32),
                        "text_mask": text_mask_np.astype(np.float32),
                    },
                )[0],
                dtype=np.float32,
            ).reshape(-1)
            dur = torch.from_numpy(dur_np).to(self.device)
        else:
            dur = self.dp_model(
                text_ids, text_mask=text_mask, style_dp=style.dp, return_log=False
            ).view(bsz).float()
            dur_np = dur.detach().cpu().numpy()
        tm_np = text_mask.detach().cpu().numpy()
        ref = float(pace_dpt_ref) if pace_dpt_ref is not None else DURATION_PACE_DPT_REF
        dur_np = blend_duration_pace(dur_np, tm_np, pace_blend, ref)
        dur = torch.from_numpy(dur_np).to(self.device) / max(float(speed), 1e-6)
        dur_out = dur.detach().cpu().numpy().astype(np.float32)

        # Text encoding.
        h_text = self.text_encoder(text_ids, style.ttl, text_mask=text_mask)

        # Init noisy latent + ONNX-matched VF sampling (internal CFG + Euler).
        xt, latent_mask = self.sample_noisy_latent(dur)
        self.vector_estimator.cfg_scale = float(cfg_scale)
        total_step_t = torch.full((bsz,), float(total_step), device=self.device)

        for step in range(total_step):
            t = torch.full((bsz,), float(step), device=self.device)
            xt = self.vector_estimator(
                noisy_latent=xt * latent_mask,
                text_emb=h_text,
                style_ttl=style.ttl,
                latent_mask=latent_mask,
                text_mask=text_mask,
                current_step=t,
                total_step=total_step_t,
                return_velocity=False,
            )

        wav = self._decode(xt, dur_out)
        return wav, dur_out

    def _decode(self, x: torch.Tensor, durations: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Latent stats (mean/std) not loaded.")
        ns = float(self.normalizer_scale) or 1.0
        bsz = int(x.shape[0])
        durations = np.asarray(durations, dtype=np.float32).reshape(-1)

        # Prefer BlueV3-onnx vocoder: Hub PT package has no matching AE weights.
        # No last-packet drop — feed the full denoised latent; trim waveform by DP duration.
        if self._vocoder_ort is not None:
            xt = x.detach().cpu().numpy().astype(np.float32)
            mean = self.mean.detach().cpu().numpy().astype(np.float32)
            std = self.std.detach().cpu().numpy().astype(np.float32)
            if mean.ndim == 1:
                mean = mean.reshape(1, -1, 1)
                std = std.reshape(1, -1, 1)
            if self.denorm_before_vocoder:
                xt = (xt / ns) * std + mean
            wav, *_ = self._vocoder_ort.run(None, {"latent": xt})
            if wav.ndim == 3 and wav.shape[1] == 1:
                wav = wav[:, 0, :]
            wav = np.asarray(wav, dtype=np.float32)
        else:
            z = (x / ns) * self.std + self.mean
            z = decompress_latents(z, factor=self.chunk_compress_factor, target_channels=self.ldim)
            wav_t = self.vocoder(z)
            if wav_t.dim() == 3 and wav_t.shape[1] == 1:
                wav_t = wav_t.squeeze(1)
            wav = wav_t.detach().cpu().numpy().astype(np.float32)

        trimmed = [
            trim_chunk_wav(wav[b], self.sample_rate, float(durations[b])).reshape(-1)
            for b in range(bsz)
        ]
        t_max = max((t.shape[0] for t in trimmed), default=0)
        out = np.zeros((bsz, t_max), dtype=np.float32)
        for b, t in enumerate(trimmed):
            out[b, : t.shape[0]] = t
        return out

    # ── Public API ──────────────────────────────────────────────────────────

    def __call__(
        self,
        text: Union[str, list[str]],
        lang: Union[str, list[str]],
        style: Style,
        total_step: int,
        speed: float = DEFAULT_SPEED,
        cfg_scale: float = 3.0,
        silence_duration: float = 0.3,
        text_is_phonemes: bool = False,
        pace_blend: Optional[float] = None,
        pace_dpt_ref: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Synthesize speech (see ``blue_onnx.TextToSpeech.__call__``).

        Set ``text_is_phonemes=True`` when ``text`` already contains phonemes to skip
        G2P while keeping the normal tokenizer/chunking path.

        See :meth:`src.blue_onnx.TextToSpeech.__call__` for ``pace_blend`` and
        ``pace_dpt_ref`` (more consistent ``speed`` across languages).
        Each chunk is already trimmed to predicted duration inside ``_infer``.
        """
        phonemize = not text_is_phonemes
        if isinstance(text, list):
            has_inline_lang = any(_INLINE_LANG_PAIR.search(t) is not None for t in text)
        else:
            has_inline_lang = _INLINE_LANG_PAIR.search(text) is not None
        pace_blend_eff = (
            float(pace_blend)
            if pace_blend is not None
            else (DEFAULT_MIXED_PACE_BLEND if has_inline_lang else DEFAULT_PACE_BLEND)
        )
        if isinstance(text, list):
            assert isinstance(lang, list) and len(text) == len(lang), (
                "Batch mode requires `lang` to be a list of the same length as `text`."
            )
            if phonemize and self.g2p is not None:
                text = [self.g2p.phonemize(t, lang=l) for t, l in zip(text, lang)]
            text = [strip_lang_tags_from_phoneme_string(t) for t in text]
            return self._infer(
                text,
                lang,
                style,
                total_step,
                speed,
                cfg_scale,
                pace_blend=pace_blend_eff,
                pace_dpt_ref=pace_dpt_ref,
            )

        assert isinstance(lang, str), "Single-text mode requires `lang` to be a str."
        assert (
            style.ttl.shape[0] == 1
        ), "Single speaker text to speech only supports single style"
        if phonemize and self.g2p is not None:
            text = self.g2p.phonemize(text, lang=lang)
        text = strip_lang_tags_from_phoneme_string(text)
        max_len = 120 if lang == "ko" else 300
        text_list = chunk_text(text, max_len=max_len)

        wav_cat = None
        dur_cat = None
        for chunk in text_list:
            wav, dur = self._infer(
                [chunk],
                [lang],
                style,
                total_step,
                speed,
                cfg_scale,
                pace_blend=pace_blend_eff,
                pace_dpt_ref=pace_dpt_ref,
            )
            if wav_cat is None:
                wav_cat = wav
                dur_cat = dur
            else:
                silence = np.zeros(
                    (1, int(silence_duration * self.sample_rate)), dtype=np.float32
                )
                if wav.ndim == 1:
                    wav = wav[None]
                if wav_cat.ndim == 1:
                    wav_cat = wav_cat[None]
                wav_cat = np.concatenate([wav_cat, silence, wav], axis=1)
                dur_cat = dur_cat + dur + silence_duration
        return wav_cat, dur_cat


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _length_to_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    max_len = int(max_len or lengths.max().item())
    ids = torch.arange(0, max_len, device=lengths.device)
    mask = (ids.unsqueeze(0) < lengths.unsqueeze(1)).float()
    return mask.view(-1, 1, max_len)


@contextmanager
def timer(name: str):
    t0 = time.time()
    print(f"{name}...")
    yield
    print(f"  -> {name} completed in {time.time() - t0:.2f} sec")


# ─── Loaders ──────────────────────────────────────────────────────────────────


def _load_sd(path: str, *candidates: str) -> dict:
    """Load ``.pt`` or flat ``.safetensors``; pick nested key or ``prefix.*`` strip."""
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
        sub = {x[len(p):]: v for x, v in raw.items() if isinstance(x, str) and x.startswith(p)}
        if sub:
            return sub
    return raw


def load_cfgs(weights_dir: str, config_path: str = "tts.json") -> dict:
    for p in (config_path, os.path.join(weights_dir, "tts.json")):
        if p and os.path.exists(p):
            return load_ttl_config(p)
    return {}


def load_stats(
    weights_dir: str, device: str
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    paths = [weights_dir] if os.path.isfile(weights_dir) else [
        os.path.join(weights_dir, name)
        for name in ("stats_multilingual.pt", "stats.pt", "stats_real_data.pt", "stats_mixed.pt")
    ]
    for path in paths:
        if not os.path.exists(path):
            continue
        stats = torch.load(path, map_location="cpu", weights_only=False)
        mean, std = stats["mean"], stats["std"]
        if mean.ndim == 1:
            mean = mean.view(1, -1, 1)
            std = std.view(1, -1, 1)
        return mean.to(device), std.to(device)
    return None, None


def load_text_processor(weights_dir: str) -> UnicodeProcessor:
    """Delegate to :func:`blue_onnx.load_text_processor` (tries
    ``unicode_indexer.json``, ``vocab.json``, then the bundled fallback)."""
    return _load_text_processor_onnx(weights_dir)


def load_voice_style(voice_style_paths: list[str], device: str = "cpu", verbose: bool = False) -> Style:
    bsz = len(voice_style_paths)
    with open(voice_style_paths[0], "r") as f:
        first = json.load(f)
    ttl_dims = first["style_ttl"]["dims"]
    dp_dims = first["style_dp"]["dims"] if "style_dp" in first else None

    ttl = np.zeros([bsz, ttl_dims[1], ttl_dims[2]], dtype=np.float32)
    dp = np.zeros([bsz, dp_dims[1], dp_dims[2]], dtype=np.float32) if dp_dims else None

    for i, path in enumerate(voice_style_paths):
        with open(path, "r") as f:
            d = json.load(f)
        ttl[i] = np.array(d["style_ttl"]["data"], dtype=np.float32).reshape(ttl_dims[1], ttl_dims[2])
        if dp is not None and "style_dp" in d:
            dd = d["style_dp"]["dims"]
            dp[i] = np.array(d["style_dp"]["data"], dtype=np.float32).reshape(dd[1], dd[2])

    if verbose:
        print(f"Loaded {bsz} voice styles")
    ttl_t = torch.from_numpy(ttl).to(device)
    dp_t = torch.from_numpy(dp).to(device) if dp is not None else None
    return Style(ttl_t, dp_t)


def load_pt_models(
    weights_dir: str,
    cfg: dict,
    device: str,
    text2latent_ckpt: Optional[str] = None,
    ae_ckpt: Optional[str] = None,
    dp_ckpt: Optional[str] = None,
) -> Tuple[
    TextEncoder,
    VectorFieldEstimator,
    DPNetwork,
    LatentDecoder1D,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    vocab_size = cfg.get("vocab_size", 256)
    se_n_style = cfg.get("se_n_style", 50)
    latent_dim = int(cfg.get("latent_dim", 24))
    ccf = int(cfg.get("chunk_compress_factor", 6))
    compressed = latent_dim * ccf

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
        uncond_path = os.path.join(weights_dir, "uncond.pt")
        if os.path.exists(uncond_path):
            un = torch.load(uncond_path, map_location="cpu", weights_only=False)
            if isinstance(un, dict):
                if "u_text" in un:
                    u_text = un["u_text"].to(device)
                if "u_ref" in un:
                    u_ref = un["u_ref"].to(device)

    te_sd = TextEncoder.remap_legacy_state_dict(te_sd)
    vf_sd = VectorFieldEstimator.remap_legacy_state_dict(vf_sd)
    ttl_cfg = cfg.get("ttl", {})
    te_cfg = ttl_cfg.get("text_encoder", {})
    te_conv_cfg = te_cfg.get("convnext", {})
    te_attn_cfg = te_cfg.get("attn_encoder", {})
    spte_cfg = ttl_cfg.get("speech_prompted_text_encoder", {})
    vf_cfg = ttl_cfg.get("vector_field", {})
    vf_time_cfg = vf_cfg.get("time_encoder", {})
    vf_text_cfg = vf_cfg.get("main_blocks", {}).get("text_cond_layer", {})
    dp_cfg = cfg.get("dp", {})

    emb_key = "text_encoder.text_embedder.char_embedder.weight"
    if emb_key in te_sd and te_sd[emb_key].shape[0] != vocab_size:
        vocab_size = te_sd[emb_key].shape[0]

    text_encoder = TextEncoder(
        vocab_size=vocab_size,
        d_model=cfg.get("te_d_model", 256),
        n_conv_layers=cfg.get("te_convnext_layers", 6),
        n_attn_layers=cfg.get("te_attn_n_layers", 4),
        expansion_factor=cfg.get("te_expansion_factor", 4),
        p_dropout=cfg.get("te_attn_p_dropout", 0.0),
        kernel_size=te_conv_cfg.get("ksz", 5),
        dilation_lst=te_conv_cfg.get("dilation_lst"),
        attn_n_heads=te_attn_cfg.get("n_heads", 4),
        attn_filter_channels=te_attn_cfg.get("filter_channels", 1024),
        spte_n_heads=spte_cfg.get("n_heads", 2),
        spte_text_dim=spte_cfg.get("text_dim", cfg.get("te_d_model", 256)),
        spte_style_dim=spte_cfg.get("style_dim", cfg.get("se_d_model", 256)),
        spte_n_units=spte_cfg.get("n_units", cfg.get("te_d_model", 256)),
        spte_n_style=se_n_style,
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
        text_n_heads=int(vf_text_cfg.get("n_heads", cfg.get("vf_text_n_heads", 8))),
        time_hdim=vf_time_cfg.get("hdim", 256),
        rotary_base=float(vf_text_cfg.get("rotary_base", 10000.0)),
        cfg_scale=float(vf_cfg.get("cfg_scale", 3.0)),
    ).to(device).eval()
    vf_estimator.load_state_dict(vf_sd, strict=False)

    dp_path = dp_ckpt or os.path.join(weights_dir, "duration_predictor.pt")
    dp_sd = _load_sd(dp_path, "state_dict")
    dp_sd = DPNetwork.remap_legacy_state_dict(dp_sd)
    dp_vocab_size = vocab_size
    dp_emb_key = "sentence_encoder.text_embedder.char_embedder.weight"
    if dp_emb_key in dp_sd and dp_sd[dp_emb_key].shape[0] != dp_vocab_size:
        dp_vocab_size = dp_sd[dp_emb_key].shape[0]
    dp_model = DPNetwork(
        vocab_size=dp_vocab_size,
        latent_channels=compressed,
        style_dp=cfg.get("dp_style_tokens", 8),
        style_dim=cfg.get("dp_style_dim", 16),
        sentence_encoder_cfg=dp_cfg.get("sentence_encoder"),
        style_encoder_cfg=dp_cfg.get("style_encoder"),
        predictor_cfg=dp_cfg.get("predictor"),
    ).to(device).eval()
    # Keep config-defined shapes; skip ckpt tensors that don't fit (Hub DP
    # ref_encoder is 16-wide while config/tts.json is 64-wide). Voice-JSON
    # inference uses style_dp and does not need ref_encoder weights.
    model_sd = dp_model.state_dict()
    dp_sd = {
        k: v
        for k, v in dp_sd.items()
        if k in model_sd and getattr(v, "shape", None) == model_sd[k].shape
    }
    dp_model.load_state_dict(dp_sd, strict=False)

    voc_path = ae_ckpt or os.path.join(weights_dir, "vocoder.pt")
    voc_sd = _load_sd(voc_path, "decoder", "state_dict")
    vocoder = LatentDecoder1D(cfg=cfg.get("ae_dec_cfg", {})).to(device).eval()
    vocoder.load_state_dict(voc_sd, strict=False)

    return text_encoder, vf_estimator, dp_model, vocoder, u_text, u_ref


def load_reference_models(
    cfg: dict,
    device: str,
    text2latent_ckpt: str,
    ae_ckpt: str,
) -> Tuple[LatentEncoder, ReferenceEncoder]:
    """Load the encoders required to derive a :class:`Style` from a WAV file."""
    ttl_cfg = cfg.get("ttl", {})
    se_cfg = ttl_cfg.get("style_encoder", {})
    se_conv_cfg = se_cfg.get("convnext", {})
    se_tokens_cfg = se_cfg.get("style_token_layer", {})

    ae_encoder = LatentEncoder(cfg=cfg["ae_enc_cfg"]).to(device).eval()
    ae_sd = _load_sd(ae_ckpt, "encoder", "state_dict")
    ae_encoder.load_state_dict(ae_sd, strict=True)

    reference_encoder = ReferenceEncoder(
        in_channels=cfg["compressed_channels"],
        d_model=cfg["se_d_model"],
        hidden_dim=cfg["se_hidden_dim"],
        num_blocks=cfg["se_num_blocks"],
        num_tokens=cfg["se_n_style"],
        num_heads=cfg["se_n_heads"],
        kernel_size=se_conv_cfg.get("ksz", 5),
        dilation_lst=se_conv_cfg.get("dilation_lst"),
        prototype_dim=se_tokens_cfg.get("prototype_dim", cfg["se_d_model"]),
        n_units=se_tokens_cfg.get("n_units", cfg["se_d_model"]),
        style_value_dim=se_tokens_cfg.get("style_value_dim", cfg["se_d_model"]),
    ).to(device).eval()
    ref_sd = _load_sd(text2latent_ckpt, "reference_encoder")
    ref_sd = ReferenceEncoder.remap_legacy_state_dict(ref_sd)
    reference_encoder.load_state_dict(ref_sd, strict=False)
    return ae_encoder, reference_encoder


@torch.inference_mode()
def encode_wav_to_style(
    wav_path: str,
    ae_encoder: LatentEncoder,
    reference_encoder: ReferenceEncoder,
    dp_model: DPNetwork,
    mean: torch.Tensor,
    std: torch.Tensor,
    cfg: dict,
    device: str,
    max_frames: int = 256,
) -> Style:
    """Extract TTL and duration-predictor style tokens from a reference WAV."""
    ae_spec_cfg = cfg.get("ae", {}).get("encoder", {}).get("spec_processor", {})
    sample_rate = int(cfg["ae_sample_rate"])
    mel_spec = LinearMelSpectrogram(
        sample_rate=sample_rate,
        n_fft=ae_spec_cfg.get("n_fft", 2048),
        win_length=ae_spec_cfg.get("win_length", ae_spec_cfg.get("n_fft", 2048)),
        hop_length=ae_spec_cfg.get("hop_length", 512),
        n_mels=ae_spec_cfg.get("n_mels", 228),
    ).to(device).eval()

    import soundfile as sf

    wav_np, source_rate = sf.read(wav_path)
    if wav_np.ndim > 1:
        wav_np = wav_np.mean(axis=1)
    wav = ensure_sr(torch.from_numpy(wav_np).float(), source_rate, sample_rate, device=device)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)

    z = ae_encoder(mel_spec(wav))
    z = compress_latents(z, factor=cfg["chunk_compress_factor"])
    z = ((z - mean) / std) * cfg["normalizer_scale"]
    valid_lengths = torch.tensor([z.shape[-1]], device=device)
    z_ref, ref_mask = build_reference_only(z, valid_lengths, device, max_frames=max_frames)

    return Style(
        reference_encoder(z_ref, mask=ref_mask),
        dp_model.ref_encoder(z_ref, mask=ref_mask),
    )


def _find_onnx_sidecar(*names: str) -> Optional[str]:
    """Locate a BlueV3-onnx file next to PT weights or under ``onnx_models/``."""
    roots = []
    env = os.environ.get("ONNX_DIR")
    if env:
        roots.append(env)
    roots.extend(["onnx_models", "onnx_slim", "."])
    for root in roots:
        for name in names:
            path = os.path.join(root, name)
            if os.path.exists(path):
                return path
    return None


def load_text_to_speech(
    weights_dir: str,
    config_path: str = "tts.json",
    device: str = "cpu",
    text2latent_ckpt: Optional[str] = None,
    ae_ckpt: Optional[str] = None,
    dp_ckpt: Optional[str] = None,
    renikud_path: Optional[str] = None,
    seed: int = 42,
    duration_onnx: Optional[str] = None,
    vocoder_onnx: Optional[str] = None,
    onnx_dir: Optional[str] = None,
) -> TextToSpeech:
    """Load PT TE/VF; pair with BlueV3-onnx DP + vocoder when Hub PT sidecars are wrong/missing.

    ``notmax123/BlueV3`` ships a mismatched duration checkpoint and no AE. When
    ``onnx_models/`` (or ``ONNX_DIR`` / ``onnx_dir``) is present, duration and
    decode use the matching ONNX graphs so PT TE/VF inference stays in parity.
    """
    cfgs = load_cfgs(weights_dir, config_path)
    text_encoder, vf_estimator, dp_model, vocoder, u_text, u_ref = load_pt_models(
        weights_dir, cfgs, device, text2latent_ckpt, ae_ckpt, dp_ckpt
    )

    if onnx_dir is None:
        onnx_dir = os.environ.get("ONNX_DIR") or (
            "onnx_models" if os.path.isdir("onnx_models") else None
        )

    # Prefer ONNX stats (same denorm as BlueV3-onnx vocoder path).
    mean, std = None, None
    if onnx_dir and os.path.exists(os.path.join(onnx_dir, "stats.npz")):
        stats = np.load(os.path.join(onnx_dir, "stats.npz"))
        mean = torch.from_numpy(np.asarray(stats["mean"], dtype=np.float32)).to(device)
        std = torch.from_numpy(np.asarray(stats["std"], dtype=np.float32)).to(device)
        if mean.ndim == 1:
            mean = mean.view(1, -1, 1)
            std = std.view(1, -1, 1)
        if "normalizer_scale" in stats.files:
            cfgs["normalizer_scale"] = float(np.asarray(stats["normalizer_scale"]).reshape(-1)[0])
    if mean is None:
        mean, std = load_stats(weights_dir, device)

    text_processor = load_text_processor(weights_dir)

    if renikud_path is None:
        for cand in ("model.onnx", os.path.join(weights_dir, "model.onnx")):
            if os.path.exists(cand):
                renikud_path = cand
                break
    g2p = TextProcessor(renikud_path)

    import onnxruntime as ort
    from ..blue_onnx import _vocoder_stats_are_identity

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = min(8, os.cpu_count() or 1)
    providers = ["CPUExecutionProvider"]

    if duration_onnx is None:
        duration_onnx = _find_onnx_sidecar("duration_predictor.onnx")
        if duration_onnx is None and onnx_dir:
            cand = os.path.join(onnx_dir, "duration_predictor.onnx")
            if os.path.exists(cand):
                duration_onnx = cand
    dp_ort = None
    if duration_onnx and os.path.exists(duration_onnx):
        dp_ort = ort.InferenceSession(duration_onnx, sess_options=opts, providers=providers)

    if vocoder_onnx is None:
        vocoder_onnx = _find_onnx_sidecar("vocoder.onnx")
        if vocoder_onnx is None and onnx_dir:
            cand = os.path.join(onnx_dir, "vocoder.onnx")
            if os.path.exists(cand):
                vocoder_onnx = cand
    vocoder_ort = None
    denorm_before = False
    if vocoder_onnx and os.path.exists(vocoder_onnx):
        vocoder_ort = ort.InferenceSession(vocoder_onnx, sess_options=opts, providers=providers)
        denorm_before = _vocoder_stats_are_identity(vocoder_onnx)

    return TextToSpeech(
        cfgs=cfgs,
        text_processor=text_processor,
        dp_model=dp_model,
        text_encoder=text_encoder,
        vector_estimator=vf_estimator,
        vocoder=vocoder,
        device=device,
        g2p=g2p,
        u_text=u_text,
        u_ref=u_ref,
        mean=mean,
        std=std,
        seed=seed,
        dp_ort=dp_ort,
        vocoder_ort=vocoder_ort,
        denorm_before_vocoder=denorm_before,
    )


__all__ = [
    "AVAILABLE_LANGS",
    "Style",
    "TextToSpeech",
    "UnicodeProcessor",
    "TextProcessor",
    "chunk_text",
    "load_cfgs",
    "load_pt_models",
    "load_reference_models",
    "load_stats",
    "load_text_processor",
    "load_text_to_speech",
    "load_voice_style",
    "encode_wav_to_style",
    "timer",
]
