import os
import sys
import json
import re
from typing import Optional, Tuple

import numpy as np
import torch

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)
from _common import Style, TextProcessor, chunk_text  # noqa: E402
from _blue_vocab import text_to_indices, text_to_indices_multilang  # noqa: E402
del _src

_HERE = os.path.dirname(os.path.abspath(__file__))
_TRAINING = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "training")
if _TRAINING not in sys.path:
    sys.path.insert(0, _TRAINING)

from models.text2latent.text_encoder import TextEncoder
from models.text2latent.vf_estimator import VectorFieldEstimator
from models.text2latent.dp_network import DPNetwork
from bluecodec import LatentDecoder1D
from bluecodec.utils import decompress_latents
from models.utils import load_ttl_config


class LightBlueTTS:
    def __init__(
        self,
        weights_dir: str,
        config_path: str = "tts.json",
        style_json: Optional[str] = None,
        steps: int = 32,
        cfg_scale: float = 3.0,
        speed: float = 1.0,
        seed: int = 42,
        device: str = "cpu",
        chunk_len: int = 150,
        silence_sec: float = 0.15,
        fade_duration: float = 0.02,
        text2latent_ckpt: Optional[str] = None,
        ae_ckpt: Optional[str] = None,
        dp_ckpt: Optional[str] = None,
        renikud_path: Optional[str] = None,
    ):
        self.weights_dir = weights_dir
        self.style_json = style_json
        self.steps = steps
        self.cfg_scale = cfg_scale
        self.speed = speed
        self.seed = seed
        self.device = device
        self.chunk_len = chunk_len
        self.silence_sec = silence_sec
        self.fade_duration = fade_duration

        self._load_config(config_path)
        self._load_models(text2latent_ckpt, ae_ckpt, dp_ckpt)
        self._load_stats()
        self._text_proc = TextProcessor(renikud_path)

    def _load_config(self, config_path: str):
        self.normalizer_scale = 1.0
        self.latent_dim = 24
        self.chunk_compress_factor = 6
        self.hop_length = 512
        self.sample_rate = 44100
        self._cfg: dict = {}

        if config_path and os.path.exists(config_path):
            self._cfg = load_ttl_config(config_path)
            self.normalizer_scale = float(self._cfg.get("normalizer_scale", self.normalizer_scale))
            self.latent_dim = int(self._cfg.get("latent_dim", self.latent_dim))
            self.chunk_compress_factor = int(self._cfg.get("chunk_compress_factor", self.chunk_compress_factor))
            self.sample_rate = int(self._cfg.get("ae_sample_rate", self.sample_rate))
            self.hop_length = int(self._cfg.get("ae_hop_length", self.hop_length))

        self.compressed_channels = self.latent_dim * self.chunk_compress_factor

    def _load_models(self, text2latent_ckpt, ae_ckpt, dp_ckpt):
        cfg = self._cfg
        vocab_size        = cfg.get("vocab_size", 384)
        se_n_style        = cfg.get("se_n_style", 50)
        dp_style_tokens   = cfg.get("dp_style_tokens", 8)
        dp_style_dim      = cfg.get("dp_style_dim", 16)

        u_text = u_ref = None
        if text2latent_ckpt:
            combined = torch.load(text2latent_ckpt, map_location="cpu", weights_only=False)
            te_sd = combined["text_encoder"]
            vf_sd = combined["vf_estimator"]
            if "u_text" in combined and "u_ref" in combined:
                u_text = combined["u_text"].to(self.device)
                u_ref  = combined["u_ref"].to(self.device)
        else:
            te_path = os.path.join(self.weights_dir, "text_encoder.pt")
            te_raw = torch.load(te_path, map_location="cpu", weights_only=False)
            te_sd = te_raw.get("text_encoder", te_raw) if isinstance(te_raw, dict) else te_raw

            vf_path = os.path.join(self.weights_dir, "vector_estimator.pt")
            vf_raw = torch.load(vf_path, map_location="cpu", weights_only=False)
            vf_sd = vf_raw.get("vf_estimator", vf_raw) if isinstance(vf_raw, dict) else vf_raw

        if ae_ckpt:
            voc_raw = torch.load(ae_ckpt, map_location="cpu", weights_only=False)
        else:
            voc_raw = torch.load(os.path.join(self.weights_dir, "vocoder.pt"), map_location="cpu", weights_only=False)
        if isinstance(voc_raw, dict) and "decoder" in voc_raw:
            voc_sd = voc_raw["decoder"]
        elif isinstance(voc_raw, dict) and "state_dict" in voc_raw:
            voc_sd = voc_raw["state_dict"]
        else:
            voc_sd = voc_raw

        dp_path = dp_ckpt or os.path.join(self.weights_dir, "duration_predictor.pt")
        dp_raw = torch.load(dp_path, map_location="cpu", weights_only=False)
        dp_sd = dp_raw.get("state_dict", dp_raw) if isinstance(dp_raw, dict) and "state_dict" in dp_raw else dp_raw

        emb_key = "text_embedder.char_embedder.weight"
        if emb_key in te_sd and te_sd[emb_key].shape[0] != vocab_size:
            vocab_size = te_sd[emb_key].shape[0]

        dp_vocab_size = vocab_size
        dp_emb_key = "sentence_encoder.text_embedder.char_embedder.weight"
        if dp_emb_key in dp_sd and dp_sd[dp_emb_key].shape[0] != dp_vocab_size:
            dp_vocab_size = dp_sd[dp_emb_key].shape[0]

        self._text_encoder = TextEncoder(
            vocab_size=vocab_size,
            d_model=cfg.get("te_d_model", 256),
            n_conv_layers=cfg.get("te_convnext_layers", 4),
            n_attn_layers=cfg.get("te_attn_n_layers", 4),
            expansion_factor=cfg.get("te_expansion_factor", 4),
            p_dropout=0.0,
        ).to(self.device).eval()
        self._text_encoder.load_state_dict(te_sd, strict=False)

        self._vf_estimator = VectorFieldEstimator(
            in_channels=self.compressed_channels,
            out_channels=self.compressed_channels,
            hidden_channels=cfg.get("vf_hidden", 256),
            text_dim=cfg.get("vf_text_dim", 256),
            style_dim=cfg.get("vf_style_dim", 256),
            num_style_tokens=se_n_style,
            num_superblocks=cfg.get("vf_n_blocks", 8),
            time_embed_dim=cfg.get("vf_time_dim", 128),
            rope_gamma=cfg.get("vf_rotary_scale", 1.0),
        ).to(self.device).eval()
        self._vf_estimator.load_state_dict(vf_sd, strict=False)
        with torch.no_grad():
            self._vf_estimator.style_key.data.copy_(
                self._text_encoder.speech_prompted_text_encoder.style_key.data
            )

        self._dp_model = DPNetwork(
            vocab_size=dp_vocab_size,
            style_tokens=dp_style_tokens,
            style_dim=dp_style_dim,
        ).to(self.device).eval()
        self._dp_model.load_state_dict(dp_sd, strict=False)

        self._vocoder = LatentDecoder1D(cfg=cfg.get("ae_dec_cfg", {})).to(self.device).eval()
        self._vocoder.load_state_dict(voc_sd, strict=False)

        self._u_text = u_text
        self._u_ref  = u_ref

    def _load_stats(self):
        self.mean = self.std = None
        candidates = ["stats_multilingual.pt", "stats.pt", "stats_real_data.pt", "stats_mixed.pt"]
        for name in candidates:
            path = os.path.join(self.weights_dir, name)
            if os.path.exists(path):
                stats = torch.load(path, map_location="cpu", weights_only=False)
                mean = stats["mean"]
                std  = stats["std"]
                if mean.ndim == 1:
                    mean = mean.view(1, -1, 1)
                    std  = std.view(1, -1, 1)
                self.mean = mean.to(self.device)
                self.std  = std.to(self.device)
                break

    def create(self, phonemes: str, lang: str = "he") -> Tuple[np.ndarray, int]:
        """Synthesize speech from pre-phonemized (IPA) text.

        Args:
            phonemes: IPA phoneme string (caller is responsible for phonemization).
            lang: language code for the tokenizer (default: "he").

        Returns:
            (samples, sample_rate) — float32 numpy array and int sample rate.
        """
        chunks = chunk_text(phonemes, self.chunk_len)
        silence = np.zeros(int(self.silence_sec * self.sample_rate), dtype=np.float32)
        parts = []
        for i, chunk in enumerate(chunks):
            parts.append(self._infer_chunk(chunk, lang))
            if i < len(chunks) - 1:
                parts.append(silence)
        wav = np.concatenate(parts) if parts else np.array([], dtype=np.float32)
        return wav, self.sample_rate

    def synthesize(self, text: str, lang: str = "he") -> Tuple[np.ndarray, int]:
        """Phonemize raw text (renikud for Hebrew, espeak for others) then synthesize.

        Returns:
            (samples, sample_rate) — float32 numpy array and int sample rate.
        """
        phonemes = self._text_proc.phonemize(text, lang=lang)
        return self.create(phonemes, lang=lang)

    def _load_style_json(self, path: str):
        with open(path) as f:
            data = json.load(f)

        def _to_tensor(entry):
            arr = np.array(entry["data"], dtype=np.float32)
            return torch.from_numpy(arr).reshape(entry["dims"]).to(self.device)

        style_ttl = _to_tensor(data["style_ttl"]) if "style_ttl" in data else None
        style_dp  = _to_tensor(data["style_dp"])  if "style_dp"  in data else None
        return style_ttl, style_dp

    @torch.inference_mode()
    def _infer_chunk(self, phonemes: str, lang: str) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Latent stats not loaded.")

        style_ttl = style_dp = None
        if self.style_json:
            style_ttl, style_dp = self._load_style_json(self.style_json)

        if style_ttl is None:
            raise ValueError("Provide style_json with style_ttl content.")

        dev = self.device

        text_plain = re.sub(r"</?[a-z]{2,8}>", "", phonemes)
        ids_dp   = torch.tensor(text_to_indices(text_plain, lang=lang), dtype=torch.long, device=dev).unsqueeze(0)
        mask_dp  = torch.ones(1, 1, ids_dp.shape[1], device=dev)

        ids_full  = torch.tensor(text_to_indices_multilang(phonemes, base_lang=lang), dtype=torch.long, device=dev).unsqueeze(0)
        text_mask = torch.ones(1, 1, ids_full.shape[1], device=dev)

        T_lat = self._predict_duration(ids_dp, mask_dp, style_dp)

        h_text = self._text_encoder(ids_full, style_ttl, text_mask=text_mask)

        x = self._flow_matching(h_text, style_ttl, text_mask, T_lat)

        return self._decode(x)

    def _predict_duration(self, ids_dp, mask_dp, style_dp) -> int:
        T_lat = None
        if style_dp is not None:
            log_dur = self._dp_model(ids_dp, text_mask=mask_dp, style_tokens=style_dp, return_log=True)
            val = torch.exp(log_dur) / max(self.speed, 1e-6)
            T_lat = int(val.round().item())

        if T_lat is None:
            T_lat = int(ids_dp.shape[1] * 1.3)

        txt_len = int(ids_dp.shape[1])
        T_cap = max(20, min(txt_len * 3 + 20, 600))
        T_lat = min(max(T_lat, 10), T_cap, 800)
        return max(10, T_lat)

    def _flow_matching(self, h_text, style_ttl, text_mask, T_lat) -> torch.Tensor:
        dev = self.device
        rng = torch.Generator(device=dev)
        rng.manual_seed(self.seed)
        x = torch.randn(1, self.compressed_channels, T_lat, device=dev, generator=rng)
        latent_mask = torch.ones(1, 1, T_lat, device=dev)
        dt = 1.0 / self.steps

        use_cfg = self.cfg_scale != 1.0 and self._u_text is not None and self._u_ref is not None
        if use_cfg:
            h_text_null = self._u_text.expand(1, -1, 1)
            h_ref_null  = self._u_ref.expand(1, -1, -1)
            u_mask      = torch.ones(1, 1, 1, device=dev)

        for i in range(self.steps):
            t = torch.full((1,), i / self.steps, device=dev)
            x_in = x * latent_mask

            v_cond = self._vf_estimator(
                noisy_latent=x_in,
                text_emb=h_text,
                style_ttl=style_ttl,
                latent_mask=latent_mask,
                text_mask=text_mask,
                current_step=t,
            )

            if use_cfg:
                v_uncond = self._vf_estimator(
                    noisy_latent=x_in,
                    text_emb=h_text_null,
                    style_ttl=h_ref_null,
                    latent_mask=latent_mask,
                    text_mask=u_mask,
                    current_step=t,
                )
                v = v_uncond + self.cfg_scale * (v_cond - v_uncond)
            else:
                v = v_cond

            x = (x + v * dt) * latent_mask

        return x

    def _apply_fade(self, wav: np.ndarray) -> np.ndarray:
        fade_samples = int(self.fade_duration * self.sample_rate)
        if fade_samples == 0 or len(wav) < 2 * fade_samples:
            return wav
        wav = wav.copy()
        wav[:fade_samples]  *= np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        wav[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        return wav

    def _decode(self, x: torch.Tensor) -> np.ndarray:
        ns = float(self.normalizer_scale)
        if ns not in (0.0, 1.0):
            z_pred = (x / ns) * self.std + self.mean
        else:
            z_pred = x * self.std + self.mean

        z_pred = decompress_latents(z_pred, factor=self.chunk_compress_factor, target_channels=self.latent_dim)
        wav = self._vocoder(z_pred)

        frame_len = self.hop_length * self.chunk_compress_factor
        if wav.shape[-1] > 2 * frame_len:
            wav = wav[..., frame_len:-frame_len]

        wav_np = wav.squeeze().cpu().numpy().astype(np.float32)
        return self._apply_fade(wav_np)
