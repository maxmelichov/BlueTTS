import os
import json
import re
from typing import List, Optional, Tuple

import numpy as np
import onnxruntime as ort

from ._blue_vocab import text_to_indices, text_to_indices_multilang
from ._common import BLUE_SYNTH_MAX_CHUNK_LEN, Style, TextProcessor, chunk_text


class BlueTTS:
    def __init__(
        self,
        onnx_dir: str,
        config_path: str = "tts.json",
        style_json: Optional[str] = None,
        steps: int = 5,
        cfg_scale: float = 3.0,
        speed: float = 1.0,
        seed: int = 42,
        use_gpu: bool = False,
        chunk_len: int = BLUE_SYNTH_MAX_CHUNK_LEN,
        silence_sec: float = 0.15,
        fade_duration: float = 0.02,
        renikud_path: Optional[str] = None,
        renikud_max_clause_chars: int = 96,
    ):
        self.onnx_dir = onnx_dir
        self.style_json = style_json
        self.steps = steps
        self.cfg_scale = cfg_scale
        self.speed = speed
        self.seed = seed
        self.chunk_len = min(max(1, chunk_len), BLUE_SYNTH_MAX_CHUNK_LEN)
        self.silence_sec = silence_sec
        self.fade_duration = fade_duration

        if renikud_path is None:
            if os.path.exists("model.onnx"):
                renikud_path = "model.onnx"
            elif os.path.exists(os.path.join(onnx_dir, "model.onnx")):
                renikud_path = os.path.join(onnx_dir, "model.onnx")

        self._load_config(config_path)
        self._init_sessions(use_gpu)
        self._load_shuffle_keys()
        self._text_proc = TextProcessor(renikud_path, renikud_max_clause_chars=renikud_max_clause_chars)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_config(self, config_path: str):
        self.latent_dim = 24
        self.chunk_compress_factor = 6
        self.hop_length = 512
        self.sample_rate = 44100

        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            self.latent_dim = int(cfg.get("ttl", {}).get("latent_dim", self.latent_dim))
            self.chunk_compress_factor = int(cfg.get("ttl", {}).get("chunk_compress_factor", self.chunk_compress_factor))
            self.sample_rate = int(cfg.get("ae", {}).get("sample_rate", self.sample_rate))
            self.hop_length = int(cfg.get("ae", {}).get("encoder", {}).get("spec_processor", {}).get("hop_length", self.hop_length))

        self.compressed_channels = self.latent_dim * self.chunk_compress_factor

    def _init_sessions(self, use_gpu: bool):
        available = ort.get_available_providers()
        if use_gpu:
            providers = [p for p in ["CUDAExecutionProvider", "OpenVINOExecutionProvider", "CPUExecutionProvider"] if p in available]
        else:
            providers = [p for p in ["OpenVINOExecutionProvider", "CPUExecutionProvider"] if p in available]

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        cpu_cores = max(1, (os.cpu_count() or 4) // 4)
        opts.intra_op_num_threads = int(os.environ.get("ORT_INTRA", cpu_cores))
        opts.inter_op_num_threads = int(os.environ.get("ORT_INTER", 1))

        self._opts = opts
        self._providers = providers

        self._text_enc = self._load_session("text_encoder.onnx")
        self._vf = self._load_session("vector_estimator.onnx")
        self._vocoder = self._load_session("vocoder.onnx")
        self._dp = self._load_session("duration_predictor.onnx", required=False)

        self._vf_inputs = {i.name for i in self._vf.get_inputs()}

    def _load_session(self, name: str, required: bool = True) -> Optional[ort.InferenceSession]:
        base = os.path.join(self.onnx_dir, name)
        slim = base.replace(".onnx", ".slim.onnx")
        path = slim if os.path.exists(slim) else base
        if not os.path.exists(path):
            if required:
                raise FileNotFoundError(f"Model not found: {base}")
            return None
        return ort.InferenceSession(path, sess_options=self._opts, providers=self._providers)

    def _load_shuffle_keys(self):
        self._model_keys: dict = {}
        keys_path = os.path.join(self.onnx_dir, "keys.npz")
        if not os.path.exists(keys_path):
            return
        data = np.load(keys_path)
        for k in data.files:
            parts = k.split("/", 1)
            if len(parts) == 2:
                model, inp = parts
                self._model_keys.setdefault(model, {})[inp] = data[k]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create(self, phonemes: str, lang: str = "he") -> Tuple[np.ndarray, int]:
        """Synthesize speech from a pre-phonemized (IPA) string."""
        chunks = chunk_text(phonemes, self.chunk_len)
        silence = np.zeros(int(self.silence_sec * self.sample_rate), dtype=np.float32)
        parts = []
        for i, chunk in enumerate(chunks):
            parts.append(self._infer_chunk(chunk, lang=lang))
            if i < len(chunks) - 1:
                parts.append(silence)
        wav = np.concatenate(parts) if parts else np.array([], dtype=np.float32)
        return wav, self.sample_rate

    def synthesize(self, text: str, lang: str = "he") -> Tuple[np.ndarray, int]:
        """Phonemize raw text (phonikud for Hebrew, espeak for others) then synthesize."""
        phonemes = self._text_proc.phonemize(text, lang=lang)
        return self.create(phonemes, lang=lang)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run(self, sess: ort.InferenceSession, feed: dict, model_name: str):
        keys = self._model_keys.get(model_name)
        if keys:
            feed = {**feed, **keys}
        return sess.run(None, feed)

    def _load_style_json(self, path: str):
        with open(path) as f:
            j = json.load(f)

        def _arr(key):
            if key not in j:
                return None
            a = np.array(j[key]["data"], dtype=np.float32)
            return a[None] if a.ndim == 2 else a

        style_ttl = _arr("style_ttl")
        style_dp = _arr("style_dp")
        return style_ttl, style_dp

    def _infer_chunk(self, phonemes: str, lang: str = "he") -> np.ndarray:
        if not self.style_json:
            raise ValueError("style_json is required (must contain style_ttl and style_dp).")
        style_ttl, style_dp = self._load_style_json(self.style_json)
        if style_ttl is None:
            raise ValueError(f"{self.style_json} missing 'style_ttl'.")

        text_plain = re.sub(r"</?[a-z]{2,8}>", "", phonemes)
        indices_dp = text_to_indices(text_plain, lang=lang)
        ids_dp = np.array([indices_dp], dtype=np.int64)
        mask_dp = np.ones((1, 1, len(indices_dp)), dtype=np.float32)

        indices_full = text_to_indices_multilang(phonemes, base_lang=lang)
        text_ids = np.array([indices_full], dtype=np.int64)
        text_mask = np.ones((1, 1, len(indices_full)), dtype=np.float32)

        if style_ttl.ndim == 2:
            style_ttl = style_ttl[None]

        # Text encoder
        te_names = {i.name for i in self._text_enc.get_inputs()}
        te_feed = {"text_ids": text_ids, "text_mask": text_mask, "style_ttl": style_ttl}
        if "ref_keys" in te_names:
            te_feed["ref_keys"] = style_ttl
        text_emb = self._run(self._text_enc, te_feed, "text_encoder")[0]

        T_lat = self._predict_duration(ids_dp, mask_dp, style_dp)
        x = self._flow_matching(text_emb, style_ttl, text_mask, T_lat)
        return self._decode(x)

    def _predict_duration(self, text_ids, text_mask, style_dp) -> int:
        T_lat = None
        if self._dp is not None and style_dp is not None:
            out = self._run(
                self._dp,
                {"text_ids": text_ids, "style_dp": style_dp, "text_mask": text_mask},
                "duration_predictor",
            )
            val = float(np.squeeze(out[0]))
            if np.isfinite(val):
                T_lat = int(np.round(val / max(self.speed, 1e-6)))

        if T_lat is None:
            T_lat = int(text_ids.shape[1] * 1.3)

        txt_len = int(np.sum(text_mask))
        T_cap = max(20, min(txt_len * 3 + 20, 600))
        T_lat = min(max(int(T_lat), 1), T_cap, 800)
        return max(10, T_lat)

    def _flow_matching(self, text_emb, style_ttl, text_mask, T_lat) -> np.ndarray:
        rng = np.random.RandomState(self.seed)
        x = rng.randn(1, self.compressed_channels, T_lat).astype(np.float32)
        latent_mask = np.ones((1, 1, T_lat), dtype=np.float32)
        cfg_scale = np.array([float(self.cfg_scale)], dtype=np.float32)

        for i in range(self.steps):
            feed = {
                "noisy_latent": x,
                "text_emb": text_emb,
                "style_ttl": style_ttl,
                "latent_mask": latent_mask,
                "text_mask": text_mask,
                "current_step": np.array([float(i)], dtype=np.float32),
                "total_step": np.array([float(self.steps)], dtype=np.float32),
                "cfg_scale": cfg_scale,
            }
            x = self._run(self._vf, feed, "vector_estimator")[0]

        return x

    def _apply_fade(self, wav: np.ndarray) -> np.ndarray:
        fade_samples = int(self.fade_duration * self.sample_rate)
        if fade_samples == 0 or len(wav) < 2 * fade_samples:
            return wav
        wav = wav.copy()
        wav[:fade_samples] *= np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        wav[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        return wav

    def _decode(self, z_pred: np.ndarray) -> np.ndarray:
        wav = self._run(self._vocoder, {"z_pred": z_pred.astype(np.float32)}, "vocoder")[0]

        frame_len = int(self.hop_length * self.chunk_compress_factor)
        if wav.shape[-1] > 2 * frame_len:
            wav = wav[..., frame_len:-frame_len]

        wav = wav.squeeze()
        return self._apply_fade(wav)


# ─── Module-level loaders ─────────────────────────────────────────────────────

def load_voice_style(style_paths: List[str]) -> Style:
    """Load pre-extracted style vectors from one or more style JSON files."""
    B = len(style_paths)
    with open(style_paths[0]) as f:
        first = json.load(f)

    ttl_dims = first["style_ttl"]["dims"]
    ttl = np.zeros([B, ttl_dims[1], ttl_dims[2]], dtype=np.float32)

    dp: Optional[np.ndarray] = None
    if "style_dp" in first:
        dp_dims = first["style_dp"]["dims"]
        dp = np.zeros([B, dp_dims[1], dp_dims[2]], dtype=np.float32)

    for i, path in enumerate(style_paths):
        with open(path) as f:
            d = json.load(f)
        ttl[i] = np.array(d["style_ttl"]["data"], dtype=np.float32).reshape(ttl_dims[1], ttl_dims[2])
        if dp is not None and "style_dp" in d:
            dp[i] = np.array(d["style_dp"]["data"], dtype=np.float32).reshape(dp_dims[1], dp_dims[2])

    return Style(ttl=ttl, dp=dp)
