"""Blue ONNX TTS — flat single-file inference module."""

import json
import os
import re
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from typing import Any, List, Optional, Tuple
from unicodedata import normalize as uni_normalize

import numpy as np
import onnxruntime as ort


# ─── Vocab (loaded from vocab.json) ───────────────────────────────────────────

_VOCAB_PATH = os.path.join(os.path.dirname(__file__), "vocab.json")
with open(_VOCAB_PATH, encoding="utf-8") as _f:
    _VOCAB = json.load(_f)

PAD_ID: int = _VOCAB["pad_id"]
BOS_ID: int = _VOCAB["bos_id"]
EOS_ID: int = _VOCAB["eos_id"]
VOCAB_SIZE: int = _VOCAB["vocab_size"]
CHAR_TO_ID: dict = {k: int(v) for k, v in _VOCAB["char_to_id"].items()}
ID_TO_CHAR: dict = {v: k for k, v in CHAR_TO_ID.items()}

BLUE_SYNTH_MAX_CHUNK_LEN = 150
_LANG_TAG_RE = re.compile(r"</?\w+>")
_INLINE_LANG_PAIR = re.compile(r"<(\w+)>(.*?)(?:</\1>|<\1>)", re.DOTALL)


# ─── Tokenization (no language tokens) ────────────────────────────────────────

def text_to_indices(text: str) -> List[int]:
    """IPA string → vocab IDs. Language tags are stripped; unknown chars → PAD_ID."""
    text = _LANG_TAG_RE.sub("", text)
    return [CHAR_TO_ID.get(ch, PAD_ID) for ch in text]


def indices_to_text(indices: List[int]) -> str:
    return "".join(ID_TO_CHAR.get(i, "?") for i in indices)


# ─── Text normalization ───────────────────────────────────────────────────────

_EMOJI_RE = re.compile(
    r"[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff"
    r"\U0001f700-\U0001f77f\U0001f780-\U0001f7ff\U0001f800-\U0001f8ff"
    r"\U0001f900-\U0001f9ff\U0001fa00-\U0001fa6f\U0001fa70-\U0001faff"
    r"\u2600-\u26ff\u2700-\u27bf\U0001f1e6-\U0001f1ff]+",
    flags=re.UNICODE,
)
_UNIVERSAL_REPLACEMENTS = {
    "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
    "\u00b4": "'", "`": "'", "\u2013": "-", "\u2011": "-", "\u2014": "-",
}


def normalize_text(text: str, apply_hebrew_fixes: bool = False) -> str:
    text = uni_normalize("NFD", text.strip())
    text = _EMOJI_RE.sub("", text)
    for k, v in _UNIVERSAL_REPLACEMENTS.items():
        text = text.replace(k, v)
    text = re.sub(r"[\u2665\u2606\u2661\u00a9]", "", text)
    for p in [",", ".", "!", "?", ";", ":", "'"]:
        text = text.replace(f" {p}", p)
    for q in ['""', "''", "``"]:
        while q in text:
            text = text.replace(q, q[0])
    if apply_hebrew_fixes:
        text = text.replace("r", "\u0281").replace("g", "\u0261")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─── Text chunking ────────────────────────────────────────────────────────────

_CHUNK_BOUNDARY = re.compile(
    r"(?<!Mr\.)(?<!Mrs\.)(?<!Ms\.)(?<!Dr\.)(?<!Prof\.)(?<!Sr\.)(?<!Jr\.)"
    r"(?<!Ph\.D\.)(?<!etc\.)(?<!e\.g\.)(?<!i\.e\.)(?<!vs\.)(?<!Inc\.)"
    r"(?<!Ltd\.)(?<!Co\.)(?<!Corp\.)(?<!St\.)(?<!Ave\.)(?<!Blvd\.)"
    r"(?<!\b[A-Z]\.)(?<=[.!?])\s+"
)


def _hard_split(s: str, max_len: int) -> List[str]:
    s = s.strip()
    if not s or max_len <= 0 or len(s) <= max_len:
        return [s] if s else []
    out, start, n = [], 0, len(s)
    while start < n:
        end = min(start + max_len, n)
        if end < n:
            cut = s[start:end].rfind(" ")
            if cut > max(max_len // 4, 8):
                end = start + cut
        piece = s[start:end].strip()
        if piece:
            out.append(piece)
        start = end
        while start < n and s[start] == " ":
            start += 1
    return out


def chunk_text(text: str, max_len: int = 300) -> List[str]:
    """Split text into sentence-boundary chunks no longer than ``max_len``."""
    text = re.sub(r"([.!?])(</\w+>)\s+", r"\1\2\n\n", text)
    chunks: List[str] = []
    for paragraph in re.split(r"\n\s*\n+", text.strip()):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        current = ""
        for sentence in _CHUNK_BOUNDARY.split(paragraph):
            if len(current) + len(sentence) + 1 <= max_len:
                current += (" " if current else "") + sentence
            else:
                if current:
                    chunks.append(current.strip())
                if len(sentence) > max_len:
                    chunks.extend(_hard_split(sentence, max_len))
                    current = ""
                else:
                    current = sentence
        if current:
            chunks.append(current.strip())
    base = chunks if chunks else ([text.strip()] if text.strip() else [])
    out: List[str] = []
    for c in base:
        out.extend(_hard_split(c, max_len))

    fixed, active = [], None
    for c in out:
        c = c.strip()
        if not c:
            continue
        if active and not c.startswith(f"<{active}>"):
            c = f"<{active}>" + c
        for m in re.finditer(r"<(/)?(\w+)>", c):
            if m.group(1):
                if active == m.group(2):
                    active = None
            else:
                active = m.group(2)
        if active and not c.endswith(f"</{active}>"):
            c = c + f"</{active}>"
        fixed.append(c)
    return fixed or ([text.strip()] if text.strip() else [])


# ─── Phonemization (renikud + espeak) ─────────────────────────────────────────

_ESPEAK_MAP = {
    "en": "en-us", "en-us": "en-us", "de": "de", "ge": "de",
    "it": "it", "es": "es",
}


class TextProcessor:
    """Renikud for Hebrew; espeak for everything else. Preserves ``<lang>…</lang>`` spans."""

    def __init__(self, renikud_path: Optional[str] = None):
        self.renikud = None
        if renikud_path is None and os.path.exists("model.onnx"):
            renikud_path = "model.onnx"
        if renikud_path and os.path.exists(renikud_path):
            try:
                from renikud_onnx import G2P
                self.renikud = G2P(renikud_path)
                print(f"[INFO] Loaded Renikud G2P from {renikud_path}")
            except ImportError as e:
                raise RuntimeError("Hebrew G2P needs `renikud-onnx`. Install: `uv sync`.") from e

    def _espeak(self, text: str, lang: str) -> str:
        espeak_lang = _ESPEAK_MAP.get(lang)
        if espeak_lang is None:
            return text
        try:
            import espeakng_loader
            EspeakBackend = import_module("phonemizer.backend").EspeakBackend
            EspeakWrapper = import_module("phonemizer.backend.espeak.wrapper").EspeakWrapper
            Separator = import_module("phonemizer.separator").Separator
            EspeakWrapper.set_library(espeakng_loader.get_library_path())
            if hasattr(EspeakWrapper, "set_data_path"):
                EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
            backend = EspeakBackend(
                espeak_lang, preserve_punctuation=True,
                with_stress=True, language_switch="remove-flags",
            )
            raw = backend.phonemize([text], separator=Separator(phone="", word=" ", syllable=""))[0]
            return normalize_text(raw)
        except Exception as e:
            print(f"[WARN] Phonemizer backend failed for lang={lang}: {e}")
        try:
            r = subprocess.run(
                ["espeak-ng", "-q", "--ipa=1", "-v", espeak_lang, text],
                check=True, capture_output=True, text=True,
            )
            return normalize_text(r.stdout.replace("\n", " ").strip())
        except Exception as e:
            print(f"[WARN] espeak-ng fallback failed for lang={lang}: {e}")
            return text

    def _phonemize_segment(self, content: str, lang: str) -> str:
        content = content.strip()
        if not content:
            return ""
        has_hebrew = any("\u0590" <= c <= "\u05ff" for c in content)
        if has_hebrew:
            if self.renikud is None:
                raise ValueError(
                    "Hebrew text requires Renikud weights. Download:\n"
                    "  wget -O model.onnx https://huggingface.co/thewh1teagle/renikud/resolve/main/model.onnx"
                )
            return normalize_text(self.renikud.phonemize(content), apply_hebrew_fixes=True)
        if lang == "he":
            return normalize_text(content, apply_hebrew_fixes=True)
        return self._espeak(content, lang)

    def phonemize(self, text: str, lang: str = "he") -> str:
        if not _INLINE_LANG_PAIR.search(text):
            return self._phonemize_segment(text, lang)
        pieces, last_end = [], 0
        for m in _INLINE_LANG_PAIR.finditer(text):
            if m.start() > last_end:
                p = self._phonemize_segment(text[last_end:m.start()], lang)
                if p:
                    pieces.append(p)
            tag = m.group(1)
            pieces.append(self._phonemize_segment(m.group(2), tag))
            last_end = m.end()
        if last_end < len(text):
            p = self._phonemize_segment(text[last_end:], lang)
            if p:
                pieces.append(p)
        return re.sub(r"\s+", " ", " ".join(pieces)).strip()


# ─── Helpers ──────────────────────────────────────────────────────────────────

@contextmanager
def timer(name: str):
    start = time.time()
    print(f"{name}...")
    yield
    print(f"  -> {name} completed in {time.time() - start:.2f} sec")


def sanitize_filename(text: str, max_len: int) -> str:
    return re.sub(r"[^\w]", "_", text[:max_len], flags=re.UNICODE)


def length_to_mask(lengths: np.ndarray, max_len: Optional[int] = None) -> np.ndarray:
    max_len = int(max_len or lengths.max())
    ids = np.arange(0, max_len)
    mask = (ids < np.expand_dims(lengths, axis=1)).astype(np.float32)
    return mask.reshape(-1, 1, max_len)


@dataclass
class Style:
    ttl: Any
    dp: Optional[Any] = None


# ─── ONNX loaders ─────────────────────────────────────────────────────────────

def make_session_options() -> ort.SessionOptions:
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    cores = max(1, (os.cpu_count() or 4) // 4)
    opts.intra_op_num_threads = int(os.environ.get("ORT_INTRA", cores))
    opts.inter_op_num_threads = int(os.environ.get("ORT_INTER", 1))
    return opts


def pick_providers(use_gpu: bool) -> List[str]:
    avail = ort.get_available_providers()
    wanted = ["CUDAExecutionProvider", "OpenVINOExecutionProvider", "CPUExecutionProvider"] if use_gpu \
        else ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
    return [p for p in wanted if p in avail]


def _resolve(onnx_dir: str, name: str) -> Optional[str]:
    base = os.path.join(onnx_dir, name)
    slim = base.replace(".onnx", ".slim.onnx")
    if os.path.exists(slim):
        return slim
    return base if os.path.exists(base) else None


def load_onnx(path: str, opts: ort.SessionOptions, providers: List[str]) -> ort.InferenceSession:
    return ort.InferenceSession(path, sess_options=opts, providers=providers)


def load_cfgs(onnx_dir: str, config_path: str = "tts.json") -> dict:
    for p in (config_path, os.path.join(onnx_dir, "tts.json")):
        if p and os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return {}


def load_shuffle_keys(onnx_dir: str) -> dict:
    p = os.path.join(onnx_dir, "keys.npz")
    if not os.path.exists(p):
        return {}
    data = np.load(p)
    out: dict = {}
    for k in data.files:
        parts = k.split("/", 1)
        if len(parts) == 2:
            out.setdefault(parts[0], {})[parts[1]] = data[k]
    return out


def load_voice_style(voice_style_paths: List[str], verbose: bool = False) -> Style:
    """Load one or more voice style JSONs into a batched ``Style``."""
    bsz = len(voice_style_paths)
    with open(voice_style_paths[0]) as f:
        first = json.load(f)
    ttl_dims = first["style_ttl"]["dims"]
    ttl = np.zeros([bsz, ttl_dims[1], ttl_dims[2]], dtype=np.float32)
    dp: Optional[np.ndarray] = None
    if "style_dp" in first:
        dd = first["style_dp"]["dims"]
        dp = np.zeros([bsz, dd[1], dd[2]], dtype=np.float32)
    for i, path in enumerate(voice_style_paths):
        with open(path) as f:
            d = json.load(f)
        ttl[i] = np.array(d["style_ttl"]["data"], dtype=np.float32).reshape(ttl_dims[1], ttl_dims[2])
        if dp is not None and "style_dp" in d:
            dd = d["style_dp"]["dims"]
            dp[i] = np.array(d["style_dp"]["data"], dtype=np.float32).reshape(dd[1], dd[2])
    if verbose:
        print(f"Loaded {bsz} voice styles")
    return Style(ttl=ttl, dp=dp)


# ─── BlueTTS ──────────────────────────────────────────────────────────────────

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
    ):
        self.steps = steps
        self.cfg_scale = cfg_scale
        self.speed = speed
        self.seed = seed
        self.chunk_len = min(max(1, chunk_len), BLUE_SYNTH_MAX_CHUNK_LEN)
        self.silence_sec = silence_sec
        self.fade_duration = fade_duration

        if renikud_path is None:
            for cand in ("model.onnx", os.path.join(onnx_dir, "model.onnx")):
                if os.path.exists(cand):
                    renikud_path = cand
                    break

        cfgs = load_cfgs(onnx_dir, config_path)
        ttl = cfgs.get("ttl", {}) or {}
        ae = cfgs.get("ae", {}) or {}
        spec = (ae.get("encoder", {}) or {}).get("spec_processor", {}) or {}
        self.latent_dim = int(ttl.get("latent_dim", 24))
        self.chunk_compress_factor = int(ttl.get("chunk_compress_factor", 6))
        self.hop_length = int(spec.get("hop_length", 512))
        self.sample_rate = int(ae.get("sample_rate", 44100))
        self.compressed_channels = self.latent_dim * self.chunk_compress_factor

        opts = make_session_options()
        providers = pick_providers(use_gpu)

        def _req(name):
            p = _resolve(onnx_dir, name)
            if p is None:
                raise FileNotFoundError(f"Model not found: {os.path.join(onnx_dir, name)}")
            return load_onnx(p, opts, providers)

        def _opt(name):
            p = _resolve(onnx_dir, name)
            return load_onnx(p, opts, providers) if p else None

        self._text_enc = _req("text_encoder.onnx")
        self._vf = _req("vector_estimator.onnx")
        self._vocoder = _req("vocoder.onnx")
        self._dp = _opt("length_pred_style.onnx") or _opt("duration_predictor.onnx")
        self._te_inputs = {i.name for i in self._text_enc.get_inputs()}
        self._vf_inputs = {i.name for i in self._vf.get_inputs()}
        self._dp_inputs = {i.name for i in self._dp.get_inputs()} if self._dp else set()
        self._vocoder_input = self._vocoder.get_inputs()[0].name

        # Optional CFG uncond embeddings and vocoder stats.
        self._u_text, self._u_ref = None, None
        if os.path.exists(os.path.join(onnx_dir, "uncond.npz")):
            un = np.load(os.path.join(onnx_dir, "uncond.npz"))
            if "u_text" in un.files:
                self._u_text = un["u_text"].astype(np.float32)
            if "u_ref" in un.files:
                self._u_ref = un["u_ref"].astype(np.float32)

        self._stats_mean, self._stats_std, self._norm_scale = None, None, 1.0
        if os.path.exists(os.path.join(onnx_dir, "stats.npz")):
            st = np.load(os.path.join(onnx_dir, "stats.npz"))
            if "mean" in st.files:
                self._stats_mean = st["mean"].astype(np.float32)
            if "std" in st.files:
                self._stats_std = st["std"].astype(np.float32)
            if "normalizer_scale" in st.files:
                self._norm_scale = float(np.squeeze(st["normalizer_scale"]))

        self._keys = load_shuffle_keys(onnx_dir)
        self._style = load_voice_style([style_json]) if style_json else None
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

    def _run(self, sess, feed, name):
        extra = self._keys.get(name)
        if extra:
            feed = {**feed, **extra}
        return sess.run(None, feed)

    def _infer_chunk(self, phonemes: str, cfg_scale: Optional[float] = None) -> np.ndarray:
        cfg_scale = self.cfg_scale if cfg_scale is None else float(cfg_scale)
        if self._style is None:
            raise ValueError("style_json is required (must contain style_ttl and style_dp).")
        style_ttl = self._style.ttl if self._style.ttl.ndim == 3 else self._style.ttl[None]
        style_dp = self._style.dp

        text_ids = np.array([text_to_indices(phonemes)], dtype=np.int64)
        text_mask = np.ones((1, 1, text_ids.shape[1]), dtype=np.float32)

        te_feed = {"text_ids": text_ids, "text_mask": text_mask, "style_ttl": style_ttl}
        if "ref_keys" in self._te_inputs:
            te_feed["ref_keys"] = style_ttl
        text_emb = self._run(self._text_enc, te_feed, "text_encoder")[0]

        # Duration predictor → T_lat in latent frames.
        if self._dp is not None:
            dp_feed: dict = {"text_ids": text_ids, "text_mask": text_mask}
            if "style_dp" in self._dp_inputs and style_dp is not None:
                dp_feed["style_dp"] = style_dp
            if "z_ref" in self._dp_inputs and style_dp is not None:
                dp_feed["z_ref"] = style_dp
            if "ref_mask" in self._dp_inputs and style_dp is not None:
                dp_feed["ref_mask"] = np.ones((1, 1, style_dp.shape[-1]), dtype=np.float32)
            raw = float(np.squeeze(self._run(self._dp, dp_feed, "duration_predictor")[0])) / max(self.speed, 1e-6)
            chunk_size = self.hop_length * self.chunk_compress_factor
            as_frames = int(np.ceil(raw))
            as_seconds = int(np.ceil(raw * self.sample_rate / chunk_size))
            ref = text_ids.shape[1]
            T_lat = max(10, as_frames if abs(as_frames - ref) <= abs(as_seconds - ref) else as_seconds)
        else:
            T_lat = max(10, int(text_ids.shape[1] * 1.3))

        # Flow matching with optional classifier-free guidance.
        rng = np.random.RandomState(self.seed)
        x = rng.randn(1, self.compressed_channels, T_lat).astype(np.float32)
        latent_mask = np.ones((1, 1, T_lat), dtype=np.float32)
        total = np.array([float(self.steps)], dtype=np.float32)
        use_cfg = cfg_scale != 1.0 and self._u_text is not None and self._u_ref is not None
        u_mask = np.ones((1, 1, 1), dtype=np.float32) if use_cfg else None
        for i in range(self.steps):
            cur = np.array([float(i)], dtype=np.float32)
            cond = {
                "noisy_latent": x, "text_emb": text_emb, "style_ttl": style_ttl,
                "latent_mask": latent_mask, "text_mask": text_mask,
                "current_step": cur, "total_step": total,
            }
            if "cfg_scale" in self._vf_inputs:
                cond["cfg_scale"] = np.array([float(cfg_scale)], dtype=np.float32)
                x = self._run(self._vf, cond, "vector_estimator")[0]
            elif use_cfg:
                # SupertonicTTS §3.4: v = v_uncond + cfg_scale * (v_cond - v_uncond).
                # The ONNX wrapper returns denoised = x + v/total_step; mixing denoiseds
                # by the same formula is mathematically equivalent (linearity of Euler step).
                v_cond = self._run(self._vf, cond, "vector_estimator")[0]
                uncond = {
                    "noisy_latent": x, "text_emb": self._u_text, "style_ttl": self._u_ref,
                    "latent_mask": latent_mask, "text_mask": u_mask,
                    "current_step": cur, "total_step": total,
                }
                v_uncond = self._run(self._vf, uncond, "vector_estimator")[0]
                x = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                x = self._run(self._vf, cond, "vector_estimator")[0]

        # Vocoder prep: denormalize in 144-ch, then pixel-shuffle time axis to 24-ch.
        if self._norm_scale and self._norm_scale != 1.0:
            x = x / self._norm_scale
        if self._stats_mean is not None and self._stats_std is not None:
            x = x * self._stats_std + self._stats_mean
        if x.shape[1] == self.compressed_channels and self.chunk_compress_factor > 1:
            B, _, T = x.shape
            C, K = self.latent_dim, self.chunk_compress_factor
            x = x.reshape(B, C, K, T).transpose(0, 1, 3, 2).reshape(B, C, T * K)

        wav = self._run(self._vocoder, {self._vocoder_input: x.astype(np.float32)}, "vocoder")[0]
        frame_len = int(self.hop_length * self.chunk_compress_factor)
        if wav.shape[-1] > 2 * frame_len:
            wav = wav[..., frame_len:-frame_len]
        wav = wav.squeeze()

        fs = int(self.fade_duration * self.sample_rate)
        if fs and len(wav) >= 2 * fs:
            wav = wav.copy()
            wav[:fs] *= np.linspace(0.0, 1.0, fs, dtype=np.float32)
            wav[-fs:] *= np.linspace(1.0, 0.0, fs, dtype=np.float32)
        return wav

