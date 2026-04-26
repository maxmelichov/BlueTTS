import json
import os
import re
import subprocess
import time
from contextlib import contextmanager
from importlib import import_module
from typing import Optional, Union
from unicodedata import normalize

import numpy as np
import onnxruntime as ort


AVAILABLE_LANGS = ["en", "es", "de", "it", "he"]
BLUE_SYNTH_MAX_CHUNK_LEN = 300
# When ``pace_blend > 0``, duration is nudged toward this many seconds of audio per
# text input token, so the same ``speed`` value tracks more closely across languages.
DURATION_PACE_DPT_REF = 0.0625
# Default blend automatically applied when inline ``<lang>...`` spans are present.
DEFAULT_MIXED_PACE_BLEND = 0.25


def blend_duration_pace(
    dur: np.ndarray,
    text_mask: np.ndarray,
    pace_blend: float,
    pace_dpt_ref: float,
) -> np.ndarray:
    """Blend seconds-per-text-token toward ``pace_dpt_ref`` to reduce language bias.

    The duration head tends to use different time-per-token for different
    languages; mixing ``<lang>…</lang>`` segments in one string keeps one
    ``speed``, but the predicted total seconds can still lean on those biases.
    Blending softens that before ``duration / speed`` so ``speed`` is a more
    consistent stretch factor across languages.
    """
    b = min(max(float(pace_blend), 0.0), 1.0)
    if b <= 0.0:
        return np.asarray(dur, dtype=np.float32).reshape(-1)
    d = np.asarray(dur, dtype=np.float64).reshape(-1)
    n = np.maximum(
        np.asarray(text_mask, dtype=np.float64).sum(axis=(1, 2)),
        1.0,
    ).reshape(-1)
    dpt = d / n
    dpt2 = (1.0 - b) * dpt + b * float(pace_dpt_ref)
    return (dpt2 * n).astype(np.float32)


_ESPEAK_MAP = {
    "en": "en-us", "en-us": "en-us",
    "de": "de", "ge": "de",
    "it": "it", "es": "es",
}
_TEXT_TO_INDICES_PROCESSOR: Optional["UnicodeProcessor"] = None

_INLINE_LANG_PAIR = re.compile(r"<(\w+)>(.*?)</\1>", re.DOTALL)
_LANG_TAG_RE = re.compile(r"</?\w+>")


class TextProcessor:
    """Renikud for Hebrew; espeak for everything else. Preserves ``<lang>…</lang>`` spans.

    Output is phonemized text re-wrapped in ``<lang>…</lang>`` tags so that
    :class:`UnicodeProcessor` can route segments by language downstream.
    """

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
                raise RuntimeError(
                    "Hebrew G2P needs `renikud-onnx`. Install: `uv sync`."
                ) from e

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
            raw = backend.phonemize(
                [text], separator=Separator(phone="", word=" ", syllable="")
            )[0]
            return re.sub(r"\s+", " ", raw).strip()
        except Exception as e:
            print(f"[WARN] Phonemizer backend failed for lang={lang}: {e}")
        try:
            r = subprocess.run(
                ["espeak-ng", "-q", "--ipa=1", "-v", espeak_lang, text],
                check=True, capture_output=True, text=True,
            )
            return re.sub(r"\s+", " ", r.stdout.replace("\n", " ")).strip()
        except Exception as e:
            print(f"[WARN] espeak-ng fallback failed for lang={lang}: {e}")
            return text

    def _phonemize_segment(self, content: str, lang: str) -> str:
        # Guard against malformed/unmatched tags leaking into phonemizer input.
        content = _LANG_TAG_RE.sub("", content).strip()
        if not content:
            return ""
        has_hebrew = any("\u0590" <= c <= "\u05ff" for c in content)
        if has_hebrew or lang == "he":
            if not has_hebrew:
                return content
            if self.renikud is None:
                raise ValueError(
                    "Hebrew text requires Renikud weights. Download:\n"
                    "  wget -O model.onnx https://huggingface.co/thewh1teagle/renikud/resolve/main/model.onnx"
                )
            return self.renikud.phonemize(content)
        return self._espeak(content, lang)

    def phonemize(self, text: str, lang: str = "he") -> str:
        """Phonemize ``text``; inline ``<xx>…</xx>`` spans are phonemized per-lang.

        Returns a string with ``<lang>…</lang>`` tags preserved around each segment.
        """
        if not _INLINE_LANG_PAIR.search(text):
            seg = self._phonemize_segment(text, lang)
            return f"<{lang}>{seg}</{lang}>" if seg else ""
        pieces, last_end = [], 0
        for m in _INLINE_LANG_PAIR.finditer(text):
            if m.start() > last_end:
                seg = self._phonemize_segment(text[last_end:m.start()], lang)
                if seg:
                    pieces.append(f"<{lang}>{seg}</{lang}>")
            tag = m.group(1)
            seg = self._phonemize_segment(m.group(2), tag)
            if seg:
                pieces.append(f"<{tag}>{seg}</{tag}>")
            last_end = m.end()
        if last_end < len(text):
            seg = self._phonemize_segment(text[last_end:], lang)
            if seg:
                pieces.append(f"<{lang}>{seg}</{lang}>")
        return re.sub(r"\s+", " ", " ".join(pieces)).strip()


class UnicodeProcessor:
    """Character-level tokenizer backed by ``vocab.json`` (``char_to_id`` map).

    The constructor accepts either:
      - a path to a ``vocab.json`` with ``{pad_id, char_to_id, ...}`` (this repo), or
      - a path to a legacy unicode codepoint ``indexer.json`` (``{codepoint: id}``).

    Unknown characters and stripped ``<lang>`` tags map to ``pad_id``.
    """

    def __init__(self, indexer_path: str):
        with open(indexer_path, "r") as f:
            raw = json.load(f)
        if isinstance(raw, dict) and "char_to_id" in raw:
            self.pad_id = int(raw.get("pad_id", 0))
            self._char_to_id = {k: int(v) for k, v in raw["char_to_id"].items()}
            self._codepoint_indexer = None
        else:
            self.pad_id = 0
            self._char_to_id = None
            self._codepoint_indexer = {int(k): int(v) for k, v in raw.items()} \
                if all(isinstance(k, str) and k.isdigit() for k in raw.keys()) \
                else {int(k): int(v) for k, v in raw.items()}

    def _preprocess_text(self, text: str, lang: str) -> str:
        # TODO: Need advanced normalizer for better performance
        text = normalize("NFKD", text)

        # Remove emojis (wide Unicode range)
        emoji_pattern = re.compile(
            "[\U0001f600-\U0001f64f"  # emoticons
            "\U0001f300-\U0001f5ff"  # symbols & pictographs
            "\U0001f680-\U0001f6ff"  # transport & map symbols
            "\U0001f700-\U0001f77f"
            "\U0001f780-\U0001f7ff"
            "\U0001f800-\U0001f8ff"
            "\U0001f900-\U0001f9ff"
            "\U0001fa00-\U0001fa6f"
            "\U0001fa70-\U0001faff"
            "\u2600-\u26ff"
            "\u2700-\u27bf"
            "\U0001f1e6-\U0001f1ff]+",
            flags=re.UNICODE,
        )
        text = emoji_pattern.sub("", text)

        # Replace various dashes and symbols
        replacements = {
            "–": "-",
            "‑": "-",
            "—": "-",
            "_": " ",
            "\u201c": '"',  # left double quote "
            "\u201d": '"',  # right double quote "
            "\u2018": "'",  # left single quote '
            "\u2019": "'",  # right single quote '
            "´": "'",
            "`": "'",
            "[": " ",
            "]": " ",
            "|": " ",
            "/": " ",
            "#": " ",
            "→": " ",
            "←": " ",
        }
        for k, v in replacements.items():
            text = text.replace(k, v)

        # Remove special symbols
        text = re.sub(r"[♥☆♡©\\]", "", text)

        # Replace known expressions
        expr_replacements = {
            "@": " at ",
            "e.g.,": "for example, ",
            "i.e.,": "that is, ",
        }
        for k, v in expr_replacements.items():
            text = text.replace(k, v)

        # Fix spacing around punctuation
        text = re.sub(r" ,", ",", text)
        text = re.sub(r" \.", ".", text)
        text = re.sub(r" !", "!", text)
        text = re.sub(r" \?", "?", text)
        text = re.sub(r" ;", ";", text)
        text = re.sub(r" :", ":", text)
        text = re.sub(r" '", "'", text)

        # Remove duplicate quotes
        while '""' in text:
            text = text.replace('""', '"')
        while "''" in text:
            text = text.replace("''", "'")
        while "``" in text:
            text = text.replace("``", "`")

        # Remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()

        # If text doesn't end with punctuation, quotes, or closing brackets, add a period
        if not re.search(r"[.!?;:,'\"')\]}…。」』】〉》›»]$", text):
            text += "."

        if lang not in AVAILABLE_LANGS:
            raise ValueError(f"Invalid language: {lang}")
        # If the text already contains <lang>…</lang> spans (e.g. output from
        # :class:`TextProcessor`), don't wrap again.
        if not _INLINE_LANG_PAIR.search(text):
            text = f"<{lang}>" + text + f"</{lang}>"
        return text

    def _get_text_mask(self, text_ids_lengths: np.ndarray) -> np.ndarray:
        text_mask = length_to_mask(text_ids_lengths)
        return text_mask

    def _encode(self, text: str) -> np.ndarray:
        # Strip any remaining language tags before encoding.
        text = _LANG_TAG_RE.sub("", text)
        if self._char_to_id is not None:
            pad = self.pad_id
            ids = [self._char_to_id.get(ch, pad) for ch in text]
        else:
            assert self._codepoint_indexer is not None
            pad = self.pad_id
            ids = [self._codepoint_indexer.get(ord(ch), pad) for ch in text]
        return np.array(ids, dtype=np.int64)

    def __call__(
        self, text_list: list[str], lang_list: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        text_list = [
            self._preprocess_text(t, lang) for t, lang in zip(text_list, lang_list)
        ]
        encoded = [self._encode(t) for t in text_list]
        text_ids_lengths = np.array([len(e) for e in encoded], dtype=np.int64)
        text_ids = np.full(
            (len(encoded), int(text_ids_lengths.max())), self.pad_id, dtype=np.int64
        )
        for i, ids in enumerate(encoded):
            text_ids[i, : len(ids)] = ids
        text_mask = self._get_text_mask(text_ids_lengths)
        return text_ids, text_mask


class Style:
    def __init__(self, style_ttl_onnx: np.ndarray, style_dp_onnx: np.ndarray):
        self.ttl = style_ttl_onnx
        self.dp = style_dp_onnx


class TextToSpeech:
    def __init__(
        self,
        cfgs: dict,
        text_processor: UnicodeProcessor,
        dp_ort: ort.InferenceSession,
        text_enc_ort: ort.InferenceSession,
        vector_est_ort: ort.InferenceSession,
        vocoder_ort: ort.InferenceSession,
        g2p: Optional[TextProcessor] = None,
        u_text: Optional[np.ndarray] = None,
        u_ref: Optional[np.ndarray] = None,
    ):
        self.cfgs = cfgs
        self.text_processor = text_processor
        self.g2p = g2p
        self.dp_ort = dp_ort
        self.text_enc_ort = text_enc_ort
        self.vector_est_ort = vector_est_ort
        self.vocoder_ort = vocoder_ort
        self.sample_rate = cfgs["ae"]["sample_rate"]
        self.base_chunk_size = cfgs["ae"]["base_chunk_size"]
        self.chunk_compress_factor = cfgs["ttl"]["chunk_compress_factor"]
        self.ldim = cfgs["ttl"]["latent_dim"]
        self._u_text = u_text
        self._u_ref = u_ref
        self._vf_inputs = {i.name for i in vector_est_ort.get_inputs()}

    def sample_noisy_latent(
        self, duration: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        bsz = len(duration)
        wav_len_max = duration.max() * self.sample_rate
        wav_lengths = (duration * self.sample_rate).astype(np.int64)
        chunk_size = self.base_chunk_size * self.chunk_compress_factor
        latent_len = ((wav_len_max + chunk_size - 1) / chunk_size).astype(np.int32)
        latent_dim = self.ldim * self.chunk_compress_factor
        noisy_latent = np.random.randn(bsz, latent_dim, latent_len).astype(np.float32)
        latent_mask = get_latent_mask(
            wav_lengths, self.base_chunk_size, self.chunk_compress_factor
        )
        noisy_latent = noisy_latent * latent_mask
        return noisy_latent, latent_mask

    def _infer(
        self,
        text_list: list[str],
        lang_list: list[str],
        style: Style,
        total_step: int,
        speed: float = 1.05,
        cfg_scale: float = 3.0,
        pace_blend: float = 0.0,
        pace_dpt_ref: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert (
            len(text_list) == style.ttl.shape[0]
        ), "Number of texts must match number of style vectors"
        bsz = len(text_list)
        text_ids, text_mask = self.text_processor(text_list, lang_list)
        dur_onnx, *_ = self.dp_ort.run(
            None, {"text_ids": text_ids, "style_dp": style.dp, "text_mask": text_mask}
        )
        dur_onnx = np.asarray(dur_onnx, dtype=np.float32).reshape(-1)
        ref = float(pace_dpt_ref) if pace_dpt_ref is not None else DURATION_PACE_DPT_REF
        dur_onnx = blend_duration_pace(dur_onnx, text_mask, pace_blend, ref)
        dur_onnx = dur_onnx / max(float(speed), 1e-6)
        text_emb_onnx, *_ = self.text_enc_ort.run(
            None,
            {"text_ids": text_ids, "style_ttl": style.ttl, "text_mask": text_mask},
        )  # dur_onnx: [bsz]
        xt, latent_mask = self.sample_noisy_latent(dur_onnx)
        total_step_np = np.array([total_step] * bsz, dtype=np.float32)

        use_cfg = (
            cfg_scale != 1.0
            and self._u_text is not None
            and self._u_ref is not None
        )
        u_text_mask = np.ones((bsz, 1, 1), dtype=np.float32) if use_cfg else None

        for step in range(total_step):
            current_step = np.array([step] * bsz, dtype=np.float32)
            cond = {
                "noisy_latent": xt,
                "text_emb": text_emb_onnx,
                "style_ttl": style.ttl,
                "text_mask": text_mask,
                "latent_mask": latent_mask,
                "current_step": current_step,
                "total_step": total_step_np,
            }
            if "cfg_scale" in self._vf_inputs:
                cond["cfg_scale"] = np.array([float(cfg_scale)], dtype=np.float32)
                xt, *_ = self.vector_est_ort.run(None, cond)
            elif use_cfg:
                # SupertonicTTS §3.4: v = v_uncond + cfg_scale * (v_cond - v_uncond)
                assert self._u_text is not None and self._u_ref is not None
                v_cond, *_ = self.vector_est_ort.run(None, cond)
                u_text_b = np.broadcast_to(
                    self._u_text, (bsz, *self._u_text.shape[1:])
                ).astype(np.float32)
                u_ref_b = np.broadcast_to(
                    self._u_ref, (bsz, *self._u_ref.shape[1:])
                ).astype(np.float32)
                uncond = {
                    "noisy_latent": xt,
                    "text_emb": u_text_b,
                    "style_ttl": u_ref_b,
                    "text_mask": u_text_mask,
                    "latent_mask": latent_mask,
                    "current_step": current_step,
                    "total_step": total_step_np,
                }
                v_uncond, *_ = self.vector_est_ort.run(None, uncond)
                xt = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                xt, *_ = self.vector_est_ort.run(None, cond)
        wav, *_ = self.vocoder_ort.run(None, {"latent": xt})
        frame_len = self.base_chunk_size * self.chunk_compress_factor
        if wav.shape[-1] > 2 * frame_len:
            wav = wav[..., frame_len:-frame_len]
        if wav.ndim == 3 and wav.shape[1] == 1:
            wav = wav[:, 0, :]
        return wav, dur_onnx

    def __call__(
        self,
        text: Union[str, list[str]],
        lang: Union[str, list[str]],
        style: Style,
        total_step: int,
        speed: float = 1.0,
        cfg_scale: float = 3.0,
        silence_duration: float = 0.0,
        text_is_phonemes: bool = False,
        pace_blend: Optional[float] = None,
        pace_dpt_ref: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Synthesize speech.

        - ``text`` as ``list[str]`` → batched inference (``lang`` and ``style`` must
          match batch size; no chunking).
        - ``text`` as ``str`` → chunked single-speaker synthesis, concatenated with
          ``silence_duration`` seconds of silence between chunks.

        ``cfg_scale`` enables classifier-free guidance when uncond embeddings are
        available (loaded from ``uncond.npz`` by :func:`load_text_to_speech`) or when
        the vector estimator natively accepts a ``cfg_scale`` input.

        If ``text_is_phonemes=False`` and a :class:`TextProcessor` was wired in via
        :func:`load_text_to_speech`, text is phonemized first (renikud for Hebrew,
        espeak for Latin langs), preserving inline ``<lang>…</lang>`` spans.

        Set ``text_is_phonemes=True`` when ``text`` already contains phonemes to skip
        G2P while keeping the normal tokenizer/chunking path.

        ``pace_blend`` in ``(0, 1]`` pulls predicted duration toward a fixed
        seconds-per-text-token (``pace_dpt_ref`` or :data:`DURATION_PACE_DPT_REF`)
        so the same ``speed`` behaves more consistently across languages and in
        mixed inline-``<lang>`` text. If omitted (``None``), mixed text defaults
        to :data:`DEFAULT_MIXED_PACE_BLEND`, while single-language defaults to 0.
        """
        phonemize = not text_is_phonemes
        if isinstance(text, list):
            has_inline_lang = any(_INLINE_LANG_PAIR.search(t) is not None for t in text)
        else:
            has_inline_lang = _INLINE_LANG_PAIR.search(text) is not None
        pace_blend_eff = (
            float(pace_blend)
            if pace_blend is not None
            else (DEFAULT_MIXED_PACE_BLEND if has_inline_lang else 0.0)
        )
        if isinstance(text, list):
            assert isinstance(lang, list) and len(text) == len(lang), (
                "Batch mode requires `lang` to be a list of the same length as `text`."
            )
            if phonemize and self.g2p is not None:
                text = [self.g2p.phonemize(t, lang=l) for t, l in zip(text, lang)]
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
        max_len = 120 if lang == "ko" else 300
        text_list = chunk_text(text, max_len=max_len)
        wav_cat = None
        dur_cat = None
        for chunk in text_list:
            wav, dur_onnx = self._infer(
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
                dur_cat = dur_onnx
            else:
                silence = np.zeros(
                    (1, int(silence_duration * self.sample_rate)), dtype=np.float32
                )
                wav_cat = np.concatenate([wav_cat, silence, wav], axis=1)
                dur_cat = dur_cat + dur_onnx + silence_duration
        return wav_cat, dur_cat

    def batch(
        self,
        text_list: list[str],
        lang_list: list[str],
        style: Style,
        total_step: int,
        speed: float = 1.05,
        pace_blend: Optional[float] = None,
        pace_dpt_ref: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        has_inline_lang = any(_INLINE_LANG_PAIR.search(t) is not None for t in text_list)
        pace_blend_eff = (
            float(pace_blend)
            if pace_blend is not None
            else (DEFAULT_MIXED_PACE_BLEND if has_inline_lang else 0.0)
        )
        return self._infer(
            text_list,
            lang_list,
            style,
            total_step,
            speed,
            cfg_scale=3.0,
            pace_blend=pace_blend_eff,
            pace_dpt_ref=pace_dpt_ref,
        )


def length_to_mask(lengths: np.ndarray, max_len: Optional[int] = None) -> np.ndarray:
    """
    Convert lengths to binary mask.

    Args:
        lengths: (B,)
        max_len: int

    Returns:
        mask: (B, 1, max_len)
    """
    max_len = max_len or lengths.max()
    ids = np.arange(0, max_len)
    mask = (ids < np.expand_dims(lengths, axis=1)).astype(np.float32)
    return mask.reshape(-1, 1, max_len)


def get_latent_mask(
    wav_lengths: np.ndarray, base_chunk_size: int, chunk_compress_factor: int
) -> np.ndarray:
    latent_size = base_chunk_size * chunk_compress_factor
    latent_lengths = (wav_lengths + latent_size - 1) // latent_size
    latent_mask = length_to_mask(latent_lengths)
    return latent_mask


def load_onnx(
    onnx_path: str, opts: ort.SessionOptions, providers: list[str]
) -> ort.InferenceSession:
    return ort.InferenceSession(onnx_path, sess_options=opts, providers=providers)


def load_onnx_all(
    onnx_dir: str, opts: ort.SessionOptions, providers: list[str]
) -> tuple[
    ort.InferenceSession,
    ort.InferenceSession,
    ort.InferenceSession,
    ort.InferenceSession,
]:
    dp_onnx_path = os.path.join(onnx_dir, "duration_predictor.onnx")
    text_enc_onnx_path = os.path.join(onnx_dir, "text_encoder.onnx")
    vector_est_onnx_path = os.path.join(onnx_dir, "vector_estimator.onnx")
    vocoder_onnx_path = os.path.join(onnx_dir, "vocoder.onnx")

    dp_ort = load_onnx(dp_onnx_path, opts, providers)
    text_enc_ort = load_onnx(text_enc_onnx_path, opts, providers)
    vector_est_ort = load_onnx(vector_est_onnx_path, opts, providers)
    vocoder_ort = load_onnx(vocoder_onnx_path, opts, providers)
    return dp_ort, text_enc_ort, vector_est_ort, vocoder_ort


def load_cfgs(onnx_dir: str, config_path: str = "config/tts.json") -> dict:
    # Prefer an explicit config next to the onnx files; otherwise fall back
    # to the single repo-level config/tts.json.
    local = os.path.join(onnx_dir, "tts.json")
    cfg_path = local if os.path.exists(local) else config_path
    with open(cfg_path, "r") as f:
        return json.load(f)


def load_text_processor(onnx_dir: str = "") -> UnicodeProcessor:
    return UnicodeProcessor(os.path.join(os.path.dirname(__file__), "..", "vocab.json"))


def text_to_indices(text: str, lang: str = "he") -> list[int]:
    """Compatibility helper used by TRT: encode phoneme text with the bundled vocab."""
    global _TEXT_TO_INDICES_PROCESSOR
    if _TEXT_TO_INDICES_PROCESSOR is None:
        _TEXT_TO_INDICES_PROCESSOR = load_text_processor()
    text_ids, _ = _TEXT_TO_INDICES_PROCESSOR([text], [lang])
    return text_ids[0].astype(np.int64).tolist()


def load_text_to_speech(
    onnx_dir: str, use_gpu: bool = False, config_path: str = "config/tts.json",
) -> TextToSpeech:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # ORT over-subscribes on many-core CPUs (NUMA/SMT contention).
    # Empirically 4–8 intra-op threads is optimal for this model on CPU.
    n_threads = int(os.environ.get("ORT_NUM_THREADS", min(8, os.cpu_count() or 1)))
    opts.intra_op_num_threads = n_threads
    opts.inter_op_num_threads = 1
    if use_gpu:
        raise NotImplementedError("GPU mode is not fully tested")
    else:
        providers = ["CPUExecutionProvider"]
        print(f"Using CPU for inference (intra_op_threads={n_threads})")
    cfgs = load_cfgs(onnx_dir, config_path)
    dp_ort, text_enc_ort, vector_est_ort, vocoder_ort = load_onnx_all(
        onnx_dir, opts, providers
    )
    text_processor = load_text_processor(onnx_dir)
    g2p = TextProcessor()
    return TextToSpeech(
        cfgs, text_processor, dp_ort, text_enc_ort, vector_est_ort, vocoder_ort,
        g2p=g2p,
    )


def load_voice_style(voice_style_paths: list[str], verbose: bool = False) -> Style:
    bsz = len(voice_style_paths)

    # Read first file to get dimensions
    with open(voice_style_paths[0], "r") as f:
        first_style = json.load(f)
    ttl_dims = first_style["style_ttl"]["dims"]
    dp_dims = first_style["style_dp"]["dims"]

    # Pre-allocate arrays with full batch size
    ttl_style = np.zeros([bsz, ttl_dims[1], ttl_dims[2]], dtype=np.float32)
    dp_style = np.zeros([bsz, dp_dims[1], dp_dims[2]], dtype=np.float32)

    # Fill in the data
    for i, voice_style_path in enumerate(voice_style_paths):
        with open(voice_style_path, "r") as f:
            voice_style = json.load(f)

        ttl_data = np.array(
            voice_style["style_ttl"]["data"], dtype=np.float32
        ).flatten()
        ttl_style[i] = ttl_data.reshape(ttl_dims[1], ttl_dims[2])

        dp_data = np.array(voice_style["style_dp"]["data"], dtype=np.float32).flatten()
        dp_style[i] = dp_data.reshape(dp_dims[1], dp_dims[2])

    if verbose:
        print(f"Loaded {bsz} voice styles")
    return Style(ttl_style, dp_style)


@contextmanager
def timer(name: str):
    start = time.time()
    print(f"{name}...")
    yield
    print(f"  -> {name} completed in {time.time() - start:.2f} sec")


def sanitize_filename(text: str, max_len: int) -> str:
    """Sanitize filename by replacing non-alphanumeric characters with underscores (supports Unicode)"""
    prefix = text[:max_len]
    return re.sub(r"[^\w]", "_", prefix, flags=re.UNICODE)


def chunk_text(text: str, max_len: int = 300) -> list[str]:
    """
    Split text into chunks by paragraphs and sentences.

    Args:
        text: Input text to chunk
        max_len: Maximum length of each chunk (default: 300)

    Returns:
        List of text chunks
    """
    # Split by paragraph (two or more newlines)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]

    chunks = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        # Split by sentence boundaries (period, question mark, exclamation mark followed by space)
        # But exclude common abbreviations like Mr., Mrs., Dr., etc. and single capital letters like F.
        pattern = r"(?<!Mr\.)(?<!Mrs\.)(?<!Ms\.)(?<!Dr\.)(?<!Prof\.)(?<!Sr\.)(?<!Jr\.)(?<!Ph\.D\.)(?<!etc\.)(?<!e\.g\.)(?<!i\.e\.)(?<!vs\.)(?<!Inc\.)(?<!Ltd\.)(?<!Co\.)(?<!Corp\.)(?<!St\.)(?<!Ave\.)(?<!Blvd\.)(?<!\b[A-Z]\.)(?<=[.!?])\s+"
        sentences = re.split(pattern, paragraph)

        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= max_len:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

    return chunks
