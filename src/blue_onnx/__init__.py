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

from .text_norm import (  # noqa: F401  (re-exported as the package's public API)
    SLOW_MARK_OPEN,
    SLOW_PACE_BLEND,
    SLOW_PACE_DPT_REF,
    SLOW_SILENCE,
    SLOW_SPEED_SCALE,
    canonical_lang,
    prepare_text_for_synthesis,
    split_slow_segments,
    strip_emoji,
    strip_slow_markers,
)

AVAILABLE_LANGS = ["en", "es", "de", "it", "he"]
BLUE_SYNTH_MAX_CHUNK_LEN = 300
# When ``pace_blend > 0``, duration is nudged toward this many seconds of audio per
# text input token, so the same ``speed`` value tracks more closely across languages.
DURATION_PACE_DPT_REF = 0.0625
# Default blend automatically applied when inline ``<lang>...`` spans are present.
DEFAULT_MIXED_PACE_BLEND = 0.25
# Default classifier-free guidance scale (vector field).
DEFAULT_CFG_SCALE = 4.0


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


def strip_lang_tags_from_phoneme_string(s: str) -> str:
    """Remove ``<lang>…</lang>`` markers from a phoneme string.

    G2P wraps spans with these tags; :meth:`UnicodeProcessor._encode` strips them
    before tokenization anyway. Removing them *before* :func:`chunk_text` keeps
    sentence splits from tearing tag pairs apart (orphan ``<en>`` / ``</he>``),
    which could confuse preprocessing or leave odd artifacts at boundaries.
    """
    t = _LANG_TAG_RE.sub("", s)
    return re.sub(r"\s+", " ", t).strip()


class TextProcessor:
    """Renikud for Hebrew; espeak for everything else. Preserves ``<lang>…</lang>`` spans.

    Output is phonemized text re-wrapped in ``<lang>…</lang>`` tags so that
    :class:`UnicodeProcessor` can route segments by language downstream.
    """

    def __init__(
        self,
        renikud_path: Optional[str] = None,
        speaker: int = 0,
        target_speaker: int = 0,
    ):
        """``renikud_path`` is optional: RenikudPlus fetches its own weights.

        The G2P is built on first Hebrew segment, so Latin-only synthesis never
        pulls the Hebrew weights. ``speaker`` / ``target_speaker`` are RenikudPlus
        speaker hints applied to every Hebrew segment (0 unknown, 1 male, 2 female).
        """
        self.renikud = None
        self.renikud_path = renikud_path
        self.speaker = speaker
        self.target_speaker = target_speaker
        if renikud_path and not os.path.exists(renikud_path):
            raise FileNotFoundError(f"Renikud weights not found: {renikud_path}")

    def _load_renikud(self):
        """Build the RenikudPlus G2P on demand (downloads weights if needed)."""
        if self.renikud is None:
            try:
                from renikud_onnx import G2P
            except ImportError as e:
                raise RuntimeError(
                    "Hebrew G2P needs `renikud-plus`. Install: `uv sync`."
                ) from e
            self.renikud = G2P(self.renikud_path)
            print(
                "[INFO] Loaded RenikudPlus G2P from "
                f"{self.renikud_path or 'auto-download'}"
            )
        return self.renikud

    # Cache EspeakBackend instances per language: the espeak-ng ctypes binding
    # leaks per backend construction and each init costs ~600 ms, so reuse them.
    _ESPEAK_BACKENDS: dict = {}

    def _espeak(self, text: str, lang: str) -> str:
        espeak_lang = _ESPEAK_MAP.get(lang)
        if espeak_lang is None:
            return text
        try:
            Separator = import_module("phonemizer.separator").Separator
            backend = TextProcessor._ESPEAK_BACKENDS.get(espeak_lang)
            if backend is None:
                import espeakng_loader
                EspeakBackend = import_module("phonemizer.backend").EspeakBackend
                EspeakWrapper = import_module("phonemizer.backend.espeak.wrapper").EspeakWrapper
                EspeakWrapper.set_library(espeakng_loader.get_library_path())
                if hasattr(EspeakWrapper, "set_data_path"):
                    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
                backend = EspeakBackend(
                    espeak_lang, preserve_punctuation=True,
                    with_stress=True, language_switch="remove-flags",
                )
                TextProcessor._ESPEAK_BACKENDS[espeak_lang] = backend
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
        # Before phonemizing, not after: espeak reads emoji aloud ("Nice 🎉 party"
        # → "nice party popper party"), so the tokenizer's own emoji filter never
        # gets to see them. Applies whether or not the caller normalized first.
        content = strip_emoji(content)
        if not content:
            return ""
        has_hebrew = any("\u0590" <= c <= "\u05ff" for c in content)
        if has_hebrew or lang == "he":
            if not has_hebrew:
                return content
            return self._load_renikud().phonemize(
                content, speaker=self.speaker, target_speaker=self.target_speaker
            )
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
        # Drop inline <lang>…</lang> spans first: they are routing metadata, never
        # speech, and the "/" -> " " replacement below would mangle a closing tag
        # into "< he>", which then survives as literal "he" characters plus pad
        # ids in the encoded sequence.
        text = strip_lang_tags_from_phoneme_string(text)
        # TODO: Need advanced normalizer for better performance
        text = normalize("NFKD", text)

        # Last-resort net: G2P already strips these, but ``text_is_phonemes=True``
        # skips that path entirely.
        text = strip_emoji(text)

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
        # Tags were stripped above, so this always wraps exactly once.
        return f"<{lang}>" + text + f"</{lang}>"

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
        latent_stats: Optional[tuple[np.ndarray, np.ndarray, float]] = None,
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
        voc_shape = vocoder_ort.get_inputs()[0].shape
        self._vocoder_in_channels = voc_shape[1] if isinstance(voc_shape[1], int) else None
        self._latent_mean, self._latent_std, self._normalizer_scale = (
            latent_stats if latent_stats is not None else (None, None, 1.0)
        )

    def _prepare_vocoder_latent(self, xt: np.ndarray) -> np.ndarray:
        """Feed the vocoder whatever shape its graph declares.

        Two export vintages ship in the wild. The ``VocoderWithStats`` wrapper in
        ``exports/export_onnx.py`` bakes the de-normalization and the
        ``latent_dim*factor -> latent_dim`` time shuffle into the graph, so it takes
        the flow output as-is. Older/leaner bundles export the bare decoder, which
        expects the already de-normalized and decompressed ``latent_dim`` tensor and
        ships ``mean``/``std`` alongside in ``stats.npz``. Detect which from the
        graph's own input channel count rather than from the directory name.
        """
        if self._vocoder_in_channels != self.ldim:
            return xt
        if self._latent_mean is None or self._latent_std is None:
            raise RuntimeError(
                f"This vocoder takes {self.ldim}-channel latents, so it needs "
                "mean/std from a stats.npz next to the graphs — none was found."
            )
        z = (xt / self._normalizer_scale) * self._latent_std + self._latent_mean
        bsz, _, t = z.shape
        # Inverse of the channel-major compression: (B, ldim*f, T) -> (B, ldim, T*f).
        z = z.reshape(bsz, self.ldim, self.chunk_compress_factor, t)
        return z.transpose(0, 1, 3, 2).reshape(bsz, self.ldim, t * self.chunk_compress_factor)

    def sample_noisy_latent(
        self, duration: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        bsz = len(duration)
        wav_lengths = (duration * self.sample_rate).astype(np.int64)
        # One expression for the width, then handed to the mask explicitly: the
        # noise width used to come from the float duration and the mask width from
        # the truncated sample count, two computations that had to agree for
        # `noisy_latent * latent_mask` to broadcast.
        latent_len = latent_frames_for_duration(
            float(duration.max()), self.sample_rate,
            self.base_chunk_size, self.chunk_compress_factor,
        )
        latent_dim = self.ldim * self.chunk_compress_factor
        noisy_latent = np.random.randn(bsz, latent_dim, latent_len).astype(np.float32)
        latent_mask = get_latent_mask(
            wav_lengths, self.base_chunk_size, self.chunk_compress_factor,
            max_len=latent_len,
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
        cfg_scale: float = DEFAULT_CFG_SCALE,
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
        wav, *_ = self.vocoder_ort.run(None, {"latent": self._prepare_vocoder_latent(xt)})
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
        cfg_scale: float = DEFAULT_CFG_SCALE,
        silence_duration: float = 0.0,
        text_is_phonemes: bool = False,
        pace_blend: Optional[float] = None,
        pace_dpt_ref: Optional[float] = None,
        normalize_text: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Synthesize speech.

        - ``text`` as ``list[str]`` → batched inference (``lang`` and ``style`` must
          match batch size; no chunking).
        - ``text`` as ``str`` → chunked single-speaker synthesis, concatenated with
          ``silence_duration`` seconds of silence between chunks.

        ``normalize_text=True`` runs :func:`prepare_text_for_synthesis` first, so
        digits, dates, clock times, prices, emails and ticket codes become words
        instead of reaching G2P as symbols. Spans it marks slow (spelled codes,
        phone numbers, dates, times) are synthesized separately at
        :data:`SLOW_SPEED_SCALE` — pass ``text`` through the normalizer yourself
        to see exactly what will be spoken.

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
        if normalize_text:
            if text_is_phonemes:
                raise ValueError(
                    "normalize_text and text_is_phonemes are mutually exclusive: "
                    "the normalizer rewrites words, not phonemes."
                )
            if isinstance(text, list):
                langs = lang if isinstance(lang, list) else [lang] * len(text)
                # Batched rows are synthesized in one pass, so there is nothing to
                # schedule slow spans against — keep the text plain.
                text = [
                    prepare_text_for_synthesis(t, lang=lg, mark_slow=False)
                    for t, lg in zip(text, langs)
                ]
            else:
                text = prepare_text_for_synthesis(text, lang=lang)
        # Computed after normalization: it can introduce <en> spans (emails,
        # spelled codes, loanwords) that should count as mixed-language text.
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
                text = [
                    self.g2p.phonemize(t, lang=lang_code)
                    for t, lang_code in zip(text, lang)
                ]
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
        max_len = 120 if lang == "ko" else 300
        # Without slow markers this is a single ``(text, False)`` segment, i.e. the
        # plain phonemize → chunk → synthesize path.
        segments = split_slow_segments(text)
        wav_cat = None
        dur_cat = None
        prev_is_slow = False
        for seg_text, is_slow in segments:
            seg_speed = speed * SLOW_SPEED_SCALE if is_slow else speed
            seg_pace_blend = SLOW_PACE_BLEND if is_slow else pace_blend_eff
            seg_pace_dpt = SLOW_PACE_DPT_REF if is_slow else pace_dpt_ref
            if phonemize and self.g2p is not None:
                seg_text = self.g2p.phonemize(seg_text, lang=lang)
            seg_text = strip_lang_tags_from_phoneme_string(seg_text)
            for chunk in chunk_text(seg_text, max_len=max_len):
                wav, dur_onnx = self._infer(
                    [chunk],
                    [lang],
                    style,
                    total_step,
                    seg_speed,
                    cfg_scale,
                    pace_blend=seg_pace_blend,
                    pace_dpt_ref=seg_pace_dpt,
                )
                if wav_cat is None:
                    wav_cat = wav
                    dur_cat = dur_onnx
                else:
                    gap = SLOW_SILENCE if (is_slow or prev_is_slow) else silence_duration
                    silence = np.zeros(
                        (1, int(gap * self.sample_rate)), dtype=np.float32
                    )
                    wav_cat = np.concatenate([wav_cat, silence, wav], axis=1)
                    dur_cat = dur_cat + dur_onnx + gap
            prev_is_slow = is_slow
        if wav_cat is None:  # nothing speakable survived normalization
            return np.zeros((1, 0), dtype=np.float32), np.zeros((1,), dtype=np.float32)
        return wav_cat, dur_cat

    def batch(
        self,
        text_list: list[str],
        lang_list: list[str],
        style: Style,
        total_step: int,
        speed: float = 1.05,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        pace_blend: Optional[float] = None,
        pace_dpt_ref: Optional[float] = None,
        text_is_phonemes: bool = False,
        normalize_text: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batched synthesis for a list of strings (no sentence chunking).

        Matches list-mode :meth:`__call__`: runs G2P when ``text_is_phonemes`` is
        ``False`` and a processor was wired via :func:`load_text_to_speech`, then
        strips inline ``<lang>…</lang>`` markers before encoding.
        ``normalize_text=True`` runs :func:`prepare_text_for_synthesis` per row;
        rows are one pass each, so slow spans are not marked.

        Every row of the returned waveform is as long as the longest item in the
        batch; use the returned per-item durations to trim. Requires graphs
        exported with a dynamic batch axis (``exports/export_onnx.py``); older
        batch-1 bundles raise an ONNX Runtime shape error.
        """
        assert len(text_list) == len(lang_list), (
            "`text_list` and `lang_list` must have the same length."
        )
        if normalize_text:
            if text_is_phonemes:
                raise ValueError(
                    "normalize_text and text_is_phonemes are mutually exclusive: "
                    "the normalizer rewrites words, not phonemes."
                )
            text_list = [
                prepare_text_for_synthesis(t, lang=lang_code, mark_slow=False)
                for t, lang_code in zip(text_list, lang_list)
            ]
        has_inline_lang = any(_INLINE_LANG_PAIR.search(t) is not None for t in text_list)
        pace_blend_eff = (
            float(pace_blend)
            if pace_blend is not None
            else (DEFAULT_MIXED_PACE_BLEND if has_inline_lang else 0.0)
        )
        phonemize = not text_is_phonemes
        if phonemize and self.g2p is not None:
            text_list = [
                self.g2p.phonemize(t, lang=lang_code)
                for t, lang_code in zip(text_list, lang_list)
            ]
        text_list = [strip_lang_tags_from_phoneme_string(t) for t in text_list]
        return self._infer(
            text_list,
            lang_list,
            style,
            total_step,
            speed,
            cfg_scale,
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
    # `max_len or lengths.max()` treated an explicit 0 as "not given", and the
    # `-1` batch dim is unresolvable when max_len is 0 (a zero-size array has no
    # unique batch size), so a zero-length row used to raise from reshape.
    max_len = int(lengths.max()) if max_len is None else int(max_len)
    ids = np.arange(0, max_len)
    mask = (ids < np.expand_dims(lengths, axis=1)).astype(np.float32)
    return mask.reshape(mask.shape[0], 1, max_len)


def latent_frames_for_duration(
    seconds: float, sample_rate: int, base_chunk_size: int, chunk_compress_factor: int
) -> int:
    """Latent frames needed to hold ``seconds`` of audio (integer ceil-div).

    The flow-matching latent runs at ``base_chunk_size * chunk_compress_factor``
    samples per frame. Shared with the PyTorch and TensorRT mirrors; TRT in
    particular has to derive its ``T_lat`` from the duration predictor's
    **seconds** output using exactly this arithmetic — reading that scalar as a
    frame count instead pins every utterance to the clamp floor.

    Negative input clamps to 0 frames rather than returning a negative width,
    which would surface as an unreadable allocation error downstream.
    """
    frame_len = base_chunk_size * chunk_compress_factor
    return max(0, (int(seconds * sample_rate) + frame_len - 1) // frame_len)


def get_latent_mask(
    wav_lengths: np.ndarray, base_chunk_size: int, chunk_compress_factor: int,
    max_len: Optional[int] = None,
) -> np.ndarray:
    latent_size = base_chunk_size * chunk_compress_factor
    latent_lengths = (wav_lengths + latent_size - 1) // latent_size
    latent_mask = length_to_mask(latent_lengths, max_len=max_len)
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
    # `_infer` feeds the duration head a `style_dp` vector. Some bundles put that
    # variant in `duration_predictor_style.onnx` and reserve `duration_predictor.onnx`
    # for the reference-audio (`z_ref`) form, which takes different inputs entirely.
    dp_onnx_path = os.path.join(onnx_dir, "duration_predictor_style.onnx")
    if not os.path.exists(dp_onnx_path):
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
    """Load the bundled vocabulary. ``onnx_dir`` is ignored — a bundle's own
    ``vocab.json`` is never used, because the ids have to match the checkpoint the
    graphs were exported from.

    The file lives *inside* the package: at ``src/vocab.json`` it resolved to
    ``site-packages/vocab.json`` in an installed wheel and was not packaged at all,
    so every entry point died with FileNotFoundError on `pip install blue-onnx`.

    A bundle may ship its own ``vocab.json``, and that one wins when present: the
    ids have to match the checkpoint the graphs were exported from, and newer
    bundles extend the alphabet. The packaged copy is the fallback.
    """
    bundled = os.path.join(onnx_dir, "vocab.json") if onnx_dir else ""
    if bundled and os.path.exists(bundled):
        return UnicodeProcessor(bundled)
    return UnicodeProcessor(os.path.join(os.path.dirname(__file__), "vocab.json"))


def text_to_indices(text: str, lang: str = "he") -> list[int]:
    """Compatibility helper used by TRT: encode phoneme text with the bundled vocab."""
    global _TEXT_TO_INDICES_PROCESSOR
    if _TEXT_TO_INDICES_PROCESSOR is None:
        _TEXT_TO_INDICES_PROCESSOR = load_text_processor()
    text_ids, _ = _TEXT_TO_INDICES_PROCESSOR([text], [lang])
    return text_ids[0].astype(np.int64).tolist()


def load_uncond(onnx_dir: str) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load CFG null embeddings from ``uncond.npz`` next to the graphs.

    Only needed for bundles whose ``vector_estimator.onnx`` has no ``cfg_scale``
    input (older exports that did not bake ``u_text``/``u_ref`` into the graph);
    without them ``cfg_scale`` has no effect.
    """
    path = os.path.join(onnx_dir, "uncond.npz")
    if not os.path.exists(path):
        return None, None
    with np.load(path) as z:
        u_text = z["u_text"].astype(np.float32) if "u_text" in z else None
        u_ref = z["u_ref"].astype(np.float32) if "u_ref" in z else None
    return u_text, u_ref


def load_latent_stats(
    onnx_dir: str,
) -> Optional[tuple[np.ndarray, np.ndarray, float]]:
    """Load ``mean``/``std``/``normalizer_scale`` from ``stats.npz`` if present.

    Only bundles whose vocoder graph expects the de-normalized latent ship this;
    the ``VocoderWithStats`` exports bake the same numbers into the graph.
    """
    path = os.path.join(onnx_dir, "stats.npz")
    if not os.path.exists(path):
        return None
    with np.load(path) as z:
        if "mean" not in z or "std" not in z:
            return None
        mean = z["mean"].astype(np.float32).reshape(1, -1, 1)
        std = z["std"].astype(np.float32).reshape(1, -1, 1)
        scale = float(np.asarray(z["normalizer_scale"]).reshape(-1)[0]) if "normalizer_scale" in z else 1.0
    return mean, std, (scale or 1.0)


def load_text_to_speech(
    onnx_dir: str, use_gpu: bool = False, config_path: str = "config/tts.json",
    renikud_path: Optional[str] = None,
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
    g2p = TextProcessor(renikud_path)
    u_text, u_ref = load_uncond(onnx_dir)
    if u_text is None and "cfg_scale" not in {i.name for i in vector_est_ort.get_inputs()}:
        print("[WARN] no cfg_scale graph input and no uncond.npz — cfg_scale will be ignored.")
    return TextToSpeech(
        cfgs, text_processor, dp_ort, text_enc_ort, vector_est_ort, vocoder_ort,
        g2p=g2p, u_text=u_text, u_ref=u_ref,
        latent_stats=load_latent_stats(onnx_dir),
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


def limit_peak(audio: np.ndarray, peak_limit: float = 0.95) -> np.ndarray:
    """Scale ``audio`` down so ``max(|audio|) <= peak_limit``; quieter audio is
    returned untouched.

    The vocoder occasionally overshoots ±1.0 (peaks of ~1.1-1.25 have been
    observed), and both ``soundfile.write`` to a PCM WAV and most players clip
    there, so scale once instead of distorting. This only ever attenuates — it
    never boosts quiet output, which would change perceived loudness.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0 or not np.isfinite(audio).all():
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak <= peak_limit or peak < 1e-9:
        return audio
    return (audio * (peak_limit / peak)).astype(np.float32)


class BlueTTS:
    """One-call TTS front end: graphs + voice style in, ``(samples, sample_rate)`` out.

    ``style_json`` takes a single voice JSON (see ``voices/``) or several, in which
    case the styles are averaged into one speaker. For per-text styles, batched
    synthesis or phoneme-level control, use :class:`TextToSpeech` directly via
    :func:`load_text_to_speech`.
    """

    def __init__(
        self,
        onnx_dir: str = "onnx_models",
        style_json: Union[str, list[str]] = "voices/noa.json",
        renikud_path: Optional[str] = None,
        config_path: str = "config/tts.json",
    ):
        self.tts = load_text_to_speech(
            onnx_dir, config_path=config_path, renikud_path=renikud_path
        )
        paths = [style_json] if isinstance(style_json, str) else list(style_json)
        style = load_voice_style(paths)
        if style.ttl.shape[0] > 1:
            style = Style(
                style.ttl.mean(axis=0, keepdims=True),
                style.dp.mean(axis=0, keepdims=True),
            )
        self.style = style

    @property
    def sample_rate(self) -> int:
        return self.tts.sample_rate

    def synthesize(
        self,
        text: str,
        lang: str = "he",
        total_step: int = 5,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        speed: float = 1.0,
        silence_duration: float = 0.0,
        text_is_phonemes: bool = False,
        pace_blend: Optional[float] = None,
        pace_dpt_ref: Optional[float] = None,
        peak_limit: Optional[float] = 0.95,
        normalize_text: bool = True,
    ) -> tuple[np.ndarray, int]:
        """Synthesize ``text`` and return mono float32 samples plus the sample rate.

        Accepts inline ``<lang>…</lang>`` spans and splits long input into chunks
        joined by ``silence_duration`` seconds of silence, as
        :meth:`TextToSpeech.__call__` does.

        Text is normalized first (``normalize_text=True``), so numbers, dates,
        times, prices, emails and ticket codes are spoken as words — this front
        end takes arbitrary text, not curated strings. Pass
        ``normalize_text=False`` to synthesize exactly what you wrote.

        Output is scaled down to ``peak_limit`` when it overshoots, so writing it
        straight to a PCM WAV never clips (see :func:`limit_peak`). Pass
        ``peak_limit=None`` for the raw vocoder output.
        """
        audio, _ = self.tts(
            text,
            lang=lang,
            style=self.style,
            total_step=total_step,
            speed=speed,
            cfg_scale=cfg_scale,
            silence_duration=silence_duration,
            text_is_phonemes=text_is_phonemes,
            pace_blend=pace_blend,
            pace_dpt_ref=pace_dpt_ref,
            normalize_text=normalize_text and not text_is_phonemes,
        )
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio[0]
        if peak_limit is not None:
            audio = limit_peak(audio, peak_limit)
        return audio, self.sample_rate


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
