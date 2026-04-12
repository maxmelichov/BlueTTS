import os
import re
import sys
from dataclasses import dataclass
from typing import Any, List, Optional

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from _blue_vocab import LANG_CODE_ALIASES, LANG_ID, normalize_text  # noqa: E402


@dataclass
class Style:
    """Pre-extracted speaker style vectors (backend-agnostic container)."""
    ttl: Any            # [B, n_style, style_dim]
    dp:  Optional[Any] = None   # [B, dp_tokens, dp_dim] or None


class TextProcessor:
    """Multilingual phonemization — Renikud G2P (renikud-onnx, HF: thewh1teagle/renikud) for Hebrew; espeak for others."""

    _ESPEAK_MAP = {
        "en": "en-us", "en-us": "en-us", "de": "de", "ge": "de", "it": "it",
        "es": "es", 
    }

    # Same pairing as ``text_to_indices_multilang`` in ``_blue_vocab``: ``<lan>…</lan>`` or toggle ``<lan>…<lan>``.
    _INLINE_LANG_PAIR = re.compile(r"<(\w+)>(.*?)(?:</\1>|<\1>)", re.DOTALL)

    _RENIKUD_MODEL_HINT = (
        "wget -O model.onnx https://huggingface.co/thewh1teagle/renikud/resolve/main/model.onnx"
    )

    def __init__(self, renikud_path: Optional[str] = None):
        self.renikud = None
        if renikud_path is None and os.path.exists("model.onnx"):
            renikud_path = "model.onnx"
            
        self._renikud_path = renikud_path
        if renikud_path and os.path.exists(renikud_path):
            try:
                from renikud_onnx import G2P
                self.renikud = G2P(renikud_path)
                print(f"[INFO] Loaded Renikud G2P from {renikud_path}")
            except ImportError as e:
                raise RuntimeError(
                    "Hebrew G2P needs the `renikud-onnx` package. Install project deps: uv sync"
                ) from e

    def _hebrew_requires_renikud_error(self) -> ValueError:
        return ValueError(
            "Hebrew text requires the Renikud ONNX weights (not bundled with the wheel). "
            f"Download: https://huggingface.co/thewh1teagle/renikud/resolve/main/model.onnx\n"
            "Then pass renikud_path='model.onnx' (or an absolute path) to the TTS class. "
            "The `renikud-onnx` PyPI package is a project dependency."
        )

    def _espeak_phonemize(self, text: str, lang: str) -> str:
        """Raw text → IPA for non-Hebrew languages (espeak-ng)."""
        espeak_lang = self._ESPEAK_MAP.get(lang)
        if espeak_lang is None:
            return text
        try:
            from phonemizer.backend import EspeakBackend
            from phonemizer.separator import Separator
            backend = EspeakBackend(
                espeak_lang, preserve_punctuation=True,
                with_stress=True, language_switch="remove-flags",
            )
            raw = backend.phonemize(
                [text], separator=Separator(phone="", word=" ", syllable="")
            )[0]
            return normalize_text(raw, lang=lang)
        except Exception as e:
            print(f"[WARN] Phonemization failed for lang={lang}: {e}")
            return text

    def _phonemize_segment(self, content: str, lang: str) -> str:
        """Phonemize one span; ``lang`` is the declared language for that span (tag or base)."""
        content = content.strip()
        if not content:
            return ""
        lang = LANG_CODE_ALIASES.get(lang, lang)
        if lang not in LANG_ID:
            lang = "he"
        has_hebrew = any("\u0590" <= c <= "\u05ff" for c in content)
        if has_hebrew:
            if self.renikud is None:
                raise self._hebrew_requires_renikud_error()
            return normalize_text(self.renikud.phonemize(content), lang="he")
        if lang == "he":
            return normalize_text(content, lang="he")
        return self._espeak_phonemize(content, lang)

    def _phonemize_mixed(self, text: str, base_lang: str) -> str:
        """Phonemize mixed raw text, preserving ``<lan>ipa</lan>`` for the tokenizer."""
        base_lang = LANG_CODE_ALIASES.get(base_lang, base_lang)
        if base_lang not in LANG_ID:
            raise ValueError(f"Unknown base_lang {base_lang!r}. Available: {list(LANG_ID.keys())}.")
        pieces: List[str] = []
        last_end = 0
        for m in self._INLINE_LANG_PAIR.finditer(text):
            if m.start() > last_end:
                chunk = text[last_end:m.start()]
                p = self._phonemize_segment(chunk, base_lang)
                if p:
                    pieces.append(p)
            open_tag = m.group(1)
            seg_lang = LANG_CODE_ALIASES.get(open_tag, open_tag)
            if seg_lang not in LANG_ID:
                seg_lang = base_lang
            inner_ipa = self._phonemize_segment(m.group(2), seg_lang)
            pieces.append(f"<{open_tag}>{inner_ipa}</{open_tag}>")
            last_end = m.end()
        if last_end < len(text):
            p = self._phonemize_segment(text[last_end:], base_lang)
            if p:
                pieces.append(p)
        return re.sub(r"\s+", " ", " ".join(pieces)).strip()

    def phonemize(self, text: str, lang: str = "he") -> str:
        if self._INLINE_LANG_PAIR.search(text):
            return self._phonemize_mixed(text, base_lang=lang)

        is_hebrew = any('\u0590' <= c <= '\u05ff' for c in text)

        if lang == "he" or is_hebrew:
            if not is_hebrew:
                return normalize_text(text, lang="he")

            if self.renikud is not None:
                ipa = self.renikud.phonemize(text)
                return normalize_text(ipa, lang="he")

            raise self._hebrew_requires_renikud_error()

        return self._espeak_phonemize(text, lang)


def _hard_split_chunk(s: str, max_len: int) -> List[str]:
    """Split ``s`` into segments of at most ``max_len`` chars (prefer last space)."""
    s = s.strip()
    if not s or max_len <= 0:
        return [s] if s else []
    if len(s) <= max_len:
        return [s]
    out: List[str] = []
    start = 0
    n = len(s)
    while start < n:
        end = min(start + max_len, n)
        if end < n:
            window = s[start:end]
            cut = window.rfind(" ")
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
    """Split text into sentence-boundary chunks no longer than max_len chars."""
    pattern = (
        r"(?<!Mr\.)(?<!Mrs\.)(?<!Ms\.)(?<!Dr\.)(?<!Prof\.)(?<!Sr\.)(?<!Jr\.)"
        r"(?<!Ph\.D\.)(?<!etc\.)(?<!e\.g\.)(?<!i\.e\.)(?<!vs\.)(?<!Inc\.)"
        r"(?<!Ltd\.)(?<!Co\.)(?<!Corp\.)(?<!St\.)(?<!Ave\.)(?<!Blvd\.)"
        r"(?<!\b[A-Z]\.)(?<=[.!?])\s+"
    )
    chunks: List[str] = []
    for paragraph in re.split(r"\n\s*\n+", text.strip()):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        current = ""
        for sentence in re.split(pattern, paragraph):
            if len(current) + len(sentence) + 1 <= max_len:
                current += (" " if current else "") + sentence
            else:
                if current:
                    chunks.append(current.strip())
                if len(sentence) > max_len:
                    chunks.extend(_hard_split_chunk(sentence, max_len))
                    current = ""
                else:
                    current = sentence
        if current:
            chunks.append(current.strip())
    base = chunks if chunks else ([text.strip()] if text.strip() else [])
    # TensorRT engines cap T_text; long IPA without ".!?" must never stay in one oversized chunk.
    out: List[str] = []
    for c in base:
        out.extend(_hard_split_chunk(c, max_len))
    return out or ([text.strip()] if text.strip() else [])
