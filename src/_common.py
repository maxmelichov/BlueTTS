import os
import re
import sys
from dataclasses import dataclass
from typing import Any, List, Optional

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from _blue_vocab import normalize_text  # noqa: E402


@dataclass
class Style:
    """Pre-extracted speaker style vectors (backend-agnostic container)."""
    ttl: Any            # [B, n_style, style_dim]
    dp:  Optional[Any] = None   # [B, dp_tokens, dp_dim] or None


class TextProcessor:
    """Multilingual phonemization — renikud for Hebrew, espeak for others."""

    _ESPEAK_MAP = {
        "en": "en-us", "en-us": "en-us", "de": "de", "ge": "de", "it": "it",
        "es": "es",    "fr": "fr-fr", "pt": "pt",
    }

    def __init__(self, renikud_path: Optional[str] = None):
        self.renikud = None
        if renikud_path and os.path.exists(renikud_path):
            try:
                from renikud_onnx import G2P
                self.renikud = G2P(renikud_path)
                print(f"[INFO] Loaded G2P from {renikud_path}")
            except ImportError:
                print("[WARN] renikud_onnx not installed; Hebrew raw-text G2P disabled.")

    def phonemize(self, text: str, lang: str = "he") -> str:
        is_hebrew = any('\u0590' <= c <= '\u05ff' for c in text)

        if lang == "he" or is_hebrew:
            if not is_hebrew:
                return text  # already IPA
            
            if self.renikud is not None:
                return self.renikud.phonemize(text)
            
            # fallback if renikud not loaded (e.g. not installed)
            print("[WARN] renikud not loaded, Hebrew text will not be phonemized properly.")
            return text

        espeak_lang = self._ESPEAK_MAP.get(lang)
        if espeak_lang is None:
            return text  # assume already IPA
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
                current = sentence
        if current:
            chunks.append(current.strip())
    return chunks or [text]
