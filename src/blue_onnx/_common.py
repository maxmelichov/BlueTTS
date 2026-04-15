import os
import re
import subprocess
from importlib import import_module
from dataclasses import dataclass
from typing import Any, List, Optional

try:
    from ._blue_vocab import LANG_CODE_ALIASES, LANG_ID, normalize_text
except ImportError:
    from _blue_vocab import LANG_CODE_ALIASES, LANG_ID, normalize_text

# Max IPA characters per synthesis forward pass (ONNX / TRT). Independent of Renikud clause splitting.
BLUE_SYNTH_MAX_CHUNK_LEN = 150


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

    def __init__(
        self,
        renikud_path: Optional[str] = None,
        *,
        renikud_max_clause_chars: int = 96,
    ):
        """renikud_max_clause_chars: only for Hebrew pre-G2P oversize handling; does not affect synthesis chunk_len."""
        self.renikud = None
        self._renikud_max_clause_chars = renikud_max_clause_chars
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
            "Download: https://huggingface.co/thewh1teagle/renikud/resolve/main/model.onnx\n"
            "Then pass renikud_path='model.onnx' (or an absolute path) to the TTS class. "
            "The `renikud-onnx` PyPI package is a project dependency."
        )

    def _espeak_phonemize(self, text: str, lang: str) -> str:
        """Raw text → IPA for non-Hebrew languages (espeak-ng)."""
        espeak_lang = self._ESPEAK_MAP.get(lang)
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
            return normalize_text(raw, lang=lang)
        except Exception as e:
            print(f"[WARN] Phonemizer backend failed for lang={lang}: {e}")

        try:
            result = subprocess.run(
                ["espeak-ng", "-q", "--ipa=1", "-v", espeak_lang, text],
                check=True,
                capture_output=True,
                text=True,
            )
            raw = result.stdout.replace("\n", " ").strip()
            return normalize_text(raw, lang=lang)
        except Exception as e:
            print(f"[WARN] espeak-ng fallback failed for lang={lang}: {e}")
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
            clauses = _split_hebrew_prephoneme(content, self._renikud_max_clause_chars)
            ipa_parts = [
                normalize_text(self.renikud.phonemize(c), lang="he")
                for c in clauses
                if c.strip()
            ]
            return re.sub(r"\s+", " ", " ".join(ipa_parts)).strip()
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
                clauses = _split_hebrew_prephoneme(text, self._renikud_max_clause_chars)
                ipa_parts = [
                    normalize_text(self.renikud.phonemize(c), lang="he")
                    for c in clauses
                    if c.strip()
                ]
                return re.sub(r"\s+", " ", " ".join(ipa_parts)).strip()

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


def _split_oversized_hebrew_clause(part: str, max_clause_chars: int) -> List[str]:
    """Only used when a single sentence is longer than ``max_clause_chars``."""
    p = part.strip()
    if not p:
        return []
    if len(p) <= max_clause_chars:
        return [p]
    # Try coarser sub-boundaries in order; recurse so we only fragment when needed.
    if re.search(r":\s", p):
        pieces = [x.strip() for x in re.split(r"(?<=:)\s+", p) if x.strip()]
        if len(pieces) > 1:
            out: List[str] = []
            for x in pieces:
                out.extend(_split_oversized_hebrew_clause(x, max_clause_chars))
            return out
    if re.search(r"[\u0590-\u05ff]-\s+[\u0590-\u05ff]", p):
        pieces = [x.strip() for x in re.split(r"(?<=[\u0590-\u05ff])-\s+", p) if x.strip()]
        if len(pieces) > 1:
            out = []
            for x in pieces:
                out.extend(_split_oversized_hebrew_clause(x, max_clause_chars))
            return out
    if re.search(r",\s", p):
        pieces = [x.strip() for x in re.split(r",\s+", p) if x.strip()]
        if len(pieces) > 1:
            out = []
            for x in pieces:
                out.extend(_split_oversized_hebrew_clause(x, max_clause_chars))
            return out
    return _hard_split_chunk(p, max_clause_chars)


def _split_hebrew_prephoneme(text: str, max_clause_chars: int = 96) -> List[str]:
    """Split raw Hebrew before Renikud G2P.

    By default only **sentence boundaries** (``.?!``); colon / hyphen / comma splits run
    only when one sentence is longer than ``max_clause_chars`` (fewer chunks, still safe
    for very long sentences).
    """
    t = text.strip()
    if not t:
        return []
    t = re.sub(r"\.+", ".", t)
    t = re.sub(r"\?+", "?", t)
    t = re.sub(r"!+", "!", t)
    t = t.replace("…", ".")
    t = re.sub(r"\s+", " ", t)

    def refine_one(s: str) -> List[str]:
        s = s.strip()
        if not s:
            return []
        out: List[str] = []
        for sent in re.split(r"(?<=[.!?])\s+", s):
            sent = sent.strip()
            if not sent:
                continue
            out.extend(_split_oversized_hebrew_clause(sent, max_clause_chars))
        return out

    clauses: List[str] = []
    for block in re.split(r"\n+", t):
        block = block.strip()
        if block:
            clauses.extend(refine_one(block))
    return clauses if clauses else [t]


def chunk_text(text: str, max_len: int = 300) -> List[str]:
    """Split text into sentence-boundary chunks no longer than max_len chars."""
    # Ensure sentence endings right before a closing tag are treated as paragraph boundaries
    # so they don't get merged with the next language's sentence.
    text = re.sub(r"([.!?])(</[a-z]{2,8}>)\s+", r"\1\2\n\n", text)

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

    # Fix language tags that span across chunks
    fixed_out = []
    active_tag = None
    for c in out:
        c = c.strip()
        if not c:
            continue
            
        if active_tag and not c.startswith(f"<{active_tag}>"):
            c = f"<{active_tag}>" + c
            
        for m in re.finditer(r"<(/)?([a-z]{2,8})>", c):
            is_close = bool(m.group(1))
            tag = m.group(2)
            if is_close:
                if active_tag == tag:
                    active_tag = None
            else:
                active_tag = tag
                
        if active_tag and not c.endswith(f"</{active_tag}>"):
            c = c + f"</{active_tag}>"
            
        fixed_out.append(c)
        
    return fixed_out or ([text.strip()] if text.strip() else [])
