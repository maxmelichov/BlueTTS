"""Phoneme-string normalisation and language tags for bundles with a ``blue_release`` section in ``tts.json``.

Those models were trained on IPA passed through exactly this normalisation (NFD, emoji strip, a few symbol
replacements, affricate ligatures, a terminal mark) and on ids wrapped in ``<lang>`` ... ``</lang>`` tags (the
characters ``<``, the language code, ``>`` and ``/`` of ``vocab.json``), sent identically to the duration predictor,
the text encoder and the vector field. Older bundles keep the ``UnicodeProcessor`` path unchanged.

Inline ``<xx>`` ... ``</xx>`` spans (written by hand, by the normalizer, or by :func:`tag_latin_runs`) get no ids of
their own: training dropped inline tags and wrapped each whole input once. A span only decides which G2P reads it
and which language's normalisation its phonemes get.
"""
import re
from unicodedata import category as _category
from unicodedata import normalize as _uni

_EMOJI = re.compile(
    "[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff\U0001f700-\U0001f77f\U0001f780-\U0001f7ff"
    "\U0001f800-\U0001f8ff\U0001f900-\U0001f9ff\U0001fa00-\U0001fa6f\U0001fa70-\U0001faff☀-⛿✀-➿"
    "\U0001f1e6-\U0001f1ff]+", flags=re.UNICODE)
_AFFRICATES = (
    (re.compile(r"t͡s|t͡s|ts(?=[^\w]|$)", re.I), "ʦ"),
    (re.compile(r"t͡ʃ|t͡ʃ|tʃ", re.I), "ʧ"),
    (re.compile(r"d͡ʒ|d͡ʒ|dʒ", re.I), "ʤ"),
    (re.compile(r"d͡z|d͡z|dz", re.I), "ʣ"),
)
_REPLACE = {
    "“": '"', "”": '"', "‘": "'", "’": "'", "´": "'", "`": "'",
    "–": "-", "‑": "-", "—": "-", "_": " ", "[": " ", "]": " ", "|": " ", "^": " ", "$": " ",
}
TERMINAL = re.compile(r"[.!?;:,'\"')\]}…。\\』】〉》›»]$")
_TAG = re.compile(r"</?\w+>")
_SPAN = re.compile(r"<(\w+)>(.*?)</\1>", re.DOTALL)
# A tag pair (kept whole) or a stray tag: never searched for Latin words.
_SPAN_OR_TAG = re.compile(r"(<(\w+)>.*?</\2>|</?\w+>)", re.DOTALL)
_PUNCT = [",", ".", "!", "?", ";", ":", "'"]
# Latin-script words inside Hebrew: letters (with Latin-1 / Extended-A/B accents), apostrophes, hyphens and dots
# inside a word ("it's", "Wi-Fi", "www.google.com"), digits glued on ("5G", "MP3", "GPT-4"), and the words and
# numbers that follow it after a space ("New York Times", "iPhone 15", "Python 3.12"), so polyg2p reads them together.
_L = "A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u024f"
_LATIN_WORD = rf"\d*[{_L}][{_L}\d]*(?:['’.\-][{_L}\d]+)*"
LATIN_RUN = re.compile(rf"{_LATIN_WORD}(?:\s+(?:{_LATIN_WORD}|\d+(?:[.,]\d+)*(?![{_L}\d])))*")
# English polyg2p reads "BlueTTS" as one word (blˈuːʦ) and "ChatGPT" as ʧˈætɡpt: an all-capitals tail after a
# lower-case letter is an acronym of its own ("Blue TTS", "Chat GPT", "Open AI", "mac OS").
_CAPS_TAIL = re.compile(r"(?<=[a-z])(?=[A-Z]{2,}(?![A-Za-z]))")


def normalize_ipa(text: str, lang: str = None, final_period: bool = True) -> str:
    """The model's view of a G2P output string. Hebrew: r->ʁ, g->ɡ and /ts/ spelled 'ts' (not 'ʦ')."""
    text = _uni("NFD", text.strip())
    text = _EMOJI.sub("", text)
    for k, v in _REPLACE.items():
        text = text.replace(k, v)
    text = re.sub(r"[♥☆♡©]", "", text)
    for p in _PUNCT:
        text = text.replace(f" {p}", p)
    for q in ['""', "''", "``"]:
        while q in text:
            text = text.replace(q, q[0])
    if lang == "he":
        text = text.replace("r", "ʁ").replace("g", "ɡ")
    for pat, lig in _AFFRICATES:
        text = pat.sub(lig, text)
    if lang == "he":
        text = text.replace("ʦ", "ts")
    text = re.sub(r"\s+", " ", text).strip()
    if final_period and not TERMINAL.search(text):
        text += "."
    return text


def tag_latin_runs(text: str, lang: str) -> str:
    """Hebrew text: wrap each run of Latin-script words in ``<en>`` ... ``</en>`` (other languages: unchanged).

    RenikudPlus passes Latin letters through as they are, so "iPhone" reached the ids as the graphemes i, P, h, o, n,
    e: lower case read as the IPA symbols they also are, capitals as symbols no training label spells with — garbled
    or dropped. As an ``<en>`` span the run goes to English polyg2p instead. Existing spans and stray tags are left
    alone, and the spaces and punctuation around a run stay in the Hebrew text where they were.
    """
    if lang != "he":
        return text
    parts = _SPAN_OR_TAG.split(text)
    # split() with two groups yields text, whole match, inner tag name, text, ...
    out = []
    for i in range(0, len(parts), 3):
        out.append(LATIN_RUN.sub(lambda m: f"<en>{_english_run(m.group(0))}</en>", parts[i]))
        if i + 1 < len(parts):
            out.append(parts[i + 1])
    return "".join(out)


def _english_run(run: str) -> str:
    # English polyg2p drops letters with accents ("café" -> kˈæf, "naïve" -> nˈæv); without them it reads the word.
    run = "".join(c for c in _uni("NFD", run) if _category(c) != "Mn")
    return _CAPS_TAIL.sub(" ", run)


def strip_outer_lang_tags(ipa: str, lang: str) -> str:
    """G2P output with ``lang``'s own tags removed and other languages' spans kept, for :func:`encode_tagged`."""
    t = _TAG.sub(lambda m: " " if m.group(0).strip("</>") == lang else m.group(0), ipa)
    return re.sub(r"\s+", " ", t).strip()


def normalize_ipa_spans(text: str, lang: str, final_period: bool = True) -> str:
    """:func:`normalize_ipa` with each other-language ``<xx>`` span normalised as ``xx`` (tags dropped).

    One pass over the whole string put the English of a Hebrew sentence through the Hebrew rules (r -> ʁ, g -> ɡ,
    'ʦ' -> 'ts'): "cats" in a Hebrew chunk was not the "cats" the model learnt from English rows.
    """
    if all(m.group(1) == lang for m in _SPAN.finditer(text)):
        return normalize_ipa(_TAG.sub("", text), lang=lang, final_period=final_period)
    parts, pos = [], 0
    for m in _SPAN.finditer(text):
        parts.append(normalize_ipa(_TAG.sub("", text[pos:m.start()]), lang=lang, final_period=False))
        parts.append(normalize_ipa(_TAG.sub("", m.group(2)), lang=m.group(1), final_period=False))
        pos = m.end()
    parts.append(normalize_ipa(_TAG.sub("", text[pos:]), lang=lang, final_period=False))
    text = " ".join(p for p in parts if p)
    for p in _PUNCT:   # the pieces were joined with spaces: "ʃalom ," -> "ʃalom,"
        text = text.replace(f" {p}", p)
    if final_period and not TERMINAL.search(text):
        text += "."
    return text


def encode_tagged(ipa: str, lang: str, char_to_id: dict, pad_id: int = 0, tag=("<", ">", "/")) -> list:
    """ids of '<lang>' + normalize_ipa_spans(ipa) + '</lang>' (unknown symbols -> pad_id, as UnicodeProcessor does)."""
    text = normalize_ipa_spans(ipa, lang)
    code = re.sub(r"[^a-z]", "", lang.lower().split("-")[0])
    o, c, s = tag
    ids = [char_to_id.get(ch, pad_id) for ch in text]
    head = [char_to_id[o]] + [char_to_id[ch] for ch in code] + [char_to_id[c]]
    tail = [char_to_id[o], char_to_id[s]] + [char_to_id[ch] for ch in code] + [char_to_id[c]]
    return head + ids + tail


def tagged_len(ipa: str, lang: str) -> int:
    """``len(encode_tagged(ipa, lang, ...))`` without a vocabulary: the ids a chunk costs, its tags included
    (inline spans counted after their own language's normalisation, as :func:`encode_tagged` does)."""
    code = re.sub(r"[^a-z]", "", lang.lower().split("-")[0])
    return len(normalize_ipa_spans(ipa, lang)) + 2 * len(code) + 5
