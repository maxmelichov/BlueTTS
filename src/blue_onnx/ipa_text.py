"""Phoneme-string normalisation and language tags for bundles with a ``blue_release`` section in ``tts.json``.

Those models were trained on IPA passed through exactly this normalisation (NFD, emoji strip, a few symbol
replacements, affricate ligatures, a terminal mark) and on ids wrapped in ``<lang>`` ... ``</lang>`` tags (the
characters ``<``, the language code, ``>`` and ``/`` of ``vocab.json``), sent identically to the duration predictor,
the text encoder and the vector field. Older bundles keep the ``UnicodeProcessor`` path unchanged.
"""
import re
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


def normalize_ipa(text: str, lang: str = None, final_period: bool = True) -> str:
    """The training labels' view of a G2P output string. Hebrew: r->ʁ, g->ɡ and /ts/ spelled 'ts' (not 'ʦ')."""
    text = _uni("NFD", text.strip())
    text = _EMOJI.sub("", text)
    for k, v in _REPLACE.items():
        text = text.replace(k, v)
    text = re.sub(r"[♥☆♡©]", "", text)
    for p in [",", ".", "!", "?", ";", ":", "'"]:
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


def encode_tagged(ipa: str, lang: str, char_to_id: dict, pad_id: int = 0, tag=("<", ">", "/")) -> list:
    """ids of '<lang>' + normalize_ipa(ipa) + '</lang>' (unknown symbols -> pad_id, as UnicodeProcessor does)."""
    text = normalize_ipa(_TAG.sub("", ipa), lang=lang)
    code = re.sub(r"[^a-z]", "", lang.lower().split("-")[0])
    o, c, s = tag
    ids = [char_to_id.get(ch, pad_id) for ch in text]
    head = [char_to_id[o]] + [char_to_id[ch] for ch in code] + [char_to_id[c]]
    tail = [char_to_id[o], char_to_id[s]] + [char_to_id[ch] for ch in code] + [char_to_id[c]]
    return head + ids + tail
