"""
Universal IPA vocabulary for TTS, built on the piper phoneme set.

Layout:
  0..156   = Piper core symbols
  157..243 = Extended symbols (uppercase, additional IPA, punctuation)
  244..255 = Unused padding to hit 256

VOCAB_SIZE = 256 (fixed, highly optimized power of 2)

Language tags have been intentionally removed to force the model 
to learn a universal acoustic space via narrow IPA.
"""

import re
from unicodedata import normalize as uni_normalize

# ============================================================
# Piper phoneme map (exact IDs from rhasspy/piper-checkpoints)
# ============================================================
_PIPER_MAP: dict[str, int] = {
    "_": 0, "^": 1, "$": 2, " ": 3, "!": 4, "'": 5, "(": 6, ")": 7, ",": 8, "-": 9, ".": 10,
    ":": 11, ";": 12, "?": 13, "a": 14, "b": 15, "c": 16, "d": 17, "e": 18, "f": 19,
    "h": 20, "i": 21, "j": 22, "k": 23, "l": 24, "m": 25, "n": 26, "o": 27, "p": 28, "q": 29, 
    "r": 30, "s": 31, "t": 32, "u": 33, "v": 34, "w": 35, "x": 36, "y": 37, "z": 38,
    "æ": 39, "ç": 40, "ð": 41, "ø": 42, "ħ": 43, "ŋ": 44, "œ": 45, "ǀ": 46, "ǁ": 47, "ǂ": 48, "ǃ": 49,
    "ɐ": 50, "ɑ": 51, "ɒ": 52, "ɓ": 53, "ɔ": 54, "ɕ": 55, "ɖ": 56, "ɗ": 57, "ɘ": 58, "ə": 59, 
    "ɚ": 60, "ɛ": 61, "ɜ": 62, "ɞ": 63, "ɟ": 64, "ɠ": 65, "ɡ": 66, "ɢ": 67, "ɣ": 68, "ɤ": 69, 
    "ɥ": 70, "ɦ": 71, "ɧ": 72, "ɨ": 73, "ɪ": 74, "ɫ": 75, "ɬ": 76, "ɭ": 77, "ɮ": 78, "ɯ": 79,
    "ɰ": 80, "ɱ": 81, "ɲ": 82, "ɳ": 83, "ɴ": 84, "ɵ": 85, "ɶ": 86, "ɸ": 87, "ɹ": 88, "ɺ": 89, 
    "ɻ": 90, "ɽ": 91, "ɾ": 92, "ʀ": 93, "ʁ": 94, "ʂ": 95, "ʃ": 96, "ʄ": 97, "ʈ": 98, "ʉ": 99, 
    "ʊ": 100, "ʋ": 101, "ʌ": 102, "ʍ": 103, "ʎ": 104, "ʏ": 105, "ʐ": 106, "ʑ": 107, "ʒ": 108, "ʔ": 109,
    "ʕ": 110, "ʘ": 111, "ʙ": 112, "ʛ": 113, "ʜ": 114, "ʝ": 115, "ʟ": 116, "ʡ": 117, "ʢ": 118, "ʲ": 119,
    "ˈ": 120, "ˌ": 121, "ː": 122, "ˑ": 123, "˞": 124, "β": 125, "θ": 126, "χ": 127, "ᵻ": 128, "ⱱ": 129,
    "0": 130, "1": 131, "2": 132, "3": 133, "4": 134, "5": 135, "6": 136, "7": 137, "8": 138, "9": 139,
    "\u0327": 140, "\u0303": 141, "\u032A": 142, "\u032F": 143, "\u0329": 144, "ʰ": 145, "ˤ": 146, 
    "ε": 147, "↓": 148, "#": 149, '"': 150, "↑": 151, "\u033A": 152, "\u033B": 153, "g": 154, "ʦ": 155, "X": 156,
}

# ============================================================
# Extended symbol map (indices 157..243)
# ============================================================
_EXTENDED_MAP: dict[str, int] = {
    "A": 157, "B": 158, "C": 159, "D": 160, "E": 161, "F": 162, "G": 163, "H": 164, "I": 165, 
    "J": 166, "K": 167, "L": 168, "M": 169, "N": 170, "O": 171, "P": 172, "Q": 173, "R": 174, 
    "S": 175, "T": 176, "U": 177, "V": 178, "W": 179, "Y": 180, "Z": 181,
    "ʤ": 182, "ɝ": 183, "ʧ": 184, "ʼ": 185, "ʴ": 186, "ʱ": 187, "ʷ": 188, "ˠ": 189, "→": 190, 
    "↗": 191, "↘": 192, "¡": 193, "¿": 194, "…": 195, "«": 196, "»": 197, "*": 198, "~": 199, 
    "/": 200, "\\": 201, "&": 202, "\u0361": 203, "\u035C": 204, "\u0325": 205, "\u032C": 206, 
    "\u0339": 207, "\u031C": 208, "\u031D": 209, "\u031E": 210, "\u031F": 211, "\u0320": 212, 
    "\u0330": 213, "\u0334": 214, "\u031A": 215, "\u0318": 216, "\u0319": 217, "\u0348": 218, 
    "\u0306": 219, "\u0308": 220, "\u031B": 221, "\u0324": 222, "\u033C": 223, "\u02C0": 224, 
    "\u02C1": 225, "\u02BE": 226, "\u02BF": 227, "\u02BB": 228, "\u02C9": 229, "\u02CA": 230, 
    "\u02CB": 231, "\u02C6": 232, "\u02E5": 233, "\u02E6": 234, "\u02E7": 235, "\u02E8": 236, 
    "\u02E9": 237, "\u0300": 238, "\u0301": 239, "\u0302": 240, "\u0304": 241, "\u030C": 242, "\u0307": 243,
}

# ============================================================
# Core Config
# ============================================================
VOCAB_SIZE = 256  # Down from 384. 256 is optimal for embedding sizes.
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2

CHAR_TO_ID: dict[str, int] = {**_PIPER_MAP, **_EXTENDED_MAP}
ID_TO_CHAR: dict[int, str] = {v: k for k, v in CHAR_TO_ID.items()}
VOCAB_LIST: list[str] = list(CHAR_TO_ID.keys())

# ============================================================
# Public API
# ============================================================

def text_to_indices(text: str, lang: str = "he") -> list[int]:
    """
    Convert an IPA phoneme string directly to vocab indices.
    No language tags are prepended. The lang argument is accepted
    to avoid breaking upstream callers but is silently ignored.
    """
    # 1. Strip out ANY HTML-style tags (<he>, </en>, etc.)
    text = re.sub(r"</?\w+>", "", text)
    
    return [CHAR_TO_ID.get(ch, PAD_ID) for ch in text]

def text_to_indices_multilang(text: str, base_lang: str = "he") -> list[int]:
    """
    Alias for text_to_indices. Language tags are silently dropped.
    """
    return text_to_indices(text, lang=base_lang)

def indices_to_text(indices: list[int]) -> str:
    return "".join(ID_TO_CHAR.get(i, "?") for i in indices)

# ============================================================
# Normalization
# ============================================================
_EMOJI_PATTERN = re.compile(r"[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff\U0001f700-\U0001f77f\U0001f780-\U0001f7ff\U0001f800-\U0001f8ff\U0001f900-\U0001f9ff\U0001fa00-\U0001fa6f\U0001fa70-\U0001faff\u2600-\u26ff\u2700-\u27bf\U0001f1e6-\U0001f1ff]+", flags=re.UNICODE)

_UNIVERSAL_REPLACEMENTS = {
    "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'", "´": "'", "`": "'",
    "–": "-", "‑": "-", "—": "-", "_": " ", "[": " ", "]": " ", "|": " ",
}

def normalize_text(text: str, apply_hebrew_fixes: bool = False) -> str:
    """
    Applies Unicode normalization and common phoneme substitutions.
    """
    text = text.strip()
    text = uni_normalize("NFD", text)
    text = _EMOJI_PATTERN.sub("", text)

    for k, v in _UNIVERSAL_REPLACEMENTS.items():
        text = text.replace(k, v)

    text = re.sub(r"[♥☆♡©]", "", text)
    
    # Fix spacing around punctuation
    for punct in [",", ".", "!", "?", ";", ":", "'"]:
        text = text.replace(f" {punct}", punct)

    # Remove duplicate quotes
    for q in ['""', "''", "``"]:
        while q in text:
            text = text.replace(q, q[0])

    # Fallback fixes for bad G2P output. (Fix your upstream G2P instead!)
    if apply_hebrew_fixes:
        text = text.replace("r", "ʁ").replace("g", "ɡ")

    text = re.sub(r"\s+", " ", text).strip()

    if not re.search(r"[.!?;:,'\"')\]}…。\\]』】〉》›»]$", text):
        text += "."

    return text

print(f"[Vocab] Universal IPA Active | VOCAB_SIZE={VOCAB_SIZE} | Max Used ID={max(CHAR_TO_ID.values())}")
