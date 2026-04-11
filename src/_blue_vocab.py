import re
from unicodedata import normalize as uni_normalize

_PIPER_MAP: dict[str, int] = {
    "_": 0, "^": 1, "$": 2, " ": 3,
    "!": 4, "'": 5, "(": 6, ")": 7, ",": 8, "-": 9, ".": 10,
    ":": 11, ";": 12, "?": 13,
    "a": 14, "b": 15, "c": 16, "d": 17, "e": 18, "f": 19,
    "h": 20, "i": 21, "j": 22, "k": 23, "l": 24, "m": 25, "n": 26,
    "o": 27, "p": 28, "q": 29, "r": 30, "s": 31, "t": 32, "u": 33,
    "v": 34, "w": 35, "x": 36, "y": 37, "z": 38,
    "æ": 39, "ç": 40, "ð": 41, "ø": 42, "ħ": 43, "ŋ": 44, "œ": 45,
    "ǀ": 46, "ǁ": 47, "ǂ": 48, "ǃ": 49,
    "ɐ": 50, "ɑ": 51, "ɒ": 52, "ɓ": 53, "ɔ": 54, "ɕ": 55,
    "ɖ": 56, "ɗ": 57, "ɘ": 58, "ə": 59, "ɚ": 60, "ɛ": 61,
    "ɜ": 62, "ɞ": 63, "ɟ": 64, "ɠ": 65, "ɡ": 66, "ɢ": 67,
    "ɣ": 68, "ɤ": 69, "ɥ": 70, "ɦ": 71, "ɧ": 72, "ɨ": 73,
    "ɪ": 74, "ɫ": 75, "ɬ": 76, "ɭ": 77, "ɮ": 78, "ɯ": 79,
    "ɰ": 80, "ɱ": 81, "ɲ": 82, "ɳ": 83, "ɴ": 84, "ɵ": 85,
    "ɶ": 86, "ɸ": 87, "ɹ": 88, "ɺ": 89, "ɻ": 90, "ɽ": 91,
    "ɾ": 92, "ʀ": 93, "ʁ": 94, "ʂ": 95, "ʃ": 96, "ʄ": 97,
    "ʈ": 98, "ʉ": 99, "ʊ": 100, "ʋ": 101, "ʌ": 102, "ʍ": 103,
    "ʎ": 104, "ʏ": 105, "ʐ": 106, "ʑ": 107, "ʒ": 108, "ʔ": 109,
    "ʕ": 110, "ʘ": 111, "ʙ": 112, "ʛ": 113, "ʜ": 114, "ʝ": 115,
    "ʟ": 116, "ʡ": 117, "ʢ": 118, "ʲ": 119,
    "ˈ": 120, "ˌ": 121, "ː": 122, "ˑ": 123, "˞": 124,
    "β": 125, "θ": 126, "χ": 127, "ᵻ": 128, "ⱱ": 129,
    "0": 130, "1": 131, "2": 132, "3": 133, "4": 134,
    "5": 135, "6": 136, "7": 137, "8": 138, "9": 139,
    "\u0327": 140, "\u0303": 141, "\u032A": 142, "\u032F": 143, "\u0329": 144,
    "ʰ": 145, "ˤ": 146, "ε": 147, "↓": 148, "#": 149,
    '"': 150, "↑": 151,
    "\u033A": 152, "\u033B": 153,
    "g": 154, "ʦ": 155, "X": 156,
}

_EXTENDED_MAP: dict[str, int] = {
    "A": 157, "B": 158, "C": 159, "D": 160, "E": 161, "F": 162, "G": 163,
    "H": 164, "I": 165, "J": 166, "K": 167, "L": 168, "M": 169, "N": 170,
    "O": 171, "P": 172, "Q": 173, "R": 174, "S": 175, "T": 176, "U": 177,
    "V": 178, "W": 179, "Y": 180, "Z": 181,
    "ʤ": 182, "ɝ": 183, "ʧ": 184, "ʼ": 185, "ʴ": 186, "ʱ": 187,
    "ʷ": 188, "ˠ": 189, "→": 190, "↗": 191, "↘": 192,
    "¡": 193, "¿": 194, "…": 195, "«": 196, "»": 197,
    "*": 198, "~": 199, "/": 200, "\\": 201, "&": 202,
    "\u0361": 203, "\u035C": 204, "\u0325": 205, "\u032C": 206,
    "\u0339": 207, "\u031C": 208, "\u031D": 209, "\u031E": 210,
    "\u031F": 211, "\u0320": 212, "\u0330": 213, "\u0334": 214,
    "\u031A": 215, "\u0318": 216, "\u0319": 217, "\u0348": 218,
    "\u0306": 219, "\u0308": 220, "\u031B": 221, "\u0324": 222,
    "\u033C": 223,
    "\u02C0": 224, "\u02C1": 225, "\u02BE": 226, "\u02BF": 227,
    "\u02BB": 228, "\u02C9": 229, "\u02CA": 230, "\u02CB": 231,
    "\u02C6": 232,
    "\u02E5": 233, "\u02E6": 234, "\u02E7": 235, "\u02E8": 236, "\u02E9": 237,
    "\u0300": 238, "\u0301": 239, "\u0302": 240, "\u0304": 241,
    "\u030C": 242, "\u0307": 243,
}

PIPER_REGION_END  = 156
LANG_REGION_START = 244
LANG_REGION_SIZE  = 140
VOCAB_SIZE        = LANG_REGION_START + LANG_REGION_SIZE  # 384

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2

LANG_ID: dict[str, int] = {
    "he": LANG_REGION_START + 0,
    "en": LANG_REGION_START + 1,
    "es": LANG_REGION_START + 2,
    "de": LANG_REGION_START + 8,
    "ge": LANG_REGION_START + 8,  # alias for German (same token as ``de``)
    "it": LANG_REGION_START + 9,
}

LANG_NAMES: dict[int, str] = {v: k for k, v in LANG_ID.items()}

CHAR_TO_ID: dict[str, int] = {**_PIPER_MAP, **_EXTENDED_MAP}
ID_TO_CHAR: dict[int, str] = {v: k for k, v in CHAR_TO_ID.items()}
for _lang_name, _lang_idx in LANG_ID.items():
    ID_TO_CHAR[_lang_idx] = f"<{_lang_name}>"

VOCAB_LIST: list[str] = list(CHAR_TO_ID.keys())

assert VOCAB_SIZE == 384
assert len(_PIPER_MAP) == 157, f"Piper map must have 157 entries, got {len(_PIPER_MAP)}"
assert max(_PIPER_MAP.values()) == PIPER_REGION_END
assert all(v < LANG_REGION_START for v in _PIPER_MAP.values())
assert all(PIPER_REGION_END < v < LANG_REGION_START for v in _EXTENDED_MAP.values())
assert len(set(_EXTENDED_MAP.values())) == len(_EXTENDED_MAP)
assert not (set(_PIPER_MAP.values()) & set(_EXTENDED_MAP.values()))


def text_to_indices(text: str, lang: str = "he") -> list[int]:
    """Convert an IPA string to vocab indices with a language token prepended."""
    if lang not in LANG_ID:
        raise ValueError(
            f"Unknown language '{lang}'. Available: {list(LANG_ID.keys())}. "
            f"Add it to LANG_ID in src/_blue_vocab.py"
        )
    return [LANG_ID[lang]] + [CHAR_TO_ID.get(ch, PAD_ID) for ch in text]


def text_to_indices_multilang(text: str, base_lang: str = "he") -> list[int]:
    """Convert an IPA string with inline <lang> tags to vocab indices.

    Tags switch the language token mid-sequence:
        "hello <en>world</en> again"
    Unknown tags fall back to base_lang.
    """
    if base_lang not in LANG_ID:
        raise ValueError(
            f"Unknown language '{base_lang}'. Available: {list(LANG_ID.keys())}."
        )
    if "<" not in text:
        return text_to_indices(text, lang=base_lang)

    segments: list[tuple[str, str]] = []
    last_end = 0
    for m in re.finditer(r"<(\w+)>(.*?)(?:</\1>|<\1>)", text, flags=re.DOTALL):
        if m.start() > last_end:
            segments.append((base_lang, text[last_end:m.start()]))
        tag_lang = m.group(1)
        segments.append((tag_lang if tag_lang in LANG_ID else base_lang, m.group(2)))
        last_end = m.end()
    if last_end < len(text):
        segments.append((base_lang, text[last_end:]))

    ids: list[int] = [LANG_ID[base_lang]]
    current_lang = base_lang
    for lang, seg in segments:
        if lang != current_lang:
            ids.append(LANG_ID.get(lang, LANG_ID[base_lang]))
            current_lang = lang
        ids.extend(CHAR_TO_ID.get(ch, PAD_ID) for ch in seg)
    return ids


def indices_to_text(indices: list[int]) -> str:
    """Convert indices back to a readable string (for debugging)."""
    return "".join(ID_TO_CHAR.get(i, "?") for i in indices)


def normalize_text(text: str, lang: str = "he") -> str:
    """Normalize IPA text before tokenization."""
    text = text.strip()
    text = uni_normalize("NFD", text)
    for k, v in {
        "\u201c": '"', "\u201d": '"',
        "\u2018": "'", "\u2019": "'",
        "´": "'", "`": "'",
        "–": "-", "‑": "-", "—": "-",
    }.items():
        text = text.replace(k, v)
    if lang == "he":
        text = text.replace("r", "ʁ").replace("g", "ɡ")
    return re.sub(r"\s+", " ", text).strip()


def check_coverage(text: str) -> dict:
    """Report which chars in text are missing from the vocab."""
    known, unknown = [], []
    for ch in text:
        (known if ch in CHAR_TO_ID else unknown).append(ch)
    return {
        "total": len(text),
        "known": len(known),
        "unknown": len(unknown),
        "missing_chars": sorted(set(unknown)),
        "missing_codepoints": [f"U+{ord(c):04X}" for c in sorted(set(unknown))],
    }


def get_supported_languages() -> list[str]:
    return list(LANG_ID.keys())
