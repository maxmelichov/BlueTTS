"""Text normalization applied *before* G2P.

The acoustic model only ever sees phonemes, so anything a phonemizer cannot read
aloud — digits, dates, clock times, prices, ticket codes, brackets, markdown
headers, Hebrew abbreviation gershayim — has to become words first.
:func:`prepare_text_for_synthesis` runs the whole chain in a fixed order (the
order matters: prices and ordinals are claimed before the code speller would
spell ``3rd`` or ``$5`` letter by letter, clock times are expanded before ratios
so ``08:15`` is not read as "eight to fifteen", and generic numbers go last).

Prices, ordinals, dates and clock times are read *inline*, as part of the
sentence. Only phone numbers and spelled codes are wrapped in ``【…】`` *slow
markers*: :meth:`TextToSpeech.__call__` splits on them and synthesizes those
spans with :data:`SLOW_SPEED_SCALE` / :data:`SLOW_PACE_BLEND`, where a slightly
slower digit-by-digit reading helps the listener write them down. Callers that do
not want the markers pass ``mark_slow=False``.

Hebrew numbers go through `hebrew-num2words <https://pypi.org/project/hebrew-num2words/>`_
when it is installed (``pip install "blue-onnx[hebrew]"``): it agrees the number
with the counted noun's gender (שלושה ילדים / שלוש בנות), which a plain
cardinal table cannot. Without it a built-in fallback reads the bare cardinals.

This module is standalone: it imports nothing from the rest of the package, so
the normalizer can be used (and tested) without ONNX Runtime or a model.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional, Union

from num2words import num2words

LANG_CODE_ALIASES: dict[str, str] = {"ge": "de", "en-us": "en"}

# ── slow segments ────────────────────────────────────────────────────────────
SLOW_MARK_OPEN = "【"
SLOW_MARK_CLOSE = "】"
# Slightly slower, clearer delivery for phone numbers and spelled codes. Kept
# moderate on purpose: heavier slow-downs over-stretch the span and the model's
# energy collapses at the tail, dropping the last digit group.
SLOW_SPEED_SCALE = 0.90
SLOW_PACE_BLEND = 0.40
SLOW_PACE_DPT_REF = 0.0625
SLOW_SILENCE = 0.12

_SLOW_WRAPPED_RE = re.compile(rf"{SLOW_MARK_OPEN}([^{SLOW_MARK_CLOSE}]+){SLOW_MARK_CLOSE}")
# Digits inside an <en> span or an already-marked span are left alone.
_PROTECTED_SPAN_RE = re.compile(
    rf"(<en>.*?</en>|{SLOW_MARK_OPEN}[^{SLOW_MARK_CLOSE}]*{SLOW_MARK_CLOSE})",
    re.IGNORECASE | re.DOTALL,
)
_INLINE_EN_BLOCK_RE = re.compile(r"(<en>.*?</en>)", re.IGNORECASE | re.DOTALL)
_INLINE_LANG_PAIR_RE = re.compile(r"<(\w+)>(.*?)</\1>", re.DOTALL)
_TERMINAL_PUNCT_RE = re.compile(r"[.!?,;:…]$")

# Pictographs, dingbats and flags. Stripped *before* G2P: espeak happily reads
# them aloud ("Nice 🎉 party" → "nice party popper party"), so filtering them at
# tokenization time — after phonemization — is far too late.
EMOJI_RE = re.compile(
    "[\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map symbols
    "\U0001f700-\U0001f77f"
    "\U0001f780-\U0001f7ff"
    "\U0001f800-\U0001f8ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff"
    "☀-⛿"  # misc symbols
    "✀-➿"  # dingbats
    "\U0001f1e6-\U0001f1ff]+",  # regional indicators (flags)
    flags=re.UNICODE,
)

_HEB = "֐-׿"
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# Latin letter+digit tokens (TKT-90254, IL4829, GPT-4, …) — spelled for TTS.
_ALNUM_MIX_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)*(?![A-Za-z0-9])"
)
# A digit may follow a Hebrew prefix letter directly (ב14:30, ו2) once
# _glue_hebrew_prefixes has dropped the hyphen, so only Latin letters and digits
# block a match on the left.
_NUM_LEFT = r"(?<![A-Za-z0-9_])"
_DATE_RE = re.compile(_NUM_LEFT + r"([0-3]?\d)([/.\-])([01]?\d)\2(\d{4}|\d{2})(?![\d.,]\d)(?!\d)")
_ISO_DATE_RE = re.compile(_NUM_LEFT + r"(\d{4})-([01]\d)-([0-3]\d)(?!\d)")
_TIME_RE = re.compile(_NUM_LEFT + r"([01]?\d|2[0-3]):([0-5]\d)(?!\d)")
# German writes "14:30 Uhr"; the expansion already says Uhr, so it is consumed.
_DE_TIME_RE = re.compile(_TIME_RE.pattern + r"(?:\s*Uhr\b)?")
_NUMBER_RE = re.compile(_NUM_LEFT + r"(?<![.,]\d)(?<![.,])\d+(?:[.,]\d+)*(?![\w])")
_LIST_MARKER_RE = re.compile(
    rf"(?:^|(?<=\n)|(?<=[:;.!?] ))(\d{{1,2}})\.\s+(?=[{_HEB}A-Za-z\"'(<])", re.MULTILINE
)
# Locales that write "1.500" for one thousand five hundred and "3,5" for three
# and a half — the mirror image of the en/he convention.
_COMMA_DECIMAL_LANGS = {"de", "es", "it"}
# A Spanish word that is not a preposition or conjunction: "21 libros" → veintiún.
_ES_NOUN_FOLLOWS_RE = re.compile(r"\s+(?!(?:de|del|y|o|a|al|en|por|con|que)\b)[a-záéíóúñ]")

_HEBREW_MONTH_ORDINALS = {
    1: "לראשון", 2: "לשני", 3: "לשלישי", 4: "לרביעי", 5: "לחמישי", 6: "לשישי",
    7: "לשביעי", 8: "לשמיני", 9: "לתשיעי", 10: "לעשירי", 11: "לאחד עשר",
    12: "לשנים עשר",
}
_HEBREW_ORDINALS_MASC = {
    1: "ראשון", 2: "שני", 3: "שלישי", 4: "רביעי", 5: "חמישי", 6: "שישי",
    7: "שביעי", 8: "שמיני", 9: "תשיעי", 10: "עשירי",
}
_MONTH_NAMES: dict[str, list[str]] = {
    "en": ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"],
    "es": ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
           "agosto", "septiembre", "octubre", "noviembre", "diciembre"],
    "de": ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli",
           "August", "September", "Oktober", "November", "Dezember"],
    "it": ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio",
           "agosto", "settembre", "ottobre", "novembre", "dicembre"],
}
_PERCENT_WORDS = {"he": "אחוז", "en": "percent", "es": "por ciento",
                  "de": "Prozent", "it": "per cento"}
_RATIO_WORDS = {"he": "ל", "en": "to", "es": "a", "de": "zu", "it": "a"}
_PLUS_WORDS = {"he": "פלוס", "en": "plus", "es": "más", "de": "plus", "it": "più"}
_MINUS_WORDS = {"he": "מינוס", "en": "minus", "es": "menos", "de": "minus", "it": "meno"}
_RANGE_WORDS = {"en": "to", "es": "a", "de": "bis", "it": "a"}
_DECIMAL_WORDS = {"en": "point", "de": "Komma", "es": "coma", "it": "virgola"}
# German ordinals take -n after these (am dritten Mai, den 2. → den zweiten).
_DE_DATIVE_WORDS = {"am", "vom", "zum", "zur", "seit", "bis", "ab", "dem", "den",
                    "des", "nach", "vor", "im", "beim", "einem", "einer", "ihrem",
                    "seinem", "unserem"}
_DE_ORDINAL_ARTICLES = _DE_DATIVE_WORDS | {"der", "die", "das", "eine", "jede", "jeder"}
# Spoken list counters (1. item, 2. item, …); Hebrew counting uses the feminine
# forms, which num2words does not produce.
_HEBREW_LIST_CARDINALS: dict[int, str] = {
    1: "אחד", 2: "שתיים", 3: "שלוש", 4: "ארבע", 5: "חמש", 6: "שש", 7: "שבע",
    8: "שמונה", 9: "תשע", 10: "עשר", 11: "אחת עשרה", 12: "שתים עשרה",
    13: "שלוש עשרה", 14: "ארבע עשרה", 15: "חמש עשרה", 16: "שש עשרה",
    17: "שבע עשרה", 18: "שמונה עשרה", 19: "תשע עשרה", 20: "עשרים",
}
_HEBREW_DIGIT_WORDS: dict[str, str] = {
    "0": "אפס", "1": "אחת", "2": "שתיים", "3": "שלוש", "4": "ארבע",
    "5": "חמש", "6": "שש", "7": "שבע", "8": "שמונה", "9": "תשע",
}

# ── currency ─────────────────────────────────────────────────────────────────
# (one, many, minor one, minor many, feminine) per language and symbol. A
# Hebrew-quoted ש"ח is a currency sign too; it has to be claimed here, before
# strip_hebrew_abbreviation_quotes turns it into the word "שח".
_CURRENCY_UNITS: dict[str, dict[str, tuple[str, str, str, str, bool]]] = {
    "en": {"$": ("dollar", "dollars", "cent", "cents", False),
           "€": ("euro", "euros", "cent", "cents", False),
           "£": ("pound", "pounds", "penny", "pence", False),
           "¥": ("yen", "yen", "", "", False),
           "₪": ("shekel", "shekels", "agora", "agorot", False)},
    "de": {"$": ("Dollar", "Dollar", "Cent", "Cent", False),
           "€": ("Euro", "Euro", "Cent", "Cent", False),
           "£": ("Pfund", "Pfund", "Penny", "Pence", False),
           "¥": ("Yen", "Yen", "", "", False),
           "₪": ("Schekel", "Schekel", "Agora", "Agorot", False)},
    "es": {"$": ("dólar", "dólares", "centavo", "centavos", False),
           "€": ("euro", "euros", "céntimo", "céntimos", False),
           "£": ("libra", "libras", "penique", "peniques", True),
           "¥": ("yen", "yenes", "", "", False),
           "₪": ("séquel", "séqueles", "agorá", "agorot", False)},
    "it": {"$": ("dollaro", "dollari", "centesimo", "centesimi", False),
           "€": ("euro", "euro", "centesimo", "centesimi", False),
           "£": ("sterlina", "sterline", "penny", "pence", True),
           "¥": ("yen", "yen", "", "", False),
           "₪": ("shekel", "shekel", "agorà", "agorot", False)},
    "he": {"$": ("דולר", "דולר", "סנט", "סנט", False),
           "€": ("יורו", "יורו", "סנט", "סנט", False),
           "£": ("פאונד", "פאונד", "פני", "פני", False),
           "¥": ("ין", "ין", "", "", False),
           "₪": ("שקל", "שקלים", "אגורה", "אגורות", False)},
}
_CURRENCY_ALIASES = {'ש"ח': "₪", "ש״ח": "₪"}
_AMOUNT = r"\d{1,3}(?:[.,\u00a0\u202f]\d{3})+(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?"
_CURRENCY_SIGN = r"[$€£¥₪]|ש[\"״]ח"
_CURRENCY_BEFORE_RE = re.compile(
    rf"(?<![\w$€£¥₪])({_CURRENCY_SIGN})\s?({_AMOUNT})(?![\d])(?![.,]\d)"
)
_CURRENCY_AFTER_RE = re.compile(
    _NUM_LEFT + rf"(?<![.,]\d)({_AMOUNT})\s?({_CURRENCY_SIGN})(?![\w])"
)


def canonical_lang(lang: str) -> str:
    """Map language aliases (``ge``, ``en-us``) onto the model's codes."""
    return LANG_CODE_ALIASES.get(lang.lower(), lang.lower())


def has_foreign_lang_span(text: str, lang: str) -> bool:
    """True when ``text`` holds an inline ``<xx>…</xx>`` span in a language other than ``lang``.

    Only those make the text mixed-language. A span in the base language
    (``<en>`` inside English text) is routed to the same G2P, so counting it
    switched on the mixed-text pace blend for a plain English sentence.
    """
    base = canonical_lang(lang)
    return any(canonical_lang(m.group(1)) != base for m in _INLINE_LANG_PAIR_RE.finditer(text))


def strip_emoji(text: str) -> str:
    """Drop emoji and pictographs, which no phonemizer should try to read."""
    return re.sub(r"\s+", " ", EMOJI_RE.sub(" ", text)).strip()


def _map_en_spans(text: str, lang: str, fn) -> str:
    """Apply ``fn(segment, lang, inside_en)`` to each part of ``text``.

    ``<en>…</en>`` spans are passed with their own language and their tags put
    back afterwards. Every expander that rewrites digits has to go through this:
    running one over a whole string lets it nest a second ``<en>`` pair — or a
    ``【…】`` marker — *inside* an existing span, and :func:`split_slow_segments`
    then cuts the span in half, stranding the rest of the English text on the
    outer language's G2P.
    """
    out: list[str] = []
    for part in _INLINE_EN_BLOCK_RE.split(text):
        if _INLINE_EN_BLOCK_RE.fullmatch(part):
            out.append(f"<en>{fn(part[4:-5], 'en', True)}</en>")
        else:
            out.append(fn(part, lang, False))
    return "".join(out)


@lru_cache(maxsize=1)
def _hebrew_num2words():
    """The optional ``hebrew_num2words`` module, or ``None`` when not installed."""
    try:
        import hebrew_num2words
    except ImportError:
        return None
    return hebrew_num2words


def _spoken_number(value: Union[int, float], lang: str) -> Optional[str]:
    """``num2words`` with a ``None`` fallback for unsupported language/value pairs.

    English drops num2words' commas, "and" and hyphens: "one thousand, two
    hundred and fifty" puts two pauses inside one number, polyg2p — the G2P the
    current models were trained with — says "one thousand two hundred fifty",
    and it reads a hyphenated "twenty-one" as one unknown compound
    ("twentiwon").
    """
    try:
        words = num2words(value, lang=lang)
    except Exception:
        return None
    return _tidy_english(words) if lang == "en" else words


def _tidy_english(words: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r",|-|\band\b", " ", words)).strip()


def _spoken_ordinal(value: int, lang: str, feminine: bool = False) -> Optional[str]:
    try:
        words = num2words(value, to="ordinal", lang=lang)
    except Exception:
        return _spoken_number(value, lang)
    if lang == "en":
        return _tidy_english(words)
    if feminine and lang in ("es", "it"):
        # num2words only has the masculine: tercero → tercera, terzo → terza.
        words = re.sub(r"o\b", "a", words)
    return words


def _spoken_year(value: int, lang: str) -> Optional[str]:
    """Years the way they are said: nineteen ninety-nine, neunzehnhundertneunundachtzig.

    English pairs the digits for 1100–1999 and 2010–2099 (2005 stays "two
    thousand five"); German does so for 1100–1999. Everything else is a plain
    cardinal, which is also how Spanish and Italian read years.
    """
    pairs = {"en": (1100 <= value <= 1999 or 2010 <= value <= 2099),
             "de": 1100 <= value <= 1999}.get(lang, False)
    if pairs:
        try:
            words = num2words(value, to="year", lang=lang)
        except Exception:
            return _spoken_number(value, lang)
        return _tidy_english(words) if lang == "en" else words
    return _spoken_number(value, lang)


def _spoken_digits(digits: str, lang: str) -> str:
    """Read ``digits`` one by one (phone numbers, ticket codes)."""
    if lang == "he":
        return " ".join(_HEBREW_DIGIT_WORDS[d] for d in digits if d.isdigit())
    words = [_spoken_number(int(d), lang) for d in digits if d.isdigit()]
    return " ".join(w for w in words if w)


def _parse_number(raw: str, lang: str) -> Optional[tuple[int, Optional[str]]]:
    """Split ``raw`` into (integer, fractional digits) for ``lang``'s separators.

    ``en``/``he`` group thousands with ``,`` and use ``.`` as the decimal point;
    ``de``/``es``/``it`` the reverse. A grouping mark is only a thousands
    separator when every group after it has exactly three digits, so ``3,5`` is
    still three and a half in English and ``3.5`` in German. A (
    narrow) no-break space between three-digit groups also groups thousands.
    """
    thou, dec = (".", ",") if lang in _COMMA_DECIMAL_LANGS else (",", ".")
    body, frac = raw, None
    if dec in body:
        body, _, frac = body.rpartition(dec)
        if not frac.isdigit() or dec in body:
            return None
    groups = re.split(rf"[{re.escape(thou)}\u00a0\u202f]", body)
    if len(groups) > 1:
        if not all(len(g) == 3 for g in groups[1:]) or not 1 <= len(groups[0]) <= 3:
            if frac is None and len(groups) == 2:
                # "3.5" in German text: a decimal written English-style.
                body, frac = groups
                groups = [body]
            else:
                return None
    digits = "".join(groups)
    if not digits.isdigit():
        return None
    return int(digits), frac


def _spoken_decimal(integer: int, frac: str, lang: str) -> Optional[str]:
    """3.14 → three point one four, 2,5 → zwei Komma fünf, 0,75 → cero coma setenta y cinco.

    English reads the fraction digit by digit. German, Spanish and Italian read
    a short fraction without a leading zero as a number, the way polyg2p does.
    """
    head = _spoken_number(integer, lang)
    if head is None:
        return None
    if lang == "he":
        return f"{head} נקודה {_spoken_digits(frac, 'he')}"
    word = _DECIMAL_WORDS.get(lang)
    if word is None:
        return None
    if lang != "en" and len(frac) <= 2 and not frac.startswith("0"):
        tail = _spoken_number(int(frac), lang)
    else:
        tail = _spoken_digits(frac, lang)
    return f"{head} {word} {tail}" if tail else None


def mark_slow_segment(inner: str) -> str:
    """Wrap ``inner`` in slow-synthesis markers."""
    return f"{SLOW_MARK_OPEN}{inner}{SLOW_MARK_CLOSE}"


def strip_slow_markers(text: str) -> str:
    """Drop slow markers, keeping the text inside (for display or plain synthesis)."""
    return _SLOW_WRAPPED_RE.sub(r"\1", text)


def split_slow_segments(text: str) -> list[tuple[str, bool]]:
    """Split prepared text into ``(segment, is_slow)`` pairs.

    Each segment is synthesized as its own utterance, and the tokenizer ends any
    utterance without final punctuation with a full stop — which the model reads
    as the end of a sentence. Every segment but the last therefore ends in a
    comma instead, so a phone number in the middle of a sentence gets a short
    pause on each side, not a sentence break.
    """
    if SLOW_MARK_OPEN not in text:
        return [(text, False)]

    parts: list[tuple[str, bool]] = []
    last = 0
    for m in _SLOW_WRAPPED_RE.finditer(text):
        if m.start() > last:
            chunk = text[last:m.start()].strip()
            if chunk:
                parts.append((chunk, False))
        inner = m.group(1).strip()
        if inner:
            parts.append((inner, True))
        last = m.end()
    if last < len(text):
        tail = text[last:].strip()
        if tail:
            parts.append((tail, False))

    # A trailing "." after a slow span would otherwise be synthesized on its own
    # (or open the next segment), which the model tends to vocalize as a stray
    # vowel. Punctuation that follows a span belongs to the span.
    merged: list[tuple[str, bool]] = []
    for chunk, is_slow in parts:
        lead = re.match(r"[.!?,;:…]+", chunk)
        if merged and lead:
            previous, previous_is_slow = merged[-1]
            merged[-1] = (f"{previous}{lead.group(0)}", previous_is_slow)
            chunk = chunk[lead.end():].strip()
            if not chunk:
                continue
        merged.append((chunk, is_slow))
    for i, (chunk, is_slow) in enumerate(merged[:-1]):
        if not _TERMINAL_PUNCT_RE.search(chunk):
            merged[i] = (f"{chunk},", is_slow)
    return merged or [(text, False)]


# ── punctuation / markup ─────────────────────────────────────────────────────
def strip_silent_separator_tokens(text: str) -> str:
    """Drop separators that are punctuation for the eye but noise for the ear."""
    text = re.sub(rf"(?<=[{_HEB}])[-–—‑]+(?=[A-Za-z0-9])", " ", text)
    text = re.sub(rf"(?<=[A-Za-z0-9])[-–—‑]+(?=[{_HEB}])", " ", text)
    text = re.sub(r"(?<![A-Za-z])\s*[-–—‑]+\s*(?![A-Za-z])", " ", text)
    text = re.sub(r"(?<!\d)\s*:+\s*(?!\d)", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_brackets(text: str) -> str:
    """Keep the words inside brackets but read them as a comma-set-off aside."""
    text = re.sub(r"\s*[(\[{]\s*", ", ", text)
    return re.sub(r"\s*[)\]}]\s*", ", ", text)


def normalize_repeated_punctuation(text: str) -> str:
    """Collapse runs of punctuation: ``!!!`` → ``!``, ``...`` → ``,``.

    Ellipses become a comma rather than a period: they mark trailing off, so a
    soft pause reads better than a full stop.
    """
    text = text.replace("…", ",")
    text = re.sub(r"(?<!\d)\.{2,}(?!\d)", ",", text)
    text = re.sub(r"!+", "!", text)
    text = re.sub(r"\?+", "?", text)
    text = re.sub(r",(?:\s*,)+", ",", text)
    # Mixed runs too: bracket stripping turns ")." into ", .", and a comma next to
    # a full stop makes the model pause twice.
    text = re.sub(r"\s*[,;:]+\s*(?=[.!?])", "", text)
    return re.sub(r"(?<=[.!?])\s*[,;:]+\s*", " ", text)


def normalize_common_text(text: str) -> str:
    """Language-independent cleanup: markdown headers, brackets, punctuation runs."""
    text = re.sub(r"(^|\s)#{1,6}\s*", r"\1", text)
    text = strip_brackets(text)
    text = normalize_repeated_punctuation(text)
    return re.sub(
        r"\banymore\b",
        lambda m: "Any more" if m.group(0)[0].isupper() else "any more",
        text,
        flags=re.IGNORECASE,
    )


def expand_dialogue_quotes(text: str, lang: str = "he") -> str:
    """Turn quotes around direct speech into a soft pause.

    The tokenizer drops quote characters silently, so quoted speech otherwise
    runs into the surrounding clause without a breath.
    """
    text = re.sub(r"(?<=\S)\s*[\"“„”]\s*$", ".", text)
    text = re.sub(r"\s*:?\s*[\"“„”]\s*(?=\S)", ", ", text)
    return re.sub(r"(?<=\S)\s*[\"“„”]", ", ", text)


# ── Hebrew spelling quirks ───────────────────────────────────────────────────
def strip_hebrew_abbreviation_quotes(text: str, lang: str = "he") -> str:
    """Remove in-word abbreviation marks: מנכ"ל → מנכל.

    Only the double marks are abbreviation markers. The single geresh is
    phonetic (ג׳=j, צ׳=ch, ז׳=zh), so it stays.
    """
    if canonical_lang(lang) != "he":
        return text
    return re.sub(rf"(?<=[{_HEB}])[\"״](?=[{_HEB}])", "", text)


def normalize_phonetic_geresh(text: str, lang: str = "he") -> str:
    """Use the Hebrew geresh for phonetic apostrophes after ג/צ/ז."""
    if canonical_lang(lang) != "he":
        return text
    return re.sub(rf"(?<=[גצז])'(?=[{_HEB}])", "׳", text)


_GERESH_LOANWORD_RE = re.compile(
    rf"(?<![{_HEB}])(?:ג['׳]מיני|מנג['׳]ר)(?![{_HEB}])"
)
_GERESH_LOANWORD_EN = {
    "ג'מיני": "Gemini", "ג׳מיני": "Gemini",
    "מנג'ר": "Manager", "מנג׳ר": "Manager",
}


def expand_geresh_loanwords(text: str, lang: str = "he") -> str:
    """Route Latin loanwords written with a geresh through English G2P."""
    if canonical_lang(lang) != "he":
        return text

    def repl(m: re.Match[str]) -> str:
        en = _GERESH_LOANWORD_EN.get(m.group(0))
        return f"<en>{en}</en>" if en else m.group(0)

    def expand(segment: str, seg_lang: str, inside_en: bool) -> str:
        return segment if inside_en else _GERESH_LOANWORD_RE.sub(repl, segment)

    return _map_en_spans(text, canonical_lang(lang), expand)


def strip_hebrew_inword_hyphens(text: str, lang: str = "he") -> str:
    """Join hyphenated Hebrew compounds: באז-וורד → באזוורד."""
    if canonical_lang(lang) != "he":
        return text
    return re.sub(rf"(?<=[{_HEB}])[-–—‑]+(?=[{_HEB}])", "", text)


def expand_hebrew_lamed_before_latin(text: str, lang: str = "he") -> str:
    """Avoid one-letter Hebrew fragments in mixed text: ל-GPU → אל GPU."""
    if canonical_lang(lang) != "he":
        return text
    return re.sub(rf"(?<![{_HEB}])ל\s*[-–—‑]?\s*(?=[A-Za-z])", "אל ", text)


def _glue_hebrew_prefixes(text: str) -> str:
    """ב-14:30 → ב14:30, so the expanded number keeps its prefix: בארבע עשרה.

    Without this the hyphen is later dropped as a separator and the prefix is
    left as a one-letter word, which the G2P reads as a letter name.
    """
    return re.sub(rf"(?<![{_HEB}])([ובכלמשה]{{1,3}})[-–‑](?=\d)", r"\1", text)


# ── codes, emails, numbers ───────────────────────────────────────────────────
def email_to_spoken_english(email: str) -> str:
    """Make an address pronounceable: a.b@c.co → "a dot b at c dot c o"."""
    local, _, domain = email.partition("@")

    def spell_short_label(label: str) -> str:
        return " ".join(label) if 0 < len(label) <= 2 and label.isalpha() else label

    local = re.sub(r"[._]+", " dot ", local)
    local = re.sub(r"[-]+", " dash ", local)
    local = re.sub(r"[+]+", " plus ", local)
    domain_parts = [spell_short_label(p) for p in domain.split(".") if p]
    return re.sub(r"\s+", " ", f"{local} at {' dot '.join(domain_parts)}").strip()


def expand_emails(text: str, lang: str = "he") -> str:
    """Route addresses through English G2P as spoken words.

    Runs before :func:`expand_alphanumeric_codes`, which would otherwise spell
    the letter+digit parts of an address one character at a time. English text
    gets the words without an ``<en>`` span: a span there is a no-op for G2P but
    would switch on the mixed-language pace blend for the whole utterance.
    """
    lang = canonical_lang(lang)

    def repl(m: re.Match[str]) -> str:
        spoken = email_to_spoken_english(m.group(0))
        return spoken if lang == "en" else f"<en>{spoken}</en>"

    def expand(segment: str, seg_lang: str, inside_en: bool) -> str:
        return segment if inside_en else _EMAIL_RE.sub(repl, segment)

    return _map_en_spans(text, lang, expand)


def should_spell_alphanumeric_token(token: str) -> bool:
    """True when a token mixes Latin letters and digits and should be spelled.

    Ordinals (``3rd``) and decades (``1990s``) mix letters and digits too, but
    they are words: the ordinal expander and the G2P read them.
    """
    if not token or "@" in token:
        return False
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9\-_/]*", token):
        return False
    if re.fullmatch(r"\d+(?:st|nd|rd|th|s)", token, re.IGNORECASE):
        return False
    letters = [c for c in token if c.isascii() and c.isalpha()]
    digits = [c for c in token if c.isdigit()]
    if not letters or not digits:
        return False
    if len(letters) < 2 and len(digits) < 2:
        return False
    return len(re.sub(r"[-_/]", "", token)) >= 3


def spell_alphanumeric_code(code: str, lang: str = "he") -> str:
    """Spell a letter+digit code (ticket, tracking, model number) as one slow span.

    Letters are spelled in ``lang`` — except in Hebrew text, where they go to
    English G2P via an ``<en>`` span because the Hebrew G2P has no Latin letter
    names. Digits are read one by one in ``lang``, and separator-delimited
    groups are joined by a short comma pause (a full stop makes the model
    stretch one pause and rush the next, so the digit tempo sounds uneven).
    """
    lang = canonical_lang(lang)
    groups: list[str] = []
    prev_seg = ""
    for seg in re.split(r"[-_/]+", code):
        # Letter and digit runs keep their order: X7K9 is "X seven K nine".
        runs = []
        for run in re.findall(r"[A-Za-z]+|\d+", seg):
            if run.isdigit():
                runs.append(_spoken_digits(run, lang))
            else:
                spelled = " ".join(run.upper())
                runs.append(f"<en>{spelled}</en>" if lang == "he" else spelled)
        if not runs:
            continue
        if groups and not any(c.isdigit() for c in prev_seg):
            # A letters-only prefix (IL-4829, GPT-4) is read with what follows it.
            groups[-1] = f"{groups[-1]} {' '.join(runs)}"
        else:
            groups.append(" ".join(runs))
        prev_seg = seg
    return mark_slow_segment(", ".join(groups)) if groups else ""


def expand_alphanumeric_codes(text: str, lang: str = "he") -> str:
    """Spell any Latin letter+digit mix (IDs, model codes, tracking numbers).

    Tokens already inside an ``<en>`` span are left alone: they are bound for
    English G2P, which reads them acceptably, and spelling them would nest a
    second ``<en>`` pair plus a slow marker inside the outer span. That matters
    for addresses in particular — :func:`expand_emails` runs first, so the local
    part of ``user123@gmail.com`` is inside a span by the time we get here.
    """
    def expand(segment: str, seg_lang: str, inside_en: bool) -> str:
        if inside_en:
            return segment

        def repl(m: re.Match[str]) -> str:
            token = m.group(0)
            if not should_spell_alphanumeric_token(token):
                return token
            return spell_alphanumeric_code(token, lang=seg_lang) or token

        return _ALNUM_MIX_TOKEN_RE.sub(repl, segment)

    return _map_en_spans(text, canonical_lang(lang), expand)


def expand_list_markers(text: str, lang: str = "he") -> str:
    """Turn ``1. item`` list markers into spoken counters (אחד / one, …).

    Only at the start of a line or after a sentence/colon break: mid-sentence,
    ``am 3. Mai`` is a German ordinal date, not a list item, and reading it as
    a counter put a full stop inside the sentence.
    """
    def expand(segment: str, seg_lang: str, inside_en: bool) -> str:
        if inside_en:
            return segment

        def repl(m: re.Match[str]) -> str:
            n = int(m.group(1))
            if seg_lang == "he" and n in _HEBREW_LIST_CARDINALS:
                word = _HEBREW_LIST_CARDINALS[n]
            else:
                word = _spoken_number(n, seg_lang)
                if word is None:
                    return m.group(0)
            return f"{word}. "

        return _LIST_MARKER_RE.sub(repl, segment)

    return _map_en_spans(text, canonical_lang(lang), expand)


def expand_plus_sign(text: str, lang: str = "he") -> str:
    """Speak ``+`` as פלוס/plus when it joins phrases (``DJ gear + laptop``)."""
    def expand(segment: str, seg_lang: str, inside_en: bool) -> str:
        if inside_en:
            return segment
        word = _PLUS_WORDS.get(seg_lang, _PLUS_WORDS["en"])
        return re.sub(r"\s+\+\s+", f" {word} ", segment)

    return _map_en_spans(text, canonical_lang(lang), expand)


def _currency_words(sign: str, amount: str, lang: str) -> Optional[str]:
    """$1,250.50 → one thousand two hundred fifty dollars and fifty cents.

    Hebrew keeps the digits and only spells the unit (20 ₪ → 20 שקלים): the
    number itself is left to the Hebrew number pass, which agrees it with the
    noun's gender.
    """
    units = _CURRENCY_UNITS.get(lang, {}).get(_CURRENCY_ALIASES.get(sign, sign))
    parsed = _parse_number(amount, lang)
    if units is None or parsed is None:
        return None
    one, many, minor_one, minor_many, feminine = units
    major, frac = parsed
    minor = int((frac + "0")[:2]) if frac else 0
    if minor and not minor_many:
        return None  # ¥12.50: no minor unit to read, leave it to the number pass
    if lang == "he":
        major_part = f"{one} אחד" if major == 1 else f"{major} {many}"
        if not minor:
            return major_part
        minor_part = f"{minor_one} אחת" if minor == 1 else f"{minor} {minor_many}"
        return f"{major_part} ו{minor_part}" if major else minor_part

    def count(n: int, singular: str, plural: str, fem: bool) -> Optional[str]:
        words = _spoken_number(n, lang)
        if words is None:
            return None
        if lang in ("es", "it") and words.endswith("uno"):
            # Before a noun "uno" shortens: un euro, veintiún dólares, una libra.
            short = "una" if fem else ("ún" if words.endswith("iuno") else "un")
            words = words[:-3] + short
        elif lang == "de":
            words = re.sub(r"eins$", "ein", words)
        return f"{words} {singular if n == 1 else plural}"

    major_part = count(major, one, many, feminine)
    if major_part is None:
        return None
    if not minor:
        return major_part
    if lang == "de":
        minor_words = _spoken_number(minor, lang)
        return f"{major_part} {minor_words}" if major else count(minor, minor_one, minor_many, False)
    minor_part = count(minor, minor_one, minor_many, False)
    if minor_part is None:
        return None
    if not major:
        return minor_part
    glue = {"en": "and", "es": "con", "it": "e"}[lang]
    return f"{major_part} {glue} {minor_part}"


def expand_currency(text: str, lang: str = "he") -> str:
    """Read prices with their unit after the amount: ``$5`` → five dollars, ``20 €`` → zwanzig Euro.

    Runs before the code speller, the abbreviation-quote stripper and the
    number pass: left alone, ``$1,250.50`` reached the number pass with the sign
    still attached and the G2P read "dollars one point two five".
    """
    def expand(segment: str, seg_lang: str, inside_en: bool) -> str:
        def repl_before(m: re.Match[str]) -> str:
            return _currency_words(m.group(1), m.group(2), seg_lang) or m.group(0)

        def repl_after(m: re.Match[str]) -> str:
            return _currency_words(m.group(2), m.group(1), seg_lang) or m.group(0)

        segment = _CURRENCY_BEFORE_RE.sub(repl_before, segment)
        return _CURRENCY_AFTER_RE.sub(repl_after, segment)

    return _map_en_spans(text, canonical_lang(lang), expand)


def expand_ordinals(text: str, lang: str = "he") -> str:
    """Read written ordinals: 3rd → third, 3º → tercero/terzo, am 3. Mai → am dritten Mai.

    English also gets its decades (1990s → nineteen nineties) and the day in
    "March 3" (→ March third) here.

    Runs before the code speller, which used to spell ``3rd`` as "R D three",
    and before the list-marker pass, which read German ``3.`` as a counter.
    Hebrew ``ה-3`` → השלישי is left to the Hebrew number pass.
    """
    lang = canonical_lang(lang)

    def expand(segment: str, seg_lang: str, inside_en: bool) -> str:
        if seg_lang == "en":
            segment = re.sub(
                _NUM_LEFT + r"(\d+)(st|nd|rd|th)\b",
                lambda m: _spoken_ordinal(int(m.group(1)), "en") or m.group(0),
                segment,
                flags=re.IGNORECASE,
            )
            segment = re.sub(_NUM_LEFT + r"(\d{3}0)s\b", _english_decade, segment)
            months = "|".join(_MONTH_NAMES["en"])
            return re.sub(
                rf"\b({months})\s+(\d{{1,2}})\b(?![:.,]?\d)",
                lambda m: f"{m.group(1)} {_spoken_ordinal(int(m.group(2)), 'en')}"
                if 1 <= int(m.group(2)) <= 31 else m.group(0),
                segment,
            )
        if seg_lang in ("es", "it"):
            return re.sub(
                _NUM_LEFT + r"(\d+)\.?([ºª])",
                lambda m: _spoken_ordinal(int(m.group(1)), seg_lang, m.group(2) == "ª")
                or m.group(0),
                segment,
            )
        if seg_lang == "de":
            return _expand_german_ordinals(segment)
        return segment

    return _map_en_spans(text, lang, expand)


def _english_decade(m: re.Match[str]) -> str:
    """1990s → nineteen nineties."""
    words = _spoken_year(int(m.group(1)), "en")
    if not words:
        return m.group(0)
    return re.sub(r"y$", "ies", words) if words.endswith("y") else f"{words}s"


def _german_ordinal(n: int, prev_word: str) -> Optional[str]:
    word = _spoken_ordinal(n, "de")
    if word is None:
        return None
    return f"{word}n" if prev_word.lower() in _DE_DATIVE_WORDS else word


def _expand_german_ordinals(segment: str) -> str:
    """``am 3. Mai`` → am dritten Mai, ``der 2. Versuch`` → der zweite Versuch.

    A number with a period is an ordinal when a month name follows, or when an
    article or preposition precedes it and a word follows; otherwise the period
    may just end a sentence ("Ich habe 3. Dann …").
    """
    months = "|".join(_MONTH_NAMES["de"])

    def repl_month(m: re.Match[str]) -> str:
        prev = re.findall(r"(\w+)\s*$", segment[:m.start()])
        word = _german_ordinal(int(m.group(1)), prev[0] if prev else "")
        return f"{word} " if word else m.group(0)

    segment = re.sub(
        rf"(?<![\d.])\b(\d{{1,2}})\.\s+(?=(?:{months})\b)", repl_month, segment
    )

    def repl_article(m: re.Match[str]) -> str:
        word = _german_ordinal(int(m.group(2)), m.group(1))
        return f"{m.group(1)} {word} " if word else m.group(0)

    articles = "|".join(sorted(_DE_ORDINAL_ARTICLES, key=len, reverse=True))
    return re.sub(
        rf"\b({articles})\s+(?<![\d.])(\d{{1,3}})\.\s+(?=[A-Za-zÄÖÜäöüß])",
        repl_article,
        segment,
        flags=re.IGNORECASE,
    )


_PHONE_RE = re.compile(
    r"(?<![\w.,+])(\+\s?\d{1,3}[\s-]?)?\d{2,4}(?:[\s-]\d{2,8}){1,4}(?![\w])(?![.,]\d)"
)


def _looks_like_phone(match: str) -> bool:
    """Separate digit groups that make a phone number rather than several numbers.

    A leading ``+`` or ``0`` always does. Hyphenated groups do once they add up
    to seven digits, unless they are a year range (1990-1995) or a date.
    Space-separated groups need three or more groups and nine digits — "1 500
    000" style thousands keep their single-digit first group, so they never
    match.
    """
    groups = re.findall(r"\d+", match)
    digits = sum(len(g) for g in groups)
    if not 7 <= digits <= 15:
        return False
    if match.lstrip().startswith(("+", "0")):
        return True
    if "-" in match:
        if len(groups) == 2 and all(len(g) == 4 and g[0] in "12" for g in groups):
            return False
        return not re.fullmatch(r"\d{1,2}-\d{1,2}-\d{2,4}|\d{4}-\d{2}-\d{2}", match)
    return len(groups) >= 3 and digits >= 9 and len(groups[0]) >= 3


def expand_phone_numbers(text: str, lang: str = "he") -> str:
    """Read phone and service numbers digit by digit, group by group, as one slow span.

    ``03-5551234`` → אפס שלוש, חמש חמש חמש אחת שתיים שלוש ארבע; ``555-123-4567``
    → five five five, one two three, four five six seven. ``*6700`` (Hebrew
    star-service numbers) → כוכבית שש שבע אפס אפס.
    """
    lang = canonical_lang(lang)

    def repl_star(m: re.Match[str]) -> str:
        return mark_slow_segment("כוכבית " + _spoken_digits(m.group(1), "he"))

    def expand(segment: str, seg_lang: str, inside_en: bool) -> str:
        if inside_en:
            return segment

        def repl_phone(m: re.Match[str]) -> str:
            raw = m.group(0)
            if not _looks_like_phone(raw):
                return raw
            groups = re.findall(r"\d+", raw)
            spoken = [_spoken_digits(g, seg_lang) for g in groups]
            if raw.lstrip().startswith("+"):
                spoken[0] = f"{_PLUS_WORDS.get(seg_lang, 'plus')} {spoken[0]}"
            return mark_slow_segment(", ".join(spoken))

        if seg_lang == "he":
            segment = re.sub(r"\*(\d{2,})", repl_star, segment)
        return _PHONE_RE.sub(repl_phone, segment)

    return _map_en_spans(text, lang, expand)


def expand_times(text: str, lang: str = "he") -> str:
    """Read ``HH:MM`` as a clock time, inline: 14:30 → fourteen thirty / vierzehn Uhr dreißig.

    Runs before :func:`expand_ratios`, which would otherwise read the colon as
    "eight to fifteen". Hebrew is left to hebrew-num2words when it is installed
    (14:30 → שתיים וחצי).
    """
    lang = canonical_lang(lang)
    if lang == "he" and _hebrew_num2words() is not None:
        return _map_en_spans(text, lang, lambda s, sl, inside: expand_times(s, "en") if inside else s)

    def expand(segment: str, seg_lang: str, inside_en: bool) -> str:
        def repl(m: re.Match[str]) -> str:
            hour, minute = int(m.group(1)), int(m.group(2))
            after = segment[m.end():]
            hour_word = _spoken_number(hour, seg_lang)
            minute_word = _spoken_number(minute, seg_lang) if minute else ""
            if hour_word is None or minute_word is None:
                return m.group(0)
            if seg_lang == "he":
                return f"{hour_word} ו{minute_word}" if minute else hour_word
            if seg_lang == "de":
                return f"{hour_word} Uhr {minute_word}".strip()
            if seg_lang == "en":
                if not minute:
                    if re.match(r"\s*[ap]\.?m\b", after, re.IGNORECASE):
                        return hour_word
                    return f"{hour_word} hundred" if hour > 12 else f"{hour_word} o'clock"
                return f"{hour_word} oh {minute_word}" if minute < 10 else f"{hour_word} {minute_word}"
            glue = {"es": " y ", "it": " e "}.get(seg_lang, " ")
            return f"{hour_word}{glue}{minute_word}" if minute else hour_word

        return (_DE_TIME_RE if seg_lang == "de" else _TIME_RE).sub(repl, segment)

    return _map_en_spans(text, lang, expand)


def _hebrew_date_fallback(day: int, month: int, year: int) -> Optional[str]:
    day_word = _spoken_number(day, "he")
    year_word = _spoken_number(year, "he")
    if day_word is None or year_word is None:
        return None
    return f"{day_word} {_HEBREW_MONTH_ORDINALS[month]} {year_word}"


def _spoken_date(day: int, month: int, year: int, lang: str, prev_word: str) -> Optional[str]:
    """Words for a date in ``lang``'s own pattern.

    en: December twenty-fifth twenty twenty-four; de: (am) zwölften Mai
    zweitausendvierundzwanzig; es: doce de mayo de dos mil veinticuatro (but
    primero for the 1st); it: dodici maggio duemilaventiquattro (primo).
    """
    if lang == "he":
        return _hebrew_date_fallback(day, month, year)
    months = _MONTH_NAMES.get(lang)
    year_word = _spoken_year(year, lang)
    if months is None or year_word is None:
        return None
    month_word = months[month - 1]
    if lang == "en":
        day_word = _spoken_ordinal(day, "en")
        return f"{month_word} {day_word} {year_word}" if day_word else None
    if lang == "de":
        day_word = _german_ordinal(day, prev_word)
        return f"{day_word} {month_word} {year_word}" if day_word else None
    if lang == "es":
        day_word = "primero" if day == 1 else _spoken_number(day, "es")
        return f"{day_word} de {month_word} de {year_word}" if day_word else None
    if lang == "it":
        day_word = "primo" if day == 1 else _spoken_number(day, "it")
        return f"{day_word} {month_word} {year_word}" if day_word else None
    return None


def expand_dates(text: str, lang: str = "he") -> str:
    """Expand numeric dates (``12/05/2024``, ``12.5.24``, ``12-05-2024``, ``2024-05-12``) into words, inline.

    Day first, as written in every language here except American English; when
    the second number cannot be a month (``12/25/2024``) the two are swapped.
    Hebrew goes to hebrew-num2words when installed (שנים עשר במאי אלפיים עשרים
    וארבע); it only reads dotted dates, so slashes become dots for it.
    """
    lang = canonical_lang(lang)
    he_n2w = lang == "he" and _hebrew_num2words() is not None

    def expand(segment: str, seg_lang: str, inside_en: bool) -> str:
        def words_for(day: int, month: int, year: int, start: int) -> Optional[str]:
            if month > 12 >= day:
                day, month = month, day
            if not (1 <= day <= 31 and 1 <= month <= 12):
                return None
            prev = re.findall(r"(\w+)\s*$", segment[:start])
            return _spoken_date(day, month, year, seg_lang, prev[0] if prev else "")

        def repl(m: re.Match[str]) -> str:
            day, month, raw_year = int(m.group(1)), int(m.group(3)), m.group(4)
            year = int(raw_year)
            if len(raw_year) == 2:
                year += 2000 if year < 70 else 1900
            if seg_lang == "he" and he_n2w:
                return f"{day}.{month}.{year}" if 1 <= month <= 12 and 1 <= day <= 31 else m.group(0)
            return words_for(day, month, year, m.start()) or m.group(0)

        def repl_iso(m: re.Match[str]) -> str:
            year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if seg_lang == "he" and he_n2w:
                return f"{day}.{month}.{year}"
            return words_for(day, month, year, m.start()) or m.group(0)

        segment = _ISO_DATE_RE.sub(repl_iso, segment)
        return _DATE_RE.sub(repl, segment)

    return _map_en_spans(text, lang, expand)


def expand_ranges_and_signs(text: str, lang: str = "he") -> str:
    """``10-20`` → 10 to 20, ``-5`` → minus 5.

    Without this the hyphen survived between the two number words ("ten-twenty"),
    which a G2P reads as one compound word, and a minus sign was dropped. Hebrew
    ranges are left to the number pass: "עד" before a number reads as a clock
    hour in hebrew-num2words.
    """
    def expand(segment: str, seg_lang: str, inside_en: bool) -> str:
        if inside_en:
            return segment
        word = _RANGE_WORDS.get(seg_lang)
        if word:
            segment = re.sub(
                r"(?<![\w.,\-–])(\d+)\s?[-–]\s?(\d+)(?![\w\-–]|[.,]\d)", rf"\1 {word} \2", segment
            )
        minus = _MINUS_WORDS.get(seg_lang, _MINUS_WORDS["en"])
        return re.sub(r"(?:(?<=\s)|^)[-−–](?=\d)", f"{minus} ", segment)

    return _map_en_spans(text, canonical_lang(lang), expand)


def expand_percent_symbols(text: str, lang: str = "he") -> str:
    """Replace ``%`` with the spoken word for ``lang`` (or ``en`` inside a span)."""
    def expand(segment: str, seg_lang: str, inside_en: bool) -> str:
        word = _PERCENT_WORDS.get(seg_lang, _PERCENT_WORDS["en"])
        segment = re.sub(r"(\d+(?:[.,]\d+)?)\s*%", rf"\1 {word}", segment)
        return re.sub(r"%", f" {word} ", segment)

    return _map_en_spans(text, canonical_lang(lang), expand)


def expand_ratios(text: str, lang: str = "he") -> str:
    """Read ``2:3`` as a ratio. Anything that is a valid clock time is left alone."""
    def expand(segment: str, seg_lang: str, inside_en: bool) -> str:
        word = _RATIO_WORDS.get(seg_lang, _RATIO_WORDS["en"])

        def repl(m: re.Match[str]) -> str:
            if _TIME_RE.fullmatch(f"{m.group(1)}:{m.group(2)}"):
                return m.group(0)
            return f"{m.group(1)} {word} {m.group(2)}"

        return re.sub(r"(?<!\d)(\d+)\s*:\s*(\d+)(?!\d)", repl, segment)

    return _map_en_spans(text, canonical_lang(lang), expand)


def _hebrew_ordinal_fallback(text: str) -> str:
    """ה3 → השלישי (masculine: without a lexicon the noun's gender is unknown)."""
    def repl(m: re.Match[str]) -> str:
        word = _HEBREW_ORDINALS_MASC.get(int(m.group(2)))
        return f"{m.group(1)}{word}" if word else m.group(0)

    return re.sub(rf"(?<![{_HEB}])([ובכלמש]{{0,2}}ה)(\d{{1,2}})(?!\d|[.,:/]\d)", repl, text)


def expand_numbers(text: str, lang: str = "he") -> str:
    """Expand remaining bare numbers into words.

    Thousands separators are resolved first, so ``1,500`` reads as "one thousand
    five hundred" and not as the decimal ``1.5``. Which mark groups and which one
    is the decimal point follows the language: ``en``/``he`` group with ``,``,
    while ``de``/``es``/``it`` group with ``.`` (so German ``1.500`` is a
    thousand and ``3,5`` is three and a half). A separator that does not form
    3-digit groups is read as a decimal point either way. Four-digit numbers
    read as years where the language pairs year digits (``1999`` → nineteen
    ninety-nine). Digits inside ``<en>`` spans or slow markers are left for those
    paths to handle.

    Hebrew goes through hebrew-num2words when installed, so the number agrees
    with the noun it counts (3 ילדים → שלושה ילדים, 2 בנות → שתי בנות).
    """
    lang = canonical_lang(lang)
    he_n2w = _hebrew_num2words() if lang == "he" else None

    def repl(m: re.Match[str]) -> str:
        raw = m.group(0)
        parsed = _parse_number(raw, lang)
        if parsed is None:
            return raw
        integer, frac = parsed
        if frac:
            return _spoken_decimal(integer, frac, lang) or raw
        if raw.isdigit() and len(raw) > 1 and raw.startswith("0"):
            return _spoken_digits(raw, lang)  # 0800, 007: a code, not a quantity
        if lang == "en" and raw in ("911", "411", "311", "999"):
            return _spoken_digits(raw, lang)  # service numbers are said digit by digit
        if raw.isdigit() and len(raw) == 4:
            return _spoken_year(integer, lang) or raw
        words = _spoken_number(integer, lang) or raw
        if lang == "es" and words.endswith("uno") and _ES_NOUN_FOLLOWS_RE.match(m.string, m.end()):
            # Before a noun: veintiún libros, un gato.
            words = words[:-3] + ("ún" if words.endswith("iuno") else "un")
        return words

    def expand_segment(segment: str) -> str:
        if he_n2w is not None:
            # hebrew-num2words re-joins tokens with single spaces; keep the
            # segment's own edges so it still abuts the protected spans around it.
            lead = segment[: len(segment) - len(segment.lstrip())]
            trail = segment[len(segment.rstrip()):]
            body = segment.strip()
            return f"{lead}{he_n2w.normalize_numbers(body)}{trail}" if body else segment
        if lang == "he":
            segment = _hebrew_ordinal_fallback(segment)
        return _NUMBER_RE.sub(repl, segment)

    parts = _PROTECTED_SPAN_RE.split(text)
    return "".join(
        p if _PROTECTED_SPAN_RE.fullmatch(p) else expand_segment(p) for p in parts
    )


def prepare_text_for_synthesis(
    text: str, lang: str = "he", mark_slow: bool = True
) -> str:
    """Run the full normalization chain and return synthesis-ready text.

    Order is deliberate: emoji and structural cleanup, then prices (whose ש"ח
    the Hebrew quote stripper would mangle), Hebrew spelling quirks, ordinals and
    codes (which claim letter+digit tokens before the number expanders see
    them), then phone numbers → times → dates → ranges → percent → ratios → bare
    numbers.

    With ``mark_slow=False`` the ``【…】`` slow markers are removed from the
    result, leaving plain text for callers that synthesize in one pass.
    """
    lang = canonical_lang(lang)
    text = strip_emoji(text)
    text = normalize_common_text(text)
    text = expand_currency(text, lang=lang)
    text = strip_hebrew_abbreviation_quotes(text, lang)
    text = normalize_phonetic_geresh(text, lang)
    text = expand_geresh_loanwords(text, lang)
    text = expand_dialogue_quotes(text, lang)
    text = strip_hebrew_inword_hyphens(text, lang)
    text = expand_hebrew_lamed_before_latin(text, lang)
    text = expand_emails(text, lang=lang)
    text = expand_ordinals(text, lang=lang)
    text = expand_alphanumeric_codes(text, lang=lang)
    text = expand_list_markers(text, lang=lang)
    text = expand_plus_sign(text, lang=lang)
    text = expand_phone_numbers(text, lang=lang)
    if lang == "he" and _hebrew_num2words() is None:
        text = _glue_hebrew_prefixes(text)
    text = expand_times(text, lang=lang)
    text = expand_dates(text, lang=lang)
    text = expand_ranges_and_signs(text, lang=lang)
    text = expand_percent_symbols(text, lang=lang)
    text = expand_ratios(text, lang=lang)
    text = expand_numbers(text, lang=lang)
    text = strip_silent_separator_tokens(text)
    return text if mark_slow else strip_slow_markers(text)
