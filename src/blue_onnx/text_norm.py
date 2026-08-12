"""Text normalization applied *before* G2P.

The acoustic model only ever sees phonemes, so anything a phonemizer cannot read
aloud — digits, dates, clock times, prices, ticket codes, brackets, markdown
headers, Hebrew abbreviation gershayim — has to become words first.
:func:`prepare_text_for_synthesis` runs the whole chain in a fixed order (the
order matters: clock times must be expanded before ratios so ``08:15`` is not
read as "eight to fifteen", and generic numbers must go last).

Spelled codes, phone numbers, dates and times are wrapped in
``【…】`` *slow markers*. They are not spoken: :meth:`TextToSpeech.__call__`
splits on them and synthesizes those spans with :data:`SLOW_SPEED_SCALE` /
:data:`SLOW_PACE_BLEND` so digit groups stay intelligible. Callers that do not
want the markers pass ``mark_slow=False``.

This module is standalone: it imports nothing from the rest of the package, so
the normalizer can be used (and tested) without ONNX Runtime or a model.
"""

from __future__ import annotations

import re
from typing import Optional, Union

from num2words import num2words

LANG_CODE_ALIASES: dict[str, str] = {"ge": "de", "en-us": "en"}

# ── slow segments ────────────────────────────────────────────────────────────
SLOW_MARK_OPEN = "【"
SLOW_MARK_CLOSE = "】"
# Slightly slower, clearer delivery for spelled IDs and expanded numbers. Kept
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

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# Latin letter+digit tokens (TKT-90254, IL4829, GPT-4, …) — spelled for TTS.
_ALNUM_MIX_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)*(?![A-Za-z0-9])"
)
_DATE_RE = re.compile(r"(?<!\d)([0-3]?\d)[/.]([01]?\d)[/.](\d{2}|\d{4})(?!\d)")
_TIME_RE = re.compile(r"(?<!\d)([01]?\d|2[0-3]):([0-5]\d)(?!\d)")
# Thousands groups first, so "1,500" is 1500 and not 1.5 (see expand_numbers).
_GROUPED_INT_COMMA_RE = re.compile(r"(?<![\w.,])\d{1,3}(?:,\d{3})+(?![\w.,])")
_GROUPED_INT_DOT_RE = re.compile(r"(?<![\w.,])\d{1,3}(?:\.\d{3})+(?![\w.,])")
_PLAIN_NUMBER_RE = re.compile(r"(?<![\w])\d+(?:[.,]\d+)?(?![\w])")
# Locales that write "1.500" for one thousand five hundred and "3,5" for three
# and a half — the mirror image of the en/he convention.
_COMMA_DECIMAL_LANGS = {"de", "es", "it"}
_LIST_MARKER_RE = re.compile(r"(?<![\d/])(\d{1,2})\.\s+(?=[֐-׿A-Za-z\"'(<])")

_HEBREW_MONTH_ORDINALS = {
    1: "לראשון", 2: "לשני", 3: "לשלישי", 4: "לרביעי", 5: "לחמישי", 6: "לשישי",
    7: "לשביעי", 8: "לשמיני", 9: "לתשיעי", 10: "לעשירי", 11: "לאחד עשר",
    12: "לשנים עשר",
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
_DATE_DAY_MONTH_GLUE = {"en": " of ", "es": " de ", "de": " ", "it": " "}
_PERCENT_WORDS = {"he": "אחוז", "en": "percent", "es": "por ciento",
                  "de": "Prozent", "it": "per cento"}
_RATIO_WORDS = {"he": "ל", "en": "to", "es": "a", "de": "zu", "it": "a"}
_PLUS_WORDS = {"he": "פלוס", "en": "plus", "es": "más", "de": "plus", "it": "più"}
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


def canonical_lang(lang: str) -> str:
    """Map language aliases (``ge``, ``en-us``) onto the model's codes."""
    return LANG_CODE_ALIASES.get(lang.lower(), lang.lower())


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


def _maybe_slow(inner: str, inside_en: bool) -> str:
    """Wrap in slow markers unless inside an ``<en>`` span, which they'd split."""
    return inner if inside_en else mark_slow_segment(inner)


def _spoken_number(value: Union[int, float], lang: str) -> Optional[str]:
    """``num2words`` with a ``None`` fallback for unsupported language/value pairs."""
    try:
        return num2words(value, lang=lang)
    except Exception:
        return None


def _spoken_ordinal(value: int, lang: str) -> Optional[str]:
    try:
        return num2words(value, to="ordinal", lang=lang)
    except Exception:
        return _spoken_number(value, lang)


def _spoken_digits(digits: str, lang: str) -> str:
    """Read ``digits`` one by one (phone numbers, ticket codes)."""
    if lang == "he":
        return " ".join(_HEBREW_DIGIT_WORDS[d] for d in digits if d.isdigit())
    words = [_spoken_number(int(d), lang) for d in digits if d.isdigit()]
    return " ".join(w for w in words if w)


def mark_slow_segment(inner: str) -> str:
    """Wrap ``inner`` in slow-synthesis markers."""
    return f"{SLOW_MARK_OPEN}{inner}{SLOW_MARK_CLOSE}"


def strip_slow_markers(text: str) -> str:
    """Drop slow markers, keeping the text inside (for display or plain synthesis)."""
    return _SLOW_WRAPPED_RE.sub(r"\1", text)


def split_slow_segments(text: str) -> list[tuple[str, bool]]:
    """Split prepared text into ``(segment, is_slow)`` pairs."""
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

    # A trailing "." after a slow span would otherwise be synthesized on its own,
    # which the model tends to vocalize as a stray vowel. Keep punctuation-only
    # fragments with the segment they belong to.
    merged: list[tuple[str, bool]] = []
    for chunk, is_slow in parts:
        if merged and re.fullmatch(r"[.!?,;:…]+", chunk):
            previous, previous_is_slow = merged[-1]
            merged[-1] = (f"{previous}{chunk}", previous_is_slow)
        else:
            merged.append((chunk, is_slow))
    return merged or [(text, False)]


# ── punctuation / markup ─────────────────────────────────────────────────────
def strip_silent_separator_tokens(text: str) -> str:
    """Drop separators that are punctuation for the eye but noise for the ear."""
    text = re.sub(r"(?<=[֐-׿])[-–—‑]+(?=[A-Za-z0-9])", " ", text)
    text = re.sub(r"(?<=[A-Za-z0-9])[-–—‑]+(?=[֐-׿])", " ", text)
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
    return re.sub(r"(?<=[֐-׿])[\"״](?=[֐-׿])", "", text)


def normalize_phonetic_geresh(text: str, lang: str = "he") -> str:
    """Use the Hebrew geresh for phonetic apostrophes after ג/צ/ז."""
    if canonical_lang(lang) != "he":
        return text
    return re.sub(r"(?<=[גצז])'(?=[֐-׿])", "׳", text)


_GERESH_LOANWORD_RE = re.compile(
    r"(?<![֐-׿])(?:ג['׳]מיני|מנג['׳]ר)(?![֐-׿])"
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
    return re.sub(r"(?<=[֐-׿])[-–—‑]+(?=[֐-׿])", "", text)


def expand_hebrew_lamed_before_latin(text: str, lang: str = "he") -> str:
    """Avoid one-letter Hebrew fragments in mixed text: ל-GPU → אל GPU."""
    if canonical_lang(lang) != "he":
        return text
    return re.sub(r"(?<![֐-׿])ל\s*[-–—‑]?\s*(?=[A-Za-z0-9])", "אל ", text)


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
    the letter+digit parts of an address one character at a time.
    """
    def repl(m: re.Match[str]) -> str:
        return f"<en>{email_to_spoken_english(m.group(0))}</en>"

    def expand(segment: str, seg_lang: str, inside_en: bool) -> str:
        return segment if inside_en else _EMAIL_RE.sub(repl, segment)

    return _map_en_spans(text, canonical_lang(lang), expand)


def should_spell_alphanumeric_token(token: str) -> bool:
    """True when a token mixes Latin letters and digits and should be spelled."""
    if not token or "@" in token:
        return False
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9\-_/]*", token):
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

    Letters are read in English via an ``<en>`` span; each separator-delimited
    digit group is read digit by digit in ``lang``, groups joined by a short
    comma pause (a full stop makes the model stretch one pause and rush the
    next, so the digit tempo sounds uneven).
    """
    lang = canonical_lang(lang)
    letters: list[str] = []
    digit_groups: list[str] = []

    for seg in re.split(r"[-_/]+", code):
        if not seg:
            continue
        seg_letters = [c.upper() for c in seg if c.isascii() and c.isalpha()]
        seg_digits = "".join(c for c in seg if c.isdigit())
        if seg_letters:
            letters.extend(seg_letters)
        if seg_digits:
            digit_groups.append(seg_digits)

    digit_parts = [w for w in (_spoken_digits(g, lang) for g in digit_groups) if w]
    if not letters and not digit_parts:
        return ""

    letters_block = f"<en>{' '.join(letters)}</en>" if letters else ""
    digits_block = " , ".join(digit_parts)
    if digits_block:
        digits_block += " ."  # guard the last group against the vocoder end-crop
    inner = " ".join(part for part in (letters_block, digits_block) if part)
    return mark_slow_segment(inner)


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
    """Turn ``1. item`` list markers into spoken counters (אחד / one, …)."""
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


def expand_phone_numbers(text: str, lang: str = "he") -> str:
    """Read phone and service numbers digit by digit rather than as cardinals.

    ``03-5551234`` → אפס שלוש חמש חמש חמש אחת שתיים שלוש ארבע, ``*6700`` →
    כוכבית שש שבע אפס אפס. Hebrew only: number formats and the "star" prefix are
    local conventions.
    """
    if canonical_lang(lang) != "he":
        return text

    def repl_star(m: re.Match[str]) -> str:
        return mark_slow_segment("כוכבית " + _spoken_digits(m.group(1), "he"))

    def repl_phone(m: re.Match[str]) -> str:
        return mark_slow_segment(_spoken_digits(m.group(0).replace("-", ""), "he"))

    def expand(segment: str, seg_lang: str, inside_en: bool) -> str:
        if inside_en:
            return segment
        segment = re.sub(r"\*(\d{2,})", repl_star, segment)
        return re.sub(r"(?<!\d)0\d{0,2}-\d{6,8}(?!\d)", repl_phone, segment)

    return _map_en_spans(text, canonical_lang(lang), expand)


def expand_times(text: str, lang: str = "he") -> str:
    """Read ``HH:MM`` as a clock time (08:15 → שמונה וחמש עשרה / eight fifteen).

    Runs before :func:`expand_ratios`, which would otherwise read the colon as
    "eight to fifteen".
    """
    def expand(segment: str, seg_lang: str, inside_en: bool) -> str:
        def repl(m: re.Match[str]) -> str:
            hour, minute = int(m.group(1)), int(m.group(2))
            hour_word = _spoken_number(hour, seg_lang)
            if hour_word is None:
                return m.group(0)
            if minute == 0:
                return _maybe_slow(hour_word, inside_en)
            minute_word = _spoken_number(minute, seg_lang)
            if minute_word is None:
                return m.group(0)
            joined = (
                f"{hour_word} ו{minute_word}"
                if seg_lang == "he"
                else f"{hour_word} {minute_word}"
            )
            return _maybe_slow(joined, inside_en)

        return _TIME_RE.sub(repl, segment)

    return _map_en_spans(text, canonical_lang(lang), expand)


def expand_dates(text: str, lang: str = "he") -> str:
    """Expand day-first numeric dates (``12/05/2024``, ``12.5.24``) into words."""
    def expand(segment: str, seg_lang: str, inside_en: bool) -> str:
        def repl(m: re.Match[str]) -> str:
            day, month, raw_year = int(m.group(1)), int(m.group(2)), m.group(3)
            if not (1 <= day <= 31 and 1 <= month <= 12):
                return m.group(0)
            year = int(raw_year)
            if len(raw_year) == 2:
                year += 2000 if year < 70 else 1900
            year_word = _spoken_number(year, seg_lang)
            if year_word is None:
                return m.group(0)
            if seg_lang == "he":
                day_word = _spoken_number(day, "he")
                if day_word is None:
                    return m.group(0)
                return _maybe_slow(
                    f"{day_word} {_HEBREW_MONTH_ORDINALS[month]} {year_word}", inside_en
                )
            months = _MONTH_NAMES.get(seg_lang)
            day_word = _spoken_ordinal(day, seg_lang)
            if months is None or day_word is None:
                return m.group(0)
            glue = _DATE_DAY_MONTH_GLUE.get(seg_lang, " ")
            return _maybe_slow(
                f"{day_word}{glue}{months[month - 1]} {year_word}", inside_en
            )

        return _DATE_RE.sub(repl, segment)

    return _map_en_spans(text, canonical_lang(lang), expand)


def expand_percent_symbols(text: str, lang: str = "he") -> str:
    """Replace ``%`` with the spoken word for ``lang`` (or ``en`` inside a span)."""
    def expand(segment: str, seg_lang: str, inside_en: bool) -> str:
        word = _PERCENT_WORDS.get(seg_lang, _PERCENT_WORDS["en"])
        segment = re.sub(r"(\d+(?:[.,]\d+)?)\s*%", rf"\1 {word}", segment)
        return re.sub(r"%", f" {word} ", segment)

    return _map_en_spans(text, canonical_lang(lang), expand)


def expand_ratios(text: str, lang: str = "he") -> str:
    """Read ``2:3`` as a ratio. Clock times are already gone by this point."""
    def expand(segment: str, seg_lang: str, inside_en: bool) -> str:
        word = _RATIO_WORDS.get(seg_lang, _RATIO_WORDS["en"])
        return re.sub(r"(?<!\d)(\d+)\s*:\s*(\d+)(?!\d)", rf"\1 {word} \2", segment)

    return _map_en_spans(text, canonical_lang(lang), expand)


def expand_numbers(text: str, lang: str = "he") -> str:
    """Expand remaining bare numbers into words.

    Thousands separators are resolved first, so ``1,500`` reads as "one thousand
    five hundred" and not as the decimal ``1.5``. Which mark groups and which one
    is the decimal point follows the language: ``en``/``he`` group with ``,``,
    while ``de``/``es``/``it`` group with ``.`` (so German ``1.500`` is a
    thousand and ``3,5`` is three and a half). A separator that does not form
    3-digit groups is read as a decimal point either way. Digits inside ``<en>``
    spans or slow markers are left for those paths to handle.
    """
    lang = canonical_lang(lang)
    comma_decimal = lang in _COMMA_DECIMAL_LANGS
    group_re = _GROUPED_INT_DOT_RE if comma_decimal else _GROUPED_INT_COMMA_RE
    group_mark = "." if comma_decimal else ","

    def repl_grouped(m: re.Match[str]) -> str:
        word = _spoken_number(int(m.group(0).replace(group_mark, "")), lang)
        return word if word else m.group(0)

    def repl_plain(m: re.Match[str]) -> str:
        raw = m.group(0)
        try:
            value: Union[int, float] = (
                float(raw.replace(",", ".")) if ("." in raw or "," in raw) else int(raw)
            )
        except ValueError:
            return raw
        word = _spoken_number(value, lang)
        return word if word else raw

    def expand_segment(segment: str) -> str:
        segment = group_re.sub(repl_grouped, segment)
        return _PLAIN_NUMBER_RE.sub(repl_plain, segment)

    parts = _PROTECTED_SPAN_RE.split(text)
    return "".join(
        p if _PROTECTED_SPAN_RE.fullmatch(p) else expand_segment(p) for p in parts
    )


def prepare_text_for_synthesis(
    text: str, lang: str = "he", mark_slow: bool = True
) -> str:
    """Run the full normalization chain and return synthesis-ready text.

    Order is deliberate: emoji and structural cleanup, then Hebrew spelling quirks, then
    codes (which claim letter+digit tokens before the number expanders see
    them), then times → dates → percent → ratios → bare numbers.

    With ``mark_slow=False`` the ``【…】`` slow markers are removed from the
    result, leaving plain text for callers that synthesize in one pass.
    """
    text = strip_emoji(text)
    text = normalize_common_text(text)
    text = strip_hebrew_abbreviation_quotes(text, lang)
    text = normalize_phonetic_geresh(text, lang)
    text = expand_geresh_loanwords(text, lang)
    text = expand_dialogue_quotes(text, lang)
    text = strip_hebrew_inword_hyphens(text, lang)
    text = expand_hebrew_lamed_before_latin(text, lang)
    text = expand_emails(text, lang=lang)
    text = expand_alphanumeric_codes(text, lang=lang)
    text = expand_list_markers(text, lang=lang)
    text = expand_plus_sign(text, lang=lang)
    text = expand_phone_numbers(text, lang=lang)
    text = expand_times(text, lang=lang)
    text = expand_dates(text, lang=lang)
    text = expand_percent_symbols(text, lang=lang)
    text = expand_ratios(text, lang=lang)
    text = expand_numbers(text, lang=lang)
    text = strip_silent_separator_tokens(text)
    return text if mark_slow else strip_slow_markers(text)
