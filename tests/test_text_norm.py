"""Unit tests for the pre-G2P text normalizer (no ONNX runtime required)."""

from __future__ import annotations

import re
import unittest
from unittest import mock

from src.blue_onnx import text_norm
from src.blue_onnx.text_norm import (
    SLOW_MARK_CLOSE,
    SLOW_MARK_OPEN,
    email_to_spoken_english,
    expand_alphanumeric_codes,
    expand_dates,
    expand_dialogue_quotes,
    expand_emails,
    expand_hebrew_lamed_before_latin,
    expand_list_markers,
    expand_numbers,
    expand_percent_symbols,
    expand_phone_numbers,
    expand_plus_sign,
    expand_ratios,
    expand_times,
    normalize_repeated_punctuation,
    prepare_text_for_synthesis,
    should_spell_alphanumeric_token,
    spell_alphanumeric_code,
    split_slow_segments,
    strip_brackets,
    strip_emoji,
    strip_hebrew_abbreviation_quotes,
    strip_hebrew_inword_hyphens,
    strip_silent_separator_tokens,
    strip_slow_markers,
)


def without_hebrew_num2words():
    """Run the built-in Hebrew fallback even when hebrew-num2words is installed."""
    return mock.patch.object(text_norm, "_hebrew_num2words", lambda: None)


HAS_HEBREW_NUM2WORDS = text_norm._hebrew_num2words() is not None


class TestNumbers(unittest.TestCase):
    def test_thousands_separator_is_not_a_decimal(self):
        # Regression: float("1,500".replace(",", ".")) == 1.5 read prices as decimals.
        self.assertEqual(expand_numbers("1,500", lang="en"), "one thousand five hundred")
        self.assertNotIn("point", expand_numbers("1,500", lang="en"))

    def test_multiple_thousands_groups(self):
        self.assertEqual(
            expand_numbers("12,345,678", lang="en"),
            "twelve million three hundred forty five thousand six hundred seventy eight",
        )

    def test_decimal_point_still_decimal(self):
        self.assertEqual(expand_numbers("3.5", lang="en"), "three point five")

    def test_comma_decimal_when_not_a_thousands_group(self):
        self.assertEqual(expand_numbers("3,5", lang="en"), "three point five")

    def test_hebrew_integer(self):
        with without_hebrew_num2words():
            self.assertEqual(expand_numbers("1,500", lang="he"), "אלף וחמש מאות")

    def test_comma_decimal_locale_groups_with_dots(self):
        # German writes 1.500 for a thousand and 3,5 for three and a half.
        self.assertEqual(expand_numbers("1.500", lang="de"), "eintausendfünfhundert")
        self.assertEqual(expand_numbers("3,5", lang="de"), "drei Komma fünf")

    def test_dot_grouping_not_applied_to_english(self):
        self.assertEqual(expand_numbers("1.500", lang="en"), "one point five zero zero")

    def test_digits_in_en_span_are_left_alone(self):
        with without_hebrew_num2words():
            self.assertEqual(
                expand_numbers("<en>GPT 4</en> ו 5", lang="he"),
                "<en>GPT 4</en> ו חמש",
            )

    def test_digits_in_slow_span_are_left_alone(self):
        marked = f"{SLOW_MARK_OPEN}4 8{SLOW_MARK_CLOSE} 9"
        self.assertEqual(expand_numbers(marked, lang="en"), f"{marked[:-2]} nine")


class TestTimesAndRatios(unittest.TestCase):
    def test_hebrew_clock_time(self):
        with without_hebrew_num2words():
            out = prepare_text_for_synthesis("בשעה 08:15", lang="he")
        self.assertIn("שמונה וחמש עשרה", out)
        self.assertNotIn(SLOW_MARK_OPEN, out)

    def test_non_hebrew_clock_time_is_a_time_not_a_ratio(self):
        # Regression: expand_times used to be Hebrew-only, so expand_ratios turned
        # "08:15" into "eight to fifteen".
        out = prepare_text_for_synthesis("the show starts at 08:15", lang="en")
        self.assertIn("eight fifteen", out)
        self.assertNotIn(" to ", out)

    def test_whole_hour_drops_minutes(self):
        self.assertEqual(expand_times("at 08:00", lang="en"), "at eight o'clock")
        self.assertEqual(expand_times("at 08:00 pm", lang="en"), "at eight pm")

    def test_out_of_range_time_untouched(self):
        self.assertEqual(expand_times("25:99", lang="he"), "25:99")

    def test_ratio(self):
        self.assertEqual(expand_ratios("2:3", lang="en"), "2 to 3")
        self.assertEqual(expand_ratios("2:3", lang="he"), "2 ל 3")


class TestDates(unittest.TestCase):
    def test_hebrew_date(self):
        with without_hebrew_num2words():
            out = expand_dates("12/05/2024", lang="he")
        self.assertIn("לחמישי", out)
        self.assertNotIn(SLOW_MARK_OPEN, out)

    def test_latin_date_becomes_words_not_digits(self):
        out = expand_dates("12/05/2024", lang="en")
        self.assertIn("May", out)
        self.assertIn("twelfth", out)
        self.assertNotIn("12", out)

    def test_two_digit_year(self):
        self.assertIn("twenty twenty four", expand_dates("12.5.24", lang="en"))

    def test_invalid_date_untouched(self):
        self.assertEqual(expand_dates("45/99/2024", lang="he"), "45/99/2024")


class TestCodesAndEmails(unittest.TestCase):
    def test_should_spell(self):
        self.assertTrue(should_spell_alphanumeric_token("IL-4829-7361-05"))
        self.assertTrue(should_spell_alphanumeric_token("GPT4"))
        self.assertFalse(should_spell_alphanumeric_token("hello"))
        self.assertFalse(should_spell_alphanumeric_token("2024"))
        self.assertFalse(should_spell_alphanumeric_token("a@b.com"))

    def test_spell_code_hebrew(self):
        out = spell_alphanumeric_code("IL-4829", lang="he")
        self.assertIn("<en>I L</en>", out)
        self.assertIn("ארבע שמונה שתיים תשע", out)
        self.assertTrue(out.startswith(SLOW_MARK_OPEN) and out.endswith(SLOW_MARK_CLOSE))

    def test_spell_code_latin_digits_become_words(self):
        out = spell_alphanumeric_code("IL-4829", lang="en")
        self.assertIn("four eight two nine", out)
        self.assertNotIn("4829", out)

    def test_code_groups_are_separated(self):
        out = spell_alphanumeric_code("IL-4829-7361-05", lang="he")
        self.assertEqual(out.count(","), 2)

    def test_expand_codes_leaves_plain_words(self):
        self.assertEqual(expand_alphanumeric_codes("שלום עולם", lang="he"), "שלום עולם")

    def test_email_spoken(self):
        self.assertEqual(
            email_to_spoken_english("max.me006@gmail.com"),
            "max dot me006 at gmail dot com",
        )

    def test_email_short_tld_is_spelled_out(self):
        # 1-2 letter labels are initialisms ("co.il"), so they are read letter by letter.
        self.assertEqual(
            email_to_spoken_english("a@site.co.il"), "a at site dot c o dot i l"
        )

    def test_emails_routed_to_english(self):
        out = expand_emails("שלח לי מייל ל a.b@gmail.com", lang="he")
        self.assertIn("<en>a dot b at gmail dot com</en>", out)

    def test_email_not_spelled_as_a_code(self):
        out = prepare_text_for_synthesis("כתוב ל user1@site.com בבקשה", lang="he")
        self.assertIn("<en>", out)
        self.assertNotIn("S I T E", out)


class TestHebrewSpelling(unittest.TestCase):
    def test_abbreviation_quotes_removed(self):
        self.assertEqual(strip_hebrew_abbreviation_quotes('מנכ"ל', "he"), "מנכל")

    def test_phonetic_geresh_kept(self):
        self.assertIn("׳", prepare_text_for_synthesis("דיג׳יי", lang="he"))

    def test_inword_hyphens_joined(self):
        self.assertEqual(strip_hebrew_inword_hyphens("באז-וורד", "he"), "באזוורד")

    def test_lamed_before_latin(self):
        self.assertEqual(
            expand_hebrew_lamed_before_latin("ל-GPU", "he"), "אל GPU"
        )

    def test_hebrew_only_rules_skip_other_langs(self):
        self.assertEqual(strip_hebrew_inword_hyphens("well-known", "en"), "well-known")


class TestPhoneNumbers(unittest.TestCase):
    def test_landline_digit_by_digit(self):
        out = expand_phone_numbers("התקשר ל 03-5551234", lang="he")
        self.assertIn("אפס שלוש, חמש חמש חמש אחת שתיים שלוש ארבע", out)

    def test_star_number(self):
        self.assertIn("כוכבית שש שבע אפס אפס", expand_phone_numbers("*6700", lang="he"))

    def test_other_languages_read_digit_groups(self):
        self.assertEqual(
            expand_phone_numbers("call 555-123-4567", lang="en"),
            f"call {SLOW_MARK_OPEN}five five five, one two three, four five six seven{SLOW_MARK_CLOSE}",
        )
        self.assertIn("null drei null, eins", expand_phone_numbers("unter 030 1234567", lang="de"))
        self.assertIn("seis uno dos, tres", expand_phone_numbers("al 612 345 678", lang="es"))
        self.assertIn("più tre nove", expand_phone_numbers("+39 06 1234 5678", lang="it"))

    def test_numbers_that_are_not_phone_numbers(self):
        for text in ("1 500 000", "1990-1995", "12-05-2024", "10-20", "2024-05-12"):
            self.assertEqual(expand_phone_numbers(text, lang="en"), text)


class TestPunctuationAndMarkup(unittest.TestCase):
    def test_repeated_punctuation(self):
        self.assertEqual(normalize_repeated_punctuation("wow!!! really???"), "wow! really?")

    def test_ellipsis_becomes_comma(self):
        self.assertEqual(normalize_repeated_punctuation("well... maybe"), "well, maybe")

    def test_comma_next_to_a_full_stop_is_dropped(self):
        # strip_brackets turns ")." into ", .", which would pause twice.
        self.assertEqual(normalize_repeated_punctuation("aside, . next"), "aside. next")
        self.assertEqual(normalize_repeated_punctuation("stop. , next"), "stop. next")
        self.assertEqual(normalize_repeated_punctuation("a, , b"), "a, b")

    def test_brackets_before_a_period_read_as_one_stop(self):
        out = prepare_text_for_synthesis("laptop (can deliver).", lang="en")
        self.assertEqual(out, "laptop, can deliver.")

    def test_brackets_become_asides(self):
        self.assertEqual(strip_brackets("laptop (can deliver) 4"), "laptop, can deliver, 4")

    def test_dialogue_quotes(self):
        self.assertNotIn('"', expand_dialogue_quotes('אמר "הציבור עשה את שלו"', "he"))

    def test_percent(self):
        self.assertEqual(expand_percent_symbols("50%", lang="en"), "50 percent")
        self.assertEqual(expand_percent_symbols("50%", lang="he"), "50 אחוז")

    def test_plus_sign(self):
        self.assertEqual(expand_plus_sign("gear + laptop", lang="en"), "gear plus laptop")

    def test_standalone_dash_is_dropped(self):
        self.assertEqual(strip_silent_separator_tokens("שלום - עולם"), "שלום עולם")
        self.assertEqual(strip_silent_separator_tokens("4 - 5"), "4 5")

    def test_in_word_hyphen_survives(self):
        self.assertEqual(strip_silent_separator_tokens("well-known"), "well-known")

    def test_list_markers_hebrew(self):
        self.assertTrue(expand_list_markers("1. שלום", lang="he").startswith("אחד."))

    def test_list_markers_latin(self):
        # Regression: the lookahead used to require a Hebrew letter, so Latin
        # lists kept a bare digit.
        self.assertTrue(expand_list_markers("1. First item", lang="en").startswith("one."))


class TestSlowSegments(unittest.TestCase):
    def test_split_without_markers(self):
        self.assertEqual(split_slow_segments("plain text"), [("plain text", False)])

    def test_split_marks_inner_span(self):
        # Every segment but the last ends in a comma: on its own a segment would
        # get the tokenizer's sentence-final full stop in mid-sentence.
        text = f"before {SLOW_MARK_OPEN}code{SLOW_MARK_CLOSE} after"
        self.assertEqual(
            split_slow_segments(text),
            [("before,", False), ("code,", True), ("after", False)],
        )

    def test_split_keeps_existing_punctuation(self):
        text = f"call me: {SLOW_MARK_OPEN}code{SLOW_MARK_CLOSE}. Bye"
        self.assertEqual(
            split_slow_segments(text),
            [("call me:", False), ("code.", True), ("Bye", False)],
        )

    def test_trailing_punctuation_merges_into_previous(self):
        text = f"{SLOW_MARK_OPEN}code{SLOW_MARK_CLOSE}."
        self.assertEqual(split_slow_segments(text), [("code.", True)])

    def test_strip_markers_keeps_content(self):
        text = f"a {SLOW_MARK_OPEN}b{SLOW_MARK_CLOSE} c"
        self.assertEqual(strip_slow_markers(text), "a b c")

    def test_prepare_can_omit_markers(self):
        out = prepare_text_for_synthesis("ההזמנה IL-4829", lang="he", mark_slow=False)
        self.assertNotIn(SLOW_MARK_OPEN, out)
        self.assertNotIn(SLOW_MARK_CLOSE, out)


class TestPrepareEndToEnd(unittest.TestCase):
    def test_plain_text_is_unchanged_apart_from_final_spacing(self):
        self.assertEqual(
            prepare_text_for_synthesis("שלום, זהו מודל דיבור בעברית.", lang="he"),
            "שלום, זהו מודל דיבור בעברית.",
        )

    def test_price_reads_as_thousands(self):
        out = prepare_text_for_synthesis("המחיר 1,500 שקלים", lang="he")
        self.assertIn("אלף וחמש מאות", out)
        self.assertNotIn("נקודה", out)

    def test_no_digits_survive_a_mixed_message(self):
        out = prepare_text_for_synthesis(
            "ההזמנה IL-4829-7361-05 תגיע ב 12/05/2024 בשעה 08:15, מחיר 1,500 ש\"ח (50%)",
            lang="he",
        )
        self.assertFalse(any(ch.isdigit() for ch in strip_slow_markers(out)), out)


class TestEnSpanProtection(unittest.TestCase):
    """No expander may nest a tag pair or a slow marker inside an ``<en>`` span.

    ``split_slow_segments`` cuts on ``【…】`` without looking at tags, so a marker
    landing inside a span orphans its ``<en>``/``</en>`` and the remaining
    English is phonemized by the outer language's G2P.
    """

    def assert_spans_are_well_formed(self, out: str) -> None:
        self.assertEqual(out.count("<en>"), out.count("</en>"), out)
        depth = 0
        for tag in re.findall(r"</?en>", out):
            depth += 1 if tag == "<en>" else -1
            self.assertIn(depth, (0, 1), f"nested or unbalanced <en>: {out!r}")
        for span in re.findall(r"<en>.*?</en>", out, re.DOTALL):
            self.assertNotIn(SLOW_MARK_OPEN, span, out)
            self.assertNotIn(SLOW_MARK_CLOSE, span, out)

    def test_email_with_digits_does_not_nest_spans(self):
        # Regression: expand_emails wraps the address, then the code speller used
        # to re-spell "user123" inside that span.
        out = prepare_text_for_synthesis("שלח לי מייל ל user123@gmail.com", lang="he")
        self.assert_spans_are_well_formed(out)
        self.assertIn("at gmail dot com", out)

    def test_slow_split_keeps_the_english_span_intact(self):
        out = prepare_text_for_synthesis("שלח לי מייל ל user123@gmail.com", lang="he")
        for segment, _is_slow in split_slow_segments(out):
            self.assert_spans_are_well_formed(segment)

    def test_time_inside_an_en_span_reads_in_english(self):
        out = prepare_text_for_synthesis("פגישה <en>at 08:15 sharp</en> מחר", lang="he")
        self.assert_spans_are_well_formed(out)
        self.assertIn("eight fifteen", out)
        self.assertNotIn("שמונה", out)

    def test_date_inside_an_en_span_reads_in_english(self):
        out = prepare_text_for_synthesis("ההזמנה <en>on 12/05/2024</en>", lang="he")
        self.assert_spans_are_well_formed(out)
        self.assertIn("May", out)

    def test_percent_and_ratio_inside_an_en_span_read_in_english(self):
        out = prepare_text_for_synthesis("<en>50% at a 2:3 rate</en>", lang="he")
        self.assert_spans_are_well_formed(out)
        self.assertIn("percent", out)
        self.assertIn(" to ", out)
        self.assertNotIn("אחוז", out)

    def test_code_inside_an_en_span_is_left_to_english_g2p(self):
        out = prepare_text_for_synthesis("הקוד <en>is TKT-90254</en> בבקשה", lang="he")
        self.assert_spans_are_well_formed(out)
        self.assertIn("TKT-90254", out)

    def test_code_outside_a_span_is_still_spelled(self):
        out = prepare_text_for_synthesis("הקוד הוא TKT-90254 בבקשה", lang="he")
        self.assert_spans_are_well_formed(out)
        self.assertIn("T K T", out)
        self.assertIn(SLOW_MARK_OPEN, out)

    def test_unmatched_open_tag_does_not_swallow_text(self):
        out = prepare_text_for_synthesis("שלום <en>hello", lang="he")
        self.assertIn("hello", out)


class TestReportedReadings(unittest.TestCase):
    """The five readings reported on BlueV3 with ``synthesize()`` defaults."""

    def test_price_reads_amount_then_unit(self):
        # Was "dollars one point two five. fifty".
        self.assertEqual(
            prepare_text_for_synthesis("The total is $1,250.50 today.", lang="en"),
            "The total is one thousand two hundred fifty dollars and fifty cents today.",
        )

    def test_english_ordinal_is_a_word_not_a_code(self):
        # Was spelled as a code: "R D three".
        self.assertEqual(
            prepare_text_for_synthesis("She finished 3rd in the race.", lang="en"),
            "She finished third in the race.",
        )

    def test_german_ordinal_date(self):
        # Was read as a list counter: "drei. Mai".
        self.assertEqual(
            prepare_text_for_synthesis("Wir treffen uns am 3. Mai.", lang="de"),
            "Wir treffen uns am dritten Mai.",
        )

    def test_times_dates_prices_are_one_inline_segment(self):
        # Were split into slow passes with a full stop after each piece.
        cases = [
            ("en", "The meeting is at 14:30 on 25/12/2024 and costs $40."),
            ("es", "La reunión es a las 14:30 del 12/05/2024 y cuesta 40 €."),
            ("it", "La riunione è alle 14:30 del 12/05/2024 e costa 40 €."),
            ("de", "Der Zug fährt am 12.05.2024 um 14:30 und kostet 40 €."),
            ("he", "הפגישה בשעה 14:30 ב-12/05/2024 ועולה 40 ₪."),
        ]
        for lang, text in cases:
            out = prepare_text_for_synthesis(text, lang=lang)
            self.assertEqual(split_slow_segments(out), [(out, False)], (lang, out))
            self.assertFalse(any(ch.isdigit() for ch in out), (lang, out))
            self.assertEqual(out.count("."), 1, (lang, out))

    def test_english_text_gets_no_language_spans(self):
        for text in (
            "She finished 3rd in the race.",
            "Your booking code is BA2490.",
            "Write to john.doe@example.com today.",
            "Call me at 555-123-4567.",
        ):
            out = prepare_text_for_synthesis(text, lang="en")
            self.assertNotIn("<en>", out, out)


class TestSlowSpansOnlyForDigitStrings(unittest.TestCase):
    def test_phone_and_code_are_slow_without_a_full_stop(self):
        out = prepare_text_for_synthesis("Call 555-123-4567 or use code BA2490 today.", lang="en")
        segments = split_slow_segments(out)
        self.assertEqual(
            segments,
            [
                ("Call,", False),
                ("five five five, one two three, four five six seven,", True),
                ("or use code,", False),
                ("B A two four nine zero,", True),
                ("today.", False),
            ],
        )

    def test_code_letters_keep_their_order(self):
        self.assertIn("X seven K nine Q two", prepare_text_for_synthesis("X7K9Q2", lang="en"))

    def test_code_letters_are_spelled_in_the_text_language(self):
        # Only Hebrew needs English letter names: its G2P has none for Latin letters.
        self.assertIn("A B eins zwei", prepare_text_for_synthesis("AB12", lang="de"))
        self.assertIn("<en>A B</en>", prepare_text_for_synthesis("AB12", lang="he"))


class TestLocales(unittest.TestCase):
    def check(self, lang, cases):
        for text, expected in cases:
            self.assertEqual(prepare_text_for_synthesis(text, lang=lang), expected, text)

    def test_english(self):
        self.check("en", [
            ("It costs €20 or £15.50.", "It costs twenty euros or fifteen pounds and fifty pence."),
            ("The fine is $1.", "The fine is one dollar."),
            ("He was born on the 21st of June.", "He was born on the twenty first of June."),
            ("We met on March 3, 2021.", "We met on March third, twenty twenty one."),
            ("The deadline is 25/12/2024.", "The deadline is December twenty fifth twenty twenty four."),
            ("The train leaves at 9:05.", "The train leaves at nine oh five."),
            ("Born in 1999.", "Born in nineteen ninety nine."),
            ("The 1990s were different.", "The nineteen nineties were different."),
            ("Sales grew by 12.5%.", "Sales grew by twelve point five percent."),
            ("Add 0.5 liters.", "Add zero point five liters."),
        ])

    def test_german(self):
        self.check("de", [
            ("Der Preis beträgt 1.250,50 €.", "Der Preis beträgt eintausendzweihundertfünfzig Euro fünfzig."),
            ("Das Ticket kostet $30.", "Das Ticket kostet dreißig Dollar."),
            ("Heute ist der 1. Januar.", "Heute ist der erste Januar."),
            ("Das ist der 2. Versuch.", "Das ist der zweite Versuch."),
            ("Geboren am 3.10.1990.", "Geboren am dritten Oktober neunzehnhundertneunzig."),
            ("Wir treffen uns um 8:00 Uhr.", "Wir treffen uns um acht Uhr."),
            ("Der Zug fährt um 14:30 ab.", "Der Zug fährt um vierzehn Uhr dreißig ab."),
            ("Der Wert liegt bei 0,75.", "Der Wert liegt bei null Komma fünfundsiebzig."),
        ])

    def test_ranges_signs_and_dashed_dates(self):
        self.check("en", [
            ("Pages 10-20.", "Pages ten to twenty."),
            ("A 50-60% rise.", "A fifty to sixty percent rise."),
            ("It was -5 degrees.", "It was minus five degrees."),
            ("Due 12-05-2024.", "Due May twelfth twenty twenty four."),
            ("Call 911.", "Call nine one one."),
        ])
        self.assertEqual(prepare_text_for_synthesis("Seiten 10–20", lang="de"), "Seiten zehn bis zwanzig")

    def test_german_sentence_final_number_is_not_an_ordinal(self):
        self.assertEqual(
            prepare_text_for_synthesis("Ich habe 3. Dann gehe ich.", lang="de"),
            "Ich habe drei. Dann gehe ich.",
        )

    def test_spanish(self):
        self.check("es", [
            ("El total es de 1.250,50 €.", "El total es de mil doscientos cincuenta euros con cincuenta céntimos."),
            ("Pagó $21.", "Pagó veintiún dólares."),
            ("Quedó 3º en la carrera.", "Quedó tercero en la carrera."),
            ("Es la 1ª vez.", "Es la primera vez."),
            ("Nació el 1/3/1990.", "Nació el primero de marzo de mil novecientos noventa."),
            ("Es a las 14:30.", "Es a las catorce y treinta."),
            ("Tengo 21 libros.", "Tengo veintiún libros."),
            ("Mide 2,5 metros.", "Mide dos coma cinco metros."),
        ])

    def test_italian(self):
        self.check("it", [
            ("Il totale è di 1.250,50 €.", "Il totale è di milleduecentocinquanta euro e cinquanta centesimi."),
            ("Costa 1 €.", "Costa un euro."),
            ("È arrivato 3º.", "È arrivato terzo."),
            ("È la 1ª volta.", "È la prima volta."),
            ("È nato il 1/3/1990.", "È nato il primo marzo millenovecentonovanta."),
            ("Parte alle 9:15.", "Parte alle nove e quindici."),
            ("Il valore è 0,75.", "Il valore è zero virgola settantacinque."),
        ])

    def test_ordinary_text_is_unchanged(self):
        for lang, text in [
            ("en", "Please read the instructions before you begin."),
            ("de", "Bitte schließen Sie die Tür hinter sich."),
            ("es", "Buenos días, ¿cómo estás hoy?"),
            ("it", "Per favore, chiudi la porta quando esci."),
            ("he", "השמיים בהירים ומזג האוויר נעים."),
        ]:
            self.assertEqual(prepare_text_for_synthesis(text, lang=lang), text)


class TestHebrewFallback(unittest.TestCase):
    """Without hebrew-num2words: bare cardinals, but nothing split or stranded."""

    def test_prefix_stays_on_the_number(self):
        with without_hebrew_num2words():
            out = prepare_text_for_synthesis("הפגישה ב-14:30.", lang="he")
        self.assertEqual(out, "הפגישה בארבע עשרה ושלושים.")

    def test_currency_sign_becomes_a_word(self):
        with without_hebrew_num2words():
            self.assertEqual(prepare_text_for_synthesis("זה עולה 20 ₪.", lang="he"), "זה עולה עשרים שקלים.")

    def test_definite_ordinal(self):
        with without_hebrew_num2words():
            out = prepare_text_for_synthesis("הוא סיים במקום ה-3.", lang="he")
        self.assertEqual(out, "הוא סיים במקום השלישי.")


@unittest.skipUnless(HAS_HEBREW_NUM2WORDS, "hebrew-num2words not installed")
class TestHebrewNum2Words(unittest.TestCase):
    def check(self, cases):
        for text, expected in cases:
            self.assertEqual(prepare_text_for_synthesis(text, lang="he"), expected, text)

    def test_gender_agreement(self):
        self.check([
            ("יש לי 3 ילדים ו-2 בנות.", "יש לי שלושה ילדים ושתי בנות."),
            ("קניתי 5 ספרים.", "קניתי חמישה ספרים."),
        ])

    def test_prices(self):
        self.check([
            ('המחיר הוא 1,250 ש"ח.', "המחיר הוא אלף מאתיים וחמישים שקלים."),
            ("הכרטיס עולה 9.90 ₪.", "הכרטיס עולה תשעה שקלים ותשעים אגורות."),
            ("שילמתי $15 על הספר.", "שילמתי חמישה עשר דולר על הספר."),
        ])

    def test_dates_times_ordinals(self):
        self.check([
            ("הפגישה תתקיים ב-12/05/2024.", "הפגישה תתקיים בשנים עשר במאי אלפיים עשרים וארבע."),
            ("הרכבת יוצאת ב-9:15.", "הרכבת יוצאת בתשע ורבע."),
            ("זו הפעם ה-2 שאני כאן.", "זו הפעם השנייה שאני כאן."),
            ("המחירים עלו ב-3.5%.", "המחירים עלו בשלושה וחצי אחוז."),
        ])

    def test_codes_and_phones_are_still_spelled_first(self):
        out = prepare_text_for_synthesis("קוד IL4829, טלפון 054-1234567", lang="he")
        self.assertIn("<en>I L</en> ארבע שמונה שתיים תשע", out)
        self.assertIn("אפס חמש ארבע, אחת שתיים", out)


class TestEmoji(unittest.TestCase):
    def test_emoji_are_dropped(self):
        self.assertEqual(strip_emoji("Nice 🎉 party"), "Nice party")
        self.assertEqual(strip_emoji("😀"), "")

    def test_emoji_do_not_fuse_neighbouring_words(self):
        self.assertEqual(strip_emoji("Nice🎉party"), "Nice party")

    def test_prepare_strips_emoji_before_g2p(self):
        # Regression: espeak reads them aloud, so the tokenizer's own filter —
        # which runs after phonemization — never sees them.
        self.assertNotIn("🎉", prepare_text_for_synthesis("Nice 🎉 party", lang="en"))

    def test_flags_and_dingbats_are_dropped(self):
        self.assertEqual(strip_emoji("done ✅ 🇮🇱"), "done")

    def test_ordinary_punctuation_survives(self):
        self.assertEqual(strip_emoji("a — b, c."), "a — b, c.")


if __name__ == "__main__":
    unittest.main()
