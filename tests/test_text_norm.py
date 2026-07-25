"""Unit tests for the pre-G2P text normalizer (no ONNX runtime required)."""

from __future__ import annotations

import unittest

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
    strip_hebrew_abbreviation_quotes,
    strip_hebrew_inword_hyphens,
    strip_silent_separator_tokens,
    strip_slow_markers,
)


class TestNumbers(unittest.TestCase):
    def test_thousands_separator_is_not_a_decimal(self):
        # Regression: float("1,500".replace(",", ".")) == 1.5 read prices as decimals.
        self.assertEqual(expand_numbers("1,500", lang="en"), "one thousand, five hundred")
        self.assertNotIn("point", expand_numbers("1,500", lang="en"))

    def test_multiple_thousands_groups(self):
        self.assertEqual(
            expand_numbers("12,345,678", lang="en"),
            "twelve million, three hundred and forty-five thousand, six hundred and seventy-eight",
        )

    def test_decimal_point_still_decimal(self):
        self.assertEqual(expand_numbers("3.5", lang="en"), "three point five")

    def test_comma_decimal_when_not_a_thousands_group(self):
        self.assertEqual(expand_numbers("3,5", lang="en"), "three point five")

    def test_hebrew_integer(self):
        self.assertEqual(expand_numbers("1,500", lang="he"), "אלף וחמש מאות")

    def test_comma_decimal_locale_groups_with_dots(self):
        # German writes 1.500 for a thousand and 3,5 for three and a half.
        self.assertEqual(expand_numbers("1.500", lang="de"), "eintausendfünfhundert")
        self.assertEqual(expand_numbers("3,5", lang="de"), "drei Komma fünf")

    def test_dot_grouping_not_applied_to_english(self):
        self.assertEqual(expand_numbers("1.500", lang="en"), "one point five")

    def test_digits_in_en_span_are_left_alone(self):
        self.assertEqual(
            expand_numbers("<en>GPT 4</en> ו 5", lang="he"),
            "<en>GPT 4</en> ו חמש",
        )

    def test_digits_in_slow_span_are_left_alone(self):
        marked = f"{SLOW_MARK_OPEN}4 8{SLOW_MARK_CLOSE} 9"
        self.assertEqual(expand_numbers(marked, lang="en"), f"{marked[:-2]} nine")


class TestTimesAndRatios(unittest.TestCase):
    def test_hebrew_clock_time(self):
        out = expand_times("בשעה 08:15", lang="he")
        self.assertIn("שמונה וחמש עשרה", out)
        self.assertIn(SLOW_MARK_OPEN, out)

    def test_non_hebrew_clock_time_is_a_time_not_a_ratio(self):
        # Regression: expand_times used to be Hebrew-only, so expand_ratios turned
        # "08:15" into "eight to fifteen".
        out = prepare_text_for_synthesis("the show starts at 08:15", lang="en")
        self.assertIn("eight fifteen", out)
        self.assertNotIn(" to ", out)

    def test_whole_hour_drops_minutes(self):
        self.assertIn("eight", expand_times("at 08:00", lang="en"))
        self.assertNotIn("zero", expand_times("at 08:00", lang="en"))

    def test_out_of_range_time_untouched(self):
        self.assertEqual(expand_times("25:99", lang="he"), "25:99")

    def test_ratio(self):
        self.assertEqual(expand_ratios("2:3", lang="en"), "2 to 3")
        self.assertEqual(expand_ratios("2:3", lang="he"), "2 ל 3")


class TestDates(unittest.TestCase):
    def test_hebrew_date(self):
        out = expand_dates("12/05/2024", lang="he")
        self.assertIn("לחמישי", out)
        self.assertIn(SLOW_MARK_OPEN, out)

    def test_latin_date_becomes_words_not_digits(self):
        out = expand_dates("12/05/2024", lang="en")
        self.assertIn("May", out)
        self.assertIn("twelfth", out)
        self.assertNotIn("12", out)

    def test_two_digit_year(self):
        self.assertIn("two thousand and twenty-four", expand_dates("12.5.24", lang="en"))

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
        self.assertIn("אפס שלוש חמש חמש חמש אחת שתיים שלוש ארבע", out)

    def test_star_number(self):
        self.assertIn("כוכבית שש שבע אפס אפס", expand_phone_numbers("*6700", lang="he"))

    def test_non_hebrew_untouched(self):
        self.assertEqual(expand_phone_numbers("call 03-5551234", lang="en"),
                         "call 03-5551234")


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
        text = f"before {SLOW_MARK_OPEN}code{SLOW_MARK_CLOSE} after"
        self.assertEqual(
            split_slow_segments(text),
            [("before", False), ("code", True), ("after", False)],
        )

    def test_trailing_punctuation_merges_into_previous(self):
        text = f"{SLOW_MARK_OPEN}code{SLOW_MARK_CLOSE}."
        self.assertEqual(split_slow_segments(text), [("code.", True)])

    def test_strip_markers_keeps_content(self):
        text = f"a {SLOW_MARK_OPEN}b{SLOW_MARK_CLOSE} c"
        self.assertEqual(strip_slow_markers(text), "a b c")

    def test_prepare_can_omit_markers(self):
        out = prepare_text_for_synthesis("ההזמנה 12/05/2024", lang="he", mark_slow=False)
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


if __name__ == "__main__":
    unittest.main()
