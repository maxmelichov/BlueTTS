"""Unit tests for blue_onnx.ipa_text (release bundles: IPA normalization and <lang> tags)."""

import json
import unittest
from pathlib import Path

from src.blue_onnx import chunk_text
from src.blue_onnx.ipa_text import (
    encode_tagged,
    normalize_ipa,
    normalize_ipa_spans,
    strip_outer_lang_tags,
    tag_latin_runs,
)

VOCAB = json.loads((Path(__file__).resolve().parents[1] / "src/blue_onnx/vocab.json").read_text())["char_to_id"]


class TestNormalizeIpa(unittest.TestCase):
    def test_terminal_mark_and_spacing(self):
        self.assertEqual(normalize_ipa("hɛloʊ  wɝld"), "hɛloʊ wɝld.")
        self.assertEqual(normalize_ipa("hɛloʊ ?"), "hɛloʊ?")

    def test_affricates(self):
        self.assertEqual(normalize_ipa("tʃɪp dʒɑb", final_period=False), "ʧɪp ʤɑb")

    def test_hebrew(self):
        # r -> ʁ, g -> ɡ, and /ts/ stays spelled 'ts'
        self.assertEqual(normalize_ipa("ʃalom ʁega bats", lang="he"), "ʃalom ʁeɡa bats.")
        self.assertEqual(normalize_ipa("ʃalom rega", lang="he"), "ʃalom ʁeɡa.")


class TestEncodeTagged(unittest.TestCase):
    def test_tags_wrap_ids(self):
        chars = "<>/enabc."
        c2i = {ch: i + 1 for i, ch in enumerate(chars)}
        ids = encode_tagged("<en>abc</en>", "en", c2i)
        head = [c2i["<"], c2i["e"], c2i["n"], c2i[">"]]
        tail = [c2i["<"], c2i["/"], c2i["e"], c2i["n"], c2i[">"]]
        self.assertEqual(ids, head + [c2i["a"], c2i["b"], c2i["c"], c2i["."]] + tail)

    def test_unknown_symbol_is_pad_and_region_code(self):
        chars = "<>/es."
        c2i = {ch: i + 1 for i, ch in enumerate(chars)}
        ids = encode_tagged("ʒ", "es-419", c2i, pad_id=0)
        self.assertEqual(ids[4], 0)
        self.assertEqual(ids[:4], [c2i["<"], c2i["e"], c2i["s"], c2i[">"]])



class TestTagLatinRuns(unittest.TestCase):
    def test_words_and_boundaries(self):
        # spaces and punctuation around a run stay in the Hebrew text, once, where they were
        self.assertEqual(tag_latin_runs("שלום Google, מה נשמע?", "he"), "שלום <en>Google</en>, מה נשמע?")
        self.assertEqual(tag_latin_runs("Hello, מה שלומך? Goodbye", "he"), "<en>Hello</en>, מה שלומך? <en>Goodbye</en>")
        self.assertEqual(tag_latin_runs("ראיתי ב-Netflix.", "he"), "ראיתי ב-<en>Netflix</en>.")
        self.assertEqual(tag_latin_runs("(OK)", "he"), "(<en>OK</en>)")
        self.assertEqual(tag_latin_runs("בGoogle", "he"), "ב<en>Google</en>")

    def test_one_run_per_phrase(self):
        self.assertEqual(tag_latin_runs("גר ב-New York, ליד Times Square!", "he"),
                         "גר ב-<en>New York</en>, ליד <en>Times Square</en>!")

    def test_inner_marks_and_digits(self):
        self.assertEqual(tag_latin_runs("it's fine, אמרתי", "he"), "<en>it's fine</en>, אמרתי")
        self.assertEqual(tag_latin_runs("אין Wi-Fi כאן", "he"), "אין <en>Wi-Fi</en> כאן")
        self.assertEqual(tag_latin_runs("ה-GPT-4 חדש", "he"), "ה-<en>GPT-4</en> חדש")
        self.assertEqual(tag_latin_runs("רשת 5G ו-MP3", "he"), "רשת <en>5G</en> ו-<en>MP3</en>")
        self.assertEqual(tag_latin_runs("ה-iPhone 15 Pro יצא", "he"), "ה-<en>iPhone 15 Pro</en> יצא")
        self.assertEqual(tag_latin_runs("Python 3.12.", "he"), "<en>Python 3.12</en>.")
        self.assertEqual(tag_latin_runs("iPhone 15, 16 ו-17", "he"), "<en>iPhone 15</en>, 16 ו-17")
        self.assertEqual(tag_latin_runs("באתר www.google.com.", "he"), "באתר <en>www.google.com</en>.")

    def test_acronym_tail_and_accents(self):
        self.assertEqual(tag_latin_runs("אני BlueTTS", "he"), "אני <en>Blue TTS</en>")
        self.assertEqual(tag_latin_runs("ChatGPT של OpenAI", "he"), "<en>Chat GPT</en> של <en>Open AI</en>")
        self.assertEqual(tag_latin_runs("NASA ו-iPhone", "he"), "<en>NASA</en> ו-<en>iPhone</en>")
        self.assertEqual(tag_latin_runs("קפה ב-café", "he"), "קפה ב-<en>cafe</en>")

    def test_spans_tags_and_other_languages_untouched(self):
        for text in ("שלום <en>Hello world</en>.", "שלום <es>hola amigo</es>!", "<he>שלום</he> end"):
            self.assertEqual(tag_latin_runs(text, "he").count("<en>"), text.count("<en>") + text.endswith("end"))
        self.assertEqual(tag_latin_runs("שלום 【<en>G P T</en> ארבע .】 X", "he"),
                         "שלום 【<en>G P T</en> ארבע .】 <en>X</en>")
        self.assertEqual(tag_latin_runs("Hello world", "en"), "Hello world")
        self.assertEqual(tag_latin_runs("שלום, מה נשמע? 12 ש\"ח.", "he"), "שלום, מה נשמע? 12 ש\"ח.")


class TestInlineSpans(unittest.TestCase):
    def test_span_gets_its_own_normalisation(self):
        # English 'ʦ' (cats) stays a ligature; the Hebrew around it is spelled 'ts' with r -> ʁ, g -> ɡ
        self.assertEqual(normalize_ipa_spans("matsˈav ʁeɡa, <en>kˈæʦ ɡoʊ</en> ! bats", "he"),
                         "matsˈav ʁeɡa, kˈæʦ ɡoʊ! bats.")
        self.assertEqual(normalize_ipa("matsˈav ʁeɡa, kˈæʦ ɡoʊ ! bats", lang="he"), "matsˈav ʁeɡa, kˈæts ɡoʊ! bats.")

    def test_boundary_punctuation_kept_once(self):
        self.assertEqual(normalize_ipa_spans("<en>həlˈoʊ</en> , mˈa ʃlomχˈa ? <en>ɡʊdbˈaɪ</en>", "he"),
                         "həlˈoʊ, mˈa ʃlomχˈa? ɡʊdbˈaɪ.")
        self.assertEqual(normalize_ipa_spans("ʃalˈom <en>ˌoʊkˈeɪ</en>.", "he"), "ʃalˈom ˌoʊkˈeɪ.")

    def test_without_foreign_spans_identical(self):
        cases = [("ʃalˈom ,  ma niʃmˈa  ? bats", "he"), ("<he>ʃalˈom  ,</he> <he>ʁeɡa</he>", "he"),
                 ("kˈæts ɡoʊ  ,", "en"), ("<en>kˈæts</en> ɡoʊ", "en")]
        for ipa, lang in cases:
            self.assertEqual(normalize_ipa_spans(ipa, lang), normalize_ipa(_strip(ipa), lang=lang))

    def test_spans_have_no_ids(self):
        ids = encode_tagged("ʃalˈom <en>kˈæʦ</en>", "he", VOCAB)
        self.assertEqual(ids, _tagged("ʃalˈom kˈæʦ.", "he"))
        self.assertEqual(sum(i == VOCAB["<"] for i in ids), 2)   # the chunk's own '<he>' and '</he>' only

    def test_pure_hebrew_ids_unchanged(self):
        ipa = "ʃalˈom, ʔanˈi medabˈeʁ ʔivʁˈit  ! mˈa ʃlomχˈa ? ʁega bats"
        self.assertEqual(encode_tagged(ipa, "he", VOCAB), _tagged(normalize_ipa(ipa, lang="he"), "he"))
        self.assertEqual(encode_tagged(f"<he>{ipa}</he>", "he", VOCAB), _tagged(normalize_ipa(ipa, lang="he"), "he"))

    def test_strip_outer_lang_tags(self):
        self.assertEqual(strip_outer_lang_tags("<he>ʃalˈom,</he> <en>blˈuː</en> <he>!</he>", "he"), "ʃalˈom, <en>blˈuː</en> !")
        self.assertEqual(strip_outer_lang_tags("<he>ʃalˈom</he>  <he>ma</he>", "he"), "ʃalˈom ma")

    def test_chunks_close_and_reopen_a_cut_span(self):
        text = "ʃalˈom. <en>həlˈoʊ. wˈɜːld.</en> ma"
        self.assertEqual(chunk_text(text, max_len=12), ["ʃalˈom.", "<en>həlˈoʊ.</en>", "<en>wˈɜːld.</en> ma"])
        # tags are not counted: the same cuts as the untagged text
        plain = "ʃalˈom. həlˈoʊ. wˈɜːld. ma"
        self.assertEqual([_strip(c) for c in chunk_text(text, max_len=16)], chunk_text(plain, max_len=16))


def _strip(s: str) -> str:
    import re
    return re.sub(r"</?\w+>", "", s)


def _tagged(text: str, code: str) -> list:
    tag = [VOCAB[c] for c in code]
    return [VOCAB["<"], *tag, VOCAB[">"]] + [VOCAB.get(c, 0) for c in text] + [VOCAB["<"], VOCAB["/"], *tag, VOCAB[">"]]


if __name__ == "__main__":
    unittest.main()
