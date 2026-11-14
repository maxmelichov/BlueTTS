"""Unit tests for blue_onnx.ipa_text (release bundles: IPA normalization and <lang> tags)."""

import json
import unittest
from pathlib import Path

from src.blue_onnx import TextProcessor
from src.blue_onnx.ipa_text import (
    encode_tagged,
    normalize_ipa,
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
    """Release bundles, mixed text: each span gets its own language's view, the join is normalised once."""

    def mixed(self, text, outputs):
        tp = TextProcessor(release={"inference": {}})
        tp._phonemize_segment = lambda content, lang: outputs[(lang, content)]
        return tp.phonemize(text, "he")

    def test_span_views_and_one_wrap(self):
        out = self.mixed("רגע <en>cats go</en> בטח", {("he", "רגע"): "rega", ("en", "cats go"): "kˈæts ɡoʊ.",
                                                     ("he", "בטח"): "betaχ"})
        # Hebrew rules on the Hebrew spans only; the English span loses the '.' its G2P added mid-sentence
        self.assertEqual(out, "<he>ʁeɡa kˈæʦ ɡoʊ betaχ</he>")
        ids = encode_tagged(out, "he", VOCAB)
        self.assertEqual(ids, _tagged(normalize_ipa("ʁeɡa kˈæʦ ɡoʊ betaχ", lang="he"), "he"))
        self.assertEqual(sum(i == VOCAB["<"] for i in ids), 2)   # the chunk's own '<he>' and '</he>' only

    def test_punctuation_between_spans_kept(self):
        out = self.mixed("<en>Hello</en>, מה שלומך?", {("en", "Hello"): "həlˈoʊ.", ("he", ", מה שלומך?"): ", mˈa ʃlomχˈa?"})
        self.assertEqual(out, "<he>həlˈoʊ , mˈa ʃlomχˈa?</he>")
        self.assertEqual(normalize_ipa(_strip(out), lang="he"), "həlˈoʊ, mˈa ʃlomχˈa?")

    def test_pure_hebrew_ids_unchanged(self):
        ipa = "ʃalˈom, ʔanˈi medabˈeʁ ʔivʁˈit  ! mˈa ʃlomχˈa ? ʁega bats"
        self.assertEqual(encode_tagged(ipa, "he", VOCAB), _tagged(normalize_ipa(ipa, lang="he"), "he"))
        self.assertEqual(encode_tagged(f"<he>{ipa}</he>", "he", VOCAB), _tagged(normalize_ipa(ipa, lang="he"), "he"))


def _strip(s: str) -> str:
    import re
    return re.sub(r"</?\w+>", "", s)


def _tagged(text: str, code: str) -> list:
    tag = [VOCAB[c] for c in code]
    return [VOCAB["<"], *tag, VOCAB[">"]] + [VOCAB.get(c, 0) for c in text] + [VOCAB["<"], VOCAB["/"], *tag, VOCAB[">"]]


if __name__ == "__main__":
    unittest.main()
