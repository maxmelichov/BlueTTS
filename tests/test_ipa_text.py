"""Unit tests for blue_onnx.ipa_text (release bundles: IPA normalization and <lang> tags)."""

import unittest

from src.blue_onnx.ipa_text import encode_tagged, normalize_ipa


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


if __name__ == "__main__":
    unittest.main()
