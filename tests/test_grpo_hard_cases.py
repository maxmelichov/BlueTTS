"""GRPO hard cases (FlowTTS-GRPO Sec 2.5, Table 1): the same word positions are repeated in the IPA and the ASR text."""

from __future__ import annotations

import random
import unittest

try:
    from training.rl.train_grpo import hard_variant
except ImportError:  # the inference install has no torch / bluecodec
    hard_variant = None


@unittest.skipIf(hard_variant is None, "training dependencies not installed")
class TestHardCases(unittest.TestCase):
    rows = {
        "en": ("ðə stˈɑːɹz bɪɡˈæn tə kɹˈʌmbəl.", "The stars began to crumble."),
        "de": ("vɪɐ tʁˈɛfən ʊns mˈɔʁɡən.", "Wir treffen uns morgen."),
        "es": ("mañˈana bˈamos al mˈeɾkado.", "Mañana vamos al mercado."),
        "he": ("ʔanˈaχnu nifɡaʃˈim maχˈaʁ.", None),               # no orthographic text: the IPA alone
    }

    def check_aligned(self, ipa, txt, src_ipa, src_txt):
        if txt is None:
            return
        self.assertEqual(len(ipa.split()), len(txt.split()))
        # every IPA word copy sits at the position of a copy of the same text word
        orig = dict(zip([w.strip(".") for w in src_ipa.split()], [w.strip(".") for w in src_txt.split()]))
        for i, t in zip(ipa.split(), txt.split()):
            self.assertEqual(orig[i.strip(".,")], t.strip(".,"))

    def test_lwr(self):
        for lang, (ipa0, txt0) in self.rows.items():
            for seed in range(20):
                kind, ipa, txt = hard_variant(ipa0, txt0, "lwr", random.Random(seed))
                self.assertEqual(kind, "lwr")
                self.assertTrue(2 <= len(ipa.split()) - len(ipa0.split()) <= 4)     # one word, 3-5 copies
                self.assertTrue(ipa.endswith(".") and ipa.count(".") == 1)
                self.check_aligned(ipa, txt, ipa0, txt0)

    def test_smr(self):
        for lang, (ipa0, txt0) in self.rows.items():
            for seed in range(20):
                kind, ipa, txt = hard_variant(ipa0, txt0, "smr", random.Random(seed))
                n0 = len(ipa0.split())
                self.assertTrue(n0 + 2 <= len(ipa.split()) <= n0 + 6)              # 2-3 words, 2-3 copies each
                self.check_aligned(ipa, txt, ipa0, txt0)

    def test_gsr(self):
        kind, ipa, txt = hard_variant(*self.rows["en"], "gsr", random.Random(0))
        self.assertEqual(ipa, "ðə stˈɑːɹz bɪɡˈæn tə kɹˈʌmbəl, ðə stˈɑːɹz bɪɡˈæn tə kɹˈʌmbəl.")
        self.assertEqual(txt, "The stars began to crumble, The stars began to crumble.")
        kind, ipa, txt = hard_variant(*self.rows["he"], "gsr", random.Random(0))
        self.assertEqual(ipa, "ʔanˈaχnu nifɡaʃˈim maχˈaʁ, ʔanˈaχnu nifɡaʃˈim maχˈaʁ.")
        self.assertIsNone(txt)

    def test_fallback_to_gsr(self):
        # word counts differ, or the text has digits / hyphens / dashes / symbols: positions cannot be matched
        for txt in ("Twenty-one stars began crumbling.", "The 3 stars began crumbling.", "Stars & moons began now."):
            for kind in ("lwr", "smr"):
                k, ipa, t = hard_variant("ðə θɹˈiː stˈɑːɹz bɪɡˈæn kɹˈʌmblɪŋ.", txt, kind, random.Random(0))
                self.assertEqual(k, "gsr")
                self.assertTrue(t.startswith(txt.rstrip(".") + ", "))
        k, _, _ = hard_variant("a b c.", "A b c.", "lwr", random.Random(0))     # no word of 2+ letters
        self.assertEqual(k, "gsr")


if __name__ == "__main__":
    unittest.main()
