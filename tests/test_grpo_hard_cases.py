"""GRPO hard cases (FlowTTS-GRPO Table 1): the same word positions are repeated in the IPA and the ASR text."""

from __future__ import annotations

import random
import unittest

try:
    from training.rl.train_grpo import hard_case
except ImportError:  # the inference install has no torch / bluecodec
    hard_case = None


@unittest.skipIf(hard_case is None, "training dependencies not installed")
class TestHardCases(unittest.TestCase):
    ipa, txt = "ðə stˈɑːɹz bɪɡˈæn tə kɹˈʌmbəl.", "The stars began to crumble."

    def test_lwr_repeats_one_word_in_both(self):
        for seed in range(20):
            ipa, txt = hard_case(self.ipa, self.txt, "lwr", random.Random(seed))
            extra = len(ipa.split()) - 5
            self.assertTrue(2 <= extra <= 4)                    # one word, 3-5 copies
            self.assertEqual(len(txt.split()), len(ipa.split()))
            self.assertTrue(ipa.endswith(".") and ipa.count(".") == 1)

    def test_smr_and_gsr(self):
        ipa, txt = hard_case(self.ipa, self.txt, "smr", random.Random(0))
        self.assertEqual(len(txt.split()), len(ipa.split()))
        self.assertTrue(7 <= len(ipa.split()) <= 11)
        ipa, txt = hard_case(self.ipa, self.txt, "gsr", random.Random(0))
        self.assertEqual(ipa, "ðə stˈɑːɹz bɪɡˈæn tə kɹˈʌmbəl, ðə stˈɑːɹz bɪɡˈæn tə kɹˈʌmbəl.")
        self.assertEqual(txt, "The stars began to crumble, The stars began to crumble.")

    def test_word_count_mismatch(self):
        self.assertIsNone(hard_case("a bb cc", "a bb", "lwr", random.Random(0)))
        self.assertIsNotNone(hard_case("a bb cc", "a bb", "gsr", random.Random(0)))


if __name__ == "__main__":
    unittest.main()
