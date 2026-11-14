"""GRPO ASR reward: word normalisation and the capped edit-distance error rate, on synthetic transcripts."""

from __future__ import annotations

import unittest

try:
    import torch  # noqa: F401
    import torchaudio  # noqa: F401
except ImportError:  # the inference install has no torch
    torch = None

if torch is not None:
    from training.rl.rewards import error_rate, words


@unittest.skipIf(torch is None, "torch not installed")
class TestRewards(unittest.TestCase):
    def test_error_rate(self):
        self.assertEqual(error_rate("a b c".split(), "a b c".split()), 0.0)
        self.assertEqual(error_rate("a b c d".split(), "a x c".split()), 0.5)       # 1 substitution + 1 deletion
        self.assertEqual(error_rate("a b".split(), "a a b b".split()), 1.0)         # 2 insertions
        self.assertEqual(error_rate("a".split(), "x y z w".split()), 1.0)           # capped at 1
        self.assertEqual(error_rate([], []), 0.0)

    def test_words(self):
        # Whisper's basic normaliser without diacritics; Russian ё -> е
        self.assertEqual(words("Mañana, ¿vamos al MERCADO?"), ["manana", "vamos", "al", "mercado"])
        self.assertEqual(words("Wir treffen uns morgen."), ["wir", "treffen", "uns", "morgen"])
        self.assertEqual(words("Всё хорошо!"), ["все", "хорошо"])
        self.assertEqual(words("È così."), ["e", "cosi"])

    def test_synthetic_transcripts(self):
        # 1 - WER: an exact transcript in another casing / punctuation / accents scores 0 error
        ref = "Mañana vamos al mercado con los niños."
        self.assertEqual(error_rate(words(ref), words("mañana vamos al mercado, con los ninos")), 0.0)
        self.assertAlmostEqual(error_rate(words(ref), words("Mañana vamos mercado con los niños.")), 1 / 7)
        self.assertEqual(error_rate(words("Всё хорошо."), words("все хорошо")), 0.0)

    def test_english_normaliser(self):
        try:
            from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
        except ImportError:
            self.skipTest("transformers not installed")
        en = EnglishTextNormalizer({})
        self.assertEqual(error_rate(words("Mr. Smith isn't here.", en), words("mister smith is not here", en)), 0.0)


if __name__ == "__main__":
    unittest.main()
