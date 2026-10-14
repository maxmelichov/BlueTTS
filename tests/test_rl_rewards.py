"""GRPO ASR reward: word normalisation and the capped edit-distance error rate."""

from __future__ import annotations

import unittest

try:
    import torch  # noqa: F401
    import torchaudio  # noqa: F401
except ImportError:  # the inference install has no torch
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class TestRewards(unittest.TestCase):
    def test_error_rate(self):
        from training.rl.rewards import error_rate

        self.assertEqual(error_rate("a b c".split(), "a b c".split()), 0.0)
        self.assertEqual(error_rate("a b c d".split(), "a x c".split()), 0.5)       # 1 substitution + 1 deletion
        self.assertEqual(error_rate("a b".split(), "a a b b".split()), 1.0)         # 2 insertions
        self.assertEqual(error_rate("a".split(), "x y z w".split()), 1.0)           # capped at 1
        self.assertEqual(error_rate([], []), 0.0)

    def test_words(self):
        from training.rl.rewards import words

        self.assertEqual(words("Hello, World! It's  fine."), ["hello", "world", "it", "s", "fine"])
        self.assertEqual(words("ʃalˈom, ʔolˈam."), ["ʃalˈom", "ʔolˈam"])


if __name__ == "__main__":
    unittest.main()
