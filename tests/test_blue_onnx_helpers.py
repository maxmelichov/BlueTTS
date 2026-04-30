"""Unit tests for pure helpers (no ONNX runtime required)."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from src.blue_onnx import (
    AVAILABLE_LANGS,
    UnicodeProcessor,
    blend_duration_pace,
    chunk_text,
    length_to_mask,
    strip_lang_tags_from_phoneme_string,
)


class TestBlendDurationPace(unittest.TestCase):
    def test_blend_zero_returns_reshaped_dur(self):
        text_mask = np.ones((2, 1, 5), dtype=np.float32)
        dur = np.array([1.0, 2.0], dtype=np.float32)
        out = blend_duration_pace(dur, text_mask, 0.0, 0.0625)
        self.assertEqual(out.shape, (2,))
        np.testing.assert_allclose(out, dur)

    def test_blend_half_changes_values(self):
        text_mask = np.ones((1, 1, 4), dtype=np.float32)
        dur = np.array([4.0], dtype=np.float32)
        out = blend_duration_pace(dur, text_mask, 1.0, 0.0625)
        self.assertEqual(out.shape, (1,))
        self.assertGreater(abs(float(out[0]) - 4.0), 1e-6)


class TestStripLangTags(unittest.TestCase):
    def test_removes_tags_and_collapses_space(self):
        s = strip_lang_tags_from_phoneme_string("<en>a b</en>  <he>ג</he>")
        self.assertNotIn("<", s)
        self.assertNotIn(">", s)


class TestChunkText(unittest.TestCase):
    def test_short_sentences_single_chunk(self):
        self.assertEqual(chunk_text("Hi. There.", max_len=300), ["Hi. There."])

    def test_paragraph_split(self):
        chunks = chunk_text("First para.\n\nSecond para.", max_len=300)
        self.assertEqual(len(chunks), 2)


class TestLengthToMask(unittest.TestCase):
    def test_shape(self):
        m = length_to_mask(np.array([2, 3], dtype=np.int64))
        self.assertEqual(m.shape, (2, 1, 3))


class TestUnicodeProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        vocab = Path(__file__).resolve().parents[1] / "src" / "vocab.json"
        cls.proc = UnicodeProcessor(str(vocab))

    def test_invalid_lang_raises(self):
        with self.assertRaises(ValueError):
            self.proc._preprocess_text("hello", lang="xx")

    def test_available_langs_nonempty(self):
        self.assertIn("he", AVAILABLE_LANGS)


if __name__ == "__main__":
    unittest.main()
