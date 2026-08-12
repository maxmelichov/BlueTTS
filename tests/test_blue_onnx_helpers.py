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
    get_latent_mask,
    latent_frames_for_duration,
    length_to_mask,
    strip_lang_tags_from_phoneme_string,
)

# Shipped config/tts.json: ae.sample_rate, ae.base_chunk_size, ttl.chunk_compress_factor.
SR, BASE_CHUNK, CCF = 44100, 512, 6
FRAME_LEN = BASE_CHUNK * CCF


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

    def test_zero_length_does_not_raise(self):
        # Regression: `mask.reshape(-1, 1, 0)` cannot infer the batch dim.
        mask = length_to_mask(np.array([0], dtype=np.int64))
        self.assertEqual(mask.shape, (1, 1, 0))

    def test_explicit_zero_max_len_is_honoured(self):
        # Regression: `max_len or lengths.max()` treated 0 as "not given".
        mask = length_to_mask(np.array([5], dtype=np.int64), max_len=0)
        self.assertEqual(mask.shape, (1, 1, 0))

    def test_padded_rows_are_masked_off(self):
        mask = length_to_mask(np.array([1, 3], dtype=np.int64))
        np.testing.assert_array_equal(mask[0, 0], [1.0, 0.0, 0.0])
        np.testing.assert_array_equal(mask[1, 0], [1.0, 1.0, 1.0])


class TestUnicodeProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        vocab = Path(__file__).resolve().parents[1] / "src" / "blue_onnx" / "vocab.json"
        cls.proc = UnicodeProcessor(str(vocab))

    def test_invalid_lang_raises(self):
        with self.assertRaises(ValueError):
            self.proc._preprocess_text("hello", lang="xx")

    def test_available_langs_nonempty(self):
        self.assertIn("he", AVAILABLE_LANGS)


class TestLatentFrameMath(unittest.TestCase):
    """The seconds → latent-frame conversion shared by blue_onnx and blue_trt."""

    def frames(self, seconds: float) -> int:
        return latent_frames_for_duration(seconds, SR, BASE_CHUNK, CCF)

    def test_ceils_to_whole_frames(self):
        self.assertEqual(self.frames(FRAME_LEN / SR), 1)
        self.assertEqual(self.frames(FRAME_LEN / SR + 1e-4), 2)
        self.assertEqual(self.frames(0.0), 0)

    def test_known_duration(self):
        # 2.489 s at 44.1 kHz over 3072-sample frames.
        self.assertEqual(self.frames(2.489233), 36)

    def test_never_shorter_than_the_audio_it_must_hold(self):
        for seconds in (0.001, 0.5, 2.489233, 9.999, 30.0, 123.456):
            self.assertGreaterEqual(self.frames(seconds) * FRAME_LEN, int(seconds * SR))

    def test_matches_the_mask_width_get_latent_mask_derives(self):
        # Regression: the noise tensor width and the mask width were computed by
        # two different expressions (one on the float duration, one on the
        # truncated sample count) that had to agree for the product to broadcast.
        for seconds in (0.0, 0.001, 0.5, 2.489233, 9.999, 30.0, 123.456):
            wav_lengths = np.array([int(seconds * SR)], dtype=np.int64)
            mask = get_latent_mask(wav_lengths, BASE_CHUNK, CCF)
            self.assertEqual(mask.shape[2], self.frames(seconds), seconds)


if __name__ == "__main__":
    unittest.main()
