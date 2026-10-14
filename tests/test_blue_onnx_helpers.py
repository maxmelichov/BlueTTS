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
    clean_chunk_audio,
    get_latent_mask,
    latent_frames_for_duration,
    latent_keep_frames,
    length_to_mask,
    limit_peak,
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

    def _ids(self, text, affricate_ligatures=False):
        ids, _ = self.proc([text], ["en"], affricate_ligatures)
        return ids[0].tolist()

    def test_tag_and_affricate_ids(self):
        c2i = self.proc._char_to_id
        self.assertEqual((c2i["<"], c2i[">"], c2i["/"], c2i["ʣ"]), (255, 48, 200, 244))
        self.assertNotIn("ǂ", c2i)
        # off by default: affricates stay as two tokens
        self.assertEqual(self._ids("dzu tʃa"), [c2i[c] for c in "dzu tʃa."])
        # on: d͡z / dz -> ʣ, tʃ -> ʧ, dʒ -> ʤ, word-final ts -> ʦ (word-internal ts stays)
        on = lambda t: self._ids(t, affricate_ligatures=True)
        self.assertEqual(on("dzu tʃa dʒa kats tsa"), on("ʣu ʧa ʤa kaʦ tsa"))
        self.assertEqual(on("d\u0361za"), on("ʣa"))

    def test_normalization(self):
        c2i = self.proc._char_to_id
        # one terminal mark, never '..' or '?.'
        self.assertEqual(self._ids("ɡaɪt?")[-1], c2i["?"])
        self.assertEqual(self._ids("falən")[-2:], [26, 10])
        self.assertEqual(self._ids("ʃalˈom」")[-1], c2i.get("」", 0))
        # NFD keeps modifier letters ('ʰ' is its own token, not 'h')
        self.assertIn(c2i["ʰ"], self._ids("pʰa"))
        # '/', '\\', '#' are word breaks and '…' is three dots, as in training rows
        self.assertEqual(self._ids("a/b\\c#d"), self._ids("a b c d"))
        self.assertEqual(self._ids("a…"), self._ids("a..."))
        # BOS/EOS characters never reach the ids
        self.assertEqual(self._ids("^a$"), self._ids("a"))


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


class TestPostDecode(unittest.TestCase):
    def test_latent_keep_frames_drops_near_zero_tail_only(self):
        rng = np.random.default_rng(0)
        z = rng.normal(0, 1, (24, 40)).astype(np.float32)
        self.assertEqual(latent_keep_frames(z), 40)
        z[:, -3:] *= 0.01
        self.assertEqual(latent_keep_frames(z), 37)
        z[:, 10] = 0.0  # a quiet frame inside the body stays
        self.assertEqual(latent_keep_frames(z), 37)

    def test_clean_chunk_cuts_to_duration_and_fades(self):
        rng = np.random.default_rng(1)
        wav = (0.1 * rng.standard_normal(SR)).astype(np.float32)
        out = clean_chunk_audio(wav, SR, 0.5)
        self.assertEqual(len(out), int(0.5 * SR))
        self.assertEqual(out[0], 0.0)
        self.assertEqual(out[-1], 0.0)

    def test_clean_chunk_cuts_hot_tail_without_speech_after(self):
        rng = np.random.default_rng(2)
        wav = (0.05 * rng.standard_normal(SR)).astype(np.float32)
        wav[-1000:] *= 20.0  # 23 ms end burst, 26 dB over the speech
        out = clean_chunk_audio(wav, SR, 2.0)
        self.assertLess(len(out), SR - 1000 + 220)
        self.assertLess(float(np.abs(out).max()), 0.5)

    def test_limit_peak_burst_vs_hot_render(self):
        rng = np.random.default_rng(3)
        quiet = (0.1 * rng.standard_normal(SR)).astype(np.float32)
        np.testing.assert_array_equal(limit_peak(quiet, 1.0, sample_rate=SR), quiet)
        burst = quiet.copy()
        burst[1000:1500] *= 30.0  # 11 ms: soft-limited, level left alone
        out = limit_peak(burst, 1.0, sample_rate=SR)
        self.assertLessEqual(float(np.abs(out).max()), 1.0)
        np.testing.assert_array_equal(out[5000:], burst[5000:])
        hot = quiet * 15.0  # wide: whole render scaled to 0.99 x the ceiling
        self.assertAlmostEqual(float(np.abs(limit_peak(hot, 1.0, sample_rate=SR)).max()), 0.99, places=5)

    def test_limit_peak_ceiling_is_backward_compatible(self):
        rng = np.random.default_rng(4)
        hot = (1.5 * rng.standard_normal(SR)).astype(np.float32)
        for ceiling in (0.95, 0.8):
            self.assertLessEqual(float(np.abs(limit_peak(hot, ceiling)).max()), ceiling)
        self.assertLessEqual(float(np.abs(limit_peak(hot)).max()), 0.95)  # default ceiling as before
        for off in (None, False, 0):
            np.testing.assert_array_equal(limit_peak(hot, off), hot)


if __name__ == "__main__":
    unittest.main()
