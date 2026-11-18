"""Chunk-by-chunk output: ``TextToSpeech.stream`` and ``BlueTTS.stream``.

The graphs are stand-ins (no model weights): the vocoder turns the noisy latent into samples,
so two runs from the same NumPy seed give identical audio and the joined stream can be compared
with ``__call__`` / ``synthesize`` sample for sample.
"""

import unittest
from unittest import mock

import numpy as np

from src.blue_onnx import SLOW_MARK_OPEN, SLOW_SILENCE, BlueTTS, Style, TextToSpeech, limit_peak
from src.blue_onnx.text_norm import SLOW_MARK_CLOSE


class _Input:
    def __init__(self, name, shape=None):
        self.name = name
        self.shape = shape


class _Session:
    def __init__(self, fn, inputs):
        self._fn = fn
        self._inputs = inputs

    def get_inputs(self):
        return self._inputs

    def run(self, _outputs, feed):
        return [self._fn(feed)]


class _TextProcessor:
    release = False

    def __call__(self, text_list, lang_list, *_args):
        n = max(len(t) for t in text_list)
        return np.ones((len(text_list), n), np.int64), np.ones((len(text_list), 1, n), np.float32)


def _fake_tts():
    cfgs = {
        "ae": {"sample_rate": 1000, "base_chunk_size": 4},
        "ttl": {"chunk_compress_factor": 2, "latent_dim": 3},
    }
    # 0.01 s per text id, so a longer chunk gives longer audio.
    dp = _Session(lambda f: f["text_mask"].sum(axis=(1, 2)).astype(np.float32) * 0.01, [_Input("text_ids")])
    te = _Session(lambda f: np.zeros((1, 4, f["text_ids"].shape[1]), np.float32), [_Input("text_ids")])
    vf = _Session(lambda f: f["noisy_latent"], [_Input(n) for n in ("noisy_latent", "text_emb", "style_ttl")])
    voc = _Session(lambda f: np.repeat(f["latent"].mean(axis=1), 8, axis=-1)[:, None, :],
                   [_Input("latent", [1, 6, "T"])])
    return TextToSpeech(cfgs, _TextProcessor(), dp, te, vf, voc)


def _style():
    return Style(np.zeros((1, 2, 4), np.float32), np.zeros((1, 1, 3), np.float32))


# Five ~110-character sentences: more than one 300-character chunk.
TEXT = " ".join(f"Sentence number {i} is here, and it goes on for a while so that the chunker has to split it up." for i in range(5))
KW = dict(lang="en", total_step=2, cfg_scale=1.0, silence_duration=0.25, text_is_phonemes=True)


class TestTextToSpeechStream(unittest.TestCase):
    def setUp(self):
        self.tts = _fake_tts()
        self.style = _style()

    def whole(self, text):
        np.random.seed(7)
        return self.tts(text, style=self.style, **KW)

    def pieces(self, text):
        np.random.seed(7)
        return list(self.tts.stream(text, style=self.style, **KW))

    def test_joined_chunks_equal_call(self):
        for text in (TEXT, f"Before the code. {SLOW_MARK_OPEN}12 34{SLOW_MARK_CLOSE} After the code. " + TEXT):
            wav, dur = self.whole(text)
            pieces = self.pieces(text)
            self.assertGreater(len(pieces), 1)
            np.testing.assert_array_equal(np.concatenate([w for w, _ in pieces], axis=1), wav)
            np.testing.assert_allclose(sum(d for _, d in pieces), dur, rtol=1e-6)

    def test_chunks_after_the_first_start_with_the_gap(self):
        pieces = self.pieces(TEXT)
        gap = int(KW["silence_duration"] * self.tts.sample_rate)
        self.assertNotEqual(float(pieces[0][0][0, 0]), 0.0)
        for w, _ in pieces[1:]:
            np.testing.assert_array_equal(w[:, :gap], 0.0)

    def test_slow_spans_use_slow_silence(self):
        pieces = self.pieces(f"Before. {SLOW_MARK_OPEN}12 34{SLOW_MARK_CLOSE} After.")
        self.assertEqual(len(pieces), 3)
        slow_gap = int(SLOW_SILENCE * self.tts.sample_rate)
        for w, _ in pieces[1:]:
            np.testing.assert_array_equal(w[:, :slow_gap], 0.0)
            self.assertNotEqual(float(w[0, slow_gap]), 0.0)

    def test_first_chunk_before_the_rest_is_synthesized(self):
        real = self.tts._infer
        with mock.patch.object(self.tts, "_infer", side_effect=real) as spy:
            gen = self.tts.stream(TEXT, style=self.style, **KW)
            next(gen)
            self.assertEqual(spy.call_count, 1)
            rest = list(gen)
        self.assertEqual(spy.call_count, 1 + len(rest))

    def test_nothing_to_say_yields_nothing(self):
        self.assertEqual(self.pieces(""), [])
        wav, _ = self.whole("")
        self.assertEqual(wav.shape, (1, 0))

    def test_list_input_is_refused(self):
        with self.assertRaises(TypeError):
            next(self.tts.stream(["a", "b"], style=self.style, **KW))


class TestBlueTTSStream(unittest.TestCase):
    def setUp(self):
        self.bt = BlueTTS.__new__(BlueTTS)  # skip loading graphs from disk
        self.bt.tts = _fake_tts()
        self.bt.style = _style()
        self.kw = dict(lang="en", total_step=2, cfg_scale=1.0, silence_duration=0.25,
                       text_is_phonemes=True)

    def test_joined_chunks_equal_synthesize(self):
        np.random.seed(3)
        whole, sr = self.bt.synthesize(TEXT, peak_limit=None, **self.kw)
        np.random.seed(3)
        pieces = list(self.bt.stream(TEXT, peak_limit=None, **self.kw))
        self.assertGreater(len(pieces), 1)
        self.assertTrue(all(s == sr for _, s in pieces))
        self.assertTrue(all(w.ndim == 1 and w.dtype == np.float32 for w, _ in pieces))
        np.testing.assert_array_equal(np.concatenate([w for w, _ in pieces]), whole)

    def test_peak_limit_is_per_chunk(self):
        np.random.seed(5)
        raw = [w for w, _ in self.bt.stream(TEXT, peak_limit=None, **self.kw)]
        np.random.seed(5)
        limited = [w for w, _ in self.bt.stream(TEXT, peak_limit=0.1, **self.kw)]
        for r, lim in zip(raw, limited):
            np.testing.assert_array_equal(lim, limit_peak(r, 0.1))
            self.assertLessEqual(float(np.max(np.abs(lim))), 0.1 + 1e-6)

    def test_release_defaults_apply(self):
        self.bt.tts.release = {"inference": {"steps": 3, "cfg_scale": 2.0, "join_silence_s": 0.5}}
        with mock.patch.object(self.bt.tts, "stream", return_value=iter(())) as spy:
            list(self.bt.stream("Hi.", lang="en"))
        kw = spy.call_args.kwargs
        self.assertEqual((kw["total_step"], kw["cfg_scale"], kw["silence_duration"]), (3, 2.0, 0.5))
        self.assertTrue(kw["normalize_text"])


if __name__ == "__main__":
    unittest.main()
