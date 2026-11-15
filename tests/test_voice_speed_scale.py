"""Per-voice, per-language speed calibration (``speed_scale`` in the voice JSON).

Internal speed = user speed * speed_scale[lang]; the duration predictor's seconds
are divided by it. Voices without the key, and languages the key does not list,
use 1.0, i.e. exactly the old ``dur / speed``.
"""

import json
import os
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

import src.blue_onnx as blue_onnx
from src.blue_onnx import (
    SLOW_MARK_OPEN,
    SLOW_SPEED_SCALE,
    Style,
    TextToSpeech,
    load_voice_style,
    mean_speed_scale,
    parse_speed_scale,
    speed_divisor,
    speed_scale_for,
)
from src.blue_onnx.style import style_from_payload
from src.blue_onnx.text_norm import SLOW_MARK_CLOSE

SCALE = {"he": 1.077, "en": 1.030, "de": 0.996, "es": 1.056, "it": 1.041, "ru": 1.054, "yi": 1.029}
# Durations the fake duration predictor returns, deliberately not round numbers.
DP_SECONDS = np.array([1.2345678, 2.7182817, 0.3333333], dtype=np.float32)


def _voice(path, speed_scale=None, fill=0.0):
    d = {
        "style_ttl": {"data": [[[fill] * 4] * 2], "dims": [1, 2, 4]},
        "style_dp": {"data": [[[fill] * 3]], "dims": [1, 1, 3]},
    }
    if speed_scale is not None:
        d["speed_scale"] = speed_scale
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f)
    return path


class _Input:
    def __init__(self, name, shape=None):
        self.name = name
        self.shape = shape


class _Session:
    """Minimal stand-in for an ``ort.InferenceSession``."""

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
        ids = np.ones((len(text_list), n), dtype=np.int64)
        mask = np.ones((len(text_list), 1, n), dtype=np.float32)
        return ids, mask


def _fake_tts(seconds=DP_SECONDS):
    cfgs = {
        "ae": {"sample_rate": 44100, "base_chunk_size": 512},
        "ttl": {"chunk_compress_factor": 6, "latent_dim": 24},
    }
    dp = _Session(lambda f: seconds[: f["text_ids"].shape[0]].copy(), [_Input("text_ids")])
    te = _Session(lambda f: np.zeros((f["text_ids"].shape[0], 4, f["text_ids"].shape[1]), np.float32),
                  [_Input("text_ids")])
    vf = _Session(lambda f: f["noisy_latent"], [_Input(n) for n in ("noisy_latent", "text_emb", "style_ttl")])
    voc = _Session(lambda f: np.zeros((f["latent"].shape[0], 1, 8), np.float32), [_Input("latent", [1, 144, "T"])])
    return TextToSpeech(cfgs, _TextProcessor(), dp, te, vf, voc)


def _style(bsz, speed_scale=None):
    return Style(np.zeros((bsz, 2, 4), np.float32), np.zeros((bsz, 1, 3), np.float32), speed_scale=speed_scale)


def _old_durations(bsz, speed):
    """The pre-calibration formula, verbatim."""
    return DP_SECONDS[:bsz].copy() / max(float(speed), 1e-6)


class TestHelpers(unittest.TestCase):
    def test_parse(self):
        self.assertIsNone(parse_speed_scale(None))
        self.assertEqual(parse_speed_scale({"he": "1.5", "en": 1}), {"he": 1.5, "en": 1.0})
        for bad in ([1.0], {"he": 0}, {"he": -1.0}, {"he": "fast"}, {"he": float("nan")}, {"he": None}):
            with self.assertRaises(ValueError):
                parse_speed_scale(bad)

    def test_lookup_fallbacks(self):
        self.assertEqual(speed_scale_for(None, 0, "he"), 1.0)
        self.assertEqual(speed_scale_for([None], 0, "he"), 1.0)
        self.assertEqual(speed_scale_for([SCALE], 0, "he"), SCALE["he"])
        self.assertEqual(speed_scale_for([SCALE], 0, "fr"), 1.0)          # unknown language
        self.assertEqual(speed_scale_for([SCALE], 0, "en-us"), SCALE["en"])  # alias -> canonical code

    def test_divisor_is_plain_float_without_scales(self):
        for speed in (1.0, 0.95, 1.3, 0.0):
            d = speed_divisor(speed, None, ["he", "en"])
            self.assertIsInstance(d, float)
            self.assertEqual(d, max(float(speed), 1e-6))

    def test_mean_speed_scale(self):
        self.assertIsNone(mean_speed_scale(None))
        self.assertIsNone(mean_speed_scale([None, None]))
        out = mean_speed_scale([{"he": 1.2, "en": 1.1}, None, {"he": 0.9}])
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0]["he"], (1.2 + 1.0 + 0.9) / 3)
        self.assertAlmostEqual(out[0]["en"], (1.1 + 1.0 + 1.0) / 3)


class TestInferDurations(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.tts = _fake_tts()

    def infer(self, style, langs, speed):
        _, dur = self.tts._infer(["abc"] * len(langs), langs, style, total_step=1, speed=speed, cfg_scale=1.0)
        return dur

    def test_no_key_is_bit_identical(self):
        for speed in (1.0, 0.95, 1.25):
            for langs in (["he"], ["he", "en", "de"]):
                dur = self.infer(_style(len(langs)), langs, speed)
                self.assertEqual(dur.dtype, np.float32)
                np.testing.assert_array_equal(dur, _old_durations(len(langs), speed))

    def test_rows_without_key_or_lang_are_identical(self):
        # A batch where the only scaled row is row 0: the others must match the old formula exactly.
        dur = self.infer(_style(3, [{"he": 1.077}, None, {"he": 1.2}]), ["he", "en", "fr"], 1.1)
        old = _old_durations(3, 1.1)
        np.testing.assert_allclose(dur[0], DP_SECONDS[0] / (1.1 * 1.077), rtol=1e-6)
        np.testing.assert_array_equal(dur[1:], old[1:])

    def test_key_divides_by_scale_for_the_lang(self):
        langs = ["he", "en", "de"]
        dur = self.infer(_style(3, [SCALE] * 3), langs, 1.0)
        expected = DP_SECONDS / np.array([SCALE[lg] for lg in langs], dtype=np.float32)
        np.testing.assert_allclose(dur, expected, rtol=1e-6)
        dur = self.infer(_style(1, [SCALE]), ["it"], 0.9)
        np.testing.assert_allclose(dur, DP_SECONDS[:1] / (0.9 * SCALE["it"]), rtol=1e-6)

    def test_unknown_lang_is_one(self):
        dur = self.infer(_style(1, [SCALE]), ["fr"], 1.2)
        np.testing.assert_array_equal(dur, _old_durations(1, 1.2))

    def test_slow_spans_multiply_on_top(self):
        """Slow spans run at speed * SLOW_SPEED_SCALE; the voice's scale multiplies on top."""
        real = self.tts._infer

        def run(style):
            calls = []

            def spy(text_list, lang_list, st, total_step, speed, *a, **k):
                wav, dur = real(text_list, lang_list, st, total_step, speed, *a, **k)
                calls.append((speed, float(dur[0])))
                return wav, dur

            with mock.patch.object(self.tts, "_infer", side_effect=spy):
                self.tts(f"abc {SLOW_MARK_OPEN}123{SLOW_MARK_CLOSE} def", "he", style, 1,
                         speed=1.0, cfg_scale=1.0, text_is_phonemes=True)
            return calls

        plain, scaled = run(_style(1)), run(_style(1, [SCALE]))
        self.assertEqual([s for s, _ in plain], [s for s, _ in scaled])
        self.assertIn(SLOW_SPEED_SCALE, [s for s, _ in scaled])
        for (_, d_plain), (_, d_scaled) in zip(plain, scaled):
            self.assertAlmostEqual(d_scaled, d_plain / SCALE["he"], places=5)


class TestLoading(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.plain = _voice(os.path.join(self.tmp, "plain.json"))
        self.scaled = _voice(os.path.join(self.tmp, "scaled.json"), SCALE, fill=1.0)

    def test_load_voice_style_reads_key(self):
        self.assertIsNone(load_voice_style([self.plain]).speed_scale)
        self.assertEqual(load_voice_style([self.scaled]).speed_scale, [SCALE])
        self.assertEqual(load_voice_style([self.plain, self.scaled]).speed_scale, [None, SCALE])

    def test_bad_key_raises(self):
        bad = _voice(os.path.join(self.tmp, "bad.json"), {"he": -1})
        with self.assertRaises(ValueError):
            load_voice_style([bad])

    def test_style_from_payload(self):
        with open(self.scaled, encoding="utf-8") as f:
            self.assertEqual(style_from_payload(json.load(f)).speed_scale, [SCALE])
        with open(self.plain, encoding="utf-8") as f:
            self.assertIsNone(style_from_payload(json.load(f)).speed_scale)

    def _bluetts(self, paths):
        fake = types.SimpleNamespace(check_style=lambda s: None)
        with mock.patch.object(blue_onnx, "load_text_to_speech", return_value=fake):
            return blue_onnx.BlueTTS(onnx_dir="unused", style_json=paths)

    def test_bluetts_single_voice(self):
        self.assertEqual(self._bluetts(self.scaled).style.speed_scale, [SCALE])
        self.assertIsNone(self._bluetts(self.plain).style.speed_scale)

    def test_bluetts_averaged_voices(self):
        other = _voice(os.path.join(self.tmp, "other.json"), {"he": 0.9, "fr": 1.1})
        style = self._bluetts([self.scaled, other, self.plain]).style
        self.assertEqual(style.ttl.shape[0], 1)
        (avg,) = style.speed_scale
        self.assertAlmostEqual(avg["he"], (SCALE["he"] + 0.9 + 1.0) / 3)
        self.assertAlmostEqual(avg["en"], (SCALE["en"] + 1.0 + 1.0) / 3)
        self.assertAlmostEqual(avg["fr"], (1.0 + 1.1 + 1.0) / 3)
        # Averaging voices that carry no key keeps the old behaviour.
        self.assertIsNone(self._bluetts([self.plain, self.plain]).style.speed_scale)

    def test_averaged_voice_durations(self):
        other = _voice(os.path.join(self.tmp, "other.json"), {"he": 0.9})
        style = self._bluetts([self.scaled, other]).style
        np.random.seed(0)
        _, dur = _fake_tts()._infer(["abc"], ["he"], style, total_step=1, speed=1.0, cfg_scale=1.0)
        np.testing.assert_allclose(dur, DP_SECONDS[:1] / ((SCALE["he"] + 0.9) / 2), rtol=1e-6)


class TestPtLoader(unittest.TestCase):
    def test_pt_loader_and_divisor(self):
        try:
            import torch  # noqa: F401
            import src.blue_pt as blue_pt
        except Exception as e:  # torch / training deps not installed
            self.skipTest(f"blue_pt unavailable: {e}")
        tmp = tempfile.mkdtemp()
        plain = _voice(os.path.join(tmp, "plain.json"))
        scaled = _voice(os.path.join(tmp, "scaled.json"), SCALE)
        self.assertIsNone(blue_pt.load_voice_style([plain]).speed_scale)
        self.assertEqual(blue_pt.load_voice_style([plain, scaled]).speed_scale, [None, SCALE])


class TestTrtFrames(unittest.TestCase):
    """blue_trt: seconds -> latent frames divides by speed * speed_scale[lang] (tensorrt stubbed out)."""

    def _trt(self, speed_scale, speed=1.0):
        import sys
        try:
            import torch
        except Exception as e:
            self.skipTest(f"torch unavailable: {e}")
        with mock.patch.dict(sys.modules, {"tensorrt": mock.MagicMock()}):
            sys.modules.pop("src.blue_trt", None)
            import src.blue_trt as blue_trt
        sys.modules.pop("src.blue_trt", None)
        eng = blue_trt.BlueTRT.__new__(blue_trt.BlueTRT)
        eng.speed, eng.sample_rate, eng.base_chunk_size, eng.chunk_compress_factor = speed, 44100, 512, 6
        eng._style = blue_trt.Style(ttl=None, speed_scale=speed_scale)
        mask = torch.ones(1, 1, 10)
        return lambda sec, lang: eng._seconds_to_latent_frames(sec, mask, 0.0, None, lang=lang)

    def test_frames(self):
        from src.blue_onnx import latent_frames_for_duration as lf
        plain, scaled = self._trt(None, 0.9), self._trt([{"he": 1.25}], 0.9)
        for sec in (1.0, 3.7, 12.3):
            self.assertEqual(plain(sec, "he"), lf(np.float32(sec).item() / 0.9, 44100, 512, 6))
            self.assertEqual(scaled(sec, "en"), plain(sec, "en"))          # language not listed
            self.assertEqual(scaled(sec, "he"), lf(np.float32(sec).item() / (0.9 * 1.25), 44100, 512, 6))


if __name__ == "__main__":
    unittest.main()
