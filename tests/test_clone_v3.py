"""Voice cloning with a BlueV3 bundle: ``reference_encoder.onnx`` (one ``wav`` input) in ``style.py``.

A tiny stand-in graph with the release encoder's interface (``wav`` [1, S] -> ``style_ttl``
[1, 50, 256], ``style_dp`` [1, 8, 16]) replaces the 135 MB model.
"""

import json
import os
import tempfile
import types
import unittest

import numpy as np
import onnx
import soundfile as sf
from onnx import TensorProto, helper

from src.blue_onnx import TextToSpeech, load_voice_style
from src.blue_onnx.style import (
    MAX_REFERENCE_SECONDS,
    VoiceStyleExtractor,
    export_voice_style,
    style_from_payload,
    style_from_wav,
)

SR = 44100
MODEL = "bluev3-702000"


def _write_encoder(path):
    """style_ttl = ones * max(wav), style_dp = ones * (number of samples)."""
    nodes = [
        helper.make_node("ReduceMax", ["wav"], ["peak"], keepdims=1),
        helper.make_node("Shape", ["wav"], ["shape"]),
        helper.make_node("Gather", ["shape", "one"], ["n_int"], axis=0),
        helper.make_node("Cast", ["n_int"], ["n"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["ttl_ones", "peak"], ["style_ttl"]),
        helper.make_node("Mul", ["dp_ones", "n"], ["style_dp"]),
    ]
    inits = [
        helper.make_tensor("one", TensorProto.INT64, [], [1]),
        helper.make_tensor("ttl_ones", TensorProto.FLOAT, [1, 50, 256], np.ones(50 * 256).tolist()),
        helper.make_tensor("dp_ones", TensorProto.FLOAT, [1, 8, 16], np.ones(8 * 16).tolist()),
    ]
    graph = helper.make_graph(
        nodes, "ref_enc",
        [helper.make_tensor_value_info("wav", TensorProto.FLOAT, [1, "S"])],
        [helper.make_tensor_value_info("style_ttl", TensorProto.FLOAT, [1, 50, 256]),
         helper.make_tensor_value_info("style_dp", TensorProto.FLOAT, [1, 8, 16])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)


def _bundle(root):
    onnx_dir = os.path.join(root, "onnx")
    os.makedirs(onnx_dir)
    _write_encoder(os.path.join(onnx_dir, "reference_encoder.onnx"))
    cfg = {
        "ae": {"sample_rate": SR, "base_chunk_size": 512},
        "ttl": {"chunk_compress_factor": 6},
        "blue_release": {"model": MODEL, "inference": {}},
    }
    with open(os.path.join(onnx_dir, "tts.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    return onnx_dir


def _wav(path, seconds, sr=SR, channels=1, peak=0.5):
    t = np.arange(int(seconds * sr)) / sr
    x = (peak * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    sf.write(path, np.stack([x] * channels, axis=1) if channels > 1 else x, sr)
    return path


class TestReleaseClone(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.onnx_dir = _bundle(self.tmp)
        self.hop = 512 * 6

    def test_style_shapes_tag_and_padding(self):
        style = style_from_wav(_wav(os.path.join(self.tmp, "a.wav"), 3.3), onnx_dir=self.onnx_dir)
        self.assertEqual(style.ttl.shape, (1, 50, 256))
        self.assertEqual(style.dp.shape, (1, 8, 16))
        self.assertEqual(style.model, MODEL)
        n = int(style.dp[0, 0, 0])
        self.assertEqual(n % self.hop, 0)                      # padded to whole latent frames
        self.assertLess(n - int(3.3 * SR), self.hop)
        self.assertAlmostEqual(float(style.ttl[0, 0, 0]), 0.5, places=3)

    def test_resampled_stereo_and_cut(self):
        # 24 kHz stereo: mixed to mono and resampled to 44.1 kHz (soxr, no resampy needed).
        style = style_from_wav(_wav(os.path.join(self.tmp, "b.wav"), 2.0, sr=24000, channels=2),
                               onnx_dir=self.onnx_dir)
        self.assertLess(abs(int(style.dp[0, 0, 0]) - 2 * SR), self.hop + 2)
        long = style_from_wav(_wav(os.path.join(self.tmp, "c.wav"), MAX_REFERENCE_SECONDS + 5),
                              onnx_dir=self.onnx_dir)
        cut = int(MAX_REFERENCE_SECONDS * SR)
        self.assertEqual(int(long.dp[0, 0, 0]), -(-cut // self.hop) * self.hop)  # cut, then padded

    def test_too_short(self):
        with self.assertRaises(ValueError):
            style_from_wav(_wav(os.path.join(self.tmp, "d.wav"), 0.2), onnx_dir=self.onnx_dir)

    def test_saved_voice_loads_and_matches_the_bundle(self):
        payload = export_voice_style(_wav(os.path.join(self.tmp, "e.wav"), 2.0), onnx_dir=self.onnx_dir)
        self.assertEqual(payload["model"], MODEL)
        self.assertEqual(payload["metadata"]["ref_wav"], "e.wav")
        path = os.path.join(self.tmp, "mine.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        style = load_voice_style([path])
        self.assertEqual(style.model, MODEL)
        self.assertEqual(style_from_payload(payload).model, MODEL)
        TextToSpeech.check_style(types.SimpleNamespace(release={"model": MODEL}), style)
        with self.assertRaises(ValueError):  # a voice of this bundle is refused by an older one
            TextToSpeech.check_style(types.SimpleNamespace(release=None), style)

    def test_dir_without_any_encoder(self):
        empty = os.path.join(self.tmp, "empty")
        os.makedirs(empty)
        with self.assertRaises(FileNotFoundError) as cm:
            VoiceStyleExtractor(onnx_dir=empty, config={"ae": {}})
        self.assertIn("reference_encoder.onnx", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
