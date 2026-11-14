"""Voice JSON "model" tag: read by load_voice_style, checked against the bundle by TextToSpeech.check_style."""

import json
import os
import tempfile
import types
import unittest

from src.blue_onnx import TextToSpeech, load_voice_style


def _voice(path, model=None):
    d = {"style_ttl": {"data": [[[0.0] * 4] * 2], "dims": [1, 2, 4]}, "style_dp": {"data": [[[0.0] * 3]], "dims": [1, 1, 3]}}
    if model:
        d["model"] = model
    with open(path, "w") as f:
        json.dump(d, f)
    return path


class TestVoiceModelTag(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.v3 = _voice(os.path.join(self.tmp, "v3.json"), "bluev3-701500")
        self.v2 = _voice(os.path.join(self.tmp, "v2.json"))

    def check(self, release, path):
        TextToSpeech.check_style(types.SimpleNamespace(release=release), load_voice_style([path]))

    def test_tag_is_read(self):
        self.assertEqual(load_voice_style([self.v3]).model, "bluev3-701500")
        self.assertIsNone(load_voice_style([self.v2]).model)

    def test_v3_voice_refused_by_an_older_bundle(self):
        with self.assertRaises(ValueError):
            self.check(None, self.v3)

    def test_matching_and_untagged(self):
        self.check({"inference": {}}, self.v3)                       # release bundle without its own tag
        self.check({"model": "bluev3-701500"}, self.v3)
        self.check(None, self.v2)                                    # untagged voice, older bundle: as before
        self.check({"model": "bluev3-701500"}, self.v2)              # untagged (own exports) are not checked
        with self.assertRaises(ValueError):
            self.check({"model": "bluev3-800000"}, self.v3)

    def test_mixed_models_in_one_style(self):
        other = _voice(os.path.join(self.tmp, "o.json"), "bluev3-800000")
        with self.assertRaises(ValueError):
            load_voice_style([self.v3, other])


if __name__ == "__main__":
    unittest.main()
