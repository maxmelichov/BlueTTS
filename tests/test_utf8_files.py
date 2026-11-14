"""Text and JSON files are read as UTF-8 whatever the platform's default encoding (Windows: cp1252)."""

import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCRIPT = r"""
import json, locale, sys
assert locale.getpreferredencoding(False).lower() not in ("utf-8", "utf8"), locale.getpreferredencoding(False)
from src.blue_onnx import load_text_processor, load_voice_style
tp = load_text_processor()
assert any(ord(c) > 127 for c in tp._char_to_id), "vocab without non-ASCII symbols"
style = load_voice_style([sys.argv[1]])
print("ok", style.ttl.shape)
"""


class TestUtf8Files(unittest.TestCase):
    def test_loaders_with_a_non_utf8_default_encoding(self):
        with tempfile.TemporaryDirectory() as d:
            voice = os.path.join(d, "voice.json")
            with open(voice, "w", encoding="utf-8") as f:
                json.dump({"style_ttl": {"data": [[[0.0] * 4] * 2], "dims": [1, 2, 4]},
                           "style_dp": {"data": [[[0.0] * 3]], "dims": [1, 1, 3]},
                           "source": "voix d’été ʃalˈom שלום"}, f, ensure_ascii=False)
            # an ASCII locale, no UTF-8 mode, no C-locale coercion: open() without encoding= would fail here,
            # exactly as cp1252 does on Windows
            env = dict(os.environ, LC_ALL="C", LANG="C", PYTHONUTF8="0", PYTHONCOERCECLOCALE="0")
            r = subprocess.run([sys.executable, "-X", "utf8=0", "-c", SCRIPT, voice], cwd=ROOT, env=env,
                               capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            self.assertIn("ok", r.stdout)


if __name__ == "__main__":
    unittest.main()
