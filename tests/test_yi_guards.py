"""Yiddish front end guards (release bundles): digits raise, dropped words are counted."""

import unittest

from src.blue_onnx import TextProcessor, _n_words


class TestYiddishGuards(unittest.TestCase):
    def test_digits_raise(self):
        tp = TextProcessor(release={"inference": {}}, frontend_dir="/nonexistent")
        tp._yi = lambda t: t
        with self.assertRaises(ValueError):
            tp._release_g2p("ס'איז 5 אַזייגער", "yi", True)

    def test_dropped_word_warns(self):
        import contextlib
        import io
        tp = TextProcessor(release={"inference": {}}, frontend_dir="/nonexistent")
        tp._yi = lambda t: " ".join(w for w in t.split() if w != "ABC")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            out = tp._release_g2p("דער ABC טאָג", "yi", True)
        self.assertEqual(out, "דער טאָג")
        self.assertIn("ABC", err.getvalue())

    def test_hyphen_piece_reported_and_nothing_readable_raises(self):
        import contextlib
        import io
        tp = TextProcessor(release={"inference": {}}, frontend_dir="/nonexistent")
        tp._yi = lambda t: t
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            tp._release_g2p("דער קאָמפּיוטער-", "yi", True)
        self.assertIn("קאָמפּיוטער-", err.getvalue())
        tp._yi = lambda t: ""
        with self.assertRaises(ValueError):
            tp._release_g2p("ABC", "yi", True)

    def test_n_words(self):
        self.assertEqual(_n_words("ɡit mɔrɡn, vɛlt ."), 3)
        self.assertEqual(_n_words("אַזייגער טאָג"), 2)


if __name__ == "__main__":
    unittest.main()
