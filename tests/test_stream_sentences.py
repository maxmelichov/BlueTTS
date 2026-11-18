import unittest

from src.blue_onnx import _stream_sentences


class StreamSentencesTest(unittest.TestCase):
    def test_splits_at_sentence_ends(self):
        self.assertEqual(_stream_sentences("One. Two? Three!"), ["One.", "Two?", "Three!"])

    def test_keeps_lang_span_whole(self):
        t = "שלום. <en>Hello there. How are you?</en> להתראות."
        self.assertEqual(_stream_sentences(t), ["שלום.", "<en>Hello there. How are you?</en> להתראות."])

    def test_no_split_inside_numbers_and_empty(self):
        self.assertEqual(_stream_sentences("It costs 3.50 dollars."), ["It costs 3.50 dollars."])
        self.assertEqual(_stream_sentences("no end"), ["no end"])


if __name__ == "__main__":
    unittest.main()
