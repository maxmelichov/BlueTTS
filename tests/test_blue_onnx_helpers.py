"""Unit tests for pure helpers (no ONNX runtime required)."""

from __future__ import annotations

import random
import re
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
from src.blue_onnx import _strip_tags_keep_spans as strip_tags_keep_spans
from src.blue_onnx.ipa_text import encode_tagged, normalize_ipa_spans, strip_outer_lang_tags, tagged_len

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

    def test_short_sentences_pack_as_before(self):
        rng = random.Random(0)
        for _ in range(200):
            text = " ".join(_sentence(rng, rng.randint(1, 12)) for _ in range(rng.randint(1, 12)))
            if rng.random() < 0.3:
                text = text.replace(". ", ".\n\n", 1)
            for max_len in (40, 80, 300):
                if all(len(s) <= max_len for s in _old_sentences(text)):
                    self.assertEqual(chunk_text(text, max_len=max_len), _old_chunk_text(text, max_len))

    def test_long_sentence_cut_at_last_comma_that_fits(self):
        s = "aaa bbb ccc, ddd eee; fff ggg hhh iii, jjj kkk lll mmm nnn."
        self.assertEqual(chunk_text(s, max_len=40), ["aaa bbb ccc, ddd eee; fff ggg hhh iii,", "jjj kkk lll mmm nnn."])
        self.assertEqual(chunk_text(s, max_len=30), ["aaa bbb ccc, ddd eee;", "fff ggg hhh iii,", "jjj kkk lll mmm nnn."])

    def test_no_punctuation_cut_at_last_space(self):
        self.assertEqual(chunk_text("aaa bbb ccc ddd eee fff", max_len=10), ["aaa bbb", "ccc ddd", "eee fff"])

    def test_no_space_cut_at_the_limit(self):
        self.assertEqual(chunk_text("x" * 25, max_len=10), ["x" * 10, "x" * 10, "x" * 5])

    def test_ids_exact_limit_and_one_over(self):
        # release bundles count ids: '<he>' + normalised IPA (final '.' added) + '</he>'
        s = "ʃalˈom ma niʃmˈa, ʔanˈi bsedˈeʁ"
        n = tagged_len(s, "he")
        self.assertEqual(n, len(encode_tagged(s, "he", {c: 1 for c in "<>/he" + s + "."})))
        self.assertEqual(chunk_text(s, max_len=300, fits=lambda c: tagged_len(c, "he") <= n), [s])
        cut = chunk_text(s, max_len=300, fits=lambda c: tagged_len(c, "he") <= n - 1)
        self.assertEqual(cut, ["ʃalˈom ma niʃmˈa,", "ʔanˈi bsedˈeʁ"])

    def test_every_chunk_fits(self):
        rng = random.Random(1)
        for _ in range(100):
            text = " ".join(_sentence(rng, rng.randint(1, 60)) for _ in range(rng.randint(1, 4)))
            fits = lambda c: tagged_len(c, "he") <= 60   # noqa: E731
            chunks = chunk_text(text, max_len=300, fits=fits)
            self.assertTrue(all(fits(c) for c in chunks), chunks)
            self.assertEqual(" ".join(chunks).split(), text.split())   # nothing lost, no word split

    def test_inline_span_not_cut(self):
        ipa = "<he>ʔanˈi ɡˈaʁ lejˈad</he> <en>tˈaɪmz skwˈɛɹ ænd sˈɛntɹəl pˈɑːɹk</en> <he>kvˈaʁ ʃanˈim ʁabˈot</he>"
        text, in_span = strip_tags_keep_spans(ipa, "he")
        self.assertEqual(text, strip_lang_tags_from_phoneme_string(ipa))
        self.assertEqual(chunk_text(text, max_len=50, in_span=in_span),
                         ["ʔanˈi ɡˈaʁ lejˈad", "tˈaɪmz skwˈɛɹ ænd sˈɛntɹəl pˈɑːɹk kvˈaʁ ʃanˈim", "ʁabˈot"])
        # without the spans the cut falls inside the English
        self.assertEqual(chunk_text(text, max_len=50)[0], "ʔanˈi ɡˈaʁ lejˈad tˈaɪmz skwˈɛɹ ænd sˈɛntɹəl")

    def test_inline_span_longer_than_a_chunk_is_cut_at_its_spaces(self):
        ipa = "<he>ʃalˈom</he> <en>wˈʌn tˈuː θɹˈiː fˈɔːɹ fˈaɪv sˈɪks</en> <he>toda</he>"
        text, in_span = strip_tags_keep_spans(ipa, "he")
        chunks = chunk_text(text, max_len=20, in_span=in_span)
        self.assertTrue(all(len(c) <= 20 for c in chunks), chunks)
        self.assertEqual(" ".join(chunks), text)

    def test_kept_spans_are_the_release_string(self):
        # release bundles keep other-language tags (flagged as inside their span) for encode_tagged
        ipa = "<he>ʔanˈi ɡˈaʁ,</he> <en>tˈaɪmz skwˈɛɹ</en> <he>!</he>"
        text, in_span = strip_tags_keep_spans(ipa, "he", keep=True)
        self.assertEqual(text, strip_outer_lang_tags(ipa, "he"))
        self.assertEqual(len(in_span), len(text))
        self.assertEqual("".join(c for c, f in zip(text, in_span) if f), "<en>tˈaɪmz skwˈɛɹ</en>")

    def test_kept_span_not_cut_and_stays_tagged(self):
        ipa = "<he>ʔanˈi ɡˈaʁ lejˈad</he> <en>tˈaɪmz skwˈɛɹ ænd sˈɛntɹəl pˈɑːɹk</en> <he>kvˈaʁ ʃanˈim ʁabˈot</he>"
        text, in_span = strip_tags_keep_spans(ipa, "he", keep=True)
        self.assertEqual(chunk_text(text, max_len=50, in_span=in_span),
                         ["ʔanˈi ɡˈaʁ lejˈad", "<en>tˈaɪmz skwˈɛɹ ænd sˈɛntɹəl pˈɑːɹk</en> kvˈaʁ ʃanˈim", "ʁabˈot"])

    def test_kept_span_longer_than_a_chunk_closed_and_reopened(self):
        ipa = "<he>ʃalˈom</he> <en>wˈʌn tˈuː θɹˈiː fˈɔːɹ fˈaɪv sˈɪks</en> <he>toda</he>"
        text, in_span = strip_tags_keep_spans(ipa, "he", keep=True)
        plain, plain_span = strip_tags_keep_spans(ipa, "he")
        chunks = chunk_text(text, max_len=20, in_span=in_span)
        # the same cuts as without the tags, each English piece in its own <en> pair
        self.assertEqual([_untag(c) for c in chunks], chunk_text(plain, max_len=20, in_span=plain_span))
        self.assertEqual(chunks[0], "ʃalˈom")
        self.assertTrue(all(c.count("<en>") == c.count("</en>") for c in chunks), chunks)
        self.assertTrue(all(c.startswith("<en>") and c.endswith("</en>") for c in chunks[1:-1]), chunks)

    def test_ids_counted_with_each_span_normalised_as_its_language(self):
        # English 'ʦ' stays one id inside a Hebrew chunk; the Hebrew rules would spell it 'ts'
        s = "ʔanˈi ʁaʔˈiti <en>kˈæʦ</en> belˈondon"
        n = tagged_len(s, "he")
        self.assertEqual(n, len(encode_tagged(s, "he", {c: 1 for c in "<>/he" + normalize_ipa_spans(s, "he")})))
        self.assertEqual(n, tagged_len(_untag(s), "he") - 1)
        text, in_span = strip_tags_keep_spans(f"<he>{s.replace('<en>', '</he> <en>').replace('</en>', '</en> <he>')}</he>",
                                              "he", keep=True)
        self.assertEqual(text, s)
        self.assertEqual(chunk_text(text, max_len=300, fits=lambda c: tagged_len(c, "he") <= n, in_span=in_span), [s])

    def test_hard_cut_never_splits_a_tag(self):
        text = "<en>" + "a" * 30 + "</en>"
        chunks = chunk_text(text, max_len=12, in_span=[True] * len(text))
        self.assertEqual("".join(_untag(c) for c in chunks), "a" * 30)
        self.assertTrue(all(re.fullmatch(r"<en>a+</en>", c) for c in chunks), chunks)


def _untag(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"</?\w+>", " ", s)).strip()


def _sentence(rng: random.Random, n: int) -> str:
    words = ["".join(rng.choice("abdeiklmnoʃʁˈ") for _ in range(rng.randint(1, 9))) for _ in range(n)]
    for i in range(len(words) - 1):
        if rng.random() < 0.15:
            words[i] += rng.choice(",;:")
    return " ".join(words) + rng.choice(".?!")


_OLD_SENTENCE = (r"(?<!Mr\.)(?<!Mrs\.)(?<!Ms\.)(?<!Dr\.)(?<!Prof\.)(?<!Sr\.)(?<!Jr\.)(?<!Ph\.D\.)(?<!etc\.)(?<!e\.g\.)"
                 r"(?<!i\.e\.)(?<!vs\.)(?<!Inc\.)(?<!Ltd\.)(?<!Co\.)(?<!Corp\.)(?<!St\.)(?<!Ave\.)(?<!Blvd\.)(?<!\b[A-Z]\.)"
                 r"(?<=[.!?])\s+")


def _old_sentences(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]
    return [s for p in paragraphs for s in re.split(_OLD_SENTENCE, p)]


def _old_chunk_text(text: str, max_len: int) -> list[str]:
    """chunk_text before long sentences were cut (v3.0.0-inference), for the texts it already handled."""
    chunks = []
    for paragraph in [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]:
        current = ""
        for sentence in re.split(_OLD_SENTENCE, paragraph):
            if len(current) + len(sentence) + 1 <= max_len:
                current += (" " if current else "") + sentence
            else:
                if current:
                    chunks.append(current.strip())
                current = sentence
        if current:
            chunks.append(current.strip())
    return chunks


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
        # Hebrew keeps /ts/ as two tokens
        he = self.proc(["kats t\u0361sa"], ["he"], True)[0][0].tolist()
        self.assertEqual(he, self.proc(["kats tsa"], ["he"])[0][0].tolist())

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


if __name__ == "__main__":
    unittest.main()
