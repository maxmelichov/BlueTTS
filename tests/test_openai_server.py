"""examples/openai_server.py against a fake engine: request validation, voices, formats, markers, streaming.

No model is loaded; the fake returns a tone whose length encodes the call, so the tests can
check what the server asked the engine for.
"""

import base64
import importlib.util
import io
import json
import struct
import sys
import threading
import unittest
import wave
from pathlib import Path

import numpy as np

try:
    from fastapi.testclient import TestClient
except ImportError:  # the server is an optional extra: `uv sync --extra server`
    TestClient = None

ROOT = Path(__file__).resolve().parents[1]


def _load_server():
    if "openai_server" in sys.modules:
        return sys.modules["openai_server"]
    spec = importlib.util.spec_from_file_location("openai_server", ROOT / "examples" / "openai_server.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look their module up here
    spec.loader.exec_module(mod)
    return mod


class FakeEngine:
    sample_rate = 1000
    join_silence = 0.01  # 10 samples

    def __init__(self, voices):
        self._voices = list(voices)
        self.calls = []
        self.lock = threading.Lock()

    @property
    def voices(self):
        return self._voices

    def synthesize(self, text, voice, lang, speed):
        with self.lock:
            self.calls.append((text, voice, lang, speed))
        n = int(round(100 / speed))
        return (0.5 * np.sin(np.arange(n) * 0.3)).astype(np.float32)


@unittest.skipIf(TestClient is None, "fastapi not installed (uv sync --extra server)")
class TestOpenAIServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = _load_server()

    def setUp(self, **settings):
        self.engine = FakeEngine(self.srv.V3_VOICES)
        settings.setdefault("encoders", {"wav": "builtin", "pcm": "builtin"})
        settings.setdefault("pcm_sample_rate", 0)
        self.client = TestClient(self.srv.create_app(self.engine, self.srv.Settings(**settings)))

    def post(self, **body):
        body.setdefault("input", "Hello world.")
        body.setdefault("response_format", "wav")
        return self.client.post("/v1/audio/speech", json=body)

    def wav_samples(self, content):
        with wave.open(io.BytesIO(content)) as w:
            self.assertEqual((w.getnchannels(), w.getsampwidth(), w.getframerate()), (1, 2, 1000))
            return w.getnframes()

    # --- listing endpoints
    def test_health_models_voices(self):
        self.assertEqual(self.client.get("/health").json()["status"], "ok")
        self.assertEqual([m["id"] for m in self.client.get("/v1/models").json()["data"]], ["bluev3"])
        v = self.client.get("/v1/audio/voices").json()
        self.assertEqual([x["id"] for x in v["data"]], list(self.srv.V3_VOICES))
        self.assertEqual(len(v["voices"]), 15)
        self.assertEqual(v["aliases"]["nova"], "noa")

    # --- validation
    def test_unknown_voice_lists_voices(self):
        r = self.post(voice="zelda")
        self.assertEqual(r.status_code, 400)
        err = r.json()["error"]
        self.assertEqual(err["param"], "voice")
        for name in self.srv.V3_VOICES:
            self.assertIn(name, err["message"])

    def test_speed_bounds(self):
        self.assertEqual(self.post(speed=0.1).status_code, 400)
        self.assertEqual(self.post(speed=4.5).status_code, 400)
        self.assertEqual(self.post(speed=0.25).status_code, 200)
        self.assertEqual(self.post(speed=4.0).status_code, 200)

    def test_speed_reaches_engine(self):
        r = self.post(speed=2.0)
        self.assertEqual(self.engine.calls[-1][3], 2.0)
        self.assertEqual(self.wav_samples(r.content), 50)

    def test_bad_input_format_language(self):
        self.assertEqual(self.post(input="   ").status_code, 400)
        self.assertEqual(self.post(input="[adam] [lily]").status_code, 400)
        self.assertEqual(self.post(input="x" * 5000).status_code, 400)
        self.assertEqual(self.client.post("/v1/audio/speech", json={"voice": "noa"}).status_code, 400)
        self.assertEqual(self.post(response_format="ogg").status_code, 400)
        r = self.post(response_format="mp3")  # no encoder configured in this test
        self.assertEqual(r.status_code, 400)
        self.assertIn("ffmpeg", r.json()["error"]["message"])
        self.assertEqual(self.post(language="fr").status_code, 400)
        self.assertEqual(self.post(stream_format="video").status_code, 400)

    def test_model_is_ignored(self):
        self.assertEqual(self.post(model="tts-1").status_code, 200)

    def test_api_key(self):
        self.setUp(api_key="k")
        self.assertEqual(self.post().status_code, 401)
        r = self.client.post("/v1/audio/speech", json={"input": "Hi.", "response_format": "wav"},
                             headers={"Authorization": "Bearer k"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertEqual(self.client.get("/v1/health").status_code, 200)
        self.assertEqual(self.client.get("/v1/models").status_code, 401)

    # --- voices and languages
    def test_voice_mapping(self):
        for openai_name, v3 in self.srv.OPENAI_VOICE_ALIASES.items():
            self.assertEqual(self.post(voice=openai_name).status_code, 200)
            self.assertEqual(self.engine.calls[-1][1], v3)
        self.post(voice="Lily")
        self.assertEqual(self.engine.calls[-1][1], "lily")

    def test_default_voice(self):
        self.setUp(default_voice="adam")
        self.client.post("/v1/audio/speech", json={"input": "Hi.", "response_format": "wav"})
        self.assertEqual(self.engine.calls[-1][1], "adam")

    def test_language_detection_and_override(self):
        self.post(input="שלום עולם.")
        self.assertEqual(self.engine.calls[-1][2], "he")
        self.post(input="Hello.")
        self.assertEqual(self.engine.calls[-1][2], "en")
        self.post(input="Hallo Welt.", language="DE")
        self.assertEqual(self.engine.calls[-1][2], "de")

    # --- formats
    def test_wav(self):
        r = self.post()
        self.assertEqual(r.headers["content-type"], "audio/wav")
        self.assertEqual(self.wav_samples(r.content), 100)

    def test_pcm(self):
        r = self.post(response_format="pcm")
        self.assertEqual(r.headers["content-type"], "audio/pcm")
        self.assertEqual(r.headers["x-sample-rate"], "1000")
        self.assertEqual(len(r.content), 200)

    def test_pcm_resampled_to_openai_rate(self):
        self.setUp(pcm_sample_rate=2000)
        r = self.post(response_format="pcm")
        self.assertEqual(r.headers["x-sample-rate"], "2000")
        self.assertEqual(len(r.content), 400)

    def test_encoded_formats_when_available(self):
        enc = self.srv.detect_encoders()
        self.setUp(encoders=enc)
        magic = {"mp3": (b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"), "flac": (b"fLaC",),
                 "opus": (b"OggS",), "aac": (b"\xff\xf1", b"\xff\xf9")}
        for fmt, heads in magic.items():
            if fmt not in enc:
                continue
            r = self.post(response_format=fmt)
            self.assertEqual(r.status_code, 200, fmt)
            self.assertEqual(r.headers["content-type"], self.srv.MEDIA_TYPES[fmt])
            self.assertTrue(r.content.startswith(heads), (fmt, r.content[:4]))

    # --- voice markers
    def test_markers_split_segments(self):
        r = self.post(input="Hello there. [adam] Hi, I am Adam. [nova]שלום.")
        segs = [(c[0], c[1], c[2]) for c in self.engine.calls]
        self.assertEqual(segs, [("Hello there.", "noa", "en"), ("Hi, I am Adam.", "adam", "en"),
                                ("שלום.", "noa", "he")])
        self.assertEqual(self.wav_samples(r.content), 3 * 100 + 2 * 10)

    def test_marker_semantics(self):
        split = self.srv.split_voice_markers
        voices = list(self.srv.V3_VOICES)
        self.assertEqual(split("[lily]Hi.", "noa", voices), [("lily", "Hi.")])
        self.assertEqual(split("See [note] here.", "noa", voices), [("noa", "See [note] here.")])
        self.assertEqual(split("Hi.[adam]", "noa", voices), [("noa", "Hi.")])
        self.assertEqual(split("<en>Hello</en>.[lily]<he>שלום</he>.", "noa", voices),
                         [("noa", "<en>Hello</en>."), ("lily", "<he>שלום</he>.")])

    def test_no_markers_is_one_call(self):
        self.post(input="One. Two. Three.")
        self.assertEqual(len(self.engine.calls), 1)

    # --- streaming
    def test_sentence_split_respects_spans(self):
        s = self.srv.split_sentences
        self.assertEqual(s("One. Two!\nThree"), ["One.", "Two!", "Three"])
        self.assertEqual(s("<en>Hi. There.</en> שלום."), ["<en>Hi. There.</en>", "שלום."])

    def test_stream_pcm(self):
        r = self.post(input="One. Two. Three.", response_format="pcm", stream_format="audio")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["content-type"], "audio/pcm")
        self.assertNotIn("content-length", r.headers)
        self.assertEqual([c[0] for c in self.engine.calls], ["One.", "Two.", "Three."])
        self.assertEqual(len(r.content), 2 * (3 * 100 + 2 * 10))

    def test_stream_wav(self):
        r = self.post(input="One. Two.", stream=True)
        self.assertEqual(r.headers["content-type"], "audio/wav")
        self.assertEqual(r.content[:4], b"RIFF")
        self.assertEqual(struct.unpack("<I", r.content[40:44])[0], 0xFFFFFFFF)
        self.assertEqual(len(r.content) - 44, 2 * (2 * 100 + 10))

    def test_stream_sse(self):
        r = self.post(input="One. Two.", response_format="pcm", stream_format="sse")
        self.assertTrue(r.headers["content-type"].startswith("text/event-stream"))
        events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
        self.assertEqual([e["type"] for e in events], ["speech.audio.delta"] * 2 + ["speech.audio.done"])
        self.assertEqual(sum(len(base64.b64decode(e["audio"])) for e in events[:-1]), 2 * 210)
        self.assertEqual(self.post(response_format="mp3", stream_format="sse").status_code, 400)

    def test_stream_non_streamable_format_falls_back(self):
        enc = self.srv.detect_encoders()
        if "flac" not in enc:
            self.skipTest("no FLAC encoder")
        self.setUp(encoders=enc)
        r = self.post(input="One. Two.", response_format="flac", stream_format="audio")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(self.engine.calls), 1)

    def test_concurrent_requests(self):
        results = []

        def go():
            results.append(self.post().status_code)

        threads = [threading.Thread(target=go) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(results, [200] * 8)


@unittest.skipIf(TestClient is None, "fastapi not installed (uv sync --extra server)")
class TestBundleHelpers(unittest.TestCase):
    def test_voice_discovery_first_dir_wins(self):
        import tempfile

        srv = _load_server()
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            (Path(a) / "noa.json").write_text("{}")
            (Path(b) / "noa.json").write_text("{}")
            (Path(b) / "Adam.json").write_text("{}")
            found = srv.discover_voices([Path(a), Path(b), Path(a) / "missing"])
            self.assertEqual(found["noa"].parent, Path(a))
            self.assertIn("adam", found)

    def test_resolve_onnx_dir(self):
        import tempfile

        srv = _load_server()
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "onnx").mkdir()
            self.assertEqual(srv.resolve_onnx_dir(d), Path(d) / "onnx")
            (Path(d) / "onnx" / "vector_estimator.onnx").touch()
            self.assertEqual(srv.resolve_onnx_dir(str(Path(d) / "onnx")), Path(d) / "onnx")
            with self.assertRaises(FileNotFoundError):
                srv.resolve_onnx_dir(str(Path(d) / "nope"))


if __name__ == "__main__":
    unittest.main()
