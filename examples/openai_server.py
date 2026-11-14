"""OpenAI-compatible text-to-speech server for BlueTTS.

Serves ``POST /v1/audio/speech`` with the request and response shapes of the OpenAI
audio API, so any OpenAI client (the ``openai`` SDK, Open WebUI, curl) can use BlueTTS
by pointing its base URL at this server.

    uv sync --extra server
    uv run python examples/openai_server.py --bundle bluev3_onnx

    curl http://localhost:8000/v1/audio/speech -H "Content-Type: application/json" \\
        -d '{"model": "bluev3", "input": "שלום, מה שלומך?", "voice": "noa", "response_format": "wav"}' \\
        -o speech.wav

Extensions to the OpenAI request (all optional):

- ``language``: ``he``, ``en``, ``de``, ``es``, ``it``, ``ru`` or ``yi``. Without it, a
  segment containing Hebrew letters is read as ``he``, anything else as ``en``.
- Voice markers: ``[name]`` inside ``input`` switches to that voice from that point on,
  e.g. ``"Hello there. [adam] And hello from me."``. Only names of available voices
  (or the OpenAI aliases) count as markers; without markers the whole input is one voice.
- ``stream_format: "audio"`` (or ``stream: true``) streams ``pcm`` / ``wav`` sentence by
  sentence as chunked transfer; ``stream_format: "sse"`` sends the same bytes as
  ``speech.audio.delta`` server-sent events.

One engine is loaded at startup and shared; synthesis calls are serialized with a lock,
so concurrent requests are safe and queue behind each other.

The idea of per-segment ``[voice]`` markers comes from the server shared in issue #22.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Protocol

import numpy as np
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("blue.openai_server")

MODEL_ID = "bluev3"
LANGUAGES = ("he", "en", "de", "es", "it", "ru", "yi")
# The 15 BlueV3 voices, in the order of examples/voices.md (female, then male).
V3_VOICES = (
    "maya", "noa", "lily", "tamar", "shira", "yael", "emma", "sara",
    "adam", "daniel", "david", "jonathan", "ethan", "michael", "ariel",
)
# OpenAI's voice names, each mapped to one fixed BlueV3 voice so OpenAI clients work unchanged.
OPENAI_VOICE_ALIASES = {
    "alloy": "maya",
    "echo": "adam",
    "fable": "daniel",
    "onyx": "david",
    "nova": "noa",
    "shimmer": "lily",
}
DEFAULT_VOICE = "noa"
FORMATS = ("mp3", "opus", "aac", "flac", "wav", "pcm")
STREAMABLE_FORMATS = ("pcm", "wav")
MEDIA_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}
# OpenAI's `pcm` is headerless 16-bit little-endian mono at 24 kHz; clients play it at that rate.
OPENAI_PCM_RATE = 24000

_HEBREW_RE = re.compile(r"[֐-׿]")
_MARKER_RE = re.compile(r"\[([A-Za-z][A-Za-z0-9_-]*)\]")
# Sentence ends (whitespace after . ! ? or an ellipsis), line breaks, and inline <xx>/</xx> tags,
# which are tracked so a sentence is never cut inside a language span.
_SPLIT_RE = re.compile(r"</?[a-z]{2,3}>|(?<=[.!?…])\s+|(?<=[.!?…]</[a-z]{2}>)\s+|\n+")


# --------------------------------------------------------------------------- engine


class Engine(Protocol):
    """What the HTTP layer needs from a synthesizer (the tests use a fake one)."""

    sample_rate: int
    join_silence: float  # seconds of silence between segments

    @property
    def voices(self) -> list[str]: ...

    def synthesize(self, text: str, voice: str, lang: str, speed: float) -> np.ndarray:
        """Mono float32 samples in [-1, 1] at ``sample_rate``."""
        ...


def resolve_onnx_dir(bundle: str) -> Path:
    """Accept either the bundle root (``bluev3_onnx``) or its ``onnx/`` directory."""
    root = Path(bundle)
    if (root / "onnx").is_dir():
        return root / "onnx"
    if (root / "vector_estimator.onnx").exists():
        return root
    raise FileNotFoundError(
        f"no BlueTTS bundle at {bundle!r}: download it with\n"
        '  hf download notmax123/BlueV3-onnx --include "onnx/*" "voices/*" "frontend/*" --local-dir bluev3_onnx'
    )


def discover_voices(dirs: list[Path]) -> dict[str, Path]:
    """Voice name -> JSON path over ``dirs``; the first directory that has a name wins."""
    found: dict[str, Path] = {}
    for d in dirs:
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.json")):
            found.setdefault(p.stem.lower(), p)
    return found


class BlueEngine:
    """``BlueTTS`` with every voice preloaded; one synthesis at a time."""

    def __init__(self, bundle: str, voices_dir: Optional[str] = None, providers: Optional[list[str]] = None):
        from src.blue_onnx import BlueTTS, load_voice_style

        onnx_dir = resolve_onnx_dir(bundle)
        dirs = [Path(voices_dir)] if voices_dir else [onnx_dir.parent / "voices", REPO_ROOT / "voices"]
        paths = discover_voices(dirs)
        if not paths:
            raise FileNotFoundError(f"no voice JSONs in {', '.join(map(str, dirs))}")
        first = paths.get(DEFAULT_VOICE) or next(iter(paths.values()))
        self._tts = BlueTTS(onnx_dir=str(onnx_dir), style_json=str(first), providers=providers)
        self._styles = {}
        for name, path in paths.items():
            style = load_voice_style([str(path)])
            try:
                self._tts.tts.check_style(style)
            except ValueError as e:  # a voice of another model: skip it, keep serving the rest
                log.warning("skipping voice %s: %s", path, e)
                continue
            self._styles[name] = style
        if not self._styles:
            raise RuntimeError("none of the voice JSONs match the loaded bundle")
        rel = (self._tts.tts.release or {}).get("inference", {})
        self.join_silence = float(rel.get("join_silence_s", 0.0))
        self.sample_rate = int(self._tts.sample_rate)
        self._lock = threading.Lock()

    @property
    def voices(self) -> list[str]:
        known = [v for v in V3_VOICES if v in self._styles]
        return known + sorted(v for v in self._styles if v not in V3_VOICES)

    def synthesize(self, text: str, voice: str, lang: str, speed: float) -> np.ndarray:
        with self._lock:
            audio, _ = self._tts.synthesize(text, lang=lang, speed=speed, style=self._styles[voice])
        return np.asarray(audio, dtype=np.float32)


# --------------------------------------------------------------------------- text


def detect_language(text: str) -> str:
    return "he" if _HEBREW_RE.search(text) else "en"


def resolve_voice(name: str, available: list[str]) -> Optional[str]:
    key = name.strip().lower()
    key = OPENAI_VOICE_ALIASES.get(key, key)
    return key if key in available else None


def split_voice_markers(text: str, voice: str, available: list[str]) -> list[tuple[str, str]]:
    """``[(voice, text), ...]``; a ``[name]`` of an available voice switches voice from there on.

    Brackets around anything else are left in the text. Segments with no words are dropped.
    """
    segments: list[tuple[str, str]] = []
    current, start = voice, 0
    for m in _MARKER_RE.finditer(text):
        v = resolve_voice(m.group(1), available)
        if v is None:
            continue
        segments.append((current, text[start:m.start()]))
        current, start = v, m.end()
    segments.append((current, text[start:]))
    return [(v, t.strip()) for v, t in segments if re.search(r"\w", t)]


def split_sentences(text: str) -> list[str]:
    """Cut at sentence ends and line breaks, never inside an inline ``<xx>…</xx>`` span."""
    pieces, start, depth = [], 0, 0
    for m in _SPLIT_RE.finditer(text):
        tok = m.group(0)
        if tok.startswith("<"):
            depth = max(0, depth - 1) if tok.startswith("</") else depth + 1
        elif depth == 0:
            pieces.append(text[start:m.start()])
            start = m.end()
    pieces.append(text[start:])
    return [p.strip() for p in pieces if re.search(r"\w", p)]


# --------------------------------------------------------------------------- audio


def to_int16(audio: np.ndarray) -> bytes:
    a = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return (np.round(a * 32767.0)).astype("<i2").tobytes()


def wav_header(sample_rate: int, n_bytes: Optional[int]) -> bytes:
    """16-bit mono WAV header; ``n_bytes=None`` writes the 0xFFFFFFFF sizes used for streams."""
    data = 0xFFFFFFFF if n_bytes is None else n_bytes
    riff = 0xFFFFFFFF if n_bytes is None else 36 + n_bytes
    return (
        b"RIFF" + struct.pack("<I", riff) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        + b"data" + struct.pack("<I", data)
    )


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or audio.size == 0:
        return audio
    import librosa

    return librosa.resample(audio, orig_sr=sr_in, target_sr=sr_out).astype(np.float32)


def _sf_supports(fmt: str, subtype: Optional[str] = None) -> bool:
    try:
        import soundfile as sf
    except Exception:
        return False
    if fmt not in sf.available_formats():
        return False
    return subtype is None or subtype in sf.available_subtypes(fmt)


def detect_encoders() -> dict[str, str]:
    """Format -> backend (``builtin``, ``soundfile`` or ``ffmpeg``); missing formats get a 400."""
    ffmpeg = shutil.which("ffmpeg")
    enc = {"wav": "builtin", "pcm": "builtin"}
    if _sf_supports("FLAC"):
        enc["flac"] = "soundfile"
    elif ffmpeg:
        enc["flac"] = "ffmpeg"
    if ffmpeg:  # LAME through ffmpeg first: libsndfile's MP3 writer is newer and less common
        enc["mp3"] = "ffmpeg"
    elif _sf_supports("MP3", "MPEG_LAYER_III"):
        enc["mp3"] = "soundfile"
    if ffmpeg:
        enc["opus"] = "ffmpeg"
        enc["aac"] = "ffmpeg"
    elif _sf_supports("OGG", "OPUS"):
        enc["opus"] = "soundfile"
    return enc


_FFMPEG_ARGS = {
    "mp3": ["-c:a", "libmp3lame", "-b:a", "128k", "-f", "mp3"],
    "opus": ["-ar", "48000", "-c:a", "libopus", "-b:a", "64k", "-f", "ogg"],
    "aac": ["-c:a", "aac", "-b:a", "128k", "-f", "adts"],
    "flac": ["-c:a", "flac", "-f", "flac"],
}


def encode(audio: np.ndarray, sample_rate: int, fmt: str, backend: str, pcm_rate: int) -> bytes:
    if fmt == "pcm":
        return to_int16(resample(audio, sample_rate, pcm_rate))
    if fmt == "wav":
        pcm = to_int16(audio)
        return wav_header(sample_rate, len(pcm)) + pcm
    if backend == "ffmpeg":
        pcm = to_int16(audio)
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "wav", "-i", "pipe:0",
               *_FFMPEG_ARGS[fmt], "pipe:1"]
        proc = subprocess.run(cmd, input=wav_header(sample_rate, len(pcm)) + pcm, capture_output=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode(errors='replace').strip()}")
        return proc.stdout
    import io

    import soundfile as sf

    buf = io.BytesIO()
    if fmt == "flac":
        sf.write(buf, audio, sample_rate, format="FLAC")
    elif fmt == "mp3":
        sf.write(buf, audio, sample_rate, format="MP3", subtype="MPEG_LAYER_III")
    elif fmt == "opus":  # Opus only runs at 8/12/16/24/48 kHz
        sf.write(buf, resample(audio, sample_rate, 48000), 48000, format="OGG", subtype="OPUS")
    else:
        raise ValueError(fmt)
    return buf.getvalue()


# --------------------------------------------------------------------------- HTTP


class SpeechRequest(BaseModel):
    model: str = MODEL_ID  # accepted and ignored: there is one model
    input: str
    voice: Optional[str] = None  # None -> the server's --default-voice
    response_format: str = "mp3"  # OpenAI's default
    speed: float = Field(1.0, ge=0.25, le=4.0)
    instructions: Optional[str] = None  # accepted and ignored
    stream_format: Optional[str] = None  # "audio" | "sse"
    # Extensions
    language: Optional[str] = None
    stream: Optional[bool] = None


@dataclass
class Settings:
    default_voice: str = DEFAULT_VOICE
    max_input_chars: int = 4096  # OpenAI's limit
    pcm_sample_rate: int = OPENAI_PCM_RATE  # 0 = the model's own rate
    api_key: Optional[str] = None
    encoders: dict[str, str] = field(default_factory=detect_encoders)


class APIError(Exception):
    def __init__(self, status: int, message: str, param: Optional[str] = None, code: Optional[str] = None):
        super().__init__(message)
        self.status, self.message, self.param, self.code = status, message, param, code


def _error(status: int, message: str, param: Optional[str] = None, code: Optional[str] = None) -> JSONResponse:
    kind = "invalid_request_error" if status < 500 else "server_error"
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": kind, "param": param, "code": code}},
    )


def create_app(engine: Optional[Engine] = None, settings: Optional[Settings] = None,
               engine_factory=None) -> FastAPI:
    """Build the app. Pass ``engine`` (or ``engine_factory``, called once at startup)."""
    settings = settings or Settings()

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.engine is None and engine_factory is not None:
            app.state.engine = engine_factory()
        yield

    app = FastAPI(title="BlueTTS OpenAI-compatible server", lifespan=lifespan)
    app.state.engine = engine
    app.state.settings = settings

    @app.exception_handler(APIError)
    async def _api_error(_request: Request, exc: APIError):
        return _error(exc.status, exc.message, exc.param, exc.code)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request: Request, exc: RequestValidationError):
        errs = exc.errors()
        first = errs[0] if errs else {}
        loc = [str(x) for x in first.get("loc", ()) if x != "body"]
        param = ".".join(loc) or None
        return _error(400, f"{param or 'body'}: {first.get('msg', 'invalid request')}", param)

    @app.middleware("http")
    async def _auth(request: Request, call_next):
        key = settings.api_key
        if key and request.url.path.startswith("/v1/") and not request.url.path.endswith("/health"):
            if request.headers.get("authorization", "") != f"Bearer {key}":
                return _error(401, "invalid API key", code="invalid_api_key")
        return await call_next(request)

    def eng() -> Engine:
        if app.state.engine is None:
            raise APIError(503, "the TTS engine is not loaded")
        return app.state.engine

    @app.get("/health")
    @app.get("/v1/health", include_in_schema=False)
    def health():
        e = app.state.engine
        if e is None:
            return JSONResponse(status_code=503, content={"status": "loading"})
        return {"status": "ok", "model": MODEL_ID, "sample_rate": e.sample_rate, "voices": len(e.voices)}

    @app.get("/v1/models")
    def list_models():
        return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "bluetts"}]}

    @app.get("/v1/audio/models")
    def list_audio_models():  # the shape Open WebUI reads for custom OpenAI-compatible TTS endpoints
        return {"models": [{"id": MODEL_ID, "name": MODEL_ID}]}

    @app.get("/v1/audio/voices")
    def list_voices():
        voices = eng().voices
        aliases = {k: v for k, v in OPENAI_VOICE_ALIASES.items() if v in voices}
        return {
            "object": "list",
            "data": [{"id": v, "name": v, "object": "voice"} for v in voices],
            "voices": [{"id": v, "name": v} for v in voices],  # Open WebUI reads this key
            "aliases": aliases,
        }

    @app.post("/v1/audio/speech")
    def speech(req: SpeechRequest):
        e = eng()
        voices = e.voices
        text = req.input
        if not text.strip():
            raise APIError(400, "input is empty", "input")
        if len(text) > settings.max_input_chars:
            raise APIError(400, f"input is longer than {settings.max_input_chars} characters", "input",
                           "string_above_max_length")
        voice = resolve_voice(req.voice or settings.default_voice, voices)
        if voice is None:
            names = ", ".join(voices)
            alias = ", ".join(f"{k}->{v}" for k, v in OPENAI_VOICE_ALIASES.items())
            raise APIError(400, f"unknown voice {req.voice!r}; available: {names} (OpenAI aliases: {alias})", "voice")
        fmt = req.response_format.lower()
        if fmt not in FORMATS:
            raise APIError(400, f"unsupported response_format {req.response_format!r}; one of {', '.join(FORMATS)}",
                           "response_format")
        if fmt not in settings.encoders:
            have = ", ".join(f for f in FORMATS if f in settings.encoders)
            raise APIError(400, f"response_format {fmt!r} needs ffmpeg (or a libsndfile that writes it), "
                           f"which this server does not have; available: {have}", "response_format")
        lang = req.language.lower() if req.language else None
        if lang is not None and lang not in LANGUAGES:
            raise APIError(400, f"unsupported language {req.language!r}; one of {', '.join(LANGUAGES)}", "language")
        sf_ = (req.stream_format or "").lower() or None
        if sf_ not in (None, "audio", "sse"):
            raise APIError(400, "stream_format must be 'audio' or 'sse'", "stream_format")
        stream = sf_ is not None or bool(req.stream)
        if sf_ == "sse" and fmt not in STREAMABLE_FORMATS:
            raise APIError(400, f"stream_format 'sse' supports {' and '.join(STREAMABLE_FORMATS)}", "response_format")

        segments = [(v, lang or detect_language(t), t) for v, t in split_voice_markers(text, voice, voices)]
        if not segments:
            raise APIError(400, "input has no words to speak", "input")
        sr = e.sample_rate
        pcm_rate = settings.pcm_sample_rate or sr
        out_rate = pcm_rate if fmt == "pcm" else sr
        headers = {"X-Sample-Rate": str(out_rate)}

        if stream and fmt in STREAMABLE_FORMATS:
            return _stream(e, segments, req.speed, fmt, pcm_rate, sf_ == "sse", headers)

        parts: list[np.ndarray] = []
        gap = np.zeros(int(round(e.join_silence * sr)), dtype=np.float32)
        for v, lg, t in segments:
            try:
                audio = e.synthesize(t, v, lg, req.speed)
            except Exception as exc:  # noqa: BLE001 — report as an API error, not a stack trace
                log.exception("synthesis failed")
                raise APIError(500, f"synthesis failed: {exc}") from exc
            if parts:
                parts.append(gap)
            parts.append(np.asarray(audio, dtype=np.float32).reshape(-1))
        audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        try:
            body = encode(audio, sr, fmt, settings.encoders[fmt], pcm_rate)
        except Exception as exc:  # noqa: BLE001
            log.exception("encoding failed")
            raise APIError(500, f"encoding {fmt} failed: {exc}") from exc
        return Response(content=body, media_type=MEDIA_TYPES[fmt], headers=headers)

    def _stream(e: Engine, segments, speed: float, fmt: str, pcm_rate: int, sse: bool, headers: dict):
        sr = e.sample_rate

        def chunks() -> Iterator[bytes]:
            if fmt == "wav":
                yield wav_header(sr, None)
            first = True
            gap = np.zeros(int(round(e.join_silence * sr)), dtype=np.float32)
            for v, lg, t in segments:
                for sentence in split_sentences(t):
                    try:
                        audio = np.asarray(e.synthesize(sentence, v, lg, speed), dtype=np.float32).reshape(-1)
                    except Exception:  # noqa: BLE001 — headers are sent; end the stream
                        log.exception("synthesis failed mid-stream")
                        return
                    if not first:
                        audio = np.concatenate([gap, audio])
                    first = False
                    yield to_int16(resample(audio, sr, pcm_rate)) if fmt == "pcm" else to_int16(audio)

        if not sse:
            return StreamingResponse(chunks(), media_type=MEDIA_TYPES[fmt], headers=headers)

        def events() -> Iterator[str]:
            for b in chunks():
                payload = {"type": "speech.audio.delta", "audio": base64.b64encode(b).decode("ascii")}
                yield f"data: {json.dumps(payload)}\n\n"
            yield f"data: {json.dumps({'type': 'speech.audio.done'})}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream", headers=headers)

    return app


# --------------------------------------------------------------------------- CLI


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bundle", default=os.environ.get("BLUE_BUNDLE", "bluev3_onnx"),
                    help="bundle root (with onnx/ and voices/) or its onnx/ directory [env BLUE_BUNDLE]")
    ap.add_argument("--voices-dir", default=os.environ.get("BLUE_VOICES_DIR"),
                    help="voice JSON directory (default: the bundle's voices/, then the repo's voices/)")
    ap.add_argument("--providers", default=os.environ.get("BLUE_PROVIDERS", "CPUExecutionProvider"),
                    help="comma-separated ONNX Runtime providers, e.g. CUDAExecutionProvider,CPUExecutionProvider")
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to accept other machines / containers")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--default-voice", default=DEFAULT_VOICE, help="voice when a request leaves it out")
    ap.add_argument("--pcm-sample-rate", type=int, default=OPENAI_PCM_RATE,
                    help="sample rate of response_format=pcm (OpenAI: 24000; 0 = the model's 44100)")
    ap.add_argument("--max-input-chars", type=int, default=4096)
    ap.add_argument("--api-key", default=os.environ.get("BLUE_API_KEY"),
                    help="require 'Authorization: Bearer <key>' (default: any key or none is accepted)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    engine = BlueEngine(args.bundle, voices_dir=args.voices_dir, providers=providers)
    default_voice = resolve_voice(args.default_voice, engine.voices)
    if default_voice is None:
        ap.error(f"--default-voice {args.default_voice!r} is not one of {', '.join(engine.voices)}")
    settings = Settings(default_voice=default_voice, max_input_chars=args.max_input_chars,
                        pcm_sample_rate=args.pcm_sample_rate, api_key=args.api_key)
    log.info("voices: %s", ", ".join(engine.voices))
    log.info("formats: %s", ", ".join(f for f in FORMATS if f in settings.encoders))

    import uvicorn

    uvicorn.run(create_app(engine, settings), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
