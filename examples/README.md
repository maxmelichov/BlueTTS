# Examples

Run these from the **repository root**, with the BlueV3 bundle in `./bluev3_onnx` (see
[Quick start](../README.md#quick-start); `ONNX_DIR` / `--onnx-dir` point elsewhere) and the `voices/*.json` that ship with the repo.

```bash
uv run python examples/basic.py       # he / en / es / it / de + mixed in one run
uv run python examples/mixed.py       # inline <en>…</en> language spans
uv run python examples/normalize.py   # numbers, dates, times, codes, emails
uv run python examples/stream.py      # chunk-by-chunk streaming
uv run python examples/benchmark.py   # speed on this machine (real-time factor)
uv run python examples/zero_shot.py   # voice cloning from a reference clip (BlueV3, or blue-onnx-v2 with ONNX_DIR)
uv run python examples/app.py --lang en --text "Hello world." --play  # play after synthesis on macOS
uv run --extra server python examples/openai_server.py   # OpenAI-compatible HTTP server on :8000
```

The server is documented in the main README under
[OpenAI-compatible server](../README.md#openai-compatible-server).

Output goes to `examples/out/`; `app.py` defaults to `examples/out/app_output.wav`.

## Pointing at a different model directory

- `basic.py`, `mixed.py`, `normalize.py` — set the `ONNX_DIR` environment variable.
- `app.py` — pass `--onnx-dir`.
- `app.py` — pass `--voice-json path/to/voice.json` for a custom or cloned voice, and `--play` to play the output on macOS.

## Imports

These files are **not** installed-package consumers. They `sys.path.append(".")` and
import from `src.blue_onnx`, so they work in a fresh clone after `uv sync`.

## Voices

See [voices.md](voices.md) for `app.py` voice selection. To build a new voice JSON from
a reference clip, see [`exports/README.md`](../exports/README.md#export-a-new-voice).
