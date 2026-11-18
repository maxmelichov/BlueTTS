# Examples

Run these from the **repository root**, with the BlueV3 bundle in `./bluev3_onnx` (see
[Quick start](../README.md#quick-start); `ONNX_DIR` / `--onnx-dir` point elsewhere) and the `voices/*.json`
that ship with the repo.

```bash
uv run python examples/basic.py       # he / en / es / it / de + mixed in one run
uv run python examples/mixed.py       # inline <en>…</en> language spans
uv run python examples/normalize.py   # numbers, dates, times, codes, emails
uv run python examples/zero_shot.py   # voice cloning from ref.wav (see its header)
uv run python examples/app.py --lang en --text "Hello world."
uv run --extra server python examples/openai_server.py   # OpenAI-compatible HTTP server on :8000
```

The server is documented in the main README under
[OpenAI-compatible server](../README.md#openai-compatible-server).

Output goes to `examples/out/`; `app.py` defaults to `examples/out/app_output.wav`.

## Pointing at a different model directory

- `basic.py`, `mixed.py`, `normalize.py`, `zero_shot.py` — set the `ONNX_DIR` environment variable.
- `app.py` — pass `--onnx-dir`.

## Imports

These files are **not** installed-package consumers. They `sys.path.append(".")` and
import from `src.blue_onnx`, so they work in a fresh clone after `uv sync`.

## Voices

See [voices.md](voices.md) for `app.py` voice selection. To build a BlueV3 voice JSON from
a reference clip, run `zero_shot.py` (it saves `examples/out/my_voice.json`); see
[Voice cloning](../README.md#voice-cloning).
