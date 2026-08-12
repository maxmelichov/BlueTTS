# Examples

Run these from the **repository root**, with graphs in `./onnx_models` (see
[Models](../README.md#models)) and the `voices/*.json` that ship with the repo.

```bash
uv run python examples/basic.py       # he / en / es / it / de + mixed in one run
uv run python examples/mixed.py       # inline <en>…</en> language spans
uv run python examples/normalize.py   # numbers, dates, times, codes, emails
uv run python examples/zero_shot.py   # voice conversion from a reference clip
uv run python examples/app.py --lang en --text "Hello world."
```

Output goes to `examples/out/`; `app.py` defaults to `examples/out/app_output.wav`.

## Pointing at a different model directory

- `basic.py`, `mixed.py`, `normalize.py` — set the `ONNX_DIR` environment variable.
- `app.py` — pass `--onnx-dir`.

## Imports

These files are **not** installed-package consumers. They `sys.path.append(".")` and
import from `src.blue_onnx`, so they work in a fresh clone after `uv sync`.

## Voices

See [voices.md](voices.md) for `app.py` voice selection. To build a new voice JSON from
a reference clip, see [`exports/README.md`](../exports/README.md#export-a-new-voice).
