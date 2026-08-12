# Voices

Four style JSONs ship with the repo in [`voices/`](../voices/):

| voice | file |
|---|---|
| `adam` | `voices/adam.json` |
| `daniel` | `voices/daniel.json` |
| `lily` | `voices/lily.json` |
| `noa` | `voices/noa.json` (default) |

Each holds a `style_ttl` `[1, 50, 256]` and a `style_dp` `[1, 8, 16]` tensor extracted
from a reference clip. A voice is language-independent — any of them can speak any of
the five supported languages.

## Picking one

```bash
uv run python examples/app.py --lang he --voice noa   --text "שלום"
uv run python examples/app.py --lang en --voice adam  --text "Hello"
```

From Python:

```python
from blue_onnx import BlueTTS
tts = BlueTTS(onnx_dir="onnx_models", style_json="voices/lily.json")
```

## Blending

`BlueTTS` averages several styles into one speaker if you pass a list:

```python
tts = BlueTTS(onnx_dir="onnx_models", style_json=["voices/noa.json", "voices/lily.json"])
```

For *per-text* styles in one batch, load them with `load_voice_style([...])` and pass the
resulting `Style` to `TextToSpeech.batch` instead — that keeps them separate.

## Making your own

Extract a style from any reference clip — see
[`exports/README.md`](../exports/README.md#export-a-new-voice), or
`examples/zero_shot.py` for doing it at runtime without writing a JSON.
