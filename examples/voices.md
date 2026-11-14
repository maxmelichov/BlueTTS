# Voices

## BlueV3 v3.0.0 voices

Fifteen style JSONs in [`voices/`](../voices/), exported from the v3.0.0 model (HF `notmax123/BlueV3-onnx`,
checkpoint 701500). A voice file only works with the model it was exported from; these carry `"model": "bluev3-701500"`.

| voice | | file |
|---|---|---|
| `maya` | female | `voices/maya.json` |
| `noa` | female | `voices/noa.json` |
| `lily` | female | `voices/lily.json` |
| `tamar` | female | `voices/tamar.json` |
| `shira` | female | `voices/shira.json` |
| `yael` | female | `voices/yael.json` |
| `emma` | female | `voices/emma.json` |
| `sara` | female | `voices/sara.json` |
| `adam` | male | `voices/adam.json` |
| `daniel` | male | `voices/daniel.json` |
| `david` | male | `voices/david.json` |
| `jonathan` | male | `voices/jonathan.json` |
| `ethan` | male | `voices/ethan.json` |
| `michael` | male | `voices/michael.json` |
| `ariel` | male | `voices/ariel.json` |

Reference clips: LibriTTS-R (Koizumi et al., 2023; CC BY 4.0), one utterance per voice (the `source` field).

## Previous model (v2.5) voices

The four voices of the previous model are in [`voices/v2/`](../voices/v2/): `adam`, `daniel`, `lily`, `noa` (default
for the current code and examples).

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
tts = BlueTTS(onnx_dir="onnx_models", style_json="voices/v2/lily.json")
```

## Blending

`BlueTTS` averages several styles into one speaker if you pass a list:

```python
tts = BlueTTS(onnx_dir="onnx_models", style_json=["voices/v2/noa.json", "voices/v2/lily.json"])
```

For *per-text* styles in one batch, load them with `load_voice_style([...])` and pass the
resulting `Style` to `TextToSpeech.batch` instead — that keeps them separate.

## Making your own

Extract a style from any reference clip — see
[`exports/README.md`](../exports/README.md#export-a-new-voice), or
`examples/zero_shot.py` for doing it at runtime without writing a JSON.
