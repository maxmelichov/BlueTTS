# `blue_onnx` — inference API

The inference package. Three entry points, in increasing order of control.

## `BlueTTS` — one voice, one call

```python
import soundfile as sf
from blue_onnx import BlueTTS

tts = BlueTTS(onnx_dir="onnx_models", style_json="voices/noa.json")
samples, sr = tts.synthesize("שלום, זהו מודל דיבור בעברית.", lang="he")
sf.write("out.wav", samples, sr)
```

Defaults suited to arbitrary user text: `normalize_text=True` and peak limiting on.
`style_json` also takes a list, in which case the styles are averaged into one speaker.

| argument | default | notes |
|---|---|---|
| `lang` | `"he"` | one of `he`, `en`, `es`, `de`, `it` |
| `total_step` | `5` | Euler steps; more is slower and slightly cleaner |
| `cfg_scale` | `4.0` | classifier-free guidance |
| `speed` | `1.0` | duration is divided by this |
| `normalize_text` | `True` | see [Text normalization](#text-normalization) |
| `peak_limit` | `0.95` | `None` for raw vocoder output |

## `load_text_to_speech` — per-call style, chunking, phonemes

```python
from blue_onnx import load_text_to_speech, load_voice_style, limit_peak

tts = load_text_to_speech("onnx_models")
style = load_voice_style(["voices/noa.json"])
wav, dur = tts("Hello world.", lang="en", style=style, total_step=5,
               normalize_text=True, silence_duration=0.15)
wav = limit_peak(wav)            # always, before writing a PCM WAV
```

`normalize_text` is **opt-in** here (unlike `BlueTTS`). Passing a `str` chunks on
sentence boundaries and concatenates with `silence_duration` seconds of silence;
passing a `list[str]` batches instead.

Set `text_is_phonemes=True` to skip G2P and feed phonemes directly. It is mutually
exclusive with `normalize_text` — the normalizer rewrites words, not phonemes.

## `TextToSpeech.batch` — batched rows

```python
wav, durs = tts.batch(["First row.", "Second row."], ["en", "en"], style, total_step=5)
```

No sentence chunking, and no slow spans — each row is one pass. Every row of the
returned waveform is as long as the longest item; use the returned durations to trim.

> **Requires graphs exported with a dynamic batch axis.** The currently published
> `notmax123/blue-onnx-v2` bundle is batch-1 and raises an ONNX Runtime shape error.
> Re-export with [`exports/export_onnx.py`](../../exports/README.md) to use this.

## Text normalization

The model only speaks phonemes, so digits and symbols have to become words first.

```python
wav, _ = tts('ההזמנה IL-4829 תגיע ב 12/05/2024 בשעה 08:15, מחיר 1,500 ש"ח (50% הנחה).',
             lang="he", style=style, total_step=8, normalize_text=True)
```

Handled: numbers, prices, percentages, ratios, dates, clock times, phone numbers,
emails, ticket/model codes, bracketed asides, markdown headers, repeated punctuation,
emoji, and Hebrew spelling quirks (gershayim, phonetic geresh, hyphenated compounds).

It is **locale-aware**: `1,500` is a thousand in English/Hebrew, while `1.500` is a
thousand in German/Spanish/Italian.

Spelled codes, phone numbers, dates and times are wrapped in `【…】` *slow markers* and
synthesized as separate, slower passes so digit groups stay intelligible. Inspect
exactly what will be spoken without synthesizing:

```python
from blue_onnx import prepare_text_for_synthesis, split_slow_segments

prepared = prepare_text_for_synthesis(text, lang="he")   # mark_slow=False for plain text
for segment, is_slow in split_slow_segments(prepared):
    print(is_slow, segment)
```

`uv run python examples/normalize.py` prints both and writes the audio.

## Inline language spans

`<en>…</en>` inside any text routes that span to a different G2P:

```python
mixed = "שלום לכולם, <en>welcome to the presentation</en>, <es>espero que lo disfruten</es>."
samples, sr = tts.synthesize(mixed, lang="he")
```

The normalizer also emits these itself, for emails and Latin loanwords. Their presence
switches `pace_blend` on automatically (see below).

## Pace blending

The duration head uses a different seconds-per-token per language, so one `speed` value
is inconsistent across mixed text. `pace_blend` in `(0, 1]` pulls predicted duration
toward a fixed seconds-per-text-token before dividing by `speed`. Left at `None` it
defaults to `0.25` for text containing inline spans and `0` otherwise.

## Accelerators

Exactly one build may own the `onnxruntime` import, so install the extra and then drop
the stock CPU wheel:

```bash
uv sync --extra openvino && uv pip uninstall onnxruntime   # Intel
uv sync --extra gpu      && uv pip uninstall onnxruntime   # NVIDIA CUDA
```

Do not combine `openvino` and `gpu`. For TensorRT, see [`exports/`](../../exports/README.md).
CPU thread count is `min(8, cpu_count)`, overridable with `ORT_NUM_THREADS`.

## Notes

- Always pass vocoder output through `limit_peak()` before writing a PCM WAV — the
  vocoder overshoots ±1.0 and would clip. `BlueTTS.synthesize` does this for you.
- Hebrew G2P weights (~310 MB) download lazily on the first Hebrew segment, so
  Latin-only synthesis never fetches them.
- Output is **not** seeded: `blue_onnx` draws its initial latent from the global NumPy
  RNG, so two identical calls differ. Seed `numpy.random` yourself if you need
  reproducibility.
