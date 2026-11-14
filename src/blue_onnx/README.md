# `blue_onnx` — inference API

The inference package. Three entry points, in increasing order of control.

## `BlueTTS` — one voice, one call

```python
import soundfile as sf
from blue_onnx import BlueTTS

tts = BlueTTS(onnx_dir="bluev3_onnx/onnx", style_json="voices/noa.json")
samples, sr = tts.synthesize("שלום, זהו מודל דיבור בעברית.", lang="he")
sf.write("out.wav", samples, sr)
```

Defaults suited to arbitrary user text: `normalize_text=True` and peak limiting on.
`style_json` also takes a list, in which case the styles are averaged into one speaker.

| argument | default | notes |
|---|---|---|
| `lang` | `"he"` | one of `he`, `en`, `es`, `de`, `it`; release bundles also `ru`, `yi` |
| `total_step` | `8` | Euler steps; more is slower and slightly cleaner |
| `cfg_scale` | `4.0` | classifier-free guidance |
| `silence_duration` | `0.0` | seconds of silence between sentence chunks |
| `speed` | `1.0` | duration is divided by this |
| `normalize_text` | `True` | see [Text normalization](#text-normalization) |
| `peak_limit` | `0.95` | `None` for raw vocoder output |

A release bundle whose `tts.json` has a `blue_release` section (v3.0.0 on) sets its own
defaults for `total_step`, `cfg_scale` and `silence_duration`; see [Release bundles](#release-bundles).

## `load_text_to_speech` — per-call style, chunking, phonemes

```python
from blue_onnx import load_text_to_speech, load_voice_style, limit_peak

tts = load_text_to_speech("bluev3_onnx/onnx")
style = load_voice_style(["voices/noa.json"])
wav, dur = tts("Hello world.", lang="en", style=style, total_step=8,
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
wav, durs = tts.batch(["First row.", "Second row."], ["en", "en"], style, total_step=8)
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

With a release bundle, Latin-script words in Hebrew text get a span of their own: "קניתי
iPhone 15 חדש" is read as "קניתי <en>iPhone 15</en> חדש" (a run of Latin words, with the
numbers after them, goes to English polyg2p as one span). An all-capitals tail after a
lower-case letter is read as an acronym ("BlueTTS" as "Blue TTS"), and accents are dropped
for the English G2P ("café" as "cafe"). Spans have no ids of their own: each chunk is still
wrapped once in the request's `<lang>` tags, and a span's phonemes get its own language's
normalisation (`ipa_text.normalize_ipa_spans`).

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

## Release bundles

A bundle with a `blue_release` section in `onnx/tts.json` (v3.0.0 on) ships its own text
front end next to `onnx/`, and `blue_onnx` switches to it:

- **G2P.** English, German, Spanish, Italian and Russian go through `polyg2p` (the wheel in the
  bundle's `frontend/`, data files from `frontend/polyg2p_data` or `POLYG2P_DATA`).
  Hebrew uses RenikudPlus with the bundle's `frontend/renikud/model.onnx`.
  `BLUE_G2P_NOLEX=1` switches German / Italian to rules only (`frontend/polyg2p_data_nolex`);
  `BLUE_ES_VARIANT=es-419` selects Latin American Spanish for the G2P.
- **Russian, Yiddish.** `ru` goes through polyg2p too (its stress tables are in `frontend/polyg2p_data`);
  `yi` uses the Yiddish G2P and its context model in the bundle's `frontend/yi`.
- **Ids.** The phonemes get the model's normalization (`ipa_text.py`) and every chunk is
  wrapped in `<lang>` … `</lang>` ids, as the model was trained.
- **Chunks.** No chunk is longer than `blue_release.inference.max_chunk_tokens` ids (300), tags
  included. Sentences are packed while they fit; a longer sentence is cut after its last comma,
  semicolon or colon that fits, else at its last space, never inside a word or an inline span
  unless nothing else fits. Inline spans count with their own language's normalisation, and a
  span cut by a chunk boundary is closed and reopened in the next chunk.
- **Defaults.** In `BlueTTS.synthesize`, `total_step`, `cfg_scale` and `silence_duration` come from
  `blue_release.inference` (`load_text_to_speech` callers pass them, e.g. 8 steps, CFG 4.0); the fixed
  edge trim is skipped when `trim_edges` is false.
- **ONNX only.** `blue_trt` and `blue_pt` do not read `blue_release` yet; use `blue_onnx` for these bundles.

```bash
uv pip install release/frontend/polyg2p-*.whl    # Linux x86-64, glibc >= 2.34
```

```python
tts = BlueTTS(onnx_dir="release/onnx", style_json="release/voices/maya.json")
samples, sr = tts.synthesize("Hello there.", lang="en")
```

Bundles without the section load exactly as before.
