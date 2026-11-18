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
| `speed` | `1.0` | duration is divided by `speed * speed_scale[lang]` (see below) |
| `normalize_text` | `True` | see [Text normalization](#text-normalization) |
| `peak_limit` | `0.95` | `None` for raw vocoder output |

A release bundle whose `tts.json` has a `blue_release` section (v3.0.0 on) sets its own
defaults for `total_step`, `cfg_scale` and `silence_duration`; see [Release bundles](#release-bundles).

### Streaming: `BlueTTS.stream`

```python
with sf.SoundFile("long.wav", "w", samplerate=tts.sample_rate, channels=1) as f:
    for samples in tts.stream(long_text, lang="he"):   # mono float32 at tts.sample_rate
        f.write(samples)
```

It yields the audio of each sentence as soon as that sentence is synthesized (a sentence longer than a
chunk is cut further, as in `synthesize`). Chunks after the first start with the bundle's joining
silence, so the pieces play back to back. `peak_limit` is applied per chunk. For
`load_text_to_speech` callers, `tts(text, ..., stream=True)` returns a generator of
`(wav, duration)` per chunk; joining the `wav` arrays gives exactly what the call returns without
`stream`.

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

## Voice cloning — `blue_onnx.style`

```python
from blue_onnx.style import export_voice_style, style_from_wav

payload = export_voice_style("ref.wav", onnx_dir="bluev3_onnx/onnx")   # voice JSON, save with json.dump
style = style_from_wav("ref.wav", onnx_dir="bluev3_onnx/onnx")         # or a Style for synthesize(style=...)
```

With a BlueV3 bundle, `onnx/reference_encoder.onnx` takes the clip (mixed to mono, resampled to
44.1 kHz with the resampler the model was trained with, padded to whole latent frames) and returns
`style_ttl` and `style_dp`. It reads only about the first 9 s; use 8-10 s of one clean speaker (see [Voice cloning tips](../../README.md#voice-cloning-tips)).
Clips under 1 s raise `ValueError`. The result carries the bundle's model tag
(`"model": "bluev3-702000"`). `BlueTTS(ref_wav=...)` and `BlueTTS.voice_from_wav(...)` wrap this.

With `notmax123/blue-onnx-v2` (`onnx_dir="onnx_models"`), the v2 graphs `codec_encoder`,
`style_encoder` and `duration_style_encoder` are used instead (pass `config="onnx_models/tts.json"`).

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
thousand in German/Spanish/Italian. Each language gets its own patterns:

| input | `en` | `de` | `es` / `it` |
|---|---|---|---|
| `$1,250.50` / `1.250,50 €` | one thousand two hundred fifty dollars and fifty cents | eintausendzweihundertfünfzig Euro fünfzig | mil doscientos cincuenta euros con cincuenta céntimos |
| `3rd` / `am 3. Mai` / `3º` | third | am dritten Mai | tercero / terzo |
| `14:30` | fourteen thirty | vierzehn Uhr dreißig | catorce y treinta / quattordici e trenta |
| `25/12/2024` | December twenty fifth twenty twenty four | fünfundzwanzigste Dezember zweitausendvierundzwanzig | veinticinco de diciembre de dos mil veinticuatro |

Prices, dates and times are read inline at the sentence's own pace. Phone numbers and
spelled codes are wrapped in `【…】` *slow markers* and synthesized as separate, slightly
slower passes, joined by commas (never a full stop) so digit groups stay intelligible.
Inspect exactly what will be spoken without synthesizing:

```python
from blue_onnx import prepare_text_for_synthesis, split_slow_segments

prepared = prepare_text_for_synthesis(text, lang="he")   # mark_slow=False for plain text
for segment, is_slow in split_slow_segments(prepared):
    print(is_slow, segment)
```

`uv run python examples/normalize.py` prints both and writes the audio.

**Hebrew numbers** agree with the noun they count (`3 ילדים` → שלושה ילדים, `3 בנות` →
שלוש בנות) when the optional [hebrew-num2words](https://pypi.org/project/hebrew-num2words/)
package is installed; it also reads times on the 12-hour clock (`14:30` → שתיים וחצי)
and dates with month names (`12/05/2024` → שנים עשר במאי אלפיים עשרים וארבע):

```bash
uv sync --extra hebrew        # or: pip install "blue-onnx[hebrew]"
```

Without it, Hebrew numbers fall back to plain cardinals.

## Inline language spans

`<en>…</en>` inside any text routes that span to a different G2P:

```python
mixed = "שלום לכולם, <en>welcome to the presentation</en>, <es>espero que lo disfruten</es>."
samples, sr = tts.synthesize(mixed, lang="he")
```

The normalizer also emits these itself in non-English text, for emails, Latin letters
in codes and loanwords. A span in another language than `lang` switches `pace_blend` on
automatically for older bundles (see below); a span in the same language (`<en>` in English text) does not.

With a release bundle, Latin-script words in Hebrew text get a span of their own: "קניתי
iPhone 15 חדש" is read as "קניתי <en>iPhone 15</en> חדש" (a run of Latin words, with the
numbers after them, goes to English polyg2p as one span). An all-capitals tail after a
lower-case letter is read as an acronym ("BlueTTS" as "Blue TTS"), and accents are dropped
for the English G2P ("café" as "cafe"). Spans have no ids of their own: each chunk is still
wrapped once in the request's `<lang>` tags. A span's phonemes get their own language's view
(Hebrew rules on Hebrew spans only), then the chunk is normalised once in the request's language.

## Voice speed calibration

`speed=1.0` means the voice's calibrated natural reading pace in the requested language. A voice
JSON may carry an optional `speed_scale` object, e.g. `{"he": 1.077, "en": 1.03, "de": 0.996}`;
the predicted duration is then divided by `speed * speed_scale[lang]`. Voice files without the key,
and languages the key does not list, use 1.0, so older voices behave exactly as before. When
`style_json` averages several voices, each language's factor is the mean over those voices (a voice
that does not list the language counts as 1.0). Slowed spans multiply on top.

## Pace blending

The duration head uses a different seconds-per-token per language, so one `speed` value
is inconsistent across mixed text. `pace_blend` in `(0, 1]` pulls predicted duration
toward a fixed seconds-per-text-token before dividing by `speed`. Left at `None` it
defaults to `0.25` for text containing inline spans in another language and `0` otherwise; with a release bundle it
defaults to `0` everywhere (the model's own durations), and is applied only when passed; the normalizer's slowed spans
(spelled codes, phone numbers) are not used with a release bundle either.

## Accelerators

Exactly one build may own the `onnxruntime` import, so install the extra and then drop
the stock CPU wheel:

```bash
uv sync --extra openvino && uv pip uninstall onnxruntime   # Intel
uv sync --extra gpu      && uv pip uninstall onnxruntime   # NVIDIA CUDA
```

Do not combine `openvino` and `gpu`. For TensorRT, see [`exports/`](../../exports/README.md).
CPU thread count is `min(8, cpu_count)`, overridable with `ORT_NUM_THREADS`. More is not always
faster: on an Apple M5 (4 performance + 6 efficiency cores) 4 threads were faster than all 10 cores
(see [Speed](../../README.md#speed)).

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
- **Russian, Yiddish.** `ru` goes through polyg2p too (its stress tables are in `frontend/polyg2p_data`; it
  reads numbers itself); `yi` uses the Yiddish G2P and its context model in the bundle's `frontend/yi`. Yiddish has
  no number reader: spell numbers out (digits raise), and a word the Yiddish G2P cannot read is reported.
  Its context model gives the release's readings on CPU onnxruntime 1.27–1.29 (the pinned range); the GPU and
  OpenVINO builds can read some Yiddish words differently.
- **Ids.** The phonemes get the model's normalization (`ipa_text.py`) and every chunk is
  wrapped in `<lang>` … `</lang>` ids, as the model was trained.
- **Chunks.** No chunk is longer than `blue_release.inference.max_chunk_tokens` ids (300), tags
  included. Sentences are packed while they fit; a longer sentence is cut after its last comma,
  semicolon or colon that fits, else at its last space, never inside a word or an inline span
  unless nothing else fits.
- **Defaults.** In `BlueTTS.synthesize`, `total_step`, `cfg_scale` and `silence_duration` come from
  `blue_release.inference` (`load_text_to_speech` callers pass them, e.g. 8 steps, CFG 4.0); the fixed
  edge trim is skipped when `trim_edges` is false.
- **Voices.** The BlueV3 voices (`voices/*.json`) carry `"model": "bluev3-702000"`; a tagged voice is refused by a
  bundle of another model (bundles without `blue_release` count as the previous model, whose voices are in
  `voices/v2/`). Untagged voice JSONs, e.g. your own exports, are not checked.
- **ONNX only.** `blue_trt` and `blue_pt` do not read `blue_release` yet; use `blue_onnx` for these bundles.

```bash
uv pip install --no-index --find-links release/frontend polyg2p==0.1.0    # Linux x86-64 (glibc >= 2.34), macOS
```

```python
tts = BlueTTS(onnx_dir="release/onnx", style_json="release/voices/maya.json")
samples, sr = tts.synthesize("Hello there.", lang="en")
```

Bundles without the section load exactly as before.
