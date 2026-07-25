# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Blue (`blue-onnx` on PyPI) is a multilingual TTS system — Hebrew, English, Spanish, Italian, German —
based on SupertonicTTS (flow-matching text-to-latent + autoencoder vocoder). The repo holds three
things that are versioned together but installed separately:

- **`src/`** — the published inference package (ONNX; plus PyTorch and TensorRT mirrors).
- **`exports/`** — PyTorch checkpoints → ONNX graphs / TensorRT engines / voice-style JSON.
- **`training/`** — a *separate* uv project with its own `pyproject.toml`, lockfile and Python pin.

The PyPI wheel ships only `blue_onnx` (inference). `exports/`, `training/`, `examples/` and `voices/`
are repo-only.

## Environments

Two independent uv projects. Never mix them.

```bash
uv sync                       # root: inference (Python >=3.12), stock onnxruntime CPU wheel
uv sync --extra export        # + torch/bluecodec, needed by exports/*.py and src/blue_pt
uv sync --extra tensorrt      # + torch (tensorrt-cu12 must be installed separately)
uv sync --extra openvino      # or --extra gpu; then `uv pip uninstall onnxruntime`

cd training && uv sync --extra cu128    # training: Python >=3.13,<3.14, CUDA wheels
```

`openvino`/`gpu`/stock `onnxruntime` all provide the same `onnxruntime` import — exactly one must be
installed (declared as a `conflicts` pair in `pyproject.toml`, but the stock wheel removal is manual).
The `dev` group (ruff) is **not** installed by `uv sync` (`default-groups = []`); use
`uv run --group dev ruff …`.

## Commands

```bash
# tests (unittest, no pytest). Run from the repo root — tests import `src.blue_onnx`.
uv run python -m unittest tests.test_text_norm tests.test_blue_onnx_helpers
uv run python -m unittest tests.test_blue_onnx_helpers.TestBlendDurationPace          # one class
uv run python -m unittest tests.test_text_norm.TestNumbers.test_thousands_separator_is_not_a_decimal
# `unittest discover` fails here: tests/ has no __init__.py.

uv run --group dev ruff check .          # ~58 pre-existing failures (exports/, training/, blue_trt)
uv run --group dev ruff check src/blue_onnx tests   # this subtree is clean — keep it that way

# smoke-test real synthesis (needs ./onnx_models, see below)
uv run python examples/basic.py
uv run python examples/app.py --lang en --text "Hello world." --out /tmp/out.wav
```

The tests cover pure helpers only (text normalization, masks, chunking, pace blending) and need no
ONNX runtime or model weights. Anything touching graphs must be verified by running an example.

## Model assets (not in git)

```bash
uv run hf download notmax123/blue-onnx-v2 --repo-type model --local-dir ./onnx_models   # FP32 graphs
uv run hf download notmax123/blue-v2      --repo-type model --local-dir ./pt_models     # for exports/
```

`onnx_models/` holds `text_encoder`, `vector_estimator`, `vocoder`, `duration_predictor` (+ optional
`codec_encoder`, `style_encoder`, `duration_style_encoder` for zero-shot VC) plus `tts.json`.
Hebrew G2P weights (RenikudPlus, ~310 MB) are fetched lazily on the first Hebrew segment.

Config resolution: `load_cfgs` prefers `<onnx_dir>/tts.json`, falling back to `config/tts.json`.
The **tokenizer vocab does not follow that rule** — `load_text_processor` always loads
`src/vocab.json` relative to the package and ignores `onnx_dir`, so a bundle's `vocab.json` is dead
weight. If you change the vocabulary, `src/vocab.json` is the file that matters.

## Architecture

### Inference pipeline (`src/blue_onnx/__init__.py`)

```
text ──normalize_text──> prepare_text_for_synthesis (text_norm.py)
     ──G2P─────────────> TextProcessor  (RenikudPlus for Hebrew, espeak for Latin langs)
     ──tokenize────────> UnicodeProcessor (char → id via vocab.json)
                         ├─> duration_predictor.onnx ──> seconds
                         ├─> text_encoder.onnx ────────> text_emb
                         └─> vector_estimator.onnx × total_step (Euler flow matching, CFG)
                             └─> vocoder.onnx ─────────> waveform
```

Three public entry points, in increasing order of control:

- `BlueTTS` — one voice, one call, normalization + peak limiting on by default.
- `load_text_to_speech()` → `TextToSpeech.__call__` — per-call style, chunking, phoneme input.
- `TextToSpeech.batch()` — batched rows, no chunking; requires graphs exported with a dynamic batch
  axis.

`__call__` with a `str` chunks and concatenates; with a `list[str]` it batches. `batch()` and
list-mode `__call__` must stay behaviorally aligned (G2P, `cfg_scale`, `pace_blend`, normalization) —
they have diverged before and were re-aligned deliberately.

### Cross-cutting concepts

**Inline language spans.** `<en>…</en>` inside any text routes that span to a different G2P. The tags
travel as far as tokenization and are then stripped (`strip_lang_tags_from_phoneme_string`) — critically
*before* `chunk_text`, so sentence splitting cannot orphan a tag. Presence of inline spans also flips
`pace_blend` to `DEFAULT_MIXED_PACE_BLEND` automatically.

**Pace blending.** The duration head uses different seconds-per-token per language, which makes one
`speed` value inconsistent in mixed text. `blend_duration_pace` pulls predicted duration toward
`DURATION_PACE_DPT_REF` before dividing by `speed`.

**Slow spans.** `text_norm.prepare_text_for_synthesis` wraps spelled codes, phone numbers, dates and
times in `【…】` markers. `split_slow_segments` then splits them out so they synthesize as separate
passes at `SLOW_SPEED_SCALE` / `SLOW_PACE_BLEND`, joined by `SLOW_SILENCE`. Batched paths pass
`mark_slow=False` — there is no way to schedule per-span speed inside a single batched pass.

**CFG.** Two mechanisms: graphs exported with a `cfg_scale` input do guidance internally; older
bundles need `uncond.npz` (`u_text`/`u_ref`) next to the graphs and the two-pass formula in `_infer`.
With neither, `cfg_scale` is silently ignored (a warning prints at load).

**Backend mirrors.** `src/blue_pt` (PyTorch) and `src/blue_trt` (TensorRT) reimplement the same
`Style` / `TextToSpeech` surface and *import shared helpers from `blue_onnx`* rather than copying
them. When changing chunking, tag handling, pace blending or the normalizer, check all three.
`blue_pt` imports `training.t2l.models.*`, `training.dp.*` and `bluecodec`, so it only works from a
repo checkout with `--extra export` and `training/` importable (`pyproject.toml` sets
`tool.pyright.extraPaths = ["training"]` for this reason).

### Text normalization (`src/blue_onnx/text_norm.py`)

Pre-G2P rewriting of everything the model cannot pronounce: numbers, prices, percentages, ratios,
dates, times, phone numbers, emails, alphanumeric codes, list markers, brackets, repeated punctuation,
and Hebrew-specific quirks (gershayim, phonetic geresh, in-word hyphens, lamed before Latin). It is
**locale-aware**: `1,500` is a thousand in en/he but `1.500` is a thousand in de/es/it — the regression
test for that is in `tests/test_text_norm.py`, and it is the easiest thing to break. Each rewrite is a
small standalone `expand_*` / `strip_*` function composed by `prepare_text_for_synthesis`; add new ones
the same way and unit-test them in isolation.

### Training (`training/`)

Stages run in order: `combine_datasets.py` → `compute_latent_stats.py` → `training.dp.cli` →
`training.t2l.cli`. Stats must come from the same CSV you train on. All model dimensions live in
`config/tts.json` under `ae` / `ttl` / `dp` — the same file the inference runtime reads, so a config
change is a change to both sides. Deeper docs in `training/docs/` (`overview.md`,
`architecture_and_legacy_training.md`, `stage2_text_to_latent.md`, `stage3_duration_predictor.md`).

## Conventions

- Examples run from the **repo root** and `sys.path.append(".")` + `from src.blue_onnx import …`;
  they are not installed-package consumers. Override the graph directory with `ONNX_DIR` (or
  `--onnx-dir` for `app.py`). Output goes to `examples/out/`.
- Always pass vocoder output through `limit_peak()` before writing a PCM WAV — the vocoder overshoots
  ±1.0 and would clip.
- espeak backends are cached per language in `TextProcessor._ESPEAK_BACKENDS`: construction costs
  ~600 ms and the ctypes binding leaks. Do not construct `EspeakBackend` per call.
- RenikudPlus is loaded lazily on the first Hebrew segment; keep it that way so Latin-only synthesis
  never downloads the Hebrew weights.
- Comments here explain *why* (a past bug, a numerical constraint), not what the line does — match
  that when adding them.
