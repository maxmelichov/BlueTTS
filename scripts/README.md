# Scripts: ONNX export, TensorRT engines, and stats conversion

These utilities live next to the **training** code: they import `models.*` from the `training/` tree. From the repository root, set `PYTHONPATH` so Python can resolve that package:

```bash
export PYTHONPATH="/path/to/blue/training${PYTHONPATH:+:$PYTHONPATH}"
```

(Replace `/path/to/blue` with your clone path.)

All paths below are relative to your **current working directory** unless you pass absolute paths. Checkpoint defaults (`checkpoints/...`) match a typical layout where you run from `training/` with checkpoints under `training/checkpoints/`. If you run from the repo root, pass `--ckpt_dir`, `--ae_ckpt`, and `--dp_ckpt` explicitly.

The TTS config (`tts.json`) is required for correct tensor shapes and `normalizer_scale`. The scripts default to `configs/tts.json`; this repo also ships `config/tts.json`, and Hugging Face bundles may use `tts.json` in the download folder—use `--config` to point at the file you actually have.

---

## Convert PyTorch checkpoints to ONNX

**Dependencies (repository root):**

```bash
cd /path/to/blue
uv sync --extra export
```

**Default run (HF `pt_weights/` layout — flat `*.safetensors` for codec, flow, and duration predictor):**

```bash
PYTHONPATH=training uv run python scripts/export_onnx.py \
  --config config/tts.json \
  --onnx_dir onnx_models
```

With those defaults the exporter reads:

- `pt_weights/vf_estimator.safetensors` — combined text2latent (text encoder + reference encoder + flow-matching estimator, flat keys)
- `pt_weights/blue_codec.safetensors` — autoencoder (the `decoder.*` keys become the vocoder)
- `pt_weights/duration_predictor.safetensors` — duration predictor

**Run against training-tree checkpoints (legacy nested `.pt`):**

```bash
PYTHONPATH=training uv run python scripts/export_onnx.py \
  --config path/to/tts.json \
  --onnx_dir onnx_models \
  --ttl_ckpt training/checkpoints/text2latent/ckpt_step_XXXX.pt \
  --ae_ckpt  training/checkpoints/ae/ae_latest.pt \
  --dp_ckpt  training/checkpoints/duration_predictor/duration_predictor_final.pt
```

If `--ttl_ckpt` is missing the script falls back to the newest `ckpt_step_*.pt` under `--ckpt_dir` (default `checkpoints/text2latent`).

Useful options:

- `--slim` — run `onnxslim` on each model for a small graph-cleanup pass (same numerics, marginally faster load, negligible speed change at inference).
- `--int8` — per-tensor `QUInt8` weight-only dynamic quantization after the slim pass. **Not recommended** (see runtime table); kept for experimentation.
- `--no-verify` — skip the ONNX Runtime vs PyTorch numerical check (faster export, less safe).

Outputs written under `--onnx_dir`:

| File | Source module |
|---|---|
| `text_encoder.onnx` | `TextEncoder` |
| `vector_estimator.onnx` | flow-matching `VectorFieldEstimator` |
| `vocoder.onnx` | `LatentDecoder1D` (codec decoder) |
| `duration_predictor.onnx` | `DPNetwork` (reference-conditioned) |

Remember to copy `stats.npz` and `uncond.npz` into the same `--onnx_dir` before calling `BlueTTS` (see the next section for `stats.npz`).

### Inference runtime

Measured on a 44.4 s mixed-language utterance, 32 flow-matching steps, CFG scale 3.0, one warm-up synth before timing. CPU = local x86 via ORT `CPUExecutionProvider`, GPU = RTX 3090 via `CUDAExecutionProvider`. SNR is vs the FP32 `regular` output on the same device.

| Variant | Device | Load (s) | Synth (s) | Audio (s) | RTF | SNR vs FP32 (dB) |
|---|---|---:|---:|---:|---:|---:|
| regular   | CPU | 0.51 |  7.50 | 44.42 | 0.169 | —    |
| slim      | CPU | 0.27 |  7.10 | 44.42 | 0.160 | 89.6 |
| int8_slim | CPU | 0.56 | 17.02 | 44.42 | 0.383 |  2.9 |
| regular   | GPU | 0.59 |  1.41 | 44.42 | 0.032 | —    |
| slim      | GPU | 0.36 |  1.59 | 44.42 | 0.036 | 34.3 |
| int8_slim | GPU | 0.58 | 20.94 | 44.42 | 0.471 |  3.3 |

On-disk directory sizes: regular **252 MB**, slim **251 MB**, int8_slim **65 MB**. `--slim` is free numerically (CPU SNR ≈ 90 dB) and recommended. `--int8` trades 4× disk savings for 2–15× slower inference and destroyed quality (~3 dB SNR) — ORT's `MatMulInteger` kernels do not beat MLAS FP32 on modern x86, and CUDA has no INT8 path so it falls back to CPU.

The flow-matching loop dominates synth time: 32 steps × 2 (conditional + unconditional for CFG) = 64 `vector_estimator.onnx` calls per chunk. To go faster, lower `steps` or set `cfg_scale=1.0` (skips the unconditional pass) when constructing `BlueTTS`.

---

## Convert `stats.pt` to `stats.npz` (Docker and ONNX inference)

`BlueTTS` ONNX inference (`src.blue_onnx`) loads normalization statistics from `onnx_models/stats.npz` via NumPy only. A Torch `.pt` stats file is fine for training or PyTorch workflows, but for **inference images and faster, simpler Docker builds** you should ship `stats.npz` next to the ONNX models: no PyTorch is needed to read it, and the runtime already expects that format.

**Dependencies:** same training layout as above; you need `torch` and NumPy (e.g. `uv sync --extra export` from the repo root, or your training environment).

**Run:**

```bash
PYTHONPATH=training uv run python scripts/convert_stats.py \
  --config path/to/tts.json \
  --pt path/to/stats_multilingual.pt \
  --out onnx_models/stats.npz
```

The script reads `mean` and `std` from the checkpoint, reshapes them to `[1, C, 1]`, embeds `normalizer_scale` from the config into the `.npz`, clamps tiny standard deviations for safety, and runs a small round-trip self-check.

Place the resulting `stats.npz` inside the directory you pass as `onnx_dir` to `BlueTTS` (or copy it into the `onnx_models` tree before `docker build`).

---

## Build TensorRT engines from ONNX

**Requirements:** NVIDIA GPU, CUDA, and TensorRT-compatible drivers. Install the optional dependency from the repository root:

```bash
uv sync --extra tensorrt
```

**Batch mode (one `.trt` per `.onnx` in a folder):**

```bash
uv run python scripts/create_tensorrt.py \
  --onnx_dir onnx_models \
  --engine_dir trt_engines \
  --precision fp16 \
  --config path/to/tts.json
```

**Single model:**

```bash
uv run python scripts/create_tensorrt.py \
  -o onnx_models/some_model.onnx \
  -e trt_engines/some_model.trt \
  --precision fp16
```

Other flags: `--workspace` (GPU workspace size in GB, default `24`), `-v` / `--verbose`, `--use_int8` (only if your platform supports fast INT8).

**Note:** `create_tensorrt.py` sets `CUDA_VISIBLE_DEVICES=1` at import time so the builder uses the second GPU by default. On a single-GPU machine, change that line in the script to `0` or remove it.

After engines are built, run inference with your usual ONNX or TensorRT runtime workflow; this repository does not ship a separate TensorRT benchmark CLI.
