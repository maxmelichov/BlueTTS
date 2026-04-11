# Scripts: ONNX export, TensorRT engines, and stats conversion

These utilities live next to the **training** code: they import `models.*` from the `training/` tree. From the repository root, set `PYTHONPATH` so Python can resolve that package:

```bash
export PYTHONPATH="/path/to/Light-BlueTTS/training${PYTHONPATH:+:$PYTHONPATH}"
```

(Replace `/path/to/Light-BlueTTS` with your clone path.)

All paths below are relative to your **current working directory** unless you pass absolute paths. Checkpoint defaults (`checkpoints/...`) match a typical layout where you run from `training/` with checkpoints under `training/checkpoints/`. If you run from the repo root, pass `--ckpt_dir`, `--ae_ckpt`, and `--dp_ckpt` explicitly.

The TTS config (`tts.json`) is required for correct tensor shapes and `normalizer_scale`. The scripts default to `configs/tts.json`; this repo also ships `config/tts.json`, and Hugging Face bundles may use `tts.json` in the download folder—use `--config` to point at the file you actually have.

---

## Convert PyTorch checkpoints to ONNX

**Dependencies (repository root):**

```bash
cd /path/to/Light-BlueTTS
uv sync --extra export
```

**Run:**

```bash
PYTHONPATH=training uv run python scripts/export_onnx.py \
  --config path/to/tts.json \
  --onnx_dir onnx_models \
  --ckpt_dir training/checkpoints/text2latent \
  --ae_ckpt training/checkpoints/ae/ae_latest.pt \
  --dp_ckpt training/checkpoints/duration_predictor/duration_predictor_final.pt
```

Useful options:

- `--ttl_ckpt` — explicit Text2Latent checkpoint file (otherwise the latest under `--ckpt_dir` is chosen).
- `--no-verify` — skip ONNX Runtime vs PyTorch numerical checks (faster, less safe).

The script writes multiple `.onnx` files under `--onnx_dir` (text encoder, vector-field estimator, latent decoder / vocoder, reference encoder, duration predictors, etc.) and `uncond.npz` when unconditional tokens are present in the checkpoint.

---

## Convert `stats.pt` to `stats.npz` (Docker and ONNX inference)

`LightBlueTTS` ONNX inference (`src.blue_onnx`) loads normalization statistics from `onnx_models/stats.npz` via NumPy only. A Torch `.pt` stats file is fine for training or PyTorch workflows, but for **inference images and faster, simpler Docker builds** you should ship `stats.npz` next to the ONNX models: no PyTorch is needed to read it, and the runtime already expects that format.

**Dependencies:** same training layout as above; you need `torch` and NumPy (e.g. `uv sync --extra export` from the repo root, or your training environment).

**Run:**

```bash
PYTHONPATH=training uv run python scripts/convert_stats.py \
  --config path/to/tts.json \
  --pt path/to/stats_multilingual.pt \
  --out onnx_models/stats.npz
```

The script reads `mean` and `std` from the checkpoint, reshapes them to `[1, C, 1]`, embeds `normalizer_scale` from the config into the `.npz`, clamps tiny standard deviations for safety, and runs a small round-trip self-check.

Place the resulting `stats.npz` inside the directory you pass as `onnx_dir` to `LightBlueTTS` (or copy it into the `onnx_models` tree before `docker build`).

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
