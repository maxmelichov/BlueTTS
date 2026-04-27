# Export Tools

Run commands from the repository root.

Install the export dependencies first:

```bash
uv sync --extra export
```

## Weights (safetensors) — **required** for new voice and ONNX

Exporting a **voice** from a reference clip or **ONNX** graphs from PyTorch both load the same trained checkpoints. Those weights live as **`.safetensors`** files (plus stats) in the [notmax123/blue-v2](https://huggingface.co/notmax123/blue-v2) model repo. Download them once, for example into `pt_models/`:

```bash
uv run hf download notmax123/blue-v2 --repo-type model --local-dir ./pt_models
```

You will need (filenames as on the Hub, paths passed via flags):

- `blue_codec.safetensors` — autoencoder (`--ae_ckpt`)
- `vf_estimetor.safetensors` — voice-flow stack (`--ttl_ckpt`)
- `duration_predictor_final.safetensors` — duration model (`--dp_ckpt`)
- `stats_multilingual.safetensors` — running-mean / stats (`--stats`)

Without these files, `export_new_voice.py` and `export_onnx.py` cannot run.

## Export A New Voice

Use one clean reference WAV. The script downmixes to mono and resamples to the model sample rate.

```bash
uv run python exports/export_new_voice.py \
  --ref_wav path/to/reference.wav \
  --out voices/my_voice.json \
  --config config/tts.json \
  --ae_ckpt pt_models/blue_codec.safetensors \
  --ttl_ckpt pt_models/vf_estimetor.safetensors \
  --dp_ckpt pt_models/duration_predictor_final.safetensors \
  --stats pt_models/stats_multilingual.safetensors
```

The output JSON can be used anywhere a voice/style JSON is expected:

```python
from blue_onnx import load_voice_style

style = load_voice_style(["voices/my_voice.json"])
```

Notes:

- If your checkpoint filenames are different, keep the same flags and change only the paths.
- `--out_pt path/to/file.pt` is optional and saves the raw reference latent for debugging.
- `--device cuda` is optional if your PyTorch install has CUDA.

## Export ONNX Models

Use this when you have PyTorch checkpoints and want an ONNX inference folder.

```bash
uv run python exports/export_onnx.py \
  --config config/tts.json \
  --onnx_dir onnx_models \
  --ttl_ckpt pt_models/vf_estimetor.safetensors \
  --ae_ckpt pt_models/blue_codec.safetensors \
  --dp_ckpt pt_models/duration_predictor_final.safetensors \
  --stats pt_models/stats_multilingual.safetensors \
  --slim
```

`export_onnx.py` loads stats with `torch.load` (mean/std tensors). Use the `.pt` stats file from the same Hub repo, not the `.safetensors` copy.

This writes:

- `onnx_models/text_encoder.onnx`
- `onnx_models/vector_estimator.onnx`
- `onnx_models/vocoder.onnx`
- `onnx_models/duration_predictor.onnx`

Useful flags:

- `--slim` cleans the ONNX graphs after export.
- `--no-verify` skips the PyTorch vs ONNX numerical check.
- `--int8` exists for experiments, but is not recommended for quality or speed.

## Build TensorRT Engines

This is optional and requires NVIDIA TensorRT.

```bash
uv sync --extra tensorrt
uv run python exports/create_tensorrt.py \
  --onnx_dir onnx_models \
  --engine_dir trt_engines \
  --precision fp16 \
  --config config/tts.json
```

For one model:

```bash
uv run python exports/create_tensorrt.py \
  --onnx onnx_models/vocoder.onnx \
  --engine trt_engines/vocoder.trt \
  --precision fp16 \
  --config config/tts.json
```
