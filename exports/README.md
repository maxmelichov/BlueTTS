# Export Tools

Run commands from the repository root.

Install the export dependencies first:

```bash
uv sync --extra export
```

## Export A New Voice

Use one clean reference WAV. The script downmixes to mono and resamples to the model sample rate.

```bash
uv run python exports/export_new_voice.py \
  --ref_wav path/to/reference.wav \
  --out voices/my_voice.json \
  --config config/tts.json \
  --ae_ckpt pt_models/blue_codec.safetensors \
  --ttl_ckpt pt_models/vf_estimator.safetensors \
  --dp_ckpt pt_models/duration_predictor.safetensors \
  --stats pt_models/stats_multilingual.pt
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
  --ttl_ckpt pt_models/vf_estimator.safetensors \
  --ae_ckpt pt_models/blue_codec.safetensors \
  --dp_ckpt pt_models/duration_predictor.safetensors \
  --stats pt_models/stats_multilingual.pt \
  --slim
```

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
