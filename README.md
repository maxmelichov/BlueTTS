# BlueTTS

Hebrew Text-to-Speech inference using ONNX Runtime with optional TensorRT acceleration.

## Installation

```bash
uv sync                    # core deps
uv sync --extra gpu        # + CUDA support
```

## Model Weights

Download the TTS weights from [notmax123/Blue](https://huggingface.co/notmax123/Blue) and the G2P ONNX model from [thewh1teagle/renikud](https://huggingface.co/thewh1teagle/renikud) (same file as in the `wget` line below).

```bash
uv run hf download notmax123/Blue --repo-type model --local-dir ./onnx_models
wget -O model.onnx https://huggingface.co/thewh1teagle/renikud/resolve/main/model.onnx
```

(`huggingface-hub` is included in the default dependencies so `uv run hf` works without a separate global install.)

## New voice JSON (reference clip)

Install export deps. Weights come from [notmax123/Blue](https://huggingface.co/notmax123/Blue). New voices need **`stats_multilingual.pt`** for latent mean/std (you can fetch just that file if the `.safetensors` checkpoints are already in `pt_weights/`):

```bash
uv sync --extra export
uv run hf download notmax123/Blue stats_multilingual.pt --local-dir ./pt_weights
```

If `pt_weights/` is empty, download the codec and head weights from the same repo once (`blue_codec.safetensors`, `vf_estimator.safetensors`, `duration_predictor.safetensors`).

From the **repo root** (so `scripts/…` resolves; `PYTHONPATH` must include `training` for `models.*`):

```bash
PYTHONPATH=training uv run python scripts/export_new_voice.py \
  --ref_wav path/to/reference.wav \
  --out voices/my_voice.json \
  --config config/tts.json \
  --ae_ckpt pt_weights/blue_codec.safetensors \
  --ttl_ckpt pt_weights/vf_estimator.safetensors \
  --dp_ckpt pt_weights/duration_predictor.safetensors \
  --stats pt_weights/stats_multilingual.pt
```

Use the JSON as `style_json` when creating `LightBlueTTS`. Details: `scripts/export_new_voice.py` (optional `--verify_hf_sizes`, `--out_pt`).

## Usage & Examples

Import `BlueTTS` from `src.blue_onnx`. Supported `lang` codes are `he`, `en`, `es`, `it`, and `ge`. For mixed text, wrap each segment in tags like `<en>...</en>` (same idea for `es`, `it`, `ge`, `he`). Pass `renikud_path` when you use Hebrew.

```python
import soundfile as sf
from src.blue_onnx import LightBlueTTS

tts = LightBlueTTS(
    onnx_dir="onnx_models",
    style_json="voices/female1.json",
    renikud_path="model.onnx",
)

samples, sr = tts.synthesize("שלום, זהו מודל דיבור בעברית.", lang="he")
sf.write("output.wav", samples, sr)

mixed = "שלום לכולם, <en>welcome to the presentation</en>, <es>espero que lo disfruten</es>."
samples, sr = tts.synthesize(mixed, lang="he")
sf.write("mixed_output.wav", samples, sr)
```

More scripts: [examples](examples/).

## TensorRT

ONLY FOR NVIDIA GPUS!

Install TensorRT-related packages, then run the builder (see [scripts/README.md](scripts/README.md) for export and other script layout notes).

```bash
uv sync --extra tensorrt
uv run python scripts/create_tensorrt.py \
  --onnx_dir onnx_models --engine_dir trt_engines --precision fp16 --config config/tts.json
```

Adjust `--config` if your `tts.json` lives elsewhere. The script defaults to GPU index `1` via `CUDA_VISIBLE_DEVICES`; on a single-GPU machine, edit `scripts/create_tensorrt.py` or set the variable before running.

## Papers

```bibtex
@ARTICLE{2025arXiv250323108K,
       author = {{Kim}, Hyeongju and {Yang}, Jinhyeok and {Yu}, Yechan and {Ji}, Seunghun and {Morton}, Jacob and {Bous}, Frederik and {Byun}, Joon and {Lee}, Juheon},
        title = "{SupertonicTTS: Towards Highly Efficient and Streamlined Text-to-Speech System}",
      journal = {arXiv e-prints},
     keywords = {Audio and Speech Processing, Machine Learning, Sound},
        pages = {arXiv:2503.23108},
}
@article{kim2025training,
  title={Training Flow Matching Models with Reliable Labels via Self-Purification},
  author={Kim, Hyeongju and Yu, Yechan and Yi, June Young and Lee, Juheon},
  journal={arXiv preprint arXiv:2509.19091},
  year={2025}
}
@misc{yi2025robustttstrainingselfpurifying,
      title={Robust TTS Training via Self-Purifying Flow Matching for the WildSpoof 2026 TTS Track}, 
      author={June Young Yi and Hyeongju Kim and Juheon Lee},
      year={2025},
      eprint={2512.17293},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2512.17293}, 
}
```

## License

MIT
