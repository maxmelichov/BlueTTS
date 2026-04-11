# Blue

Hebrew Text-to-Speech inference using ONNX Runtime with optional TensorRT acceleration.

## Installation

```bash
uv sync                    # core deps
uv sync --extra gpu        # + CUDA support
```

## Model Weights

Download the TTS weights from [notmax123/Blue](https://huggingface.co/notmax123/Blue) and the Hebrew G2P ONNX model from **[thewh1teagle/renikud](https://huggingface.co/thewh1teagle/renikud)** (Renikud: grapheme → IPA, ONNX Runtime, ~20 MB — weights are not bundled with the Python wheel).

The **`renikud-onnx`** package is included in the default dependencies; you only need the ONNX file:

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

Use the JSON as `style_json` when creating `BlueTTS`. Details: `scripts/export_new_voice.py` (optional `--verify_hf_sizes`, `--out_pt`).

## Usage & Examples

Import `BlueTTS` from `src.blue_onnx`. Supported `lang` codes are `he`, `en`, `es`, `it`, and `de` (alias `ge` for German). **Mixed-language raw text** must use explicit XML-style tags **`</lan>`** closers, e.g. `<en>English words here</en>`, `<es>…</es>`, `<he>…</he>`, `<ge>…</ge>` — the same form the tokenizer expects after phonemization. Hebrew outside tags uses Renikud; tagged Latin segments use espeak for that language. Pass `renikud_path` when any Hebrew letters appear.

```python
import soundfile as sf
from src.blue_onnx import BlueTTS

tts = BlueTTS(
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

**Examples** (from repo root):

```bash
uv run python examples/basic.py               # he / en / es / it / ge + mixed → examples/out/
uv run python examples/all_langs_and_mix.py   # all LANG_ID langs + mixed + langs_manifest.json → examples/out/
# TensorRT batch (after `uv sync --extra tensorrt` and engine build):
uv run python examples/all_langs_and_mix.py --tensorrt
uv run python examples/app.py --lang en --text "Hello world."
```

## TensorRT

ONLY FOR NVIDIA GPUS!

Install TensorRT-related packages, then run the builder (see [scripts/README.md](scripts/README.md) for export and other script layout notes).

```bash
uv sync --extra tensorrt
uv run python scripts/create_tensorrt.py \
  --onnx_dir onnx_models --engine_dir trt_engines --precision fp16 --config config/tts.json
```

Text-sequence engines allow up to **2048** tokens per profile axis (needed for long or mixed-language IPA). If mixed utterances fail or sound wrong with older `trt_engines/`, rebuild after updating `scripts/create_tensorrt.py` (previously capped at 512).

Adjust `--config` if your `tts.json` lives elsewhere. The optional extra installs **`tensorrt-cu12`** (CUDA 12.x wheels). If the build fails with CUDA error 35, your driver stack likely needs **`tensorrt-cu13`** instead—install it into the same venv (see [NVIDIA pip install](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/install-pip.html)). By default the script sets `CUDA_VISIBLE_DEVICES` to `1` only when that variable is unset; use `CUDA_VISIBLE_DEVICES=0` (or edit the script) on a single-GPU machine.

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
