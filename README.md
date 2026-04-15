# Blue

Text-to-speech inference with ONNX Runtime; optional TensorRT acceleration on NVIDIA GPUs.

<p align="center">
  <a href="https://pypi.org/project/blue-onnx/"><img src="https://img.shields.io/pypi/v/blue-onnx?style=for-the-badge&amp;label=PyPI" alt="PyPI version"></a>
  &nbsp;
  <a href="https://huggingface.co/spaces/notmax123/Blue"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Try%20Live%20Demo-FFD21E?style=for-the-badge" alt="Try Live Demo on Hugging Face"></a>
  &nbsp;
  <a href="https://lightbluetts.com/"><img src="https://img.shields.io/badge/%F0%9F%8C%90%20Website-lightbluetts.com-2563EB?style=for-the-badge" alt="lightbluetts.com"></a>
</p>

<p align="center">Hebrew, English, Spanish, Italian, and German — samples and a live demo on the site and Space above.</p>

## Install

Requires **Python 3.12+** (see `requires-python` in `pyproject.toml`).

**Users (PyPI)**

```bash
pip install blue-onnx
```

Optional accelerators (PyPI): install the extra, then drop the stock CPU wheel so a single build owns `onnxruntime`:

- Intel OpenVINO: `pip install "blue-onnx[openvino]"` then `pip uninstall onnxruntime`
- NVIDIA CUDA: `pip install "blue-onnx[gpu]"` then `pip uninstall onnxruntime`

**This repository**

```bash
git clone https://github.com/maxmelichov/BlueTTS.git
cd BlueTTS
uv sync
```

Optional extras:

```bash
uv sync --extra openvino   # Intel OpenVINO EP (then: uv pip uninstall onnxruntime)
uv sync --extra gpu        # NVIDIA CUDA ORT (then: uv pip uninstall onnxruntime)
uv sync --extra tensorrt   # TensorRT path (see below)
uv sync --extra export     # PyTorch / export tooling
```

The default environment uses the stock `onnxruntime` CPU wheel. **OpenVINO** and **CUDA (`gpu`)** are optional; each adds a second ONNX Runtime distribution until you remove the stock CPU wheel so the accelerator build owns the `onnxruntime` import (`uv pip uninstall onnxruntime`, or the same with `pip` after a PyPI install with `[openvino]` or `[gpu]`). Do not combine the `openvino` and `gpu` extras.

## Models

**ONNX bundle** (model weights; excludes the large `all_voices` tree on the Hub):

```bash
uv run hf download notmax123/blue-onnx --repo-type model --local-dir ./onnx_models \
  --exclude "voices/all_voices/**"
```

That download does **not** include small **voice JSON** files such as `voices/female1.json`. Copy the `voices/` directory from **this repository** (or browse it on [GitHub](https://github.com/maxmelichov/BlueTTS/tree/main/voices)) next to your working directory so paths like `voices/female1.json` resolve. If you use `pip` without `uv`, the same CLI is available after install because `blue-onnx` depends on `huggingface-hub` — run `hf download ...` with the same arguments.

**Optional**

- Hebrew G2P: 
  ```bash
  wget -O model.onnx https://huggingface.co/thewh1teagle/renikud/resolve/main/model.onnx
  ```
- [2000+ voice JSONs](https://huggingface.co/notmax123/blue-onnx/tree/main/voices/all_voices):
  ```bash
  uv run hf download notmax123/blue-onnx voices/all_voices/ --repo-type model --local-dir ./onnx_models
  ```
- PyTorch weights (export new voices): `uv sync --extra export` then
  ```bash
  uv run hf download notmax123/blue --repo-type model --local-dir ./pt_models
  ```

## Usage

Examples below use `voices/female1.json` from this repo. If you downloaded `all_voices`, pick paths under `onnx_models/voices/all_voices/` (see `manifest.tsv`).

## Quick start

```python
import soundfile as sf
from blue_onnx import BlueTTS

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

If you are editing **this repo** without installing the package, use `from src.blue_onnx import BlueTTS` (as in `examples/`) or put `src` on `PYTHONPATH`.

## Examples

Outputs go to `examples/out/` when run from the repo root.

```bash
uv run python examples/hebrew.py
uv run python examples/english.py
uv run python examples/spanish.py
uv run python examples/italian.py
uv run python examples/german.py
uv run python examples/mixed.py
uv run python examples/app.py --lang en --text "Hello world."
```

## TensorRT (NVIDIA only)

1. Dependencies:

```bash
uv sync --extra tensorrt
uv pip install tensorrt-cu12   # separate install; see astral-sh/uv#14313
```

2. Build engines (details in `scripts/README.md`):

```bash
uv run python scripts/create_tensorrt.py \
  --onnx_dir onnx_models --engine_dir trt_engines --precision fp32 --config config/tts.json
```

Note: `examples/all_langs_and_mix.py --tensorrt`

## Citations

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

## Acknowledgments

Hebrew G2P uses [renikud](https://github.com/thewh1teagle/renikud). Thanks to [thewh1teagle](https://github.com/thewh1teagle).

## License

MIT
