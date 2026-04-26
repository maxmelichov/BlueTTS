# Blue

Text-to-speech inference with ONNX Runtime; optional TensorRT acceleration on NVIDIA GPUs.

<p align="center">
  <a href="https://pypi.org/project/blue-onnx/"><img src="https://img.shields.io/pypi/v/blue-onnx?style=for-the-badge&amp;label=PyPI" alt="PyPI version"></a>
  &nbsp;
  <a href="https://huggingface.co/spaces/notmax123/BlueV2"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Try%20Live%20Demo-FFD21E?style=for-the-badge" alt="Try Live Demo on Hugging Face"></a>
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
```

The default environment uses the stock `onnxruntime` CPU wheel. **OpenVINO** and **CUDA (`gpu`)** are optional; each adds a second ONNX Runtime distribution until you remove the stock CPU wheel so the accelerator build owns the `onnxruntime` import (`uv pip uninstall onnxruntime`, or the same with `pip` after a PyPI install with `[openvino]` or `[gpu]`). Do not combine the `openvino` and `gpu` extras. For **TensorRT**, use `uv sync --extra tensorrt` in [TensorRT](#tensorrt-nvidia-only). For **voice or ONNX export**, add `--extra export` in [Models](#models) (with the PyTorch checkpoint download).

## Models

**ONNX bundle** (slim graph; [notmax123/blue-onnx-v2](https://huggingface.co/notmax123/blue-onnx-v2)):

```bash
uv run hf download notmax123/blue-onnx-v2 --repo-type model --local-dir ./onnx_models
```

The Hub bundle does **not** include per-voice **style JSON**; use the sample `voices/*.json` from **this repository** (or on [GitHub](https://github.com/maxmelichov/BlueTTS/tree/main/voices)), or **export a new voice** from a reference clip (see [exports/README.md](exports/README.md), PyTorch weights below). If you use `pip` without `uv`, the same CLI is available after install because `blue-onnx` depends on `huggingface-hub` — run `hf download ...` with the same arguments.

**Optional**

- Hebrew G2P: 
  ```bash
  wget -O model.onnx https://huggingface.co/thewh1teagle/renikud/resolve/main/model.onnx
  ```
- PyTorch checkpoints ([notmax123/blue-v2](https://huggingface.co/notmax123/blue-v2)) for **exporting new voice JSON** and ONNX: `uv sync --extra export` then
  ```bash
  uv run hf download notmax123/blue-v2 --repo-type model --local-dir ./pt_models
  ```

## Usage

Examples below use `voices/female1.json` from this repo, or a JSON you produced with `exports/export_new_voice.py`.

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
uv run python examples/basic.py   # he / en / es / it / de + mixed in one run
uv run python examples/mixed.py
uv run python examples/app.py --lang en --text "Hello world."
```

Edit `onnx_dir` and voice JSON in each example if your paths differ. See [examples/voices.md](examples/voices.md) for `app.py` and voice selection.

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
