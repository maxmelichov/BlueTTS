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

The PyPI package is **inference only** (ONNX TTS in Python). It does not ship training code or the repo examples.

```bash
pip install "blue-onnx"
```

**Inference with pip in three steps:** (1) install as above, (2) put ONNX in `./onnx_models` (see [Models](#models) — `hf download` from `notmax123/blue-onnx-v2`), (3) copy a style JSON (e.g. from [voices/](https://github.com/maxmelichov/BlueTTS/tree/main/voices)) and Hebrew G2P `model.onnx` (see [Models](#models)), then use this from your project:

```python
import soundfile as sf
from blue_onnx import BlueTTS

tts = BlueTTS(
    onnx_dir="onnx_models",
    style_json="voices/female1.json",
    renikud_path="model.onnx",
)
s, sr = tts.synthesize("Hello", lang="en")
sf.write("out.wav", s, sr)
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

**ONNX bundle** ([notmax123/blue-onnx-v2](https://huggingface.co/notmax123/blue-onnx-v2)): the published graphs are **onnx-slim**–cleaned, **full precision** (not INT8). Do not substitute a `--int8` export from [exports/export_onnx.py](exports/export_onnx.py); that path is experimental and not recommended for quality.

```bash
uv run hf download notmax123/blue-onnx-v2 --repo-type model --local-dir ./onnx_models
```

The Hub bundle does **not** include per-voice **style JSON**; use the sample `voices/*.json` from **this repository** (or on [GitHub](https://github.com/maxmelichov/BlueTTS/tree/main/voices)), or **export a new voice** from a reference clip (see [exports/README.md](exports/README.md), PyTorch weights below). If you use `pip` without `uv`, the same CLI is available after install because `blue-onnx` depends on `huggingface-hub` — run `hf download ...` with the same arguments and point `style_json` at a file under `voices/` (e.g. `voices/female1.json`).

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

After the `BlueTTS(...)` setup in [Install](#install) (PyPI) or the same paths in a clone (`import soundfile as sf` and build `tts` as there):

```python
samples, sr = tts.synthesize("שלום, זהו מודל דיבור בעברית.", lang="he")
sf.write("output.wav", samples, sr)

mixed = "שלום לכולם, <en>welcome to the presentation</en>, <es>espero que lo disfruten</es>."
samples, sr = tts.synthesize(mixed, lang="he")
sf.write("mixed_output.wav", samples, sr)
```

If you are editing **this repo** without installing the package, use `from src.blue_onnx import BlueTTS` (as in `examples/`) or put `src` on `PYTHONPATH`.

## Examples

Get models into `./onnx_models` (see [Models](#models)) and `voices/*.json` from the repo, then from the **repository root** use either `uv` or a PyPI install:

```bash
uv run python examples/basic.py   # he / en / es / it / de + mixed in one run
uv run python examples/mixed.py
uv run python examples/app.py --lang en --text "Hello world."
```

With **`pip install "blue-onnx==0.2.5"`** (and the same `onnx_models` + `voices/`), the same files use `import blue_onnx` automatically; a dev tree without the package falls back to `src.blue_onnx`. If your graphs live somewhere else, set `ONNX_DIR` for `basic.py` / `mixed.py`, or pass `--onnx-dir` to `app.py`. Default `app` output: `examples/out/app_output.wav`.

Edit voice JSON paths or `ONNX_DIR` if your layout differs. See [examples/voices.md](examples/voices.md) for `app.py` and voice selection.

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

## Voice cloning and responsibility

This software can produce speech that mimics a reference voice. **The maintainers and contributors are not responsible** for what you do with it—compliance with law, consent from voice owners, and ethical use are **entirely your responsibility**. Do not use it to deceive, impersonate without permission, or infringe anyone’s rights.
