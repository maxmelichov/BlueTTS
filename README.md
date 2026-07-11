# Blue

Text-to-speech inference with ONNX Runtime; optional TensorRT acceleration on NVIDIA GPUs.

<p align="center">
  <a href="https://huggingface.co/spaces/notmax123/BlueV2"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Try%20Live%20Demo-FFD21E?style=for-the-badge" alt="Try Live Demo on Hugging Face"></a>
  &nbsp;
  <a href="https://lightbluetts.com/"><img src="https://img.shields.io/badge/%F0%9F%8C%90%20Website-lightbluetts.com-2563EB?style=for-the-badge" alt="lightbluetts.com"></a>
</p>

<p align="center">Hebrew, English, Spanish, Italian, and German — samples and a live demo on the site and Space above.</p>

## Install

Requires **Python 3.12+**.

```bash
git clone https://github.com/maxmelichov/BlueTTS.git
cd BlueTTS
uv sync
```

Optional accelerators (do not combine `openvino` and `gpu`; then `uv pip uninstall onnxruntime` so the accelerator build owns the `onnxruntime` import):

```bash
uv sync --extra openvino   # Intel OpenVINO EP
uv sync --extra gpu        # NVIDIA CUDA ORT
```

## Models

**ONNX** — [notmax123/BlueV3-onnx](https://huggingface.co/notmax123/BlueV3-onnx): the four core graphs (`text_encoder`, `vector_estimator`, `vocoder`, `duration_predictor`), runtime `tts.json` / `vocab.json`, and zero-shot voice conversion graphs (`codec_encoder`, `style_encoder`, `duration_style_encoder`).

```bash
uv run hf download notmax123/BlueV3-onnx --repo-type model --local-dir ./onnx_models
```

**Hebrew G2P**:

```bash
wget -O model.onnx https://huggingface.co/thewh1teagle/renikud/resolve/main/model.onnx
```

**PyTorch checkpoints** ([notmax123/BlueV3](https://huggingface.co/notmax123/BlueV3)) — only needed to export a new voice JSON or ONNX graphs (`uv sync --extra export`):

```bash
uv run hf download notmax123/BlueV3 --repo-type model --local-dir ./pt_models
```

Voice style JSONs are not on the Hub; use `voices/*.json` from this repo (`Rotem`, `Roi`) or export your own with [`exports/export_new_voice.py`](exports/export_new_voice.py).

## Usage

```python
import soundfile as sf
from src.blue_onnx import BlueTTS

tts = BlueTTS(
    onnx_dir="onnx_models",
    style_json="voices/Rotem.json",
    renikud_path="model.onnx",
)

samples, sr = tts.synthesize("שלום, זהו מודל דיבור בעברית.", lang="he")
sf.write("output.wav", samples, sr)

mixed = "שלום לכולם, <en>welcome to the presentation</en>, <es>espero que lo disfruten</es>."
samples, sr = tts.synthesize(mixed, lang="he")
sf.write("mixed_output.wav", samples, sr)
```

## Examples

```bash
uv run python examples/basic.py   # he / en / es / it / de + mixed in one run
uv run python examples/mixed.py
uv run python examples/app.py --lang en --text "Hello world."
```

Set `ONNX_DIR` for `basic.py` / `mixed.py`, or pass `--onnx-dir` to `app.py` if your graphs live elsewhere. See [examples/voices.md](examples/voices.md) for voice selection.

## TensorRT (NVIDIA only)

```bash
uv sync --extra tensorrt
uv pip install tensorrt-cu12   # separate install; see astral-sh/uv#14313

uv run python exports/create_tensorrt.py \
  --onnx_dir onnx_models --engine_dir trt_engines --precision fp32 --config config/tts.json
```

See [exports/README.md](exports/README.md#build-tensorrt-engines) for details.

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

This software can produce speech that mimics a reference voice. **The maintainers and contributors are not responsible** for what you do with it—compliance with law, consent from voice owners, and ethical use are **entirely your responsibility**. Do not use it to deceive, impersonate without permission, or infringe anyone's rights.
