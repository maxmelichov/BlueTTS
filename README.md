# Blue

Hebrew Text-to-Speech inference using ONNX Runtime with optional TensorRT acceleration.

## ✨ Demo

<p align="center">
  <a href="https://huggingface.co/spaces/notmax123/Blue"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Try%20Live%20Demo-FFD21E?style=for-the-badge" alt="Try Live Demo on Hugging Face"></a>
  &nbsp;
  <a href="https://lightbluetts.com/"><img src="https://img.shields.io/badge/%F0%9F%8C%90%20Website-lightbluetts.com-2563EB?style=for-the-badge" alt="lightbluetts.com"></a>
  &nbsp;
  <iframe src="https://github.com/sponsors/maxmelichov/button" title="Sponsor maxmelichov" height="32" width="114" style="border: 0; border-radius: 6px;"></iframe>
</p>
<p align="center">🎙️ Human-sounding TTS in <b>Hebrew</b>, <b>English</b>, <b>Spanish</b>, <b>Italian</b> &amp; <b>German</b> — try samples and the live demo on the site.</p>

## Installation

Install the core dependencies:

```bash
uv sync
```

For CUDA (GPU) support:

```bash
uv sync --extra gpu
```

## Download Models

```bash
uv run hf download notmax123/blue-onnx --repo-type model --local-dir ./onnx_models \
  --exclude "voices/all_voices/**"
```

Optional:

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

Examples use `voices/female1.json` from this repo. After the optional voice download, use paths under `onnx_models/voices/all_voices/` (`manifest.tsv` lists them).

Here is a basic example of how to use `BlueTTS` in Python:

```python
import soundfile as sf
from src.blue_onnx import BlueTTS

tts = BlueTTS(
    onnx_dir="onnx_models",
    style_json="voices/female1.json",
    renikud_path="model.onnx",
)

# Single language
samples, sr = tts.synthesize("שלום, זהו מודל דיבור בעברית.", lang="he")
sf.write("output.wav", samples, sr)

# Mixed languages
mixed = "שלום לכולם, <en>welcome to the presentation</en>, <es>espero que lo disfruten</es>."
samples, sr = tts.synthesize(mixed, lang="he")
sf.write("mixed_output.wav", samples, sr)
```

### Running Examples

You can run the provided example scripts to test the model. Outputs will be saved in the `examples/out/` directory.

```bash
# Generate samples for individual languages
uv run python examples/hebrew.py
uv run python examples/english.py
uv run python examples/spanish.py
uv run python examples/italian.py
uv run python examples/german.py

# Generate a mixed-language sample
uv run python examples/mixed.py

# Run the CLI app
uv run python examples/app.py --lang en --text "Hello world."
```

## TensorRT (NVIDIA GPUs Only)

For faster inference on NVIDIA GPUs, you can build TensorRT engines.

1. Install TensorRT dependencies:

```bash
uv sync --extra tensorrt
uv pip install tensorrt-cu12  # installed separately due to astral-sh/uv#14313
```

2. Build the engines (see `scripts/README.md` for details):

```bash
uv run python scripts/create_tensorrt.py \
  --onnx_dir onnx_models --engine_dir trt_engines --precision fp16 --config config/tts.json
```

*(Note: The `examples/all_langs_and_mix.py --tensorrt` flag is currently bugged).*

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

## Acknowledgments

This project uses [renikud](https://github.com/thewh1teagle/renikud) for Hebrew G2P. Special thanks to [thewh1teagle](https://github.com/thewh1teagle) for his work on Hebrew phonemization.

## License

MIT
