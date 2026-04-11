# Blue

Hebrew Text-to-Speech inference using ONNX Runtime with optional TensorRT acceleration.

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

You need to download the ONNX models and the Hebrew G2P model (`renikud`). We use the `hf` CLI (included in dependencies) to download from Hugging Face.

```bash
# 1. Download Blue ONNX models
uv run hf download notmax123/BlueOnnx --repo-type model --local-dir ./onnx_models

# 2. Download Hebrew G2P ONNX model
wget -O model.onnx https://huggingface.co/thewh1teagle/renikud/resolve/main/model.onnx
```

*(Optional)* If you want to create new voices (extract latent mean/std), download the PyTorch weights and the multilingual stats:

```bash
uv sync --extra export
uv run hf download notmax123/Blue --repo-type model --local-dir ./pt_models
```

## Usage

Here is a basic example of how to use `BlueTTS` in Python:

```python
import soundfile as sf
from src.blue_onnx import BlueTTS

tts = BlueTTS(
    onnx_dir="onnx_models",
    style_json="voices/female1_new.json",
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
# Generate samples for all supported languages (he, en, es, it, ge) + mixed
uv run python examples/basic.py

# Generate all languages and save a manifest JSON
uv run python examples/all_langs_and_mix.py

# Run the CLI app
uv run python examples/app.py --lang en --text "Hello world."
```

## TensorRT (NVIDIA GPUs Only)

For faster inference on NVIDIA GPUs, you can build TensorRT engines.

1. Install TensorRT dependencies:

```bash
uv sync --extra tensorrt
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

## License

MIT
