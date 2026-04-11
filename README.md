# Light-BlueTTS

Hebrew Text-to-Speech inference using ONNX Runtime with optional TensorRT acceleration.

## Installation

```bash
uv sync                    # core deps
uv sync --extra gpu        # + CUDA support
```

## Model Weights

Download the TTS weights from [notmax123/LightBlue](https://huggingface.co/notmax123/LightBlue) and the G2P model from [thewh1teagle/renikud-onnx](https://huggingface.co/thewh1teagle/renikud-onnx).

```bash
uv run hf download notmax123/LightBlue --repo-type model --local-dir ./onnx_models
wget https://huggingface.co/thewh1teagle/renikud/resolve/main/model.onnx
```

## Usage & Examples

You can use the `src.blue_onnx` module directly to run inference. The model supports 5 languages (`he`, `en`, `es`, `it`, `ge`) and you can even mix them in a single sentence using `<lan></lan>` tags!

### Basic Example

```python
import soundfile as sf
from src.blue_onnx import LightBlueTTS

# Initialize the TTS engine with the ONNX models and a voice style
tts = LightBlueTTS(
    onnx_dir="onnx_models", 
    style_json="voices/female1.json",
    renikud_path="model.onnx" # Required for Hebrew text
)

# Synthesize speech
text = "שלום, זהו מודל דיבור בעברית."
samples, sr = tts.synthesize(text, lang="he")

# Save to file
sf.write("output.wav", samples, sr)
print("Saved output.wav")
```

### Multi-Language Example

You can combine multiple languages in a single prompt by using `<lan></lan>` tags (e.g., `<en></en>`, `<es></es>`, `<it></it>`, `<ge></ge>`, `<he></he>`). 

```python
import soundfile as sf
from src.blue_onnx import LightBlueTTS

tts = LightBlueTTS(
    onnx_dir="onnx_models", 
    style_json="voices/female1.json",
    renikud_path="model.onnx"
)

# Mix Hebrew, English, and Spanish in one sentence!
mixed_text = "שלום לכולם, <en>welcome to the presentation</en>, <es>espero que lo disfruten</es>."

# Synthesize the mixed text (default base language is Hebrew)
samples, sr = tts.synthesize(mixed_text, lang="he")

sf.write("mixed_output.wav", samples, sr)
print("Saved mixed_output.wav")
```

See the [examples](examples/) folder for more scripts.

## TensorRT

ONLY FOR NVIDIA GPUS!

```bash
uv run scripts/create_tensorrt.py --onnx_dir onnx_models --engine_dir trt_engines --precision fp16
uv run scripts/benchmark_trt.py --style_json voices/female1.json --steps 32
```

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
