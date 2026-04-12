import sys
import soundfile as sf

sys.path.append(".")
from src.blue_onnx import BlueTTS

tts = BlueTTS(
    onnx_dir="onnx_models",
    style_json="voices/female1.json"
)

text = "שלום לכולם, <en>welcome to the demo</en>, <es>gracias por escuchar</es>."
audio, sr = tts.synthesize(text, lang="he")
sf.write("mixed_example.wav", audio, sr)
print("Saved mixed_example.wav")
