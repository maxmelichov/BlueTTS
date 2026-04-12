import sys
import soundfile as sf

sys.path.append(".")
from src.blue_onnx import BlueTTS

tts = BlueTTS(
    onnx_dir="onnx_models",
    style_json="voices/female1.json"
)

text = "Hello, this is a short English test."
audio, sr = tts.synthesize(text, lang="en")
sf.write("english_example.wav", audio, sr)
print("Saved english_example.wav")
