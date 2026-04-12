import sys
import soundfile as sf

sys.path.append(".")
from src.blue_onnx import BlueTTS

tts = BlueTTS(
    onnx_dir="onnx_models",
    style_json="voices/female1.json"
)

text = "Hallo, das ist ein kurzer deutscher Test."
audio, sr = tts.synthesize(text, lang="ge")
sf.write("german_example.wav", audio, sr)
print("Saved german_example.wav")
