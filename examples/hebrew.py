import sys
import soundfile as sf

sys.path.append(".")
from src.blue_onnx import BlueTTS

tts = BlueTTS(
    onnx_dir="onnx_models",
    style_json="voices/female1.json"
)

text = "שלום, זהו מודל דיבור בעברית."
audio, sr = tts.synthesize(text, lang="he")
sf.write("hebrew_example.wav", audio, sr)
print("Saved hebrew_example.wav")
