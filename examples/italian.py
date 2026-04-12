import sys
import soundfile as sf

sys.path.append(".")
from src.blue_onnx import BlueTTS

tts = BlueTTS(
    onnx_dir="onnx_models",
    style_json="voices/female1.json"
)

text = "Ciao, questa è una prova breve in italiano."
audio, sr = tts.synthesize(text, lang="it")
sf.write("italian_example.wav", audio, sr)
print("Saved italian_example.wav")
