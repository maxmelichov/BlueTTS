import sys
import soundfile as sf

sys.path.append(".")
from src.blue_onnx import BlueTTS

tts = BlueTTS(
    onnx_dir="onnx_models",
    style_json="voices/female1.json"
)

text = "Hola, esta es una prueba breve en español."
audio, sr = tts.synthesize(text, lang="es")
sf.write("spanish_example.wav", audio, sr)
print("Saved spanish_example.wav")
