import sys
import soundfile as sf

sys.path.append(".")
from src.blue_onnx import BlueTTS

tts = BlueTTS(
    onnx_dir="onnx_models",
    style_json="voices/female1.json"
)

per_lang = [
    ("he", "שלום, זהו מודל דיבור בעברית."),
    ("en", "Hello, this is a short English test."),
    ("es", "Hola, esta es una prueba breve en español."),
    ("it", "Ciao, questa è una prova breve in italiano."),
    ("ge", "Hallo, das ist ein kurzer deutscher Test."),
]

for code, text in per_lang:
    audio, sr = tts.synthesize(text, lang=code)
    sf.write(f"sample_{code}.wav", audio, sr)
    print(f"Saved sample_{code}.wav")

mixed = "שלום לכולם, <en>welcome to the demo</en>, <es>gracias por escuchar</es>."
audio, sr = tts.synthesize(mixed, lang="he")
sf.write("sample_mixed.wav", audio, sr)
print("Saved sample_mixed.wav")
