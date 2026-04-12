import sys
from pathlib import Path

import soundfile as sf

sys.path.append(".")
from src.blue_onnx import BlueTTS


tts = BlueTTS(
    onnx_dir="onnx_models",
    style_json="voices/female1.json",
)

text = (
    "שלום וברוכים הבאים להדגמה הרב לשונית. "
    "<en>Hello and welcome. This short sample shows a clean switch into English.</en> "
    "<es>Hola y bienvenidos. Ahora escuchan una frase clara y agradable en espanol.</es> "
    "<it>Ciao a tutti. Adesso sentite una breve frase in italiano, fluida e naturale.</it> "
    "<de>Hallo zusammen. Zum Schluss horen Sie einen kurzen, klaren Satz auf Deutsch.</de> "
)

audio, sr = tts.synthesize(text, lang="he")
sf.write("mixed_example.wav", audio, sr)
print("Saved mixed_example.wav")
