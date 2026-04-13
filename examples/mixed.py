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
    "<en>Hello and welcome. We begin in English, then take a quick trip across a few other languages.</en> "
    "שלום וברוכים הבאים. עכשיו נעבור בטבעיות בין כמה שפות שונות. "
    "<es>Hola y bienvenidos. Ahora damos un pequeno salto al espanol, con una frase clara y alegre.</es> "
    "<it>Ciao a tutti. Adesso facciamo una breve sosta in italiano, con un tono naturale e scorrevole.</it> "
    "<de>Hallo zusammen. Und zum Schluss besuchen wir noch das Deutsche mit einem kurzen, klaren Satz.</de> "
)

audio, sr = tts.synthesize(text, lang="he")
sf.write("mixed_example.wav", audio, sr)
print("Saved mixed_example.wav")
