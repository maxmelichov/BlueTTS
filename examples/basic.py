"""
Usage:
    wget https://huggingface.co/thewh1teagle/renikud/resolve/main/model.onnx
    wget https://github.com/maxmelichov/Light-BlueTTS/releases/download/model-files-v1.0/onnx_models.tar.gz
    wget https://github.com/maxmelichov/Light-BlueTTS/releases/download/model-files-v1.0/voices.tar.gz
    tar -xf onnx_models.tar.gz
    tar -xf voices.tar.gz
    uv pip install renikud-onnx
    uv run examples/basic.py
"""

import soundfile as sf
from renikud_onnx import G2P
from lightblue_onnx import LightBlueTTS

g2p = G2P("model.onnx")
tts = LightBlueTTS("onnx_models", style_json="voices/female1.json")

text = "שימו לב נוסעים יקרים, הרכבת תכנס לתחנת תל אביב מרכז בעוד מספר דקות, אנא התרחקו מקצה הרציף והמתינו מאחורי הקו הצהוב, תודה."
phonemes = g2p.phonemize(text)
print(f"Phonemes: {phonemes}")

samples, sr = tts.create(phonemes)

sf.write("audio.wav", samples, sr)
print("Saved audio.wav")
