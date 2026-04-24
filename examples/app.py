import sys
import argparse
import soundfile as sf

sys.path.append(".")
from src.blue_onnx import BlueTTS

parser = argparse.ArgumentParser()
parser.add_argument("--text", default="שלום עולם")
parser.add_argument("--lang", default="he")
parser.add_argument("--voice", default="female1")
parser.add_argument("--out", default="output.wav")
args = parser.parse_args()

tts = BlueTTS(
    onnx_dir="onnx_models",
    style_json=f"voices/{args.voice}.json"
)

audio, sr = tts.synthesize(args.text, lang=args.lang)
sf.write(args.out, audio, sr)
print(f"Saved to {args.out}")
