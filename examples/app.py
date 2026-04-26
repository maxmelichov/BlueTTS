import sys
import argparse
import soundfile as sf

sys.path.append(".")
from src.blue_onnx import load_text_to_speech, load_voice_style

parser = argparse.ArgumentParser()
parser.add_argument("--text", default="שלום עולם")
parser.add_argument("--lang", default="he")
parser.add_argument("--voice", default="female1")
parser.add_argument("--out", default="output.wav")
parser.add_argument("--onnx-dir", default="onnx_models")
parser.add_argument("--text-is-phonemes", action="store_true")
args = parser.parse_args()

tts = load_text_to_speech(onnx_dir=args.onnx_dir)
style = load_voice_style([f"voices/{args.voice}.json"])

audio, _ = tts(
    args.text,
    lang=args.lang,
    style=style,
    total_step=5,
    cfg_scale=3.0,
    text_is_phonemes=args.text_is_phonemes,
)
if audio.ndim == 2:
    audio = audio[0]
sr = tts.sample_rate
sf.write(args.out, audio, sr)
print(f"Saved to {args.out}")
