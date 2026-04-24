import sys
from pathlib import Path
import soundfile as sf

sys.path.append(".")
from src.blue_onnx import load_text_to_speech, load_voice_style

tts = load_text_to_speech(onnx_dir="onnx_slim")
style = load_voice_style(["voices/reference_pt.json"])

audio, _ = tts("Hola, esta es una prueba breve en español.", lang="es", style=style, total_step=16, cfg_scale=3.0)
if audio.ndim == 2: audio = audio[0]
out = Path("examples/out/spanish.wav"); out.parent.mkdir(parents=True, exist_ok=True)
sf.write(out, audio, tts.sample_rate)
print(f"Saved {out}  ({len(audio)/tts.sample_rate:.2f}s)")
