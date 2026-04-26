import sys
from pathlib import Path
import soundfile as sf

sys.path.append(".")
from src.blue_onnx import load_text_to_speech, load_voice_style

tts = load_text_to_speech(onnx_dir="onnx_slim")
style = load_voice_style(["voices/reference_pt.json"])
OUT = Path("examples/out"); OUT.mkdir(parents=True, exist_ok=True)


def save(text: str, lang: str, name: str, text_is_phonemes: bool = False):
    audio, _ = tts(
        text,
        lang=lang,
        style=style,
        total_step=5,
        cfg_scale=3.0,
        text_is_phonemes=text_is_phonemes,
    )
    if audio.ndim == 2: audio = audio[0]
    sf.write(OUT / name, audio, tts.sample_rate)
    print(f"Saved {OUT / name}  ({len(audio)/tts.sample_rate:.2f}s)")


for code, text in [
    ("he", "שלום, זהו מודל דיבור בעברית."),
    ("en", "Hello, this is a short English test."),
    ("es", "Hola, esta es una prueba breve en español."),
    ("it", "Ciao, questa è una prova breve in italiano."),
    ("de", "Hallo, das ist ein kurzer deutscher Test."),
]:
    save(text, code, f"basic_{code}.wav")

save(
    "שלום לכולם, <en>welcome to the demo</en>, <es>gracias por escuchar</es>.",
    "he", "basic_mixed.wav",
)

assert tts.g2p is not None
ready_phonemes = tts.g2p.phonemize("Hello, this sentence is already phonemes.", lang="en")
save(ready_phonemes, "en", "basic_en_ready_phonemes.wav", text_is_phonemes=True)
