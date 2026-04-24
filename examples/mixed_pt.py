"""Mixed-language PT smoke test against pt_models/."""
import sys
from pathlib import Path

import soundfile as sf

sys.path.append(".")
from src.blue_pt import load_text_to_speech, load_voice_style


tts = load_text_to_speech(
    weights_dir="pt_models",
    config_path="config/tts.json",
    device="cpu",
    text2latent_ckpt="pt_models/vf_estimetor.pt",
    ae_ckpt="pt_models/blue_codec.safetensors",
    dp_ckpt="pt_models/duration_predictor_final.pt",
)
style = load_voice_style(["voices/reference_new.json"], device="cpu")

text = (
    "<en>Hello and welcome. We begin in English, then take a quick trip across a few other languages.</en> "
    "שלום וברוכים הבאים. עכשיו נעבור בטבעיות בין כמה שפות שונות. "
    "<es>Hola y bienvenidos. Ahora damos un pequeno salto al espanol, con una frase clara y alegre.</es> "
    "<it>Ciao a tutti. Adesso facciamo una breve sosta in italiano, con un tono naturale e scorrevole.</it> "
    "<de>Hallo zusammen. Und zum Schluss besuchen wir noch das Deutsche mit einem kurzen, klaren Satz.</de> "
)

audio, dur = tts(text, lang="he", style=style, total_step=5, speed=1.0, cfg_scale=3.0)
if audio.ndim == 2:
    audio = audio[0]
out = Path("examples/out/mixed_pt_reference.wav")
out.parent.mkdir(parents=True, exist_ok=True)
sf.write(out, audio, tts.sample_rate)
print(f"Saved {out}  ({len(audio)/tts.sample_rate:.2f}s)")
