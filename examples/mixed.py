import sys
from pathlib import Path
import soundfile as sf

sys.path.append(".")
from src.blue_onnx import load_text_to_speech, load_voice_style

tts = load_text_to_speech(onnx_dir="onnx_slim")
style = load_voice_style(["voices/female1.json"])

text = (
    "<en>Yeah, but listen to this. It doesn't bleed the accents any more when you mix them. So I can be talking in English.</en> "
    "<es>y de repente pasar al español con un acento perfecto.</es> "
        "ובעברית זה פשוט נשמע טבעי לגמרי."
    "<it>mantenendo la voce ma cambiando l'accento.</it> "
    "<de>und auf Deutsch klingt es wie ein Muttersprachler.</de>"
)
text_is_phonemes = False

audio, _ = tts(
    text,
    lang="he",
    style=style,
    total_step=16,
    cfg_scale=3.0,
    text_is_phonemes=text_is_phonemes,
    speed=0.9,
)
if audio.ndim == 2: audio = audio[0]
out = Path("examples/out/mixed.wav"); out.parent.mkdir(parents=True, exist_ok=True)
sf.write(out, audio, tts.sample_rate)
print(f"Saved {out}  ({len(audio)/tts.sample_rate:.2f}s)")
