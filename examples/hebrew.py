#!/usr/bin/env python3
import soundfile as sf
import sys
from pathlib import Path

# Ensure we can import from src
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.blue_onnx import BlueTTS

def main():
    print("Loading BlueTTS...")
    tts = BlueTTS(
        onnx_dir=str(repo_root / "onnx_models"),
        style_json=str(repo_root / "voices" / "female1_new.json"),
        renikud_path=str(repo_root / "model.onnx"),
    )

    text = "שלום, זהו מודל דיבור בעברית."
    print(f"Synthesizing Hebrew: {text}")
    
    audio, sr = tts.synthesize(text, lang="he")
    
    out_dir = repo_root / "examples" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hebrew_example.wav"
    
    sf.write(str(out_path), audio, sr)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
