"""
Measure synthesis speed on this machine (real-time factor = seconds of audio per second of compute).

The hosted demo shares one server and adds network and queue time, so it does not show the model's
speed; this does.

Run:
    uv run python examples/benchmark.py                 # CPU
    uv run python examples/benchmark.py --threads 4 --steps 4
    uv run python examples/benchmark.py --ref-wav my_voice.wav      # a cloned voice
    uv run python examples/benchmark.py --providers CUDAExecutionProvider
"""

import argparse
import os
import sys
import time

sys.path.append(".")

ap = argparse.ArgumentParser()
ap.add_argument("--onnx-dir", default=os.environ.get("ONNX_DIR", "bluev3_onnx/onnx"))
ap.add_argument("--voice", default="voices/noa.json")
ap.add_argument("--ref-wav", default=None, help="clone this clip instead of loading --voice")
ap.add_argument("--steps", type=int, default=None, help="flow steps (default: the bundle's, 8)")
ap.add_argument("--providers", default="CPUExecutionProvider")
ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = the library default)")
ap.add_argument("--runs", type=int, default=5)
args = ap.parse_args()
if args.threads:
    os.environ["ORT_NUM_THREADS"] = str(args.threads)

from src.blue_onnx import BlueTTS  # noqa: E402

tts = BlueTTS(onnx_dir=args.onnx_dir, style_json=args.voice, providers=args.providers.split(","))
if args.ref_wav:
    t0 = time.perf_counter()
    tts.style = tts.voice_from_wav(args.ref_wav)   # one-off: cloning does not change the synthesis cost
    print(f"cloned {args.ref_wav} in {time.perf_counter() - t0:.2f} s")
TEXTS = {
    "short (1 sentence)": "The weather is lovely today, so I think we should go for a walk in the park.",
    "long (5 sentences)": "The weather is lovely today, so I think we should go for a walk in the park. "
    "Could you tell me how to get to the central station from here? "
    "First, heat the oil, then add the onions, the garlic, and a pinch of salt. "
    "My brother, who lives abroad, called last night, but I was already asleep. "
    "When the rain finally stopped, we packed the car and drove north along the coast.",
}
# warm-up: graph init and G2P load are not synthesis time
tts.synthesize(TEXTS["short (1 sentence)"], lang="en", total_step=args.steps)
print(f"providers {args.providers}; threads {args.threads or 'default'}; steps {args.steps or 'bundle default'}; "
      f"best of {args.runs} runs")
for name, text in TEXTS.items():
    best = None
    for _ in range(args.runs):
        t0 = time.perf_counter()
        samples, sr = tts.synthesize(text, lang="en", total_step=args.steps)
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    t0 = time.perf_counter()
    first = next(iter(tts.stream(text, lang="en", total_step=args.steps)))
    ttfa = time.perf_counter() - t0
    audio_s = len(samples) / sr
    print(f"{name}, {len(text)} characters: {audio_s:.1f} s of audio in {best:.2f} s = {audio_s / best:.0f}x real time; "
          f"first audio (streaming) after {ttfa:.2f} s")
