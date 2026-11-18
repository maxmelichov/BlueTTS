<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img src="assets/logo.svg" alt="BlueTTS: Text To Speech, by Maxim Melichov" width="280">
  </picture>
</p>

<p align="center">
  <b>Multilingual text-to-speech on ONNX Runtime</b><br>
  Hebrew &middot; English &middot; Spanish &middot; Italian &middot; German &middot; Russian &middot; Yiddish
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/notmax123/BlueV3"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Try%20Live%20Demo-FFD21E?style=for-the-badge" alt="Try Live Demo on Hugging Face"></a>
  &nbsp;
  <a href="https://bluetts.com/"><img src="https://img.shields.io/badge/%F0%9F%8C%90%20Website-bluetts.com-2563EB?style=for-the-badge" alt="bluetts.com"></a>
</p>

---

## Quick start

```bash
git clone https://github.com/maxmelichov/BlueTTS.git
cd BlueTTS
uv sync
uv run hf download notmax123/BlueV3-onnx --include "onnx/*" "voices/*" "frontend/*" --local-dir ./bluev3_onnx
uv pip install ./bluev3_onnx/frontend/polyg2p-*.whl     # G2P for en / de / es / it / ru (Linux x86-64, glibc >= 2.34)
```

```python
import soundfile as sf
from blue_onnx import BlueTTS

tts = BlueTTS(onnx_dir="bluev3_onnx/onnx", style_json="voices/noa.json")
samples, sr = tts.synthesize("שלום, זהו מודל דיבור בעברית.", lang="he")
sf.write("out.wav", samples, sr)

samples, sr = tts.synthesize("Hello, this is BlueTTS.", lang="en")
```

The bundle sets the operating point itself (8 steps, CFG 4.0, speed 1.0, language tags on).
Speed 1.0 is each voice's calibrated natural reading pace in each language: a voice JSON can carry
`"speed_scale": {"he": 1.077, "en": 1.03, ...}`, and the duration is divided by `speed * speed_scale[lang]`.
Older voice files without `speed_scale` (or a language it does not list) use 1.0, exactly as before.
Numbers, dates, prices and codes are spoken as words automatically (Yiddish: spell numbers out). Mix languages
inline with `<en>…</en>`:

```python
samples, sr = tts.synthesize("שלום לכולם, <en>welcome to the presentation</en>.", lang="he")
```

Hebrew grapheme-to-phoneme is [RenikudPlus](https://github.com/maxmelichov/RenikudPlus), with the
weights shipped in the bundle (`frontend/renikud/model.onnx`); English, German, Spanish, Italian
and Russian use polyg2p (its wheel in the bundle is for Linux x86-64); Yiddish uses the bundle's
`frontend/yi`. The bundle runs on `blue_onnx`; `blue_trt` does not
read it yet. Details: [Release bundles](src/blue_onnx/README.md#release-bundles).

## Voice cloning

Speak in the voice of a reference recording (3-10 s of clean speech, any sample rate). The bundle's
`reference_encoder.onnx` does it; no PyTorch needed:

```python
tts = BlueTTS(onnx_dir="bluev3_onnx/onnx", ref_wav="my_voice.wav")
samples, sr = tts.synthesize("Hello, this is my cloned voice.", lang="en")

tts.voice_from_wav("my_voice.wav", save_json="my_voice.json")   # keep it: BlueTTS(style_json="my_voice.json")
```

Full script: [`examples/clone.py`](examples/clone.py). A voice works only with the bundle it was made from.

## Streaming

`stream()` yields the audio chunk by chunk, so playback starts after the first chunk instead of after the
whole text ([`examples/stream.py`](examples/stream.py)):

```python
for chunk in tts.stream(long_text, lang="en"):   # mono float32 arrays at tts.sample_rate
    play(chunk)
```

The [OpenAI-compatible server](#openai-compatible-server) streams the same way (`stream_format: "audio"`).

## Speed

Measure it on your machine; the hosted demo shares one server and adds queue and network time:

```bash
uv run python examples/benchmark.py              # CPU; --threads N, --providers CUDAExecutionProvider
```

On a Ryzen 9 7950X with `blue_onnx` (8 steps, CFG 4, float32 graphs): 15-18x real time with 8 threads, 4x with
one thread, first audio after about 0.3 s.

## Documentation

| | |
|---|---|
| [`src/blue_onnx/`](src/blue_onnx/README.md) | **Inference API** — entry points, text normalization, language spans, accelerators |
| [`examples/`](examples/README.md) | Runnable scripts for every feature |
| [`exports/`](exports/README.md) | New voices, ONNX export, TensorRT engines |
| [`training/`](training/README.md) | Dataset prep and the three training stages |

## Install

Requires **Python 3.12+** and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/maxmelichov/BlueTTS.git
cd BlueTTS
uv sync
```

Optional extras are documented per use case: [accelerators](src/blue_onnx/README.md#accelerators)
(OpenVINO, CUDA), [`--extra export`](exports/README.md) for voice/ONNX export, and
[`--extra tensorrt`](exports/README.md#build-tensorrt-engines).

## OpenAI-compatible server

[`examples/openai_server.py`](examples/openai_server.py) serves BlueTTS behind the OpenAI speech API
(`POST /v1/audio/speech`), so OpenAI clients, Open WebUI and plain curl can use it unchanged. Based on the
server shared by [@MaximS](https://github.com/MaximS) in [#22](https://github.com/maxmelichov/BlueTTS/issues/22).

```bash
uv sync --extra server              # fastapi + uvicorn; mp3 / opus / aac also need ffmpeg on PATH
uv run hf download notmax123/BlueV3-onnx --include "onnx/*" "voices/*" "frontend/*" --local-dir ./bluev3_onnx
uv run python examples/openai_server.py --bundle bluev3_onnx --host 0.0.0.0 --port 8000
```

```bash
curl http://localhost:8000/v1/audio/speech -H "Content-Type: application/json" \
  -d '{"model": "bluev3", "input": "שלום, מה שלומך?", "voice": "noa", "response_format": "wav"}' -o speech.wav
```

One engine is loaded at startup (CPU by default; `--providers CUDAExecutionProvider,CPUExecutionProvider` for a
GPU build of ONNX Runtime) and requests are served one synthesis at a time. `--bundle` also reads `BLUE_BUNDLE`;
`--api-key` (or `BLUE_API_KEY`) makes the server require `Authorization: Bearer <key>`, otherwise any key is accepted.

| field | |
|---|---|
| `input` | text, up to 4096 characters |
| `voice` | one of the 15 voices below; OpenAI's names map to fixed voices: `alloy`→`maya`, `echo`→`adam`, `fable`→`daniel`, `onyx`→`david`, `nova`→`noa`, `shimmer`→`lily` |
| `response_format` | `mp3` (default), `wav`, `pcm`, `flac`, `opus`, `aac`. `wav` and `pcm` always work; the others need ffmpeg or a libsndfile that writes them, else 400. `pcm` is 16-bit mono at 24 kHz like OpenAI's (`--pcm-sample-rate 0` for the model's 44.1 kHz) |
| `speed` | 0.25–4.0 |
| `model` | accepted and ignored (`GET /v1/models` lists `bluev3`) |
| `language` | extension: `he`, `en`, `de`, `es`, `it`, `ru`, `yi`. Default: `he` when the text has Hebrew letters, else `en` (Yiddish needs `yi` explicitly) |
| `stream_format` | `audio` (or `"stream": true`) streams `wav` / `pcm` sentence by sentence as chunked transfer; `sse` sends the same bytes as `speech.audio.delta` events. Other formats come back whole |

**Voice markers** (extension): `[name]` in the input switches voice from that point on, e.g.
`"Hello there. [adam] And hello from me. [lily] שלום!"`. Only names of available voices (or the OpenAI
aliases) count as markers; text without markers is read in the request's `voice`, and each segment gets its own
language detection. Inline `<en>…</en>` spans work as in `synthesize`.

Other endpoints: `GET /v1/audio/voices` (voice list), `GET /v1/models`, `GET /health`.

**Open WebUI**: Admin Panel → Settings → Audio → Text-to-Speech engine **OpenAI**, API base URL
`http://<host>:8000/v1` (from a Docker container on the same machine: `http://host.docker.internal:8000/v1`),
any API key (or the `--api-key` value), TTS voice e.g. `noa`, model `bluev3`.

Voices: female `maya`, `noa`, `lily`, `tamar`, `shira`, `yael`, `emma`, `sara`; male `adam`, `daniel`, `david`,
`jonathan`, `ethan`, `michael`, `ariel`. The server reads the bundle's `voices/` and then the repo's `voices/`
(`--voices-dir` to use another directory).

## Models

**Current — BlueV3 `v3.0.1`** (checkpoint `702000`): the text-to-latent EMA at 700k, trained with the RobustSpeechFlow
contrastive loss ([arXiv 2605.22083](https://arxiv.org/abs/2605.22083)), plus 1,500 iterations
of FlowTTS-GRPO reinforcement learning ([arXiv 2606.23190](https://arxiv.org/abs/2606.23190)) and 500 more with a
pause reward, a new duration model (Hebrew paced closer to read speech), and calibrated speed: `speed=1.0` is natural
read-speech pace for every voice and language (`speed_scale` in each voice JSON). v3.0.0 (checkpoint `701500`) stays
on HF under the git tag `v3.0.0`.

| | |
|---|---|
| [notmax123/BlueV3-onnx](https://huggingface.co/notmax123/BlueV3-onnx) | ONNX bundle: `onnx/` (the `blue_onnx` layout + `reference_encoder`), `voices/`, `frontend/` |
| [notmax123/BlueV3](https://huggingface.co/notmax123/BlueV3) | PyTorch checkpoints (text-to-latent `ckpt_step_702000.pt`, duration predictor), latent stats, config, voices |
| [notmax123/blue-codec](https://huggingface.co/notmax123/blue-codec) | the audio autoencoder, one file (`model.safetensors`) |

Languages: Hebrew, English, German, Spanish, Italian, Russian, Yiddish. Operating point: 8 Euler steps,
CFG 4.0, speed 1.0, language tags on.

```python
samples, sr = tts.synthesize("Привет, как дела?", lang="ru")
samples, sr = tts.synthesize("גוט מאָרגן, ווי גייט עס?", lang="yi")
```

**Voices.** 15 voice JSONs (LibriTTS-R speakers, CC BY 4.0) in [`voices/`](voices/) and in the bundle's
`voices/` ([list](examples/voices.md)): female `maya`,
`noa`, `lily`, `tamar`, `shira`, `yael`, `emma`, `sara`; male `adam`, `daniel`, `david`, `jonathan`, `ethan`,
`michael`, `ariel`. A voice JSON is valid only for the checkpoint it was exported from; voices of earlier
releases do not work with BlueV3.

**Scores** (unseen test set and unseen voices, 8 steps, CFG 4, both with the v3.0.1 duration model):

| | v3.0.0 | v3.0.1 |
|---|---|---|
| Hebrew phone error rate | 1.81 % | 1.80 % (no change) |
| Hebrew speaker similarity | 0.664 | **0.670** |
| Hebrew DNSMOS OVRL | 3.41 | **3.43** |
| English word error rate | 1.81 % | 1.95 % (no change) |
| English speaker similarity | 0.676 | **0.679** |
| Pause share, he / en / de (read speech 11.4 / 11.0 / 13.4 %) | 8.9 / 8.7 / 8.9 % | 9.6 / 9.6 / 10.0 % |

**Previous — [notmax123/BlueTTS2.5-onnx](https://huggingface.co/notmax123/BlueTTS2.5-onnx)** and
[notmax123/blue-onnx-v2](https://huggingface.co/notmax123/blue-onnx-v2) still load (no `blue_release` section in
their `tts.json`), with the previous voices in [`voices/v2/`](voices/v2/), e.g.
`BlueTTS(onnx_dir="onnx_models", style_json="voices/v2/noa.json")`. The BlueV3 voices in `voices/` carry
`"model": "bluev3-702000"`, and `blue_onnx` refuses them with an older bundle. blue-onnx-v2 is the only bundle with the zero-shot
voice-conversion graphs that `examples/zero_shot.py` and `blue_onnx.style` need.

## Citations

```bibtex
@ARTICLE{2025arXiv250323108K,
       author = {{Kim}, Hyeongju and {Yang}, Jinhyeok and {Yu}, Yechan and {Ji}, Seunghun and {Morton}, Jacob and {Bous}, Frederik and {Byun}, Joon and {Lee}, Juheon},
        title = "{SupertonicTTS: Towards Highly Efficient and Streamlined Text-to-Speech System}",
      journal = {arXiv e-prints},
     keywords = {Audio and Speech Processing, Machine Learning, Sound},
        pages = {arXiv:2503.23108},
}
@article{kim2025training,
  title={Training Flow Matching Models with Reliable Labels via Self-Purification},
  author={Kim, Hyeongju and Yu, Yechan and Yi, June Young and Lee, Juheon},
  journal={arXiv preprint arXiv:2509.19091},
  year={2025}
}
@misc{yi2025robustttstrainingselfpurifying,
      title={Robust TTS Training via Self-Purifying Flow Matching for the WildSpoof 2026 TTS Track},
      author={June Young Yi and Hyeongju Kim and Juheon Lee},
      year={2025},
      eprint={2512.17293},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2512.17293},
}
@misc{yang2026robustspeechflow,
      title={RobustSpeechFlow: Learning Robust Text-to-Speech Trajectories via Augmentation-based Contrastive Flow Matching},
      author={Jinhyeok Yang and Hyeongju Kim and Yechan Yu and Joon Byun and Frederik Bous and Juheon Lee},
      year={2026},
      eprint={2605.22083},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2605.22083},
}
@misc{wang2026flowttsgrpo,
      title={FlowTTS-GRPO: Online Reinforcement Learning with Multi-Objective Reward Optimization for Flow-Matching Based Text-to-Speech},
      author={Haoxu Wang and Biao Tian and Weiqin Li and Xiang Lv and Han Zhao and Xiangang Li},
      year={2026},
      eprint={2606.23190},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2606.23190},
}
```

## License

MIT.

## Voice cloning and responsibility

This software can produce speech that mimics a reference voice. **The maintainers and
contributors are not responsible** for what you do with it — compliance with law, consent
from voice owners, and ethical use are **entirely your responsibility**. Do not use it to
deceive, impersonate without permission, or infringe anyone's rights.
