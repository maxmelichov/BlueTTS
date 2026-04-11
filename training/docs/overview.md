# Blue training overview

Hebrew TTS based on [SupertonicTTS](https://arxiv.org/abs/2503.23108). Stages run **in order**: autoencoder → latent statistics → text-to-latent (TTL) and duration predictor (DP).

| Doc | Contents |
|-----|----------|
| [`../README.md`](../README.md) | Commands: dataset prep, stats, DP, TTL |
| [`architecture_and_legacy_training.md`](architecture_and_legacy_training.md) | **Full architecture**, **training duration/settings**, parameter tables, flow matching, DP details, config |

---

## Pipeline (compact)

```
Audio → [Stage 1: AE] → z (24-D @ ~86 Hz)
           ↓
     latent mean/std (*.pt)
           ↓
    ┌──────┴──────┐
    ▼             ▼
 [TTL: flow]   [DP: duration]
    └──────┬──────┘
           ▼
    AE decoder → waveform
```

**TTL inference:** Euler **NFE=32**, **CFG=3**. **DP** predicts utterance length only.

---

## Dataset (typical)

| | |
|--|--|
| Scale (large run) | ~5.9M files / ~10k hours |
| Language | Hebrew |
| Sample rate | 44,100 Hz |
| Metadata | CSV: paths, text/phonemes, `speaker_id` |

---

## Where things live

All dimensions and hyperparameters: **`configs/tts.json`** (`ae`, `ttl`, `dp`).
