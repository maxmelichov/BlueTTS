# Stage 2: Text-to-latent (TTL)

**Script:** `src/train_text_to_latent.py`  
**Prerequisites:** trained AE, `stats_*.pt` from `compute_latent_stats.py`.

Flow-matching model: text + reference audio → compressed latents; inference uses Euler (**NFE=32**) and classifier-free guidance (**CFG=3**).

**Architecture** (block diagrams, attention, flow-matching math, reference crops), **parameter counts**, **700k-step hyperparameters**, and **config knobs** are documented in **[`architecture_and_legacy_training.md`](architecture_and_legacy_training.md)**.

```bash
cd training
python src/train_text_to_latent.py --config configs/tts.json
```

Multi-GPU: `torchrun --nproc_per_node=N src/train_text_to_latent.py --config configs/tts.json`  
Finetune: add `--finetune` (see script for LR / SPFM behavior).
