# Stage 3: Duration predictor

**Script:** `src/train_duration_predictor.py`  
**Prerequisites:** frozen AE encoder weights, `stats_*.pt`.

Utterance-level **log-duration** (not phoneme-level). At inference, sets the latent **length** for TTL; does not set timbre quality.

**Architecture** (DP text/ref encoders, MLP head), **reference sampling**, **loss**, **speaker-balanced sampling**, **data flow**, **3k–6k step settings**, and the **TTL+DP inference** coupling are in **[`architecture_and_legacy_training.md`](architecture_and_legacy_training.md)**.

```bash
cd training
python src/train_duration_predictor.py \
    --config configs/tts.json \
    --max_steps 6000 \
    --batch_size 64 \
    --lr 1e-4
```

Checkpoints: `checkpoints/duration_predictor/` (see script for naming).
