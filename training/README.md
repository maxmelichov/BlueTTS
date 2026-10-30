# Training

Run everything from this folder:

```bash
cd training
uv sync --extra cu128
```

Order:

```text
combine dataset -> compute stats -> train DP -> train T2L
```

Stats are required. Use the stats file made from the same CSV you train on.

The autoencoder is not trained here. It comes from the [bluecodec](https://github.com/maxmelichov/blue-codec)
package, whose trainer also has the two-stage recipe on the official Supertonic-3 decoder
([SupertonicTTS, arXiv:2503.23108](https://arxiv.org/abs/2503.23108); see its `docs/training.md`).
Those decoder weights are OpenRAIL-M and are not bundled with either repo. An autoencoder trained
with that recipe uses edge padding: copy its `"pad_mode": "replicate"` into `ae.encoder` and
`ae.decoder` of the config you train and export with, then recompute the stats with its encoder.
Without the key both default to `"zeros"`, as in the published `blue_codec.safetensors`.

## 1. Combine Dataset

LibriTTS:

```bash
uv run python combine_datasets.py \
  --libritts /path/to/LibriTTS_R \
  --splits train-clean-360 \
  --lang en \
  --output generated_audio/libritts.csv \
  --clean-output generated_audio/libritts_cleaned.csv
```

Single CSV:

```bash
uv run python combine_datasets.py \
  --csv /path/to/metadata.csv \
  --audio-dir /path/to/wavs \
  --speaker-id 1 \
  --lang he \
  --output generated_audio/my_voice.csv \
  --clean-output generated_audio/my_voice_cleaned.csv
```

Smoke test: add `--limit 32`.

## 2. Compute Stats

```bash
uv run python compute_latent_stats.py \
  --tts-json ../config/tts.json \
  --metadata generated_audio/libritts_cleaned.csv \
  --ae-ckpt ../pt_models/blue_codec.safetensors \
  --output runs/libritts/stats_multilingual.pt \
  --device cuda:0
```

`compute_latent_stats.py` will not overwrite an existing stats file unless you pass `--overwrite`.

## 3. Train DP

```bash
uv run python -m training.dp.cli \
  --config ../config/tts.json \
  --data generated_audio/libritts_cleaned.csv \
  --ae_checkpoint ../pt_models/blue_codec.safetensors \
  --stats_path runs/libritts/stats_multilingual.pt \
  --out runs/libritts/dp \
  --device cuda:0
```

## 4. Train T2L

```bash
uv run python -m training.t2l.cli \
  --config ../config/tts.json \
  --data generated_audio/libritts_cleaned.csv \
  --ae_checkpoint ../pt_models/blue_codec.safetensors \
  --stats_path runs/libritts/stats_multilingual.pt \
  --out runs/libritts/t2l \
  --device cuda:0
```
