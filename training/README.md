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

## 1. Combine Dataset

Speaker/chapter tree:

```bash
uv run python combine_datasets.py \
  --spktree /path/to/speech_corpus_R \
  --splits train-clean-360 \
  --lang en \
  --output generated_audio/spktree.csv \
  --clean-output generated_audio/spktree_cleaned.csv
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
  --metadata generated_audio/spktree_cleaned.csv \
  --ae-ckpt ../pt_models/blue_codec.safetensors \
  --output runs/spktree/stats_multilingual.pt \
  --device cuda:0
```

`compute_latent_stats.py` will not overwrite an existing stats file unless you pass `--overwrite`.

## 3. Train DP

```bash
uv run python -m training.dp.cli \
  --config ../config/tts.json \
  --data generated_audio/spktree_cleaned.csv \
  --ae_checkpoint ../pt_models/blue_codec.safetensors \
  --stats_path runs/spktree/stats_multilingual.pt \
  --out runs/spktree/dp \
  --device cuda:0
```

## 4. Train T2L

```bash
uv run python -m training.t2l.cli \
  --config ../config/tts.json \
  --data generated_audio/spktree_cleaned.csv \
  --ae_checkpoint ../pt_models/blue_codec.safetensors \
  --stats_path runs/spktree/stats_multilingual.pt \
  --out runs/spktree/t2l \
  --device cuda:0
```

## 5. GRPO (optional)

FlowTTS-GRPO post-training of the T2L vector field (arXiv 2606.23190), rewarding speaker similarity (ECAPA) and
quality (DNSMOS P.835). Only the vector field trains, so voices exported from the starting checkpoint stay valid.
Extra dependencies: `uv pip install speechbrain speechmos onnxruntime transformers`.

The intelligibility reward (Whisper large-v3 WER, `--w_asr`, default 1) needs the words of each row in a `transcript`
column, which `combine_datasets.py` does not write. Add that column to use it; without it, pass `--w_asr 0` as below.

```bash
uv run python -m training.rl.cli \
  --ckpt runs/spktree/t2l/ckpt_step_1000000.pt \
  --config ../config/tts.json \
  --data generated_audio/spktree_cleaned.csv \
  --ae_checkpoint ../pt_models/blue_codec.safetensors \
  --stats_path runs/spktree/stats_multilingual.pt \
  --dp_ckpt runs/spktree/dp/duration_predictor_final.pt \
  --out runs/spktree/grpo \
  --w_asr 0 \
  --device cuda:0
```

Score `ckpt_step_*.pt` like any T2L checkpoint, at the usual steps and CFG; the sampler is unchanged.
