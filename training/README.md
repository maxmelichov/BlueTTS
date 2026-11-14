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

The autoencoder comes from [bluecodec](https://github.com/maxmelichov/blue-codec). If it uses edge padding,
set `"pad_mode": "replicate"` in `ae.encoder` / `ae.decoder` (default `"zeros"`).

## 1. Combine Dataset

LibriTTS:

```bash
uv run python combine_datasets.py \
  --libritts /path/to/speech_corpus_R \
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

## 5. GRPO (optional)

FlowTTS-GRPO post-training of the T2L vector field (arXiv 2606.23190), rewarding speaker similarity, Whisper WER and
DNSMOS. Only the vector field trains, so exported voices stay valid. Extra dependencies:
`uv pip install speechbrain speechmos onnxruntime transformers`. The WER reward (`--w_asr`) needs a `transcript`
column with each row's words; without one, pass `--w_asr 0` as below. Whisper transcribes in the row's language;
English is scored through Whisper's English normaliser, other languages through its basic normaliser without
diacritics. `train_grpo(cfg, asr=...)` takes any other `(wav, sr, target, lang) -> error rate` callable, e.g. a phone
error rate against the IPA for a language without orthographic text.

Hard cases (Sec 2.5): `--hard_frac` of the prompts get a word repeated 3-5 times (lwr), 2-3 words repeated 2-3 times
(smr) or the sentence said twice (gsr), at the same word positions in the IPA and the ASR text. Each rollout draws
5-8 steps with a 2-step SDE window starting at step 1-3. BlueV3 v3.0.0 (checkpoint 701500) is its 700k EMA after
`--iters 1500 --rollout_cfg 4 --hard_frac 0.33 --ref_max_frames 129 --dp_ref_max_frames 144 --score_workers 2`, the
defaults otherwise (`--rollout_cfg 4` is Flow-GRPO's guided form; the paper rolls out without CFG).

```bash
uv run python -m training.rl.cli \
  --ckpt runs/libritts/t2l/ckpt_step_1000000.pt \
  --config ../config/tts.json \
  --data generated_audio/libritts_cleaned.csv \
  --ae_checkpoint ../pt_models/blue_codec.safetensors \
  --stats_path runs/libritts/stats_multilingual.pt \
  --dp_ckpt runs/libritts/dp/duration_predictor_final.pt \
  --out runs/libritts/grpo \
  --w_asr 0 \
  --device cuda:0
```
