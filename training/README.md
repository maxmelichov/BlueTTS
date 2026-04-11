# Training — Light-BlueTTS

Install from the `training/` directory:

```bash
cd training
uv sync
# GPU — pick one (do not combine):
uv sync --extra cu128   # PyTorch + CUDA 12.8 (stable)
uv sync --extra cu132   # PyTorch + CUDA 13.2 (nightly wheels; experimental)
```

`bluecodec` is already a Git dependency in `pyproject.toml` and is installed with `uv sync`.

*Pretrained weights for the codec: [`notmax123/blue-codec`](https://huggingface.co/notmax123/blue-codec) on Hugging Face.*

Training scripts live under `training/src/`. Run dataset and stats commands from `training/` unless you adjust paths. Defaults inside some scripts still mention `combined_dataset_cleaned_real_data.csv`, `checkpoints/ae/`, or `stats_real_data.pt` — align filenames with your layout or edit those constants / `configs/tts.json` (`ae_ckpt_path`).

---

## 🧹 3. Prepare Dataset for Text-to-Latent & Duration Predictor

Before training the downstream models (Duration Predictor and flow matching), combine and clean labeled datasets (audio + phonemes).

1. Keep labeled data (e.g. `voice1`, `voice2`) under `generated_audio/` (or paths you list in the JSON config).
2. Copy `datasets.example.json` to a real config (for example `datasets.json`) and set `datasets`, `output`, and `clean_output` (for example `generated_audio/two_voices_cleaned.csv`).
3. Run:

```bash
cd training
python combine_datasets.py --config datasets.json
```

- **Input:** CSVs and audio paths defined in the config (see `datasets.example.json`).
- **Output:** Combined then cleaned CSV; default example name `generated_audio/two_voices_cleaned.csv` if you set `clean_output` that way.
- **Note:** Phoneme cleaning includes Hebrew-specific rules; non-Hebrew rows can be phonemized via espeak when configured.

---

## 📊 4. Compute Latent Statistics

After the autoencoder is trained and weights are in place, compute the mean and standard deviation of **compressed** latents. Flow matching and the duration predictor expect normalized latents.

```bash
cd training
python compute_latent_stats.py --tts-json configs/tts.json
```

- **Input:** AE checkpoint from `ae_ckpt_path` in `configs/tts.json` (fallback in script: `checkpoints/ae/ae_latest_newer.pt`), and a metadata CSV the script can find (see `compute_latent_stats.py` for the candidate list — point your data at one of those paths or extend the script).
- **Output:** Latent stats `.pt` file (script default: `stats_real_data.pt`; use your own name such as `stats_voice1.pt` by editing the script or re-saving).

---

## ⏱️ 5. Train Duration Predictor

The duration predictor learns utterance length from text (and reference audio in the training loop).

```bash
cd training
python src/train_duration_predictor.py --config configs/tts.json
```

Optional flags include `--max_steps`, `--batch_size`, `--lr`, `--device`.

- **Input:** Frozen AE encoder weights and stats file paths as set inside `src/train_duration_predictor.py` (defaults: `checkpoints/ae/ae_latest.pt`, `stats_multilingual.pt`) — change them to match `checkpoints/working_weights/ae_latest_newer.pt` and `stats_voice1.pt` if that is your layout.
- **Dataset:** Metadata CSV path is set in the same script (default: `generated_audio/combined_dataset_cleaned_real_data.csv`); point it at `generated_audio/two_voices_cleaned.csv` when ready.
- **Output:** Checkpoints under `checkpoints/duration_predictor/` (see script for filenames).

---

## 🌊 6. Train Text-to-Latent (Flow Matching)

Core TTS model: text (+ reference) → audio latents via flow matching and classifier-free guidance.

**Single GPU:**

```bash
cd training
python src/train_text_to_latent.py --config configs/tts.json
```

**Multi-GPU (example: 2 GPUs):**

```bash
cd training
torchrun --nproc_per_node=2 src/train_text_to_latent.py --config configs/tts.json
```

Equivalent launcher:

```bash
cd training
python -m torch.distributed.run --nproc_per_node=2 src/train_text_to_latent.py --config configs/tts.json
```

**Finetune mode** (as implemented: `lr=5e-4`, SPFM warm-up from step `40_000`):

```bash
cd training
python src/train_text_to_latent.py --config configs/tts.json --finetune
```

```bash
cd training
torchrun --nproc_per_node=2 src/train_text_to_latent.py --config configs/tts.json --finetune
```

- **Method:** Flow matching with classifier-free guidance.
- **Input:** AE checkpoint and stats paths configured in training code / `configs/tts.json` (same caveat as duration predictor — align with `checkpoints/working_weights/` and your stats file).
- **Dataset:** Same metadata CSV convention as the DP script.
- **Output:** Checkpoints under `checkpoints/text2latent/` (see script).
- **Options:** `--finetune`, `--Ke`, `--accumulation_steps`.

---

## 🎙️ 7. Inference

```bash
python inference_tts.py
```

This entrypoint is **not** present in this repository yet; add or use your own inference script that loads the trained checkpoints, runs synthesis, compares CFG scales, and toggles the duration predictor. Intended behavior: write outputs under `debug_inference/`.

---

More detail (architecture, autoencoder stage, hyperparameter tables, config snippets, citation) lives in [`training/docs/`](docs/).
