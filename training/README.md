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

*Pretrained weights for all stages (Codec, Text-to-Latent, Duration Predictor, and Latent Stats) are available at: [`notmax123/Blue`](https://huggingface.co/notmax123/Blue).*

Training scripts live under `training/src/`. Run dataset and stats commands from `training/` unless you adjust paths. Defaults inside some scripts still mention `combined_dataset_cleaned_real_data.csv`, `checkpoints/ae/`, or `stats_real_data.pt` — align filenames with your layout or edit those constants / `configs/tts.json` (`ae_ckpt_path`).

---

## 🏗️ Full Training Structure

The full training process consists of two primary parts: the standalone audio codec (Stage 1), followed by the interconnected acoustic models (Stages 2 & 3: Duration Predictor and Flow Matching).

### Stage 1: Train the Autoencoder (blue-codec) [Standalone]

Before training the TTS acoustic models, you need a trained autoencoder to compress audio into discrete/continuous latents.  
We use **blue-codec** for this. The training instructions for the autoencoder are maintained in its own repository:  
🔗 [**How to train blue-codec**](https://github.com/maxmelichov/blue-codec/blob/main/docs/training.md)

*If you are skipping Stage 1, you can use our pretrained codec weights (`blue_codec.safetensors` from `notmax123/Blue`).*

---

## 🧹 Prepare Dataset for Text-to-Latent & Duration Predictor (Stages 2 & 3)

Before training the downstream models (Duration Predictor and flow matching), combine and clean labeled datasets (audio + phonemes).

### Adding a New Language
The text vocabulary has a fixed size built to accommodate many languages without changing the model architecture. To add a new language and train a multilingual model for Stages 2/3:
1. Open [`training/data/text_vocab.py`](data/text_vocab.py) and locate the `LANG_ID` dictionary at the bottom.
2. Add your language code (e.g., `"fr"`) and increment the offset `LANG_REGION_START + X` (where `X` is the next available index, up to 139).
3. Ensure your generated training CSV data contains this exact language code in the `lang` column.
4. Continue with training exactly as normal.

### Data Preparation Steps
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

## 📊 Compute Latent Statistics (CRITICAL FOR NEW DATA)

> **⚠️ IMPORTANT:** Whenever you prepare a **new dataset**, you MUST compute the latent statistics (mean and standard deviation). Flow matching and the duration predictor strictly expect these latents to be properly normalized for your data.

After your training datasets are set up and the autoencoder weights are ready (or you downloaded `blue_codec.safetensors`), compute the stats:

```bash
cd training
python compute_latent_stats.py --tts-json configs/tts.json
```

- **Input:** AE checkpoint from `ae_ckpt_path` in `configs/tts.json` (fallback in script: `checkpoints/ae/blue_codec.safetensors`), and a metadata CSV the script can find (see `compute_latent_stats.py` for the candidate list — point your data at one of those paths or extend the script).
- **Output:** Latent stats `.pt` file (script default: `stats_real_data.pt`; use your own name such as `stats_voice1.pt` by editing the script or re-saving).

---

## ⏱️ Stage 2: Train Duration Predictor

The duration predictor learns utterance length from text (and reference audio in the training loop).

```bash
cd training
python src/train_duration_predictor.py --config configs/tts.json \
    --ae_checkpoint ../pt_weights/blue_codec.safetensors \
    --stats_path ../pt_weights/stats_multilingual.pt \
    --checkpoint_dir ../pt_weights
```

Optional flags include `--max_steps`, `--batch_size`, `--lr`, `--device`.

- **Input:** Frozen AE encoder weights and stats file paths as set inside `src/train_duration_predictor.py` (defaults: `blue_codec.safetensors`, `stats_multilingual.pt`).
- **Dataset:** Metadata CSV path is set in the same script.
- **Output:** Checkpoints under `checkpoints/duration_predictor/` (e.g. `duration_predictor_final.pt`).

---

## 🌊 Stage 3: Train Text-to-Latent (Flow Matching)

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

**Finetune mode:**
To fine-tune from our pretrained Text-to-Latent weights (`vf_estimator.safetensors`, which includes the `reference_encoder`, `text_encoder`, and `vf_estimator`), place the checkpoint in your target folder (e.g. `pt_weights/`) and specify the paths:

```bash
cd training
python src/train_text_to_latent.py --config configs/tts.json --finetune \
    --lr 5e-4 --spfm_warmup 40000 \
    --ae_checkpoint ../pt_weights/blue_codec.safetensors \
    --stats_path ../pt_weights/stats_multilingual.pt \
    --checkpoint_dir ../pt_weights
```

- **Method:** Flow matching with classifier-free guidance.
- **Input:** AE checkpoint and stats paths configured in training code (e.g., `blue_codec.safetensors`, `stats_multilingual.pt`).
- **Dataset:** Same metadata CSV convention as the DP script.
- **Output:** Checkpoints under `pt_weights/` or `checkpoints/text2latent/` (e.g. `ckpt_step_X.pt`).
- **Options:** `--finetune`, `--lr`, `--spfm_warmup`, `--Ke`, `--accumulation_steps`, `--ae_checkpoint`, `--stats_path`, `--checkpoint_dir`.

---

## 🎙️ 7. Inference

```bash
python inference_tts.py
```

This entrypoint is **not** present in this repository yet; add or use your own inference script that loads the trained checkpoints, runs synthesis, compares CFG scales, and toggles the duration predictor. Intended behavior: write outputs under `debug_inference/`.

---

More detail (architecture, autoencoder stage, hyperparameter tables, config snippets, citation) lives in [`training/docs/`](docs/).
