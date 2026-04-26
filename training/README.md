# Training — Blue

Install from the `training/` directory:

```bash
cd training
uv sync
# GPU — pick one (do not combine):
uv sync --extra cu128   # PyTorch + CUDA 12.8 (stable)
uv sync --extra cu132   # PyTorch + CUDA 13.2 (nightly wheels; experimental)
```

`bluecodec` is already a Git dependency in `pyproject.toml` and is installed with `uv sync`.

*Pretrained weights for all stages (Codec, Text-to-Latent, Duration Predictor, and Latent Stats) are available at: [`notmax123/blue`](https://huggingface.co/notmax123/blue).*

You can download them directly into the `pt_weights` directory using the Hugging Face CLI:

```bash
cd training
uv run hf download notmax123/blue --repo-type model --local-dir ./pt_weights
```

Run dataset, stats, DP, and T2L commands from `training/` unless you pass absolute paths.

---

## 🏗️ Full Training Structure

The full training process consists of two primary parts: the standalone audio codec (Stage 1), followed by the interconnected acoustic models (Stages 2 & 3: Duration Predictor and Flow Matching).

### Stage 1: Train the Autoencoder (blue-codec) [Standalone]

Before training the TTS acoustic models, you need a trained autoencoder to compress audio into discrete/continuous latents.  
We use **blue-codec** for this. The training instructions for the autoencoder are maintained in its own repository:  
🔗 [**How to train blue-codec**](https://github.com/maxmelichov/blue-codec/blob/main/docs/training.md)

*If you are skipping Stage 1, you can use our pretrained codec weights (`blue_codec.safetensors` from `notmax123/blue`).*

---

## 🧹 Prepare Dataset for Text-to-Latent & Duration Predictor (Stages 2 & 3)

Before training the downstream models (Duration Predictor and flow matching), combine and clean labeled datasets (audio + phonemes).

The order is:

```text
combine_datasets.py -> compute_latent_stats.py -> DP training -> T2L training
```

Do not skip `compute_latent_stats.py` for a new dataset. DP and T2L must use the stats file computed from the same combined CSV.

### Adding a New Language
The text vocabulary has a fixed size built to accommodate many languages without changing the model architecture. To add a new language and train a multilingual model for Stages 2/3:
1. Open [`training/data/text_vocab.py`](data/text_vocab.py) and locate the `LANG_ID` dictionary at the bottom.
2. Add your language code (e.g., `"fr"`) and increment the offset `LANG_REGION_START + X` (where `X` is the next available index, up to 139).
3. Ensure your generated training CSV data contains this exact language code in the `lang` column.
4. Continue with training exactly as normal.

### Simple Dataset Commands

For a speaker/chapter tree, you can skip writing JSON:

```bash
cd training
uv run python combine_datasets.py \
    --spktree /path/to/speech_corpus_R \
    --splits train-clean-100,train-clean-360 \
    --lang en \
    --output generated_audio/spktree.csv \
    --clean-output generated_audio/spktree_cleaned.csv
```

For one CSV:

```bash
uv run python combine_datasets.py \
    --csv /path/to/metadata.csv \
    --audio-dir /path/to/wavs \
    --speaker-id 1 \
    --lang he \
    --output generated_audio/my_voice.csv \
    --clean-output generated_audio/my_voice_cleaned.csv
```

Use `--limit 32` for a smoke-test CSV. For more complicated mixes, copy `datasets.example.json` to `datasets.json` and run `uv run python combine_datasets.py --config datasets.json`.

---

## 📊 Compute Latent Statistics (CRITICAL FOR NEW DATA)

> **⚠️ IMPORTANT:** Whenever you prepare a **new dataset**, you MUST compute the latent statistics (mean and standard deviation). Flow matching and the duration predictor strictly expect these latents to be properly normalized for your data.

After your combined CSV is ready and the autoencoder weights are available, compute stats into a new file:

```bash
cd training
uv run python compute_latent_stats.py \
    --tts-json ../config/tts.json \
    --metadata generated_audio/spktree_cleaned.csv \
    --ae-ckpt ../pt_models/blue_codec.safetensors \
    --output runs/spktree/stats_multilingual.pt \
    --device cuda:0
```

- **Input:** the exact combined/cleaned CSV you will train on.
- **Output:** a `.pt` stats file. The script refuses to overwrite existing files unless you pass `--overwrite`.

---

## ⏱️ Stage 2: Train Duration Predictor

The duration predictor learns utterance length from text (and reference audio in the training loop).

```bash
cd training
uv run python -m training.dp.cli \
    --config ../config/tts.json \
    --data generated_audio/spktree_cleaned.csv \
    --ae_checkpoint ../pt_models/blue_codec.safetensors \
    --stats_path runs/spktree/stats_multilingual.pt \
    --out runs/spktree/dp
```

Optional flags include `--max_steps`, `--batch_size`, `--lr`, `--device`.

- **Input:** Frozen AE encoder weights and stats file paths as set inside `src/train_duration_predictor.py` (defaults: `blue_codec.safetensors`, `stats_multilingual.pt`).
- **Output:** Checkpoints under `checkpoints/duration_predictor/` (e.g. `duration_predictor_final.pt`).

---

## 🌊 Stage 3: Train Text-to-Latent (Flow Matching)

Core TTS model: text (+ reference) → audio latents via flow matching and classifier-free guidance.

**Single GPU:**

```bash
cd training
uv run python -m training.t2l.cli \
    --config ../config/tts.json \
    --data generated_audio/spktree_cleaned.csv \
    --ae_checkpoint ../pt_models/blue_codec.safetensors \
    --stats_path runs/spktree/stats_multilingual.pt \
    --out runs/spktree/t2l
```

**Multi-GPU (example: 2 GPUs):**

```bash
cd training
uv run torchrun --nproc_per_node=2 -m training.t2l.cli \
    --config ../config/tts.json \
    --data generated_audio/spktree_cleaned.csv \
    --ae_checkpoint ../pt_models/blue_codec.safetensors \
    --stats_path runs/spktree/stats_multilingual.pt \
    --out runs/spktree/t2l
```

**Finetune mode:**
To fine-tune from our pretrained Text-to-Latent weights (`vf_estimator.safetensors`, which includes the `reference_encoder`, `text_encoder`, and `vf_estimator`), place the checkpoint in your target folder (e.g. `pt_weights/`) and specify the paths:

```bash
cd training
uv run python src/train_text_to_latent.py --config configs/tts.json --finetune \
    --lr 5e-4 --spfm_warmup 40000 \
    --ae_checkpoint pt_weights/blue_codec.safetensors \
    --stats_path pt_weights/stats_multilingual.pt \
    --checkpoint_dir pt_weights
```

- **Method:** Flow matching with classifier-free guidance.
- **Input:** AE checkpoint and stats paths configured in training code (e.g., `blue_codec.safetensors`, `stats_multilingual.pt`).
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
