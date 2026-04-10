# Text-to-Latent (TTL) Training

This directory contains the core generative models for Light-BlueTTS: a flow-matching model that maps phoneme sequences and speaker style into compressed audio latents, and a duration predictor.

## Installation

For training the Text-to-Latent (TTL) and Duration Predictor models, you need to install the `bluecodec` package. This is a speech autoencoder that compresses 44.1 kHz audio into a compact 24-dimensional continuous latent representation at ~86 Hz.

```bash
uv add "bluecodec @ git+https://github.com/maxmelichov/blue-codec.git"
```

*Pretrained weights for the codec: `notmax123/blue-codec` on Hugging Face.*

## Architecture Overview

The generation pipeline consists of several key networks:

1. **Reference Encoder**: Extracts a fixed-size style representation (`style_ttl`) from a reference audio latent.
2. **Text Encoder**: Converts phoneme IDs and the extracted speaker style into a context sequence.
3. **Duration Predictor**: Predicts the total latent frame count (`T_lat`) from the text and reference audio. Runs once per utterance.
4. **Vector Field Estimator (VFE)**: The core flow-matching denoiser. It predicts the vector field (velocity) from a noisy latent, conditioned on the text and style embeddings.

## How to Train

Training scripts are located in `training/src/`. Ensure your dataset and configuration (`configs/tts.json`) are properly set up before starting.

### 1. Train the Text-to-Latent (Flow Matching) Model

To train the main flow-matching model (which includes the Text Encoder, Reference Encoder, and Vector Field Estimator):

**Single GPU:**
```bash
uv run training/src/train_text_to_latent.py --config configs/tts.json
```

**Multi-GPU (Distributed Data Parallel):**
To train using multiple GPUs, use `torchrun` and specify the number of GPUs (`N`):
```bash
uv run torchrun --nproc_per_node=N training/src/train_text_to_latent.py --config configs/tts.json
```
*(Replace `N` with the number of GPUs you want to use, e.g., `4`)*

*Optional flags:*
*   `--finetune`: Use if you are fine-tuning an existing checkpoint (adjusts learning rate and SPFM warmup).
*   `--accumulation_steps M`: Adjust gradient accumulation.

### 2. Train the Duration Predictor

The duration predictor is trained separately to predict the length of the latent sequence:

```bash
uv run training/src/train_duration_predictor.py --config configs/tts.json
```
