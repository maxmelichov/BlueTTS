# Light-BlueTTS Training

This directory contains the training modules, datasets, and execution loops for Light-BlueTTS. The architecture has been refactored into a highly modular design to separate concerns and make the codebase easily extensible.

## Directory Structure

The training codebase is split into two primary subpackages alongside shared data utilities:

*   **`t2l/` (Text-to-Latent):** 
    Handles the core generative Flow Matching model. This includes the `TextEncoder`, `ReferenceEncoder`, and the `VectorFieldEstimator`.
*   **`dp/` (Duration Predictor):** 
    Houses the `DPNetwork` used to predict segment durations (lengths) based on text and reference speech styles.
*   **`data/`:**
    Shared data utilities for audio processing (`audio_utils.py`), text phonemization and indexing (`text_vocab.py`).
*   **`utils.py` & `models/`:** 
    Shared building blocks, mel-spectrogram generation, and latent compression helpers.

### Subpackage Breakdown

Within both `t2l` and `dp`, you will find a similar standard structure:
- `models/`: PyTorch `nn.Module` definitions.
- `data_module.py`: PyTorch `Dataset` and `collate_fn` implementations.
- `trainer.py`: The core training loop, validation logic, and inference checks (e.g., SPFM handling in `t2l`).
- `builders.py`: Factory functions for instantiating the models from configuration.
- `cli.py`: Command-line interface definitions via `argparse`.
- `cfg_utils.py`: Helpers for parsing and validating the JSON config.

## Running Training

You can initiate training runs using the provided CLI entry points or via the `uv run` commands if you have them registered in `pyproject.toml`.

### 1. Text-to-Latent (Flow Matching)

To train the core Text-to-Latent model, use the `t2l` trainer:

```bash
python -m training.t2l.cli --config configs/tts.json --checkpoint_dir checkpoints/text2latent
```

**Key Arguments:**
- `--finetune`: Enables fine-tuning mode (starts with a lower learning rate, adjusts SPFM warmup).
- `--Ke`: Overrides the Context-sharing expansion factor defined in the config.
- `--resume_from`: If the `checkpoint_dir` is empty, load the initial weights from this alternative directory.

### 2. Duration Predictor

To train the Duration Predictor network:

```bash
python -m training.dp.cli --config configs/tts.json --checkpoint_dir checkpoints/duration_predictor
```

## Distributed Data Parallel (DDP)

Both the `t2l` and `dp` trainers support multi-GPU training via PyTorch DDP. You can run them using `torchrun`:

```bash
torchrun --nproc_per_node=8 -m training.t2l.cli --config configs/tts.json
```

The script automatically detects the `RANK` and `WORLD_SIZE` environment variables and wraps the model in `DistributedDataParallel` and uses a `DistributedSampler` for the dataloader.

## Configuration (`tts.json`)

The architecture and training parameters are heavily governed by the `configs/tts.json` file. 

The `t2l` config includes:
- `latent_dim` & `chunk_compress_factor`: Dictates the compression ratios of the autoencoder latents.
- `text_encoder`, `style_encoder`, `vector_field`: Architectural layer counts, dimensions, and attention head configurations.
- `uncond_masker`: Probabilities for Classifier-Free Guidance (CFG) dropping text/style.
- `batch_expander.n_batch_expand`: The $K_e$ batch expansion factor for parallel flow matching integration.

## Advanced Features

### Self-Purifying Flow Matching (SPFM)

The `t2l` trainer implements SPFM (Self-Purifying Flow Matching) to handle noisy data dynamically.
- During training, the model evaluates its conditional and unconditional velocity estimates.
- Based on the MSE, it determines "dirty" candidates dynamically and routes them through unconditional dropout paths to prevent the model from learning bad phonetic alignments.
- Diagnostics are automatically printed every 1,000 steps.

### Voice Conversion (VC) Checks

During validation steps (every 1,000 iterations), the `t2l` trainer attempts to run Inference using internal pre-configured sentences across multiple languages (Hebrew, English, German, Italian, Spanish) and logs Voice Conversion checks against `reference.wav` to ensure speaker identity cloning stays intact.