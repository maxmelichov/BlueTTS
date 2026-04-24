# Duration Predictor Training

This folder contains the training loop and modules for the Duration Predictor in Light-BlueTTS.

## Running Training
You can start training with the `blue-train-dp` CLI provided by `pyproject.toml`:

```bash
uv run blue-train-dp --config configs/tts.json --data generated_audio/combined_dataset_cleaned_real_data.csv --out checkpoints/duration_predictor
```
