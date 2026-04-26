# Duration Predictor Training

This folder contains the training loop and modules for the Duration Predictor in Light-BlueTTS.

## Running Training
From the **main repository root** (inference package on PyPI does not include this):

```bash
uv run python -m training.dp.cli --config config/tts.json --data generated_audio/combined_dataset_cleaned_real_data.csv --out checkpoints/duration_predictor
```

Adjust config paths as needed; use a full clone with training dependencies.
