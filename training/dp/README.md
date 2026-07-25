# Duration Predictor Training

This folder contains the training loop and modules for the Duration Predictor.

## Running Training
From the **main repository root** (inference package on PyPI does not include this):

```bash
uv run python -m training.dp.cli \
  --config config/tts.json \
  --data generated_audio/combined_dataset_cleaned_real_data.csv \
  --out checkpoints/duration_predictor \
  --ae_checkpoint checkpoints/ae/blue_codec.safetensors \
  --stats_path stats_multilingual.pt
```

Default recipe (paper-aligned): **3000** steps, batch **128**, lr **5e-4**, L1 in **linear** seconds, worker-side 5%–95% reference crops.

Optional: `--balance` flattens language / length skew; `--loss log` uses relative-error L1 in log seconds.
