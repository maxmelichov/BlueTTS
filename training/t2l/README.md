# Light-BlueTTS Training

This folder contains the training loop and modules for Light-BlueTTS.

## Data Format
The dataset should be a CSV file with the following expected columns (or similar, depending on your dataset loader setup):
- `audio_path`: Path to the .wav file
- `text`: Transcription text
- `speaker_id`: Speaker identifier
- `lang`: Language code

Expected sample rate for training is typically 44.1kHz.

## Example Data
Create a small file in `data/sample/train.csv`:
```csv
audio_path,text,speaker_id,lang
data/sample/wavs/1.wav,"שלום עולם",spk1,he
data/sample/wavs/2.wav,"מה שלומך",spk1,he
data/sample/wavs/3.wav,"מזג אוויר יפה היום",spk1,he
```

## Running Training
From the **main repository root**, with a venv that includes the training stack (Python deps for `training/`, e.g. PyTorch and pandas, per your `uv` setup):

```bash
uv run python -m training.t2l.cli --config config/tts.json --data generated_audio/combined_dataset_cleaned_real_data.csv --out runs/exp1
```

Adjust config and paths to your layout. The `blue-onnx` PyPI package is inference-only and does not install this CLI; use a full clone for training.

## Config knobs
The `configs/tts.json` has several settings:
- `latent_dim`, `chunk_compress_factor`: Autoencoder latent dimensions and compression factor.
- `text_encoder`, `vector_field`: Architectural dimensions and layers.
- `flow_matching.sig_min`: The minimum noise scale for Flow Matching.
- `batch_expander.n_batch_expand`: The $K_e$ factor. 

## Healthy Loss Curve
A healthy Flow Matching training loss curve will start high and exponentially decay, flattening out over time. If using SPFM, you will see it drop further when fine-tuning kicks in.
