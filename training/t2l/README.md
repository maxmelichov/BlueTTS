# Light-BlueTTS Training

This folder contains the training loop and modules for Light-BlueTTS.

## Data Format
The dataset should be a CSV file with the following expected columns (or similar, depending on your dataset loader setup):
- `audio_path`: Path to the .wav file
- `text`: Transcription text
- `speaker_id`: Speaker identifier
- `lang`: Language code
- `sample_weight` (optional): per-row draw probability, up to normalisation. When present, every epoch draws rows with replacement by these weights, `floor(len(dataset) / world_size)` per rank (one weighted draw split across DDP ranks, or all of it on one GPU); rows with weight 0 are never drawn. Without it, one GPU samples speaker-balanced and DDP samples uniformly.

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
- `train.weight_decay` (optional, default `0.01`): AdamW weight decay. RobustSpeechFlow (arXiv 2605.22083, Sec. 4.3) trains with `0.0`.
- `train.lr_milestones` (optional, default `[300000, 600000]`): steps at which the learning rate is halved. A resume uses these, not the milestones stored in the checkpoint; our current recipe uses `[350000, 700000]` with `weight_decay` `0.0`.

`config/tts_st3.json` is the 64M layout (the Supertonic-3 `ttl` shapes): vector-field ConvNeXt
`intermediate_dim` 2048, text attention `n_units` 512 over 8 heads, text-encoder dilations
`[1, 1, 2, 2, 4, 4]`, `sig_min` 1e-8, no attention dropout, the Supertonic-3 time-encoder frequency
table (`time_encoder.supertonic3_freqs`) and `layer_scale_init` 0.1 (the initial ConvNeXt gamma of the
text encoder, reference encoder and vector field; default 1e-6; the DP sets it on its sentence encoder
only, under `dp.sentence_encoder.layer_scale_init`).
It also sets `edge_fill` (in `ttl` and `dp`): inside every ConvNeXt the padded positions of a row take
that row's last valid column, so each row of a padded training batch computes exactly what it computes
alone at inference (B=1). It is a no-op on an unpadded batch.
The style keys follow Supertonic-3 as well: `speech_prompted_text_encoder.style_key_mode` `"projected"`
computes the text-encoder style keys as `tanh(W_key(style_key))` from one `style_key` tensor that is also the
vector field's `tile`, and `uncond_masker.style_key_uncond` adds `u_key`, the style-attention keys of every
row whose style is dropped (CFG both-drop, SPFM-dirty, and the CFG unconditional pass). Pass it with `--config` to train
or export that model; `config/tts.json` stays the 33M model.

## Healthy Loss Curve
A healthy Flow Matching training loss curve will start high and exponentially decay, flattening out over time. If using SPFM, you will see it drop further when fine-tuning kicks in.
