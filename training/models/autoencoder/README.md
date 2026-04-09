# Autoencoder

Latent audio codec used by Light-Blue TTS. The encoder compresses spectral features into a compact latent space; the decoder (vocoder) reconstructs the waveform from those latents in a causal, streaming-friendly manner.

---

## Architecture overview

```
Mel/spectrogram  [B, 1253, T]
       │
  LatentEncoder              (non-causal, ConvNeXt backbone)
       │
  latent z       [B, 24, T]
       │  chunk-compress (ccf=6)
  compressed z   [B, 144, T/6]    ← what the TTL model predicts
       │  decompress
  latent z       [B, 24, T]
       │
  LatentDecoder1D            (causal, CausalConvNeXt backbone)
       │  VocoderHead  [B, 512, T] → reshape
  waveform       [B, T × 512]     ← 512 samples per latent frame
```

At 44 100 Hz with `hop_length = 512`, each latent frame corresponds to exactly 512 audio samples (~11.6 ms). The TTL model works at 1/6 of that rate (`chunk_compress_factor = 6`), so each compressed frame covers ~69.6 ms.

---

## Files

### `modules.py` — shared building blocks

| Class | Description |
|---|---|
| `LayerNorm1d` | Channel-wise LayerNorm for `[B, C, T]` tensors (transposes before/after). |
| `ConvNeXtBlock` | Standard (non-causal) 1-D ConvNeXt block with dilation and layer-scale γ. Used in the encoder and `StyleTTS2Vocoder`. |
| `CausalConv1d` | `nn.Conv1d` with left-only padding — strictly causal, no future context. |
| `CausalDWConv1d` | Thin wrapper around `CausalConv1d` for depthwise use; exposes weight at `dwconv.net.*` to match ONNX trace paths. |
| `CausalConvNeXtBlock` | Causal variant of `ConvNeXtBlock`; uses `CausalDWConv1d` for the depthwise step. Used in the decoder. |

---

### `latent_encoder.py` — `LatentEncoder`

Non-causal encoder that maps spectral features to 24-dimensional latents.

**Forward:** `[B, idim, T] → [B, odim, T]`

**Default config** (matches `configs/tts.json → ae.encoder`):

| Key | Default | Description |
|---|---|---|
| `idim` | 1253 | Input channels (mel/spec bins) |
| `hdim` | 512 | Hidden dimension |
| `intermediate_dim` | 2048 | FFN expansion in each ConvNeXt block |
| `ksz` | 7 | Kernel size |
| `dilation_lst` | `[1]*10` | Dilation per block |
| `odim` | 24 | Latent channels |

Architecture: stem (`Conv1d + BN`) → 10 × `ConvNeXtBlock` → projection `Conv1d(1×1)` → `LayerNorm1d`.

---

### `latent_decoder.py` — `LatentDecoder1D`

Causal decoder that reconstructs waveform from latents. Accepts both raw latents `[B, 24, T]` and chunk-compressed latents `[B, 144, T/6]` via `_prepare_latents`.

**Forward:** `[B, 24|144, T] → [B, T_wav]`

**Default config** (matches `configs/tts.json → ae.decoder`):

| Key | Default | Description |
|---|---|---|
| `idim` | 24 | Latent channels |
| `hdim` | 512 | Hidden dimension |
| `intermediate_dim` | 2048 | FFN expansion |
| `ksz` | 7 | Kernel size |
| `dilation_lst` | `[1,2,4,1,2,4,1,1,1,1]` | Dilation per block |
| `chunk_compress_factor` | 6 | Compression ratio used by TTL |
| `normalizer_scale` | 1.0 | Scale applied during decompression |
| `head.idim` | 512 | Head input dim (must equal `hdim`) |
| `head.hdim` | 2048 | Head intermediate dim |
| `head.odim` | 512 | Head output channels (= samples per frame) |
| `head.ksz` | 3 | Head causal conv kernel size |

Architecture: `CausalInputProjection` → 10 × `CausalConvNeXtBlock` → `BatchNorm1d` → `VocoderHead` (causal conv → PReLU → 1×1 conv → reshape to waveform).

`load_state_dict` contains a remapping layer that handles several historical checkpoint key layouts (`input_conv.*`, `blocks.*`, `head.conv1.*`, etc.) so older checkpoints load without manual surgery.

---

### `vocoder.py` — `StyleTTS2Vocoder`

Legacy non-causal vocoder kept for compatibility with StyleTTS2-traced checkpoints. Architecture mirrors the decoder but uses standard (non-causal) `ConvNeXtBlock` and a non-causal `VocoderHead`.

`load_from_checkpoint` maps trace-style keys (`decoder/embed/net/Conv.weight`, etc.) to PyTorch module paths.

---

### `discriminators.py` — GAN discriminators

Used only during training; not needed at inference.

| Class | Description |
|---|---|
| `DiscriminatorP` | Single-period waveform discriminator (2-D conv on period-folded waveform). |
| `MultiPeriodDiscriminator` | Ensemble of `DiscriminatorP` at periods `[2, 3, 5, 7, 11]` (HiFi-GAN MPD). |
| `DiscriminatorR` | Single-resolution log-magnitude STFT discriminator. |
| `MultiResolutionDiscriminator` | Ensemble of `DiscriminatorR` at `(n_fft, hop, win)` = `(512,128,512)`, `(1024,256,1024)`, `(2048,512,2048)`. Uses spectral norm. |

Both multi-discriminators return `(real_scores, fake_scores, real_fmaps, fake_fmaps)` for feature-matching loss.

---

## Latent normalization and chunk compression

The TTL model predicts **normalized, chunk-compressed** latents. Before passing to the decoder:

1. **Denormalize:** `z = (z_compressed / normalizer_scale) * std + mean`
2. **Decompress:** reshape `[B, 24×6, T] → [B, 24, T×6]`

```python
z = z.reshape(B, latent_dim, ccf, T).permute(0, 1, 3, 2).reshape(B, latent_dim, T * ccf)
```

`mean` / `std` come from `stats.npz` (saved alongside ONNX/TRT exports). `normalizer_scale` is saved in the same file under `normalizer_scale`.
