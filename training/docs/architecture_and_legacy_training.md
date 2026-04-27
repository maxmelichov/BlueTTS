# Architecture and training reference

Single reference for **model architecture**, **latent layout**, **parameter counts**, and **per-stage training settings** (iterations, hardware from the paper, hyperparameters).

Operational commands: [`../README.md`](../README.md). Short pipeline diagram: [`overview.md`](overview.md).

---

## Training stages overview

Stages run **in order**. Wall-clock time depends on GPU throughput, dataset I/O, and implementation; the table below uses **SupertonicTTS paper** settings (Sec. 4.1–4.3) as a reproducible baseline—not a guarantee on your hardware.

| Stage | What | Typical script / step | GPUs (paper) | Iterations / steps | Key training settings |
|-------|------|------------------------|--------------|-------------------|------------------------|
| **1 — Speech autoencoder** | Mel/spectrogram → 24-D latent @ ~86 Hz; decoder reconstructs waveform | `train_autoencoder.py` (often in **bluecodec** or separate AE repo) | 2× RTX 3090 (paper); often **4× GPU DDP** in recipes | **1,500,000** | AdamW β₁=0.8, β₂=0.99, wd=0.01; **lr** 2e-4 → cosine to 1e-6; **batch** 128; crop ~**0.19 s** (~8,379 samples @ 44.1 kHz) |
| **— Latent stats** | Per-channel mean/std on **compressed** latents (required before TTL & DP) | `compute_latent_stats.py` | 1× GPU (typical) | One full metadata pass | Uses frozen AE encoder + training CSV |
| **2 — Text-to-latent (TTL)** | Flow matching: text + ref → compressed latents | `src/train_text_to_latent.py` | 2× RTX 3090 | **700,000** | AdamW; **lr** 5e-4, **halved every 300k** steps; **batch** 64; **K_e**=4; σ_min≈1e-8; **p_uncond**=0.05; ref crop **0.2–9 s**, ≤50% utterance — *`config/tts.json` may override (e.g. `n_batch_expand`, `sig_min`)* |
| **3 — Duration predictor (DP)** | Utterance-level **log-duration** | `src/train_duration_predictor.py` | 1× RTX 3090 | **3,000–6,000** | AdamW; **lr** 1e-4; **batch** 64–128; ref crop **5%–95%** of utterance; AE encoder **frozen** |

**Inference (synthesis) stack:** the **AE encoder is not used** on the text→waveform path—you only need **decoder + TTL + DP** (**~70.6M** with default `config/tts.json`; see **Verified parameter counts** below). The **AE encoder** (**~25.6M**) still matters for **training** and **offline** latent encoding; it is not counted in that synthesis total.

---

## Full pipeline

```
 RAW AUDIO (44.1 kHz)
        │
        ▼
 ┌─────────────────────────────────┐
 │   Stage 1: Speech Autoencoder   │
 │  Audio ──► Encoder ──► z (24-dim @ ~86 Hz)
 │            z ──► Decoder ──► Waveform
 └─────────────────────────────────┘
        │
        ▼
 ┌────────────────────────────────────────┐
 │   compute_latent_stats.py              │
 │   Per-channel mean / std → *.pt        │
 └────────────────────────────────────────┘
        │
        ├────────────────────────────┬──────────────────────────────┐
        ▼                            ▼                              │
 ┌─────────────────────────┐   ┌──────────────────────────┐         │
 │  Stage 2: TTL           │   │  Stage 3: Duration pred.  │         │
 │  Flow matching          │   │  L1 on log(duration)      │         │
 │  Euler NFE=32, CFG=3    │   │  3k–6k steps              │         │
 └─────────────────────────┘   └──────────────────────────┘         │
        │                                                            │
        ▼                                                            │
 AE Decoder: z → waveform                                           │
```

---

## Latent space and compression

Shared representation: **24-dimensional** continuous latent at **~86 Hz** (sample_rate / hop, e.g. 44100 / 512).

For TTL and DP, latents are **temporally compressed** by **K_c = 6**:

```
z   [B, 24, T]    @ ~86 Hz
        │  reshape: pack 6 consecutive frames
        ▼
z_c [B, 144, T/6] @ ~14 Hz   (144 = 24 × 6)
```

Example: T = 860 AE frames → T/6 ≈ 143 compressed frames.

---

## Parameter budget

| Component | ~Params | Role |
|-----------|---------|------|
| AE Encoder | ~25.56M | Mel/spec → 24-dim latent (**not** in synthesis forward pass; training + latent prep only) |
| AE Decoder | ~25.34M | 24-dim latent → waveform (**inference**) |
| Reference encoder (TTL) | ~4.80M | Ref **compressed** latent → style (**inference**) |
| Text encoder (TTL) | ~6.97M | Text → conditioned sequence (**inference**) |
| Vector field estimator | ~33.02M | Flow-matching backbone (**inference**) |
| Duration predictor | ~0.51M | Utterance duration (**inference**) |
| Uncond (CFG), TTL train only | ~0.013M | Learnable null text/ref for classifier-free guidance (usually **not** in a minimal synthesis bundle) |
| **Synthesis subtotal (decoder + TTL + DP)** | **~70.63M** | Decoder + text + ref + VF + DP — matches `config/tts.json` architecture |
| **Optional: + AE encoder** | **+~25.56M** | If waveform→latent runs in the same deployed graph |
| **Full checkpoint (enc+dec+TTL+DP)** | **~96.19M** | Encoder included; add **~13k** if Uncond params are stored |

### Verified parameter counts

Instantiate modules from your `tts.json` and count tensors (no weights required):

```bash
cd training
uv run python scripts/count_model_params.py --config ../config/tts.json
```

Example output for **`config/tts.json` (`tts_version` v1.0.0)**:

| Module | Parameters |
|--------|------------|
| AE encoder | 25,558,088 |
| AE decoder | 25,337,345 |
| Text encoder | 6,968,832 |
| Reference encoder | 4,797,696 |
| Vector field | 33,022,784 |
| Uncond (CFG) | 13,056 |
| TTL generative (text+ref+VF) | 44,789,312 |
| Duration predictor | 505,986 |
| **Synthesis (decoder + TTL gen + DP)** | **70,632,643** |
| **Encoder + decoder + TTL gen + DP** | **96,190,731** |

**Config sanity (same run):** `latent_dim=24`, `chunk_compress_factor=6` → **144** compressed channels; **44.1 kHz**, **hop 512** → latent frame rate **≈86.13 Hz**; TTL **n_style=50**; DP **8×16** style tokens.

---

## Architecture — speech autoencoder

44.1 kHz audio → concatenated **log-linear (1025-ch) + log-mel (228-ch)** spectrogram (FFT 2048, hop 512) → **24-dim** latent @ ~86 Hz.

| Part | Details |
|------|---------|
| Input | 1253-channel spectrogram |
| Encoder (~25.56M) | Conv1d stem (1253→512) + 10 ConvNeXt blocks (intermediate 2048) + proj (512→24) |
| Decoder (~25.34M) | CausalConv1d stem (24→512) + 10 causal dilated ConvNeXt + vocoder head |
| Decoder dilations | `[1, 2, 4, 1, 2, 4, 1, 1, 1, 1]` |
| Discriminators | MPD (periods 2,3,5,7,11) + MRD (FFTs 512/1024/2048) |

**Generator loss:**

```
L_G = 45 * L_recon + 1 * L_adv + 0.1 * L_fm
```

Reconstruction: multi-resolution **mel L1** on three scales: (FFT 1024, 64 mels), (FFT 2048, 128 mels), (FFT 4096, 128 mels).

---

## Architecture — text-to-latent (TTL)

Flow-matching model: **text + reference speech** → **compressed** latent `z_c` (144 ch @ ~14 Hz). Inference: **Euler**, **NFE = 32**, **CFG = 3**.

### Block diagram

```
 Text string
      │  char → token IDs (`VOCAB_SIZE` from `data/text_vocab.py`; multilingual, not a tiny 37-char set)
      ▼
 ┌──────────────────────────── TEXT ENCODER (~6.97M) ───────────────────────────┐
 │  CharEmbedding 256 → ConvNeXt×6 (k=5, d=1) → Self-Attn×4 (LARoPE, 4 heads)   │
 │       → Style cross-attn×2 → proj 256→256                                     │
 └──────────────────────────────────────────────────────────────────────────────┘
      ▼  h_text [B, 256, T_text]

 Reference audio → AE encoder (frozen) → z_ref → compress → z_ref_c [B, 144, T_ref]
      ▼
 ┌──────────────────────── REFERENCE ENCODER (~4.80M) ──────────────────────────┐
 │  Conv1d 144→256 → ConvNeXt×6 → pos emb → learnable queries [50,256]           │
 │       → cross-attn×2 → ref_values [B,50,256], ref_keys [B,50,256]             │
 └──────────────────────────────────────────────────────────────────────────────┘

 ┌────────────────────── VECTOR FIELD ESTIMATOR (~33.02M) ────────────────────────┐
 │  z_t [B,144,T] → proj_in 144→512                                              │
 │  Time t → sinusoidal emb → MLP → time injection per superblock                │
 │  Superblock ×4: dilated ConvNeXt [1,2,4,8] → time → ConvNeXt → text cross-attn │
 │       → ConvNeXt → style cross-attn                                            │
 │  Final ConvNeXt stack → proj_out 512→144  →  v̂ [B,144,T]                      │
 └──────────────────────────────────────────────────────────────────────────────┘
```

### Summary table (same as compact spec)

| Block | Details |
|-------|---------|
| Reference encoder | Conv1d (144→256) + 6 ConvNeXt (k=5) + 2 cross-attn → 50 style tokens |
| Text encoder | Char emb 256 + 6 ConvNeXt + 4 self-attn (RoPE) + 2 style cross-attn |
| VFE | proj_in 144→512 + 4 superblocks + 4 final ConvNeXt + proj_out 512→144 |
| Superblock | 4× dilated ConvNeXt (1,2,4,8) + time + 2× ConvNeXt + text CA + style CA |

### Flow matching

**Interpolation:**

```
z_t = t·z_1 + (1 - (1 - σ_min)·t)·z_0 ,  σ_min = 1e-8
target field ≈ z_1 - z_0
```

**Loss (masked L1):**

```
L_TTL = E[ || m ⊙ (v̂(z_t, z_ref, c, t) - (z_1 - (1-σ_min)·z_0)) ||₁ ]
```

`m` = mask (e.g. 0 on reference crop, 1 on target region). **p_uncond = 0.05** for classifier-free guidance.

**CFG at inference:** `v_guided = v_uncond + cfg * (v_cond - v_uncond)`; the library default is **cfg = 4** (`DEFAULT_CFG_SCALE` in `src/blue_onnx/__init__.py`; same numeric default in PyTorch and TensorRT frontends).

**Euler:** `z_0 ~ N(0,I)`; for `s = 0..NFE-1`, `t = s/NFE`, `z += (1/NFE) * v̂`.

### Attention notes

- **LARoPE (text):** positions normalized by sequence length before rotary embedding (length-aware relative bias).
- **Style cross-attn:** no RoPE; keys through `tanh` for stability.

### TTL reference crop (training)

Self-reference: random crop **0.2–9 s** of the utterance, **≤ 50%** of total length; loss mask **m** zero on reference positions.

---

## Architecture — duration predictor (~0.51M)

Utterance-level **scalar** log-duration (not per-phoneme). **DP text encoder** and **DP reference encoder** are smaller than TTL’s.

### Block diagram

```
 text_ids                          z_ref [B,144,T_ref] (norm. compressed latent)
     │                                        │
     ▼                                        ▼
 ┌─────────────────────┐            ┌──────────────────────────┐
 │ DP text encoder     │            │ DP reference encoder    │
 │ CharEmb 64, utter.  │            │ Linear 144→64, ConvNeXt×4│
 │ token, ConvNeXt×6,  │            │ learnable queries [8,16] │
 │ self-attn×2, take   │            │ cross-attn×2 → [B,128]   │
 │ token → text_emb 64 │            └──────── style_emb ───────┘
 └──────────┬──────────┘                       │
            └─────────────── concat [B,192] ─────┘
                              │
                    Linear 192→128 → PReLU → Linear 128→1
                              │
                    L1( log_pred, log(T_frames) )
```

### DP reference sampling (training)

Random segment **5%–95%** of utterance length as `z_ref`; target is total latent frame count.

### Speaker balancing

**WeightedRandomSampler** with `weight[i] ∝ 1 / count(speaker[i])` so rare speakers are not drowned out.

### Inference coupling (TTL + DP)

1. DP predicts `T_frames ≈ exp(log_pred)`.
2. TTL generates `z_c` of length `T_frames` via Euler.
3. Decompress → AE decoder → waveform.

DP sets **length**, not timbre quality.

---

## Per-stage training settings (detail)

### Stage 1 — autoencoder (paper Sec. 4.1)

| Parameter | Value |
|-----------|--------|
| Optimizer | AdamW (β₁=0.8, β₂=0.99, weight decay 0.01) |
| Learning rate | 2e-4, cosine to 1e-6 |
| Batch size | 128 |
| Crop | ~0.19 s |
| Iterations | 1,500,000 |

### Stage 2 — TTL (paper Sec. 4.2)

| Parameter | Value |
|-----------|--------|
| Optimizer | AdamW |
| Learning rate | 5e-4, halved every 300k iterations |
| Batch size | 64 |
| Batch expansion K_e | 4 |
| σ_min | 1e-8 |
| p_uncond | 0.05 |
| Reference crop | 0.2–9 s, ≤50% utterance |
| Iterations | 700,000 |

### Stage 3 — DP (paper Sec. 4.3)

| Parameter | Value |
|-----------|--------|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Batch size | 64–128 |
| Reference crop | 5%–95% utterance |
| Iterations | 3,000–6,000 |
| AE encoder | Frozen |

Default **script** flags for DP often include `--max_steps 6000`, `--batch_size 64`, `--lr 1e-4` (see `train_duration_predictor.py`).

---

## Key `configs/tts.json` knobs (TTL)

| Key | Typical | Effect |
|-----|---------|--------|
| `ttl.vector_field.proj_in.odim` | 512 | VFE width (quadratic cost) |
| `ttl.vector_field.main_blocks.n_blocks` | 4 | Superblock count |
| `ttl.vector_field.main_blocks.convnext_0.dilation_lst` | [1,2,4,8] | Receptive field |
| `ttl.text_encoder.convnext.num_layers` | 6 | Text ConvNeXt depth |
| `ttl.text_encoder.attn_encoder.n_layers` | 4 | Self-attention depth |
| `ttl.chunk_compress_factor` | 6 | Temporal compression K_c |

---

## Reducing model size

Knobs in `configs/tts.json`.

### Autoencoder (~51M → smaller)

| Change | ~Param delta |
|--------|----------------|
| `encoder.idim` 1253 → 228 (mel-only) | −3.7M |
| `encoder.hdim` 512 → 256 | −10M |
| `encoder.intermediate_dim` 2048 → 1024 | −10.5M |
| `encoder`/`decoder` layers 10 → 6 | −8.4M each |

Mel-only requires matching feature extraction (e.g. `LinearMelSpectrogram` / `idim`).

### TTL (~45M → smaller)

| Config | Smaller choice | ~Effect |
|--------|----------------|---------|
| `vector_field.proj_in.odim` | 256 | ~−24M |
| `main_blocks.n_blocks` | 2 | ~−14M |
| `dilation_lst` | [1,2] | ~−5M per superblock |
| `text_encoder.convnext.num_layers` | 4 | ~−1.1M |
| `text_encoder.attn_encoder.n_layers` | 2 | ~−1.6M |

Practical small TTL (~10M): `proj_in.odim=256`, `n_blocks=2`, `convnext.num_layers=4`.

---

## Directory layout (this repo)

```
training/
├── src/
│   ├── train_text_to_latent.py
│   └── train_duration_predictor.py
├── models/text2latent/
│   ├── text_encoder.py
│   ├── reference_encoder.py
│   ├── vf_estimator.py
│   ├── duration_predictor.py
│   └── dp_network.py
├── combine_datasets.py
├── compute_latent_stats.py
└── configs/tts.json
```

Autoencoder training may live in **bluecodec** (see root README / HF weights).

---

## Configuration snippet

```json
{
  "ae": {
    "encoder": { "ksz": 7, "hdim": 512, "intermediate_dim": 2048, "odim": 24 },
    "decoder": { "ksz": 7, "hdim": 512, "dilation_lst": [1,2,4,1,2,4,1,1,1,1] },
    "data": { "sample_rate": 44100, "segment_size": 8379, "batch_size": 128 }
  },
  "ttl": {
    "latent_dim": 24,
    "chunk_compress_factor": 6,
    "batch_expander": { "n_batch_expand": 4 },
    "flow_matching": { "sig_min": 1e-8 }
  },
  "dp": {
    "style_encoder": { "style_token_layer": { "n_style": 8, "style_value_dim": 16 } }
  }
}
```

---

## Dataset scale (reference)

Large Hebrew runs may use on the order of **~5.9M** utterances / **~10k** hours; CSV with paths, phonemes/text, `speaker_id`. Exact paths are project-specific.

---

## Stage 1 commands (when AE script is in-tree)

```bash
cd training
torchrun --nproc_per_node=4 src/train_autoencoder.py --arch_config configs/tts.json
```

Resume:

```bash
torchrun --nproc_per_node=4 src/train_autoencoder.py --resume checkpoints/ae/ae_latest.pt
```

---

## Reference

```bibtex
@article{kim2025supertonictts,
  title={SupertonicTTS: Towards Highly Efficient and Streamlined Text-to-Speech System},
  author={Kim, Hyeongju and Yang, Jinhyeok and Yu, Yechan and Ji, Seunghun and Morton, Jacob and Bous, Frederik and Byun, Joon and Lee, Juheon},
  journal={arXiv preprint arXiv:2503.23108},
  year={2025}
}
```
