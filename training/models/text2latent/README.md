# Text-to-Latent (TTL)

Flow-matching generative model that maps phoneme sequences and speaker style into compressed audio latents. At inference the pipeline is:

```
text_ids  [B, T]           style JSON / z_ref
     │                           │
TextEncoder               ReferenceEncoder
     │                           │
text_emb [B, 256, T]    style_ttl [B, 50, 256]
     │                           │
     └──────── VectorFieldEstimator ──────────┐
                       │                      │
               (flow matching loop, N steps)  │
                       │                      │
              denoised_latent [B, 144, T_lat]◄┘
                       │
              denormalize + decompress
                       │
             LatentDecoder1D  →  waveform
```

The duration predictor (`DPNetwork`) runs separately before the VFE loop to determine `T_lat`.

---

## Files

### `text_encoder.py` — `TextEncoder` and shared building blocks

**`TextEncoder`** converts phoneme IDs and speaker style into a context sequence for the VFE.

Forward: `(text_ids [B,T], style_ttl [B,50,256], text_mask [B,1,T]) → text_emb [B,256,T]`

Pipeline:
1. **Embedding** — `TextEmbedderWrapper` (vocab size 384, matching `_vocab.py`). The language token at position 0 is broadcast-added to every phoneme position so all tokens carry language identity regardless of sequence length.
2. **ConvNeXt** — 6 × `ConvNeXtBlock` (dim 256, expansion 4, kernel 5, edge padding).
3. **Self-attention** — `AttnEncoder`: 4 × relative multi-head self-attention with windowed relative position bias (window=4) + feed-forward. Post-norm residual.
4. **ConvNeXt residual** — output of step 2 added back before style injection.
5. **Style cross-attention** — `StyleAttention`: 2 × `StyleAttentionLayer` that attends from text queries to `style_ttl` values using baked-in learnable style keys. Tanh on keys, scale = `1/√dim`.

**Shared building blocks** (also used by `reference_encoder.py` and `duration_predictor.py`):

| Class | Description |
|---|---|
| `LayerNorm` | Channel-wise LayerNorm for `[B,C,L]` — transposes to `[B,L,C]`, norms, transposes back. |
| `ConvNeXtBlock` | Explicit edge-padding (`replicate`) before the depthwise conv. Matches ONNX `Pad(mode='edge')` exactly. |
| `ConvNeXtWrapper` | `nn.ModuleList` wrapper that exposes keys as `convnext.convnext.N.*`. |
| `RelativeMultiHeadAttention` | Windowed relative self-attention (window=4). Relative position bias for both keys and values. |
| `AttnEncoder` | Stack of `RelativeMultiHeadAttention + FeedForward` with pre-add + post-norm. |
| `StyleAttentionLayer` | Single cross-attention layer with split-stack `[H,B,T,D]` layout matching the ONNX trace. |
| `StyleAttention` | Two `StyleAttentionLayer` stacked with a shared residual from the input — layer 2 queries come from layer 1 output but residual connects back to the original input. |
| `TextEmbedderWrapper` | `nn.Embedding` under `char_embedder` key to match checkpoint paths. |
| `LinearWrapped` / `StyleNorm` | Structural wrappers for ONNX key compatibility. |

---

### `reference_encoder.py` — `ReferenceEncoder`

Extracts a fixed-size style representation from a reference audio latent.

Forward: `(z_ref [B,144,T], mask [B,1,T]) → ref_values [B,50,256]`

Pipeline:
1. **Input projection** — `Conv1d(144 → 256, 1×1)`.
2. **ConvNeXt** — 6 × `ConvNeXtWrapper` blocks (from `text_encoder.py`) with mask.
3. **Positional embedding** — `SinusoidalPositionalEmbedding` added to the key/value sequence `[B,T,256]`.
4. **Cross-attention** — 2 × cross-attention layers where **queries are 50 learnable tokens** (`ref_keys`) and keys/values are the encoded audio. Each layer: pre-norm on Q and KV, `nn.MultiheadAttention`, residual, FFN.

Output `ref_values [B,50,256]` is `style_ttl` in the VFE and text encoder. The static `ref_keys` parameter is returned as a separate output for legacy callers but is not used in the current TTL forward path (the VFE bakes in its own `style_key`).

**Config defaults** (`configs/tts.json → ttl.reference_encoder`):

| Key | Value | Description |
|---|---|---|
| `in_channels` | 144 | Compressed latent channels |
| `d_model` | 256 | Style token dimension |
| `hidden_dim` | 1024 | ConvNeXt intermediate dim (ratio = 1024/256 = 4) |
| `num_blocks` | 6 | ConvNeXt layers |
| `num_tokens` | 50 | Number of style tokens |
| `num_heads` | 2 | Cross-attention heads |

---

### `vf_estimator.py` — `VectorFieldEstimator`

Flow-matching denoiser. Predicts the vector field (velocity) from a noisy latent conditioned on text and style.

Forward:
```
(noisy_latent [B,144,T], text_emb [B,256,T_text], style_ttl [B,50,256],
 latent_mask [B,1,T], text_mask [B,1,T_text], current_step [B], total_step [B])
 → denoised_latent [B,144,T]   (inference, total_step provided)
 → vector_field   [B,144,T]    (training, total_step=None)
```

**Superblock structure** (4 × repeated):

| Step | Module | Description |
|---|---|---|
| 0 | `ConvNeXtStack([1,2,4,8])` | Dilated ConvNeXt — grows receptive field |
| 1 | `TimeCondBlock` | Additive time shift: `x + Linear(t_emb)` |
| 2 | `ConvNeXtStack([1])` | Single ConvNeXt |
| 3 | `CrossAttentionBlock(use_rope=True)` | Text cross-attention with **LARoPE** |
| 4 | `ConvNeXtStack([1])` | Single ConvNeXt |
| 5 | `CrossAttentionBlock(use_rope=False)` | Style cross-attention (tanh keys, no RoPE) |

After the 4 superblocks: `ConvNeXtStack([1,1,1,1])` → `proj_out`.

**Time conditioning** (`TimeEncoder`): sinusoidal embedding (dim 64, scale 1000) → MLP (64 → 256 → 64, Mish activation). At inference, `t_norm = current_step / total_step ∈ [0,1]`.

**Length-Aware RoPE (LARoPE)** — used in text cross-attention (step 3). Positions are normalized by sequence length before computing frequencies: `pos_normalized = pos / len`. This induces a diagonal attention bias that scales with the ratio `T_latent / T_text`, helping alignment regardless of absolute length. Scaling factor `γ = 10` (`configs/tts.json → ttl.rotary_scale`).

**Style cross-attention** (step 5) uses baked-in learnable keys (`style_key [1,50,256]`, expanded to batch) and applies tanh to keys before scoring. No positional encoding.

**Attention scale**: all attention blocks use `√(attn_dim) = √256 = 16.0`, not `√head_dim`.

**ONNX vs training dual mode**: when `total_step` is provided the model applies one Euler step internally and returns `denoised_latent`; when `total_step` is `None` (training) it returns the raw vector field prediction.

**Config defaults** (`configs/tts.json → ttl.vector_field`):

| Key | Value | Description |
|---|---|---|
| `in_channels` | 144 | Latent channels (= latent_dim × ccf) |
| `hidden_channels` | 512 | Internal hidden dim |
| `text_dim` | 256 | Text embedding dim |
| `style_dim` | 256 | Style token dim |
| `num_style_tokens` | 50 | Style sequence length |
| `num_superblocks` | 4 | Number of repeated superblocks |
| `time_embed_dim` | 64 | Sinusoidal time embedding dim |
| `rope_gamma` | 10.0 | LARoPE frequency scale |

---

### `duration_predictor.py` — `TTSDurationModel`

Utterance-level duration predictor. Predicts total latent frame count from text and reference audio. Runs once per utterance before the VFE loop.

Forward: `(text_ids, z_ref, text_mask, ref_mask) → duration [B]`  (or `log(duration)` with `return_log=True`)

Three sub-modules:

**`DPTextEncoder`** — compact text encoder producing a single utterance embedding.
- Embed text IDs (dim 64) → prepend learnable utterance token → 6 × `ConvNeXtWrapper` → residual + 2 × `AttnEncoder` → take position-0 token → `proj_out [B,64]`.

**`DPReferenceEncoder`** — compact reference encoder.
- `Conv1d(144→64)` → 4 × `ConvNeXtBlock` → 2 × cross-attention with 8 learnable queries (dim 16) → flatten to `[B, 128]` (= 8 × 16).

**`DurationEstimator`** — two-layer MLP.
- `cat([text_emb, style_emb])` → `Linear(192→128)` → PReLU → `Linear(128→1)` → `exp()`.

---

### `dp_network.py` — `DPNetwork`

Thin backward-compatible subclass of `TTSDurationModel`. Keeps the same checkpoint key layout (`sentence_encoder.*`, `ref_encoder.*`, `predictor.*`) and adds float32 mask coercion and automatic `ref_mask` creation when `z_ref` is provided without a mask.

Use this class when loading checkpoints — it is the exported ONNX target.

---

## Data flow summary

```
text_ids [B, T]  +  text_mask [B, 1, T]
    │
    ├── DPNetwork ──────────────────────────────────────────── duration [B]
    │       └── T_lat = round(duration / speed)
    │
    └── TextEncoder(text_ids, style_ttl, text_mask) ──────── text_emb [B, 256, T]


z_ref [B, 144, T_ref]
    │
    └── ReferenceEncoder(z_ref, mask) ──────────────────── style_ttl [B, 50, 256]


x_0 ~ N(0, I)  [B, 144, T_lat]
    │
    └── for t in range(N_steps):
            VectorFieldEstimator(x_t, text_emb, style_ttl, ..., t, N)
            → x_{t+1} = x_t + (1/N) * v_t      # (training: x_t + reciprocal * diff_out)
    │
    x_N [B, 144, T_lat]  ← denoised latent
```

All masks follow the convention `[B, 1, T]` with `1 = valid, 0 = padding`.
