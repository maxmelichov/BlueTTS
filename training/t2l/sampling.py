"""Flow-matching sampling, latent masks, and reference construction for T2L."""

import random
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional

from training.utils import decompress_latents


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_reference_only(z_ref_input, valid_z_ref_len, device, max_frames=72):
    """
    Left-align reference latents for inference. Crops to ``max_frames`` (default 72 ≈ 5s
    at ~14.35 Hz) to match training. Returns ``(z_ref_left, ref_mask_left)``.
    """
    B, C, T_ref = z_ref_input.shape
    if max_frames is not None and T_ref > max_frames:
        z_ref_input = z_ref_input[:, :, :max_frames]
        T_ref = max_frames
    arange_T = torch.arange(T_ref, device=device).unsqueeze(0)
    valid_len = valid_z_ref_len.clamp(min=0, max=T_ref).unsqueeze(1)
    ref_mask_left = (arange_T < valid_len).unsqueeze(1).float()
    z_ref_left = z_ref_input * ref_mask_left
    return z_ref_left, ref_mask_left


def build_reference_from_latents(
    z_1, valid_z_len, z_ref_input, valid_z_ref_len, is_self_ref, device, chunk_compress_factor=6
):
    """
    Sample a reference segment from ``z_ref_input`` (or self-ref from ``z_1``),
    with target loss mask ``m`` (paper) zeroing the reference crop for self-ref.

    Returns:
        z_ref_left, ref_mask_left, train_T_lat (``valid_z_len`` copy), target_loss_mask
    """
    B, C, T = z_1.shape
    _, _, T_ref_in = z_ref_input.shape

    sr = 44100
    hop = 512
    compressed_rate = (sr / hop) / chunk_compress_factor

    min_frames = max(1, int(round(0.2 * compressed_rate)))
    max_frames = int(round(9.0 * compressed_rate))

    z_ref_left = torch.zeros(B, C, T, device=device)
    ref_mask_left = torch.zeros(B, 1, T, device=device)
    target_loss_mask = torch.ones(B, 1, T, device=device)
    train_T_lat = valid_z_len.clone()

    for i in range(B):
        sample_T = int(valid_z_len[i].item())
        ref_T = int(valid_z_ref_len[i].item())
        ref_T = min(ref_T, T_ref_in)

        if is_self_ref[i]:
            half_len = max(1, sample_T // 2)
            upper_bound = min(max_frames, half_len)
            upper_bound = max(1, upper_bound)

            if upper_bound < min_frames:
                 length = int(torch.randint(1, upper_bound + 1, (1,), device=device).item())
            else:
                 length = int(torch.randint(min_frames, upper_bound + 1, (1,), device=device).item())
            length = min(length, sample_T)
            if length < 1: length = 1
            max_start = max(0, sample_T - length)
            start = int(torch.randint(0, max_start + 1, (1,), device=device).item())

            mask_start = start
            mask_end = min(start + length, sample_T)
            target_loss_mask[i, :, mask_start:mask_end] = 0.0
            copy_len = min(length, T)
            z_ref_left[i, :, :copy_len] = z_1[i, :, mask_start:mask_start + copy_len]
            ref_mask_left[i, :, :copy_len] = 1.0

        else:
            half_ref = max(1, ref_T // 2)
            upper_bound = min(max_frames, half_ref)
            upper_bound = max(1, upper_bound)
            if upper_bound < min_frames:
                 length = int(torch.randint(1, upper_bound + 1, (1,), device=device).item())
            else:
                 length = int(torch.randint(min_frames, upper_bound + 1, (1,), device=device).item())
            length = min(length, ref_T)
            if length < 1: length = 1
            max_start = max(0, ref_T - length)
            start = int(torch.randint(0, max_start + 1, (1,), device=device).item())
            copy_len = min(length, T)
            z_ref_left[i, :, :copy_len] = z_ref_input[i, :, start:start+copy_len]
            ref_mask_left[i, :, :copy_len] = 1.0

    return z_ref_left, ref_mask_left, train_T_lat, target_loss_mask


def length_to_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    max_len = max_len or int(lengths.max().item())
    ids = torch.arange(0, max_len, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1)).float()
    return mask.unsqueeze(1)


def get_latent_mask(
    wav_lengths: torch.Tensor, base_chunk_size: int, chunk_compress_factor: int
) -> torch.Tensor:
    latent_size = base_chunk_size * chunk_compress_factor
    latent_lengths = (wav_lengths + latent_size - 1) // latent_size
    latent_mask = length_to_mask(latent_lengths)
    return latent_mask


def sample_noisy_latent(
    duration: torch.Tensor, sample_rate: int, base_chunk_size: int,
    chunk_compress_factor: int, latent_dim: int, device: str | torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz = len(duration)
    wav_len_max = duration.max() * sample_rate
    wav_lengths = (duration * sample_rate).long()
    chunk_size = base_chunk_size * chunk_compress_factor
    latent_len = int((wav_len_max + chunk_size - 1) // chunk_size)
    latent_channels = latent_dim * chunk_compress_factor

    noisy_latent = torch.randn(bsz, latent_channels, latent_len, device=device)
    latent_mask = get_latent_mask(wav_lengths, base_chunk_size, chunk_compress_factor)

    if latent_mask.shape[2] < latent_len:
        latent_mask = F.pad(latent_mask, (0, latent_len - latent_mask.shape[2]))
    elif latent_mask.shape[2] > latent_len:
        latent_mask = latent_mask[:, :, :latent_len]

    noisy_latent = noisy_latent * latent_mask
    return noisy_latent, latent_mask


@torch.no_grad()
def sample_audio(
    vf_estimator,
    text_encoder,
    reference_encoder,
    ae_decoder,
    text_ids,
    text_mask,
    z_ref,
    ref_enc_mask,
    mean,
    std,
    duration_predictor=None,
    steps=32,
    device: str | torch.device = 'cuda',
    debug_label=None,
    speed=1.0,
    style_ttl=None,
    style_dp=None,
    uncond_params=None,
    cfg_scale=1.0,
    # Config-derived params (from ttl section of tts.json)
    latent_dim=24,
    chunk_compress_factor=6,
    normalizer_scale=1.0,
):
    """
    Plain flow-matching sampling:

      1. Encode reference -> style values h_ref
      2. Encode text (+ style) -> h_text
      3. Choose target length T (from duration predictor or ref length)
      4. Sample x_0 ~ N(0, I)
      5. Integrate dx/dt = v_theta(x_t, cond, t) from t=0..1
      6. Decode latents to waveform

    VF estimator uses its own internal self.tile (ONNX /Tile_output_0) for timing (not passed explicitly).
    """
    if debug_label:
        print(f"[{debug_label}] Starting sampling...")

    B = text_ids.shape[0]
    C = latent_dim * chunk_compress_factor  # e.g. 24 * 6 = 144

    # -------------------------
    # 1. Encode style / reference
    # -------------------------
    # ref_values: [B, 50, 256] (speaker-specific style values)
    if style_ttl is not None:
        ref_values = style_ttl
    else:
        ref_values = reference_encoder(z_ref, mask=ref_enc_mask)

    # -------------------------
    # 2. Duration / target length
    # -------------------------
    if duration_predictor is not None:
        dur_pred = duration_predictor(
            text_ids,
            z_ref=z_ref,
            text_mask=text_mask,
            ref_mask=ref_enc_mask,
            style_dp=style_dp,
            return_log=True,
        )

        duration = torch.exp(dur_pred) / speed

        sample_rate = 44100
        base_chunk_size = 512

        xt, latent_mask = sample_noisy_latent(
            duration, sample_rate, base_chunk_size, chunk_compress_factor, latent_dim, device
        )

        if debug_label:
             print(f"[{debug_label}] DP latent_lengths max: {latent_mask.shape[2]}")

        T = latent_mask.shape[2]
    else:
        dur_pred = None
        if z_ref is not None:
            T = z_ref.shape[2]
        else:
            T = 200
        latent_mask = torch.ones(B, 1, T, device=device)
        xt = torch.randn(B, C, T, device=device) * latent_mask

    print("[DBG] text_len =", text_mask.sum(dim=(1,2)).detach().cpu().tolist())
    if z_ref is not None:
        print("[DBG] z_ref_T  =", z_ref.shape[2])
    if duration_predictor is None:
        print("[DBG] duration_predictor is None")
    else:
        print("[DBG] duration_predictor is set")
        print("[DBG] dur_pred  =", dur_pred.detach().cpu().numpy() if dur_pred is not None else None)
        print("[DBG] latent_lengths    =", latent_mask.sum(dim=2).squeeze(1).detach().cpu().numpy())

    print("[DBG] T chosen =", T)
    print("[DBG] latent_mask sum =", latent_mask.sum().item())

    # -------------------------
    # 3. Encode text conditioned on style
    # -------------------------
    # TextEncoder returns:
    #   h_text:    [B, 256, T_txt]
    #   style_key: [B, 50, 256] (baked-in constant) - accessed via property
    h_text = text_encoder(
        text_ids,
        ref_values,
        text_mask=text_mask,
    )

    T_txt = h_text.shape[2]

    # -------------------------
    # 4. Sampler init
    # -------------------------
    x = xt
    dt = 1.0 / steps

    # -------------------------
    # 5. Euler integration
    # -------------------------
    for i in range(steps):
        t_val = i / steps
        t = torch.full((B,), t_val, device=device)

        # Conditional velocity
        x_in = x * latent_mask # Zero out padding before forward for hygiene
        v_cond = vf_estimator(
            noisy_latent=x_in,
            text_emb=h_text,      # [B, 256, T_txt]
            style_ttl=ref_values, # [B, 50, 256] (values)
            latent_mask=latent_mask,
            text_mask=text_mask,
            current_step=t,
        )

        if cfg_scale > 1.0 and uncond_params is not None:
            # Unconditional velocity
            if hasattr(uncond_params, 'module'):
                u_text = uncond_params.module.u_text.expand(B, -1, T_txt)
                u_ref = uncond_params.module.u_ref.expand(B, -1, -1)
            else:
                u_text = uncond_params.u_text.expand(B, -1, T_txt)
                u_ref = uncond_params.u_ref.expand(B, -1, -1)

            v_uncond = vf_estimator(
                noisy_latent=x_in,
                text_emb=u_text,
                style_ttl=u_ref,
                latent_mask=latent_mask,
                text_mask=text_mask,
                current_step=t,
            )
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = v_cond

        # Stabilize boundaries: apply mask to velocity
        v = v * latent_mask

        x = x + v * dt

        x = x * latent_mask  # Zero out padded frames per-sample
    # -------------------------
    # 6. Decode to waveform
    # -------------------------
    # Un-normalize: reverse z_1 = ((z - mean) / std) * normalizer_scale
    if normalizer_scale != 1.0 and normalizer_scale != 0.0:
        z_pred = (x / normalizer_scale) * std + mean
    else:
        z_pred = x * std + mean

    print("[DBG] x shape pre-decode:", x.shape)
    print("[DBG] z_pred shape pre-decompress:", z_pred.shape)

    z_pred = decompress_latents(
        z_pred,
        factor=chunk_compress_factor,
        target_channels=latent_dim
    )                                             # [B, latent_dim, T_dec]

    print("[DBG] z_pred shape post-decompress:", z_pred.shape)

    wav_pred = ae_decoder(z_pred)                 # [B, 1, T_wav]

    print("[DBG] wav_pred shape:", wav_pred.shape)

    # 7. Enforce exact length contract
    # Contract: 1 latent frame = hop_length * chunk_compress_factor samples
    frame_len = 512 * chunk_compress_factor
    # Truncate to exact expected frames
    wav_pred = wav_pred[..., frame_len:-frame_len]

    return wav_pred
