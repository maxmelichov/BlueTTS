"""Train the utterance-level duration predictor (SupertonicTTS recipe)."""

from __future__ import annotations

import json
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from bluecodec import LatentEncoder
from training.data.text_vocab import VOCAB_SIZE
from training.dp.data_module import PACKET_SAMPLES, get_dp_dataloader
from training.dp.models.dp_network import DPNetwork
from training.utils import LinearMelSpectrogram, compress_latents


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def _load_dp_config(config_path: str) -> dict:
    """Load a TTS config and return its DP section plus full config."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        full = json.load(f)

    dp_cfg = full.get("dp")
    if isinstance(dp_cfg, dict) and dp_cfg:
        return {"_full": full, **dp_cfg}

    # Compatibility fallback for older configs.
    ttl_cfg = full.get("ttl", {}) if isinstance(full.get("ttl"), dict) else {}
    return {
        "_full": full,
        "latent_dim": ttl_cfg.get("latent_dim", 24),
        "chunk_compress_factor": ttl_cfg.get("chunk_compress_factor", 6),
        "normalizer": {"scale": ttl_cfg.get("normalizer", {}).get("scale", 1.0)},
        "style_encoder": {
            "style_token_layer": {"n_style": 8, "style_value_dim": 16}
        },
    }


def _load_ae_encoder(
    checkpoint_path: str,
    encoder_cfg: dict,
    device: str,
) -> LatentEncoder:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"AE checkpoint not found: {checkpoint_path}")

    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        payload = load_file(checkpoint_path)
    else:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(payload, dict) and "encoder" in payload:
        state_dict = payload["encoder"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    elif isinstance(payload, dict) and any(
        k.startswith("encoder.") for k in payload.keys()
    ):
        state_dict = {
            k.replace("encoder.", ""): v
            for k, v in payload.items()
            if k.startswith("encoder.")
        }
    else:
        state_dict = payload

    encoder = LatentEncoder(cfg=encoder_cfg).to(device)
    encoder.load_state_dict(state_dict, strict=True)
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder


def _load_stats(
    stats_path: str,
    compressed_channels: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Latent stats not found: {stats_path}")

    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    if "mean" not in stats or "std" not in stats:
        raise KeyError(f"{stats_path} must contain 'mean' and 'std'")

    mean = stats["mean"].float()
    std = stats["std"].float()
    if mean.ndim == 1:
        mean = mean.view(1, -1, 1)
    if std.ndim == 1:
        std = std.view(1, -1, 1)
    expected_shape = (1, compressed_channels, 1)
    if tuple(mean.shape) != expected_shape or tuple(std.shape) != expected_shape:
        raise ValueError(
            f"Stats shape mismatch: mean={tuple(mean.shape)}, std={tuple(std.shape)}, "
            f"expected={expected_shape}"
        )
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError(f"Non-finite values found in {stats_path}")
    if (std <= 0).any():
        raise ValueError(f"Non-positive standard deviation found in {stats_path}")

    return mean.to(device), std.to(device)


@torch.no_grad()
def _style_diagnostic(
    model: DPNetwork,
    text_features: torch.Tensor,
    style_tokens: torch.Tensor,
    predicted_duration: torch.Tensor,
    target_duration: torch.Tensor,
    speaker_ids: torch.Tensor,
) -> tuple[float, float, float]:
    """Measure whether matching reference style helps duration prediction."""
    permutation = torch.roll(
        torch.arange(style_tokens.shape[0], device=style_tokens.device), shifts=1
    )
    different_speaker = speaker_ids != speaker_ids[permutation]
    if not different_speaker.any():
        different_speaker = torch.ones_like(different_speaker, dtype=torch.bool)

    shuffled_prediction = model.predictor(
        text_features.detach(),
        style_tokens.detach()[permutation],
        return_log=False,
    )
    style_effect = (
        shuffled_prediction[different_speaker]
        - predicted_duration.detach()[different_speaker]
    ).abs().mean()
    matched_mae = (
        predicted_duration.detach()[different_speaker]
        - target_duration[different_speaker]
    ).abs().mean()
    shuffled_mae = (
        shuffled_prediction[different_speaker] - target_duration[different_speaker]
    ).abs().mean()
    style_gain = shuffled_mae - matched_mae
    different_fraction = different_speaker.float().mean()
    return (
        float(style_effect.item()),
        float(style_gain.item()),
        float(different_fraction.item()),
    )


def train_duration_predictor(
    metadata_path: str = "generated_audio/combined_dataset_cleaned_real_data.csv",
    checkpoint_dir: str = "checkpoints/duration_predictor",
    ae_checkpoint: Optional[str] = None,
    stats_path: str = "stats_multilingual.pt",
    config_path: str = "config/tts.json",
    max_steps: int = 3000,
    batch_size: int = 128,
    lr: float = 5e-4,
    loss_domain: str = "linear",
    balance_sampling: bool = False,
    max_wav_sec: float = 30.0,
    max_text_len: int = 800,
    num_workers: int = 16,
    save_every: int = 500,
    log_every: int = 20,
    style_check_every: int = 100,
    seed: int = 42,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
) -> str:
    """Train a paper-aligned DP and return the final checkpoint path."""
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if lr <= 0:
        raise ValueError(f"lr must be positive, got {lr}")
    if loss_domain not in ("linear", "log"):
        raise ValueError(
            f"loss_domain must be 'linear' or 'log', got {loss_domain!r}"
        )

    set_seed(seed)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Initializing Duration Predictor training on {device}...")

    dp_cfg = _load_dp_config(config_path)
    full_cfg = dp_cfg["_full"]
    if ae_checkpoint is None:
        ae_checkpoint = full_cfg.get("ae_ckpt_path")
    if not ae_checkpoint:
        raise ValueError(
            f"No AE checkpoint was provided and {config_path} has no "
            "'ae_ckpt_path'"
        )
    latent_dim = int(dp_cfg.get("latent_dim", 24))
    chunk_compress_factor = int(dp_cfg.get("chunk_compress_factor", 6))
    normalizer_scale = float(dp_cfg.get("normalizer", {}).get("scale", 1.0))
    compressed_channels = latent_dim * chunk_compress_factor

    style_token_cfg = dp_cfg.get("style_encoder", {}).get("style_token_layer", {})
    style_dp = int(style_token_cfg.get("n_style", 8))
    style_dim = int(style_token_cfg.get("style_value_dim", 16))

    ae_encoder_cfg = full_cfg["ae"]["encoder"]
    ae_spec_cfg = ae_encoder_cfg.get("spec_processor", {})
    sample_rate = int(ae_spec_cfg.get("sample_rate", 44100))
    hop_length = int(ae_spec_cfg.get("hop_length", 512))
    packet_samples = hop_length * chunk_compress_factor
    if packet_samples != PACKET_SAMPLES:
        raise ValueError(
            f"Config packet size {packet_samples} != collate PACKET_SAMPLES "
            f"{PACKET_SAMPLES}; update the collate constant"
        )

    print(
        f"Config: {config_path} "
        f"(version={full_cfg.get('tts_version', 'unknown')}, "
        f"split={full_cfg.get('split', 'unknown')})"
    )
    print(
        f"Parameters: sr={sample_rate}, hop={hop_length}, latent_dim={latent_dim}, "
        f"compression={chunk_compress_factor}, style={style_dp}x{style_dim}"
    )
    print(f"AE checkpoint: {ae_checkpoint}")
    print(
        f"Recipe: steps={max_steps}, batch={batch_size}, lr={lr:g}, "
        f"loss=L1({loss_domain} domain), balance_sampling={balance_sampling}, "
        "reference_length=5%-95%"
    )

    mel_spec = (
        LinearMelSpectrogram(
            sample_rate=sample_rate,
            n_fft=ae_spec_cfg.get("n_fft", 2048),
            win_length=ae_spec_cfg.get(
                "win_length", ae_spec_cfg.get("n_fft", 2048)
            ),
            hop_length=hop_length,
            n_mels=ae_spec_cfg.get("n_mels", 228),
        )
        .to(device)
        .eval()
    )
    ae_encoder = _load_ae_encoder(ae_checkpoint, ae_encoder_cfg, device)
    mean, std = _load_stats(stats_path, compressed_channels, device)

    model = DPNetwork(
        vocab_size=VOCAB_SIZE,
        style_dp=style_dp,
        style_dim=style_dim,
        sentence_encoder_cfg=dp_cfg.get("sentence_encoder", {}),
        style_encoder_cfg=dp_cfg.get("style_encoder", {}),
        predictor_cfg=dp_cfg.get("predictor", {}),
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"DP parameters: {parameter_count:,}")
    optimizer = AdamW(model.parameters(), lr=lr)

    dataloader = get_dp_dataloader(
        metadata_path=metadata_path,
        batch_size=batch_size,
        sample_rate=sample_rate,
        hop_length=hop_length,
        max_wav_sec=max_wav_sec,
        max_text_len=max_text_len,
        num_workers=num_workers,
        balance_sampling=balance_sampling,
        seed=seed,
        device=device,
        packet_samples=packet_samples,
    )
    data_iterator = iter(dataloader)

    model.train()
    running_loss = 0.0
    progress = tqdm(total=max_steps, desc="DP training", dynamic_ncols=True)

    for global_step in range(1, max_steps + 1):
        try:
            batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(dataloader)
            batch = next(data_iterator)

        (
            ref_wavs,
            ref_crop_lengths,
            text_ids,
            text_masks,
            wav_lengths,
            speaker_ids,
        ) = batch
        ref_wavs = ref_wavs.to(device, non_blocking=True)
        ref_crop_lengths = ref_crop_lengths.to(device, non_blocking=True)
        text_ids = text_ids.to(device, non_blocking=True)
        text_masks = text_masks.to(device, non_blocking=True)
        wav_lengths = wav_lengths.to(device, non_blocking=True)
        speaker_ids = speaker_ids.to(device, non_blocking=True)

        # Frozen audio path: only the reference crop is encoded.
        with torch.no_grad():
            spectrogram = mel_spec(ref_wavs)
            z_ref = ae_encoder(spectrogram)
            z_ref = compress_latents(z_ref, factor=chunk_compress_factor)
            z_ref = ((z_ref - mean) / std) * normalizer_scale
            ref_mask = (
                torch.arange(z_ref.shape[-1], device=device).unsqueeze(0)
                < ref_crop_lengths.unsqueeze(1)
            ).unsqueeze(1).float()
            z_ref = z_ref * ref_mask
            target_duration = wav_lengths.float() / float(sample_rate)

        optimizer.zero_grad(set_to_none=True)

        # Expanded for style diagnostic access to tokens.
        text_features = model.sentence_encoder(text_ids, mask=text_masks)
        style_tokens = model.ref_encoder(z_ref, mask=ref_mask)
        predicted_duration = model.predictor(
            text_features, style_tokens, return_log=False
        )

        if loss_domain == "log":
            predicted_log = model.predictor(
                text_features, style_tokens, return_log=True
            )
            loss = F.l1_loss(
                predicted_log, torch.log(target_duration.clamp_min(0.05))
            )
        else:
            # Paper objective: L1 directly in seconds.
            loss = F.l1_loss(predicted_duration, target_duration)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite DP loss at step {global_step}: {loss.item()}"
            )
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        running_loss += loss_value
        progress.update(1)
        progress.set_postfix(
            loss=f"{loss_value:.3f}",
            mean=f"{running_loss / global_step:.3f}",
        )

        if global_step == 1 or global_step % log_every == 0:
            with torch.no_grad():
                absolute_error = (predicted_duration - target_duration).abs()
                relative_error = absolute_error / target_duration.clamp_min(1e-3)
                crop_seconds = (
                    ref_crop_lengths.float() * packet_samples / sample_rate
                )
            print(
                f"\n[Step {global_step}] "
                f"L1={loss_value:.4f}s  "
                f"MAE={absolute_error.mean().item():.4f}s  "
                f"RelErr={100.0 * relative_error.mean().item():.2f}%"
            )
            print("  Pred:  ", predicted_duration[:4].detach().cpu().numpy())
            print("  Target:", target_duration[:4].cpu().numpy())
            print(
                f"  Ref crop: min={crop_seconds.min().item():.2f}s "
                f"mean={crop_seconds.mean().item():.2f}s "
                f"max={crop_seconds.max().item():.2f}s"
            )

        if global_step == 1 or global_step % style_check_every == 0:
            effect, gain, different_fraction = _style_diagnostic(
                model,
                text_features,
                style_tokens,
                predicted_duration,
                target_duration,
                speaker_ids,
            )
            print(
                f"  Style check: prediction_delta={effect:.4f}s, "
                f"matched_MAE_gain={gain:+.4f}s, "
                f"different_speaker={100.0 * different_fraction:.1f}%"
            )
            if global_step >= 500 and effect < 0.01:
                print(
                    "  [WARN] DP is nearly insensitive to reference style; "
                    "do not promote this checkpoint without an inference A/B."
                )

        if save_every > 0 and global_step % save_every == 0:
            save_path = os.path.join(
                checkpoint_dir, f"duration_predictor_{global_step}.pt"
            )
            torch.save(model.state_dict(), save_path)
            print(f"Saved DP checkpoint to {save_path}")

    progress.close()
    final_path = os.path.join(checkpoint_dir, "duration_predictor_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Duration Predictor training complete. Saved to {final_path}")
    return final_path
