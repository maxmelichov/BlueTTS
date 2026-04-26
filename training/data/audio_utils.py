import torch
import torchaudio.functional as AF


def ensure_sr(
    wav: torch.Tensor,
    sr_in: int,
    sr_out: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Resample audio while preserving the caller's tensor rank."""
    if device is None:
        device = wav.device
    original_dim = wav.dim()
    if original_dim == 1:
        wav = wav.unsqueeze(0)
    if sr_in != sr_out:
        wav = AF.resample(
            wav,
            sr_in,
            sr_out,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method="sinc_interp_kaiser",
            beta=14.769656459379492,
        )
    if original_dim == 1:
        wav = wav.squeeze(0)
    return wav.to(device)
