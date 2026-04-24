"""Self-Purifying Flow Matching (SPFM) — batch mask and optional step diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SpfmStepResult:
    spfm_mask: torch.Tensor
    """[B, 1, 1] — 0 means treat sample as dirty and force uncond for FM."""
    ran_spfm: bool
    """True if the SPFM probe ran (inside the time window)."""
    dirty_count: int
    spfm_score: Optional[torch.Tensor]
    """Per-sample ``L_cond - L_uncond``; only set when ``ran_spfm``."""

    def spfm_score_mean(self) -> float:
        if self.spfm_score is None:
            return 0.0
        return float(self.spfm_score.mean().item())


def spfm_forward_mask(
    *,
    global_step: int,
    max_steps: int,
    spfm_start_override: Optional[int],
    z_1: torch.Tensor,
    h_text: torch.Tensor,
    ref_values: torch.Tensor,
    latent_mask: torch.Tensor,
    target_loss_mask: torch.Tensor,
    text_masks: torch.Tensor,
    valid_z_len: torch.Tensor,
    vf_estimator: torch.nn.Module,
    u_text: torch.Tensor,
    u_ref: torch.Tensor,
    sigma_min: float,
    device: torch.device,
    B: int,
    rank: int = 0,
) -> SpfmStepResult:
    """
    If outside the SPFM step window, returns an all-ones mask and ``ran_spfm=False``.

    Otherwise runs the t=0.5 probe, compares cond vs uncond MSE to ``v_target``,
    and sets ``spfm_mask[b]=0`` for dirty samples (conditional worse than unconditional).
    May print ``[SPFM]`` / ``[SPFM Diag]`` lines on rank 0.
    """
    spfm_start = spfm_start_override if spfm_start_override is not None else 40_000
    end_spfm = max_steps
    if not (global_step >= spfm_start and global_step <= end_spfm):
        return SpfmStepResult(
            spfm_mask=torch.ones(B, 1, 1, device=device),
            ran_spfm=False,
            dirty_count=0,
            spfm_score=None,
        )

    _, C, T = z_1.shape
    D_text = h_text.shape[1]

    with torch.no_grad():
        t_spfm = torch.full((B,), 0.5, device=device)
        t_b = t_spfm.view(B, 1, 1)

        x0 = torch.randn(B, C, T, device=device)
        x_t = (1 - (1 - sigma_min) * t_b) * x0 + t_b * z_1
        v_target_spfm = z_1 - (1 - sigma_min) * x0
        x_t_in = x_t * latent_mask

        v_cond = vf_estimator(
            noisy_latent=x_t_in,
            text_emb=h_text,
            style_ttl=ref_values,
            latent_mask=latent_mask,
            text_mask=text_masks,
            current_step=t_spfm,
        )

        u_text_spfm = u_text.expand(B, D_text, 1)
        u_ref_spfm = u_ref.expand(B, -1, -1)
        u_text_mask_spfm = torch.ones(B, 1, 1, device=device)

        v_uncond = vf_estimator(
            noisy_latent=x_t_in,
            text_emb=u_text_spfm,
            style_ttl=u_ref_spfm,
            latent_mask=latent_mask,
            text_mask=u_text_mask_spfm,
            current_step=t_spfm,
        )

        final_mask_spfm = latent_mask * target_loss_mask
        mask_ct = final_mask_spfm.expand(-1, C, -1)
        denom = (final_mask_spfm.sum(dim=(1, 2)) * C).clamp_min(1)

        err_c2 = (v_cond - v_target_spfm).pow(2)
        err_u2 = (v_uncond - v_target_spfm).pow(2)
        L_cond = (err_c2 * mask_ct).sum(dim=(1, 2)) / denom
        L_uncond = (err_u2 * mask_ct).sum(dim=(1, 2)) / denom

        is_dirty_candidate = L_cond > L_uncond
        spfm_score = L_cond - L_uncond

        spfm_mask = torch.ones(B, 1, 1, device=device)
        dirty_indices = torch.where(is_dirty_candidate)[0]
        if dirty_indices.numel() > 0:
            spfm_mask[dirty_indices] = 0.0

        if global_step % 1000 == 0:
            print(f"[SPFM] Detected Dirty: {dirty_indices.numel()}/{B}")

    dirty_bool = (spfm_mask.squeeze(-1).squeeze(-1) < 0.5)
    dirty_count = int(dirty_bool.sum().item())

    if global_step % 1000 == 0 and rank == 0:
        _log_spfm_diag(
            global_step=global_step,
            B=B,
            dirty_bool=dirty_bool,
            spfm_score=spfm_score,
            text_masks=text_masks,
            valid_z_len=valid_z_len,
        )

    return SpfmStepResult(
        spfm_mask=spfm_mask,
        ran_spfm=True,
        dirty_count=dirty_count,
        spfm_score=spfm_score,
    )


def _log_spfm_diag(
    *,
    global_step: int,
    B: int,
    dirty_bool: torch.Tensor,
    spfm_score: torch.Tensor,
    text_masks: torch.Tensor,
    valid_z_len: torch.Tensor,
) -> None:
    dirty_count = int(dirty_bool.sum().item())
    clean_bool = ~dirty_bool
    txt_len = text_masks.sum(dim=(1, 2)).float()
    lat_len = valid_z_len.float()
    avg_txt_clean = txt_len[clean_bool].mean().item() if clean_bool.any() else 0.0
    avg_txt_dirty = txt_len[dirty_bool].mean().item() if dirty_bool.any() else 0.0
    avg_lat_clean = lat_len[clean_bool].mean().item() if clean_bool.any() else 0.0
    avg_lat_dirty = lat_len[dirty_bool].mean().item() if dirty_bool.any() else 0.0
    print(
        f"\n[SPFM Diag] Step {global_step} | Dirty: {dirty_count}/{B} ({dirty_count / B:.1%}) | "
        f"Score mean: {spfm_score.mean().item():.3f} | "
        f"TxtLen clean/dirty: {avg_txt_clean:.1f}/{avg_txt_dirty:.1f} | "
        f"LatLen clean/dirty: {avg_lat_clean:.1f}/{avg_lat_dirty:.1f}"
    )
