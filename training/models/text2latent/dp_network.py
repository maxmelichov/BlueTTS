import torch
import torch.nn as nn

from .duration_predictor import TTSDurationModel


class DPNetwork(TTSDurationModel):
    """Inheritance keeps state_dict keys (`sentence_encoder.*` / `predictor.*`) aligned with ONNX/checkpoints."""

    def __init__(
        self,
        vocab_size: int = 37,
        latent_channels: int = 144,
        style_dp: int = 8,
        style_dim: int = 16,
        sentence_encoder_cfg: dict = None,
        style_encoder_cfg: dict = None,
        predictor_cfg: dict = None,
    ):
        super().__init__(
            vocab_size=vocab_size,
            style_dp=style_dp,
            style_dim=style_dim,
            sentence_encoder_cfg=sentence_encoder_cfg,
            style_encoder_cfg=style_encoder_cfg,
            predictor_cfg=predictor_cfg,
        )

    def forward(
        self,
        text_ids: torch.Tensor,
        z_ref: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
        ref_mask: torch.Tensor | None = None,
        style_dp: torch.Tensor | None = None,
        return_log: bool = False,
    ) -> torch.Tensor:
        if text_mask is not None and text_mask.dtype != torch.float32:
            text_mask = text_mask.float()

        if ref_mask is not None and ref_mask.dtype != torch.float32:
            ref_mask = ref_mask.float()
        elif ref_mask is None and z_ref is not None:
            B, C, T_ref = z_ref.shape
            ref_mask = torch.ones(B, 1, T_ref, device=z_ref.device, dtype=torch.float32)

        return super().forward(
            text_ids,
            z_ref=z_ref,
            text_mask=text_mask,
            ref_mask=ref_mask,
            style_dp=style_dp,
            return_log=return_log,
        )
