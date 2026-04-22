import torch
import torch.nn as nn

from models.text2latent.text_encoder import ConvNeXtWrapper


class ReferenceEncoder(nn.Module):
    """
    Paper-aligned Reference Encoder (Section A.2.1 + Fig. 4a of the paper,
    NANSY++ timbre-token style).

    Math flow (keeps your current dimensions; only the topology changes):

        z_ref : (B, 144, T)
            Linear (144 -> d_model)                               [input_proj]
            6 ConvNeXt blocks, kernel=5, intermediate=hidden_dim  [convnext]
            kv = transpose                                        (B, T, d_model)

        Q0   = learnable_query                                    (num_tokens, d_model)
        Q1   = CrossAttn1(Q=Q0, K=kv, V=kv)                       (first CA)
        Q2   = Q0 + Q1                                            (paper Fig.4a ⊕)
        out  = CrossAttn2(Q=Q2, K=kv, V=kv)                       (second CA)

        out = self.out_proj(out)
        return out : (B, num_tokens, d_model)

    Deliberate departures from a literal reading of the paper (kept):
      - d_model=256 instead of paper's 128
      - ConvNeXt intermediate 1024 instead of paper's 512
        (These are your prior design choices; only the architecture *math* is
         being aligned here, not the widths.)

    Deliberate departures from your previous implementation (removed):
      - Sinusoidal positional embedding on K/V. The reference encoder receives
        random crops and must output a position-invariant style. Absolute PE
        hurts zero-shot timbre transfer.
      - Per-layer pre-norm + FFN inside the cross-attention stack. The paper
        diagram shows bare Q/K/V cross-attention with a single skip connection
        between layer 1 and layer 2, and no residual / FFN on layer 2.
      - Two residuals (one per layer). Paper has exactly one ⊕, between the
        two cross-attention layers.

    Interface is unchanged: forward(z_ref, mask) -> (B, num_tokens, d_model).
    """

    def __init__(
        self,
        in_channels: int = 144,
        d_model: int = 256,      # kept wider than paper (paper: 128)
        hidden_dim: int = 1024,  # kept wider than paper (paper: 512)
        num_blocks: int = 6,
        num_tokens: int = 50,
        num_heads: int = 2,
        kernel_size: int = 5,
        dilation_lst: list = None,
        prototype_dim: int = 256,
        n_units: int = 256,
        style_value_dim: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_tokens = num_tokens

        if hidden_dim % d_model != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by d_model ({d_model})"
            )
        mlp_ratio = hidden_dim // d_model

        self.input_proj = nn.Conv1d(in_channels, d_model, kernel_size=1)

        self.convnext = ConvNeXtWrapper(
            d_model,
            n_layers=num_blocks,
            expansion_factor=mlp_ratio,
            kernel_size=kernel_size,
            dilation_lst=dilation_lst,
        )

        self.ref_keys = nn.Parameter(torch.randn(num_tokens, prototype_dim) * 0.02)
        self.q_proj = nn.Linear(prototype_dim, n_units) if prototype_dim != n_units else nn.Identity()
        self.out_proj = nn.Linear(n_units, style_value_dim) if n_units != style_value_dim else nn.Identity()

        self.attn1 = nn.MultiheadAttention(
            embed_dim=n_units, num_heads=num_heads, kdim=d_model, vdim=d_model, batch_first=True
        )
        self.attn2 = nn.MultiheadAttention(
            embed_dim=n_units, num_heads=num_heads, kdim=d_model, vdim=d_model, batch_first=True
        )

    def forward(self, z_ref: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            z_ref: (B, 144, T) compressed latents
            mask:  (B, 1, T)   1 = valid, 0 = padding (optional)
        Returns:
            ref_values: (B, num_tokens, d_model)
        """
        B = z_ref.shape[0]

        x = self.input_proj(z_ref)
        x = self.convnext(x, mask=mask)

        kv = x.transpose(1, 2)  # (B, T, d_model)

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = (mask.squeeze(1) == 0)  # True = pad

        q0 = self.ref_keys.unsqueeze(0).expand(B, -1, -1)  # (B, N, prototype_dim)
        q0 = self.q_proj(q0)  # (B, N, n_units)

        q1, _ = self.attn1(
            query=q0, key=kv, value=kv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        q2 = q0 + q1  # paper Fig. 4(a) ⊕

        out, _ = self.attn2(
            query=q2, key=kv, value=kv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        out = self.out_proj(out)
        return out

    # ------------------------------------------------------------------
    # Legacy warm-start utility.
    #
    # The previous implementation had per-layer pre-norm + FFN and stored
    # attention weights under `attn_layers.{0,1}.attn.*`. The new paper-aligned
    # layout is `attn{1,2}.*` with no norm/FFN.
    #
    # Use this to continue training from e.g. ckpt_step_761000.pt:
    #     sd = torch.load(ckpt_path, map_location="cpu")["reference_encoder"]
    #     sd = ReferenceEncoder.remap_legacy_state_dict(sd)
    #     missing, unexpected = model.load_state_dict(sd, strict=False)
    # `missing` should be empty; `unexpected` will list the dropped norm/FFN
    # tensors, which is expected.
    # ------------------------------------------------------------------
    @staticmethod
    def remap_legacy_state_dict(state_dict: dict) -> dict:
        remapped = {}
        legacy_prefix_map = {
            "attn_layers.0.attn.": "attn1.",
            "attn_layers.1.attn.": "attn2.",
        }
        drop_substrings = (
            ".norm_q.", ".norm_kv.", ".ffn.",
            "pos_emb.",
        )
        for k, v in state_dict.items():
            if any(s in k for s in drop_substrings):
                continue

            new_key = k
            for old, new in legacy_prefix_map.items():
                if new_key.startswith(old):
                    new_key = new + new_key[len(old):]
                    break
            remapped[new_key] = v
        return remapped
