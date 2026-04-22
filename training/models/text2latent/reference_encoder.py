import torch
import torch.nn as nn

from models.text2latent.text_encoder import ConvNeXtWrapper


class ReferenceEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 144,
        d_model: int = 256,
        hidden_dim: int = 1024,
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
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by d_model ({d_model})")
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
        B = z_ref.shape[0]
        x = self.input_proj(z_ref)
        x = self.convnext(x, mask=mask)
        kv = x.transpose(1, 2)

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = (mask.squeeze(1) == 0)

        q0 = self.ref_keys.unsqueeze(0).expand(B, -1, -1)
        q0 = self.q_proj(q0)

        q1, _ = self.attn1(query=q0, key=kv, value=kv, key_padding_mask=key_padding_mask, need_weights=False)
        q2 = q0 + q1
        out, _ = self.attn2(query=q2, key=kv, value=kv, key_padding_mask=key_padding_mask, need_weights=False)
        return self.out_proj(out)

    @staticmethod
    def remap_legacy_state_dict(state_dict: dict) -> dict:
        """Remap pre-refactor checkpoints (per-layer pre-norm + FFN) onto current layout."""
        remapped = {}
        legacy_prefix_map = {
            "attn_layers.0.attn.": "attn1.",
            "attn_layers.1.attn.": "attn2.",
        }
        drop_substrings = (".norm_q.", ".norm_kv.", ".ffn.", "pos_emb.")
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
