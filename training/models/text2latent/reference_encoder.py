import math
import torch
import torch.nn as nn

from models.text2latent.text_encoder import ConvNeXtWrapper

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        self.dim = dim
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        return self.pe[:, :T, :]

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
    ):
        super().__init__()
        self.d_model = d_model

        if hidden_dim % d_model != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by d_model ({d_model})")
        
        mlp_ratio = hidden_dim // d_model 

        self.input_proj = nn.Conv1d(in_channels, d_model, kernel_size=1)
        self.convnext = ConvNeXtWrapper(d_model, n_layers=num_blocks, expansion_factor=mlp_ratio, kernel_size=kernel_size, dilation_lst=dilation_lst)
        self.pos_emb = SinusoidalPositionalEmbedding(d_model)
        self.ref_keys = nn.Parameter(torch.randn(num_tokens, d_model) * 0.02)

        self.attn_layers = nn.ModuleList([
            nn.ModuleDict({
                "norm_q": nn.LayerNorm(d_model),
                "norm_kv": nn.LayerNorm(d_model),
                "attn": nn.MultiheadAttention(d_model, num_heads, batch_first=True),
                "ffn": nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model * mlp_ratio),
                    nn.GELU(),
                    nn.Linear(d_model * mlp_ratio, d_model),
                ),
            })
            for _ in range(2)
        ])

    def forward(self, z_ref: torch.Tensor, mask: torch.Tensor = None):
        B, _, _ = z_ref.shape
        x = self.input_proj(z_ref)
        x = self.convnext(x, mask=mask)
        kv_seq = x.transpose(1, 2)
        kv_seq = kv_seq + self.pos_emb(kv_seq)

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = (mask.squeeze(1) == 0)

        q = self.ref_keys.unsqueeze(0).expand(B, -1, -1)

        for layer in self.attn_layers:
            q_norm = layer["norm_q"](q)
            kv_norm = layer["norm_kv"](kv_seq)
            attn_out, _ = layer["attn"](
                query=q_norm,
                key=kv_norm,
                value=kv_norm,
                key_padding_mask=key_padding_mask
            )
            q = q + attn_out
            q = q + layer["ffn"](q)

        return q