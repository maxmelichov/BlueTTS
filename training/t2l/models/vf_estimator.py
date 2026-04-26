import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearWrapper(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class LayerNormWrapper(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        return x


class ProjectionWrapper(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Mish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, scale: float = 1000.0):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.scale
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimeEncoder(nn.Module):
    def __init__(self, embed_dim: int, hdim: int = 256):
        super().__init__()
        self.sinusoidal = SinusoidalPosEmb(embed_dim, scale=1000.0)
        self.mlp = nn.Sequential(
            LinearWrapper(embed_dim, hdim),
            Mish(),
            LinearWrapper(hdim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sinusoidal(x)
        x = self.mlp(x)
        return x


class TimeCondBlock(nn.Module):
    def __init__(self, time_dim: int, channels: int):
        super().__init__()
        self.linear = LinearWrapper(time_dim, channels)
        # Zero-init so the block starts as identity.
        nn.init.zeros_(self.linear.linear.weight)
        nn.init.zeros_(self.linear.linear.bias)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        cond = self.linear(time_emb)
        cond = cond.unsqueeze(-1)
        return x + cond


class ConvNeXtBlock1D(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 5, expansion: int = 2, dropout: float = 0.0, dilation: int = 1):
        super().__init__()
        self.pad = ((kernel_size - 1) // 2) * dilation
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=0, groups=dim, dilation=dilation)
        self.norm = LayerNormWrapper(dim)
        self.pwconv1 = nn.Conv1d(dim, dim * expansion, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(dim * expansion, dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1, dim, 1) * 1e-6)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            x = x * mask
        residual = x

        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.dwconv(x)
        if mask is not None:
            x = x * mask

        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = self.dropout(x)

        x = x + residual
        if mask is not None:
            x = x * mask
        return x


class ConvNeXtStack(nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        self.convnext = nn.ModuleList([
            ConvNeXtBlock1D(channels, kernel_size=kernel_size, dilation=d, expansion=2)
            for d in dilations
        ])

    def forward(self, x, mask=None):
        for blk in self.convnext:
            x = blk(x, mask)
        return x


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    B, H, T, D = x.shape
    assert D % 2 == 0, "head_dim must be even for RoPE"

    x1 = x[..., : D // 2]
    x2 = x[..., D // 2 :]

    if cos.dim() == 2:
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    x1_rot = x1 * cos - x2 * sin
    x2_rot = x1 * sin + x2 * cos

    return torch.cat([x1_rot, x2_rot], dim=-1)


class AttentionModule(nn.Module):
    """Text path uses LARoPE; style path uses tanh on keys (no RoPE)."""

    def __init__(
        self,
        d_model: int,
        d_context: int,
        num_heads: int,
        attn_dim: int,
        use_rope: bool,
        dropout: float = 0.0,
        rope_gamma: float = 10.0,
        attn_scale: Optional[float] = None,
        rotary_base: float = 10000.0,
        use_residual: bool = True,
    ):
        super().__init__()
        assert attn_dim % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self.attn_dim = attn_dim
        self.use_rope = use_rope
        self.use_residual = use_residual
        self.rope_gamma = rope_gamma
        self.attn_scale = attn_scale if attn_scale is not None else math.sqrt(self.attn_dim)

        self.W_query = LinearWrapper(d_model, attn_dim)
        self.W_key = LinearWrapper(d_context, attn_dim)
        self.W_value = LinearWrapper(d_context, attn_dim)
        self.out_fc = LinearWrapper(attn_dim, d_model)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        if use_rope:
            inv_freq = 1.0 / (rotary_base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
            theta = (inv_freq * rope_gamma).view(1, 1, -1)
            self.register_buffer("theta", theta, persistent=True)
            self.register_buffer("increments", torch.arange(1000).view(1, 1000, 1), persistent=True)
            self.tanh = None
        else:
            self.theta = None
            self.increments = None
            self.tanh = nn.Tanh()

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_keys: Optional[torch.Tensor] = None,
        x_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, d_model, T = x.shape
        L = context.shape[1]

        x_t = x.transpose(1, 2)
        q = self.W_query(x_t)

        k_src = context_keys if context_keys is not None else context
        k = self.W_key(k_src)
        v = self.W_value(context)

        if not self.use_rope and self.tanh is not None:
            k = self.tanh(k)

        H = self.num_heads
        D = self.head_dim

        q = q.view(B, T, H, D).permute(2, 0, 1, 3)
        k = k.view(B, L, H, D).permute(2, 0, 1, 3)
        v = v.view(B, L, H, D).permute(2, 0, 1, 3)

        if self.use_rope:
            device = x.device

            if x_mask is not None:
                len_q = x_mask.sum(dim=(-2, -1)).reshape(-1, 1, 1)
            else:
                len_q = torch.tensor([T], device=device, dtype=torch.float32).reshape(1, 1, 1)

            if context_mask is not None:
                len_k = context_mask.sum(dim=(-2, -1)).reshape(-1, 1, 1)
            else:
                len_k = torch.tensor([L], device=device, dtype=torch.float32).reshape(1, 1, 1)

            if self.increments is not None and self.increments.shape[1] >= max(T, L):
                pos_q = self.increments[:, :T, :].to(device).float()
                pos_k = self.increments[:, :L, :].to(device).float()
            else:
                pos_q = torch.arange(T, device=device, dtype=torch.float32).reshape(1, -1, 1)
                pos_k = torch.arange(L, device=device, dtype=torch.float32).reshape(1, -1, 1)

            norm_pos_q = pos_q / len_q
            norm_pos_k = pos_k / len_k

            theta = self.theta if self.theta is not None else (
                (1.0 / (10000 ** (torch.arange(0, D, 2, device=device).float() / D))) * self.rope_gamma
            ).view(1, 1, -1)

            freqs_q = norm_pos_q * theta
            freqs_k = norm_pos_k * theta

            cos_q, sin_q = freqs_q.cos(), freqs_q.sin()
            cos_k, sin_k = freqs_k.cos(), freqs_k.sin()

            cos_q, sin_q = cos_q.unsqueeze(0), sin_q.unsqueeze(0)
            cos_k, sin_k = cos_k.unsqueeze(0), sin_k.unsqueeze(0)

            q = apply_rotary_pos_emb(q, cos_q, sin_q)
            k = apply_rotary_pos_emb(k, cos_k, sin_k)

        attn_logits = torch.matmul(q, k.transpose(-1, -2)) / self.attn_scale

        if context_mask is not None:
            if context_mask.dim() == 2:
                context_mask = context_mask.unsqueeze(1)
            cm = (context_mask == 0)
            attn_logits = attn_logits.masked_fill(cm.unsqueeze(0), float("-inf"))

        attn = torch.softmax(attn_logits, dim=-1)

        if x_mask is not None:
            if x_mask.dim() == 2:
                x_mask = x_mask.unsqueeze(1)
            qm = (x_mask == 0).permute(1, 0, 2).unsqueeze(-1)
            attn = attn.masked_fill(qm, 0.0)

        out = torch.matmul(attn, v)
        out = out.permute(1, 2, 0, 3).contiguous().view(B, T, self.attn_dim)
        out = self.out_fc(out)
        out = self.dropout(out)

        if x_mask is not None:
            out = out * x_mask.transpose(1, 2)

        out = out.transpose(1, 2)
        return out


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_context: int,
        num_heads: int = 8,
        attn_dim: int = 256,
        use_rope: bool = True,
        rope_gamma: float = 10.0,
        attn_scale: Optional[float] = None,
        use_residual: bool = True,
        rotary_base: float = 10000.0,
    ):
        super().__init__()
        self.use_rope = use_rope
        self.use_residual = use_residual
        attn_module = AttentionModule(
            d_model, d_context, num_heads, attn_dim, use_rope,
            rope_gamma=rope_gamma, attn_scale=attn_scale, rotary_base=rotary_base, use_residual=use_residual,
        )
        # Checkpoint naming: text (RoPE) -> 'attn'; style (no RoPE) -> 'attention'.
        if use_rope:
            self.attn = attn_module
        else:
            self.attention = attn_module
        self.norm = LayerNormWrapper(d_model)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_keys: Optional[torch.Tensor],
        x_mask: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if x_mask is not None:
            x = x * x_mask

        residual = x

        if self.use_rope:
            attn_out = self.attn(x, context, context_keys, x_mask, context_mask)
        else:
            attn_out = self.attention(x, context, context_keys, x_mask, context_mask)

        if self.use_residual:
            x = residual + attn_out
        else:
            x = attn_out

        x = self.norm(x)
        if x_mask is not None:
            x = x * x_mask
        return x


class VectorFieldEstimator(nn.Module):
    def __init__(
        self,
        in_channels: int = 144,
        hidden_channels: int = 512,
        out_channels: int = 144,
        text_dim: int = 256,
        style_dim: int = 256,
        num_style_tokens: int = 50,
        num_superblocks: int = 4,
        time_embed_dim: int = 64,
        rope_gamma: float = 10.0,
        main_blocks_cfg: dict = None,
        last_convnext_cfg: dict = None,
        text_n_heads: int = 4,
        time_hdim: int = 256,
        use_residual: bool = True,
        rotary_base: float = 10000.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.text_dim = text_dim
        self.style_dim = style_dim
        self.rope_gamma = rope_gamma

        # Shared tiled constant ([1, 50, 256]) consumed by every style-attn W_key.
        self.tile = nn.Parameter(torch.randn(1, num_style_tokens, style_dim) * 0.02)

        self.proj_in = ProjectionWrapper(in_channels, hidden_channels)
        self.time_encoder = TimeEncoder(time_embed_dim, hdim=time_hdim)

        self.main_blocks = nn.ModuleList()

        shared_attn_scale = math.sqrt(256)

        mb_cfg = main_blocks_cfg or {}
        lc_cfg = last_convnext_cfg or {}

        c0_cfg = mb_cfg.get("convnext_0", {})
        c1_cfg = mb_cfg.get("convnext_1", {})
        c2_cfg = mb_cfg.get("convnext_2", {})

        for _ in range(num_superblocks):
            self.main_blocks.append(
                ConvNeXtStack(hidden_channels, kernel_size=c0_cfg.get("ksz", 5), dilations=c0_cfg.get("dilation_lst", [1, 2, 4, 8]))
            )
            self.main_blocks.append(
                TimeCondBlock(time_dim=time_embed_dim, channels=hidden_channels)
            )
            self.main_blocks.append(
                ConvNeXtStack(hidden_channels, kernel_size=c1_cfg.get("ksz", 5), dilations=c1_cfg.get("dilation_lst", [1]))
            )
            self.main_blocks.append(
                CrossAttentionBlock(
                    d_model=hidden_channels,
                    d_context=text_dim,
                    num_heads=text_n_heads,
                    attn_dim=256,
                    use_rope=True,
                    rope_gamma=self.rope_gamma,
                    attn_scale=shared_attn_scale,
                    use_residual=use_residual,
                    rotary_base=rotary_base,
                )
            )
            self.main_blocks.append(
                ConvNeXtStack(hidden_channels, kernel_size=c2_cfg.get("ksz", 5), dilations=c2_cfg.get("dilation_lst", [1]))
            )
            self.main_blocks.append(
                CrossAttentionBlock(
                    d_model=hidden_channels,
                    d_context=style_dim,
                    num_heads=2,
                    attn_dim=256,
                    use_rope=False,
                    attn_scale=shared_attn_scale,
                    use_residual=use_residual,
                    rotary_base=rotary_base,
                )
            )

        self.last_convnext = ConvNeXtStack(
            hidden_channels, kernel_size=lc_cfg.get("ksz", 5), dilations=lc_cfg.get("dilation_lst", [1, 1, 1, 1])
        )
        self.proj_out = ProjectionWrapper(hidden_channels, out_channels)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Back-compat: older checkpoints stored the tiled style-key under `style_key`.
        legacy_key = prefix + "style_key"
        new_key = prefix + "tile"
        if legacy_key in state_dict and new_key not in state_dict:
            state_dict[new_key] = state_dict.pop(legacy_key)
        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def forward(
        self,
        noisy_latent: torch.Tensor,
        text_emb: torch.Tensor,
        style_ttl: torch.Tensor,
        latent_mask: torch.Tensor,
        text_mask: torch.Tensor,
        current_step: torch.Tensor,
        total_step: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = noisy_latent.shape[0]

        if total_step is not None:
            t_norm = current_step.reshape(B, 1, 1) / total_step.reshape(B, 1, 1)
            reciprocal = 1.0 / total_step.reshape(B, 1, 1)
            t_norm_flat = t_norm.reshape(B)
        else:
            t_norm_flat = current_step.reshape(B)

        t_emb = self.time_encoder(t_norm_flat)
        text_blc = text_emb.transpose(1, 2)

        x = self.proj_in(noisy_latent)
        x = x * latent_mask

        for i, block in enumerate(self.main_blocks):
            idx_in_super = i % 6
            if idx_in_super == 0:
                x = block(x, mask=latent_mask)
            elif idx_in_super == 1:
                x = block(x, t_emb)
                x = x * latent_mask
            elif idx_in_super == 2:
                x = block(x, mask=latent_mask)
            elif idx_in_super == 3:
                x = block(x, context=text_blc, context_keys=None,
                          x_mask=latent_mask, context_mask=text_mask)
            elif idx_in_super == 4:
                x = block(x, mask=latent_mask)
            elif idx_in_super == 5:
                x = block(x, context=style_ttl,
                          context_keys=self.tile.expand(B, -1, -1),
                          x_mask=latent_mask, context_mask=None)

        x = self.last_convnext(x, mask=latent_mask)
        diff_out = self.proj_out(x) * latent_mask

        if total_step is not None:
            denoised = noisy_latent + reciprocal * diff_out
            return denoised * latent_mask
        return diff_out
