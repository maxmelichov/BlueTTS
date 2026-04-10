import math
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearWrapper(nn.Module):
    """Wraps nn.Linear to match checkpoint path `linear.weight` -> `module.linear.weight`."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class LayerNormWrapper(nn.Module):
    """Wraps nn.LayerNorm. Expects [B, C, T] input, normalizes over C."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x.transpose(1, 2)

class ProjectionWrapper(nn.Module):
    """Wraps Conv1d 1x1 to match `proj.net.weight`."""
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
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class TimeEncoder(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.sinusoidal = SinusoidalPosEmb(embed_dim, scale=1000.0)
        self.mlp = nn.Sequential(
            LinearWrapper(embed_dim, embed_dim * 4),
            Mish(),
            LinearWrapper(embed_dim * 4, embed_dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sinusoidal(x)
        return self.mlp(x)

class TimeCondBlock(nn.Module):
    def __init__(self, time_dim: int, channels: int):
        super().__init__()
        self.linear = LinearWrapper(time_dim, channels)
        nn.init.zeros_(self.linear.linear.weight)
        nn.init.zeros_(self.linear.linear.bias)
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        cond = self.linear(time_emb).unsqueeze(-1)
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
        if mask is not None: x = x * mask
        residual = x
        x = F.pad(x, (self.pad, self.pad), mode='replicate')
        x = self.dwconv(x)
        if mask is not None: x = x * mask
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = self.dropout(x)
        x = x + residual
        if mask is not None: x = x * mask
        return x

class ConvNeXtStack(nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        self.convnext = nn.ModuleList([ConvNeXtBlock1D(channels, kernel_size=kernel_size, dilation=d, expansion=2) for d in dilations])
    def forward(self, x, mask=None):
        for blk in self.convnext:
            x = blk(x, mask)
        return x

def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    B, H, T, D = x.shape
    assert D % 2 == 0, "head_dim must be even for RoPE"
    x1, x2 = x[..., : D // 2], x[..., D // 2 :]
    if cos.dim() == 2:
        cos, sin = cos[None, None, :, :], sin[None, None, :, :]
    elif cos.dim() == 3:
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class AttentionModule(nn.Module):
    def __init__(self, d_model: int, d_context: int, num_heads: int, attn_dim: int, use_rope: bool, dropout: float = 0.0, rope_gamma: float = 10.0, attn_scale: Optional[float] = None):
        super().__init__()
        assert attn_dim % num_heads == 0
        self.d_model, self.num_heads, self.head_dim, self.attn_dim, self.use_rope, self.rope_gamma = d_model, num_heads, attn_dim // num_heads, attn_dim, use_rope, rope_gamma
        self.attn_scale = attn_scale if attn_scale is not None else math.sqrt(self.attn_dim)
        self.W_query, self.W_key, self.W_value, self.out_fc = LinearWrapper(d_model, attn_dim), LinearWrapper(d_context, attn_dim), LinearWrapper(d_context, attn_dim), LinearWrapper(attn_dim, d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        if use_rope:
            inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
            self.register_buffer("theta", (inv_freq * rope_gamma).view(1, 1, -1), persistent=True)
            self.register_buffer("increments", torch.arange(1000).view(1, 1000, 1), persistent=True)
            self.tanh = None
        else:
            self.theta, self.increments, self.tanh = None, None, nn.Tanh()

    def forward(self, x: torch.Tensor, context: torch.Tensor, context_keys: Optional[torch.Tensor] = None, x_mask: Optional[torch.Tensor] = None, context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, d_model, T = x.shape
        L = context.shape[1]
        x_t = x.transpose(1, 2)
        q = self.W_query(x_t)
        if context_keys is not None:
            if context_keys.dim() == 2: context_keys = context_keys.unsqueeze(0)
            if context_keys.shape[0] == 1: context_keys = context_keys.expand(B, -1, -1)
            k = self.W_key(context_keys)
        else:
            k = self.W_key(context)
        v = self.W_value(context)
        if not self.use_rope and self.tanh is not None: k = self.tanh(k)
        H, D = self.num_heads, self.head_dim
        q, k, v = q.view(B, T, H, D).permute(2, 0, 1, 3), k.view(B, L, H, D).permute(2, 0, 1, 3), v.view(B, L, H, D).permute(2, 0, 1, 3)
        if self.use_rope:
            device = x.device
            len_q = x_mask.sum(dim=(-2, -1)).reshape(-1, 1, 1) if x_mask is not None else torch.tensor([T], device=device, dtype=torch.float32).reshape(1, 1, 1)
            len_k = context_mask.sum(dim=(-2, -1)).reshape(-1, 1, 1) if context_mask is not None else torch.tensor([L], device=device, dtype=torch.float32).reshape(1, 1, 1)
            if self.increments is not None and self.increments.shape[1] >= max(T, L):
                pos_q, pos_k = self.increments[:, :T, :].to(device).float(), self.increments[:, :L, :].to(device).float()
            else:
                pos_q, pos_k = torch.arange(T, device=device, dtype=torch.float32).reshape(1, -1, 1), torch.arange(L, device=device, dtype=torch.float32).reshape(1, -1, 1)
            theta = self.theta if self.theta is not None else ((1.0 / (10000 ** (torch.arange(0, D, 2, device=device).float() / D))) * self.rope_gamma).view(1, 1, -1)
            freqs_q, freqs_k = (pos_q / len_q) * theta, (pos_k / len_k) * theta
            cos_q, sin_q, cos_k, sin_k = freqs_q.cos().unsqueeze(0), freqs_q.sin().unsqueeze(0), freqs_k.cos().unsqueeze(0), freqs_k.sin().unsqueeze(0)
            q, k = apply_rotary_pos_emb(q, cos_q, sin_q), apply_rotary_pos_emb(k, cos_k, sin_k)
        attn_logits = torch.matmul(q, k.transpose(-1, -2)) / self.attn_scale
        if context_mask is not None:
            if context_mask.dim() == 2: context_mask = context_mask.unsqueeze(1)
            attn_logits = attn_logits.masked_fill((context_mask == 0).unsqueeze(0), float('-inf'))
        attn = torch.softmax(attn_logits, dim=-1)
        if x_mask is not None:
            if x_mask.dim() == 2: x_mask = x_mask.unsqueeze(1)
            attn = attn.masked_fill((x_mask == 0).permute(1, 0, 2).unsqueeze(-1), 0.0)
        out = torch.matmul(attn, v).permute(1, 2, 0, 3).contiguous().view(B, T, self.attn_dim)
        out = self.out_fc(out)
        out = self.dropout(out)
        if x_mask is not None: out = out * x_mask.transpose(1, 2)
        return out.transpose(1, 2)

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, d_context: int, num_heads: int = 8, attn_dim: int = 256, use_rope: bool = True, rope_gamma: float = 10.0, attn_scale: Optional[float] = None):
        super().__init__()
        self.use_rope = use_rope
        attn_module = AttentionModule(d_model, d_context, num_heads, attn_dim, use_rope, rope_gamma=rope_gamma, attn_scale=attn_scale)
        if use_rope: self.attn = attn_module
        else: self.attention = attn_module
        self.norm = LayerNormWrapper(d_model)

    def forward(self, x: torch.Tensor, context: torch.Tensor, context_keys: Optional[torch.Tensor], x_mask: Optional[torch.Tensor], context_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if x_mask is not None: x = x * x_mask
        residual = x
        attn_out = self.attn(x, context, context_keys, x_mask, context_mask) if self.use_rope else self.attention(x, context, context_keys, x_mask, context_mask)
        x = self.norm(residual + attn_out)
        if x_mask is not None: x = x * x_mask
        return x

class VectorFieldEstimator(nn.Module):
    def __init__(self, in_channels: int = 144, hidden_channels: int = 512, out_channels: int = 144, text_dim: int = 256, style_dim: int = 256, num_style_tokens: int = 50, num_superblocks: int = 4, time_embed_dim: int = 64, rope_gamma: float = 10.0, main_blocks_cfg: dict = None, last_convnext_cfg: dict = None, text_n_heads: int = 4):
        super().__init__()
        self.in_channels, self.hidden_channels, self.out_channels, self.text_dim, self.style_dim, self.rope_gamma = in_channels, hidden_channels, out_channels, text_dim, style_dim, rope_gamma
        self.style_key = nn.Parameter(torch.randn(1, num_style_tokens, style_dim) * 0.02)
        self.proj_in, self.time_encoder, self.main_blocks = ProjectionWrapper(in_channels, hidden_channels), TimeEncoder(time_embed_dim), nn.ModuleList()
        shared_attn_scale = math.sqrt(256)
        mb_cfg, lc_cfg = main_blocks_cfg or {}, last_convnext_cfg or {}
        c0_cfg, c1_cfg, c2_cfg = mb_cfg.get("convnext_0", {}), mb_cfg.get("convnext_1", {}), mb_cfg.get("convnext_2", {})
        for _ in range(num_superblocks):
            self.main_blocks.append(ConvNeXtStack(hidden_channels, kernel_size=c0_cfg.get("ksz", 5), dilations=c0_cfg.get("dilation_lst", [1, 2, 4, 8])))
            self.main_blocks.append(TimeCondBlock(time_dim=time_embed_dim, channels=hidden_channels))
            self.main_blocks.append(ConvNeXtStack(hidden_channels, kernel_size=c1_cfg.get("ksz", 5), dilations=c1_cfg.get("dilation_lst", [1])))
            self.main_blocks.append(CrossAttentionBlock(d_model=hidden_channels, d_context=text_dim, num_heads=text_n_heads, attn_dim=256, use_rope=True, rope_gamma=self.rope_gamma, attn_scale=shared_attn_scale))
            self.main_blocks.append(ConvNeXtStack(hidden_channels, kernel_size=c2_cfg.get("ksz", 5), dilations=c2_cfg.get("dilation_lst", [1])))
            self.main_blocks.append(CrossAttentionBlock(d_model=hidden_channels, d_context=style_dim, num_heads=2, attn_dim=256, use_rope=False, attn_scale=shared_attn_scale))
        self.last_convnext = ConvNeXtStack(hidden_channels, kernel_size=lc_cfg.get("ksz", 5), dilations=lc_cfg.get("dilation_lst", [1, 1, 1, 1]))
        self.proj_out = ProjectionWrapper(hidden_channels, out_channels)

    def forward(self, noisy_latent: torch.Tensor, text_emb: Optional[torch.Tensor] = None, style_ttl: Optional[torch.Tensor] = None, latent_mask: Optional[torch.Tensor] = None, text_mask: Optional[torch.Tensor] = None, current_step: Optional[torch.Tensor] = None, total_step: Optional[torch.Tensor] = None, style_keys: Optional[torch.Tensor] = None, text_context: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if text_emb is None: text_emb = text_context
        assert text_emb is not None, "Must provide text_emb or text_context"
        B = noisy_latent.shape[0]
        if style_keys is None: style_keys = self.style_key.expand(B, -1, -1)
        if total_step is not None:
            t_norm = current_step.reshape(B, 1, 1) / total_step.reshape(B, 1, 1)
            reciprocal, t_norm_flat = (1.0 / total_step.reshape(B, 1, 1)), t_norm.reshape(B)
        else:
            t_norm_flat = current_step.reshape(B)
        t_emb, text_blc = self.time_encoder(t_norm_flat), text_emb.transpose(1, 2)
        x = self.proj_in(noisy_latent)
        if latent_mask is not None: x = x * latent_mask
        for i, block in enumerate(self.main_blocks):
            idx_in_super = i % 6
            if idx_in_super == 0: x = block(x, mask=latent_mask)
            elif idx_in_super == 1:
                x = block(x, t_emb)
                if latent_mask is not None: x = x * latent_mask
            elif idx_in_super == 2: x = block(x, mask=latent_mask)
            elif idx_in_super == 3: x = block(x, context=text_blc, context_keys=None, x_mask=latent_mask, context_mask=text_mask)
            elif idx_in_super == 4: x = block(x, mask=latent_mask)
            elif idx_in_super == 5: x = block(x, context=style_ttl, context_keys=style_keys, x_mask=latent_mask, context_mask=None)
        x = self.last_convnext(x, mask=latent_mask)
        diff_out = self.proj_out(x)
        if latent_mask is not None: diff_out = diff_out * latent_mask
        if total_step is not None:
            denoised = noisy_latent + reciprocal * diff_out
            return denoised * latent_mask if latent_mask is not None else denoised
        return diff_out

if __name__ == "__main__":
    B, T_latent, T_text = 1, 100, 60
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")
    model = VectorFieldEstimator(in_channels=144, hidden_channels=512, out_channels=144, text_dim=256, style_dim=256, num_superblocks=4, time_embed_dim=64).to(device).eval()
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    noisy_latent, text_emb, style_ttl, latent_mask, text_mask, current_step, total_step, style_keys = torch.randn(B, 144, T_latent, device=device), torch.randn(B, 256, T_text, device=device), torch.randn(B, 50, 256, device=device), torch.ones(B, 1, T_latent, device=device), torch.ones(B, 1, T_text, device=device), torch.tensor([5.0], device=device), torch.tensor([32.0], device=device), torch.randn(1, 50, 256, device=device)
    with torch.no_grad():
        out = model(noisy_latent, text_emb, style_ttl, latent_mask, text_mask, current_step, total_step, style_keys=style_keys)
    print(f"Output: {out.shape}")
    assert out.shape == (B, 144, T_latent)
    print("OK")