import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        return x


class ConvNeXtBlock(nn.Module):
    def __init__(self,
                 dim: int,
                 expansion_factor: int = 4,
                 kernel_size: int = 5,
                 dilation: int = 1,
                 layer_scale_init_value: float = 1e-6):
        super().__init__()
        hidden_dim = dim * expansion_factor

        if (kernel_size % 2) != 1:
            raise ValueError(f"ConvNeXtBlock expects odd kernel_size, got {kernel_size}")
        self.pad = ((kernel_size - 1) // 2) * dilation
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=0,
                                groups=dim, dilation=dilation)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Conv1d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(hidden_dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((1, dim, 1)),
            requires_grad=True
        )

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is not None:
            x = x * mask
        residual = x

        x = F.pad(x, (self.pad, self.pad), mode='replicate')
        x = self.dwconv(x)
        if mask is not None:
            x = x * mask

        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x

        x = residual + x
        if mask is not None:
            x = x * mask
        return x



class ConvNeXtWrapper(nn.Module):
    def __init__(self, d_model, n_layers, expansion_factor, kernel_size=5, dilation_lst=None):
        super().__init__()
        if dilation_lst is None:
            dilation_lst = [1] * n_layers
        self.convnext = nn.ModuleList([
            ConvNeXtBlock(d_model, expansion_factor=expansion_factor, kernel_size=kernel_size, dilation=dilation_lst[i])
            for i in range(n_layers)
        ])

    def forward(self, x, mask=None):
        for block in self.convnext:
            x = block(x, mask=mask)
        return x



class RelativeMultiHeadAttention(nn.Module):
    def __init__(self,
                 channels: int,
                 n_heads: int,
                 window_size: int = 4,
                 p_dropout: float = 0.0):
        super().__init__()
        assert channels % n_heads == 0
        self.channels = channels
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size

        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, channels, 1)

        self.emb_rel_k = nn.Parameter(torch.randn(1, 2 * window_size + 1, self.head_dim) * 0.02)
        self.emb_rel_v = nn.Parameter(torch.randn(1, 2 * window_size + 1, self.head_dim) * 0.02)

        self.drop = nn.Dropout(p_dropout)

    def forward(self,
                x: torch.Tensor,
                attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, C, L = x.shape

        q = self.conv_q(x).view(B, self.n_heads, self.head_dim, L).transpose(2, 3)
        q = q * self.scale
        k = self.conv_k(x).view(B, self.n_heads, self.head_dim, L).transpose(2, 3)
        v = self.conv_v(x).view(B, self.n_heads, self.head_dim, L).transpose(2, 3)

        scores = torch.matmul(q, k.transpose(-2, -1))

        t = torch.arange(L, device=x.device)
        diff = t[None, :] - t[:, None]
        window_mask = (diff.abs() <= self.window_size)
        diff_clamped = torch.clamp(diff, -self.window_size, self.window_size)
        indices = diff_clamped + self.window_size

        rel_k = self.emb_rel_k[0][indices]
        rel_scores = torch.einsum('bhld,ljd->bhlj', q, rel_k)
        rel_scores = rel_scores * window_mask[None, None, :, :]

        scores = scores + rel_scores

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e4)

        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)

        out = torch.matmul(attn, v)

        rel_v = self.emb_rel_v[0][indices]
        rel_v = rel_v * window_mask[:, :, None]
        out_rel = torch.einsum('bhlj,ljd->bhld', attn, rel_v)

        out = out + out_rel

        out = out.transpose(2, 3).contiguous().view(B, C, L)
        out = self.conv_o(out)
        return out


# =========================================================
# FeedForward
# =========================================================

class FeedForward(nn.Module):
    def __init__(self,
                 channels: int,
                 filter_channels: int,
                 kernel_size: int = 1,
                 p_dropout: float = 0.0):
        super().__init__()
        self.conv_1 = nn.Conv1d(channels, filter_channels, kernel_size)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p_dropout)
        self.conv_2 = nn.Conv1d(filter_channels, channels, kernel_size)

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is not None:
            x = x * mask
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.drop(x)
        if mask is not None:
            x = x * mask
        x = self.conv_2(x)
        if mask is not None:
            x = x * mask
        return x


# =========================================================
# AttnEncoder
# =========================================================

class AttnEncoder(nn.Module):
    def __init__(self,
                 channels: int,
                 n_heads: int,
                 filter_channels: int,
                 n_layers: int,
                 p_dropout: float = 0.0):
        super().__init__()
        self.attn_layers = nn.ModuleList(
            [RelativeMultiHeadAttention(channels, n_heads, window_size=4, p_dropout=p_dropout)
             for _ in range(n_layers)]
        )
        self.norm_layers_1 = nn.ModuleList(
            [LayerNorm(channels) for _ in range(n_layers)]
        )
        self.ffn_layers = nn.ModuleList(
            [FeedForward(channels, filter_channels, p_dropout=p_dropout) for _ in range(n_layers)]
        )
        self.norm_layers_2 = nn.ModuleList(
            [LayerNorm(channels) for _ in range(n_layers)]
        )

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is not None:
            x = x * mask

        # Build 2D attention mask [B, 1, L, L] matching ONNX:
        # Unsqueeze(mask, -1) * Unsqueeze(mask, -2)
        attn_mask = None
        if mask is not None:
            attn_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)  # [B, 1, L, L]

        for i in range(len(self.attn_layers)):
            # Self-attention block
            residual = x
            x = self.attn_layers[i](x, attn_mask=attn_mask)
            x = residual + x
            x = self.norm_layers_1[i](x)

            # FFN block
            residual_ffn = x
            x_ffn = self.ffn_layers[i](x, mask=mask)
            x = residual_ffn + x_ffn
            x = self.norm_layers_2[i](x)
        
        if mask is not None:
            x = x * mask
            
        return x


# =========================================================
# Helper Modules for Exact Structure Matching
# =========================================================

class LinearWrapped(nn.Module):
    """Wraps Linear to match keys like W_query.linear.weight"""
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x)

class StyleNorm(nn.Module):
    def __init__(self, dim, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        x = self.norm(x)
        x = x.transpose(1, 2)
        return x

class TextEmbedderWrapper(nn.Module):
    """Wraps Embedding to match keys like text_embedder.char_embedder.weight"""
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.char_embedder = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.char_embedder(x)


# =========================================================
# Style Cross-Attention Layer
# =========================================================

class StyleAttentionLayer(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int = 2):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        # ONNX divides by sqrt(dim), not sqrt(head_dim)
        self.scale = dim ** -0.5

        # Wrapped linears to match ONNX structure
        self.W_query = LinearWrapped(dim)
        self.W_key = LinearWrapped(dim)
        self.W_value = LinearWrapped(dim)
        self.out_fc = LinearWrapped(dim)

    def forward(self,
                x: torch.Tensor,
                keys: torch.Tensor,
                values: torch.Tensor,
                mask_t: torch.Tensor | None = None) -> torch.Tensor:
        
        B, T, _ = x.shape
        
        q = self.W_query(x)
        
        qs = q.chunk(self.num_heads, dim=-1)
        q = torch.stack(qs, dim=0)

        k = self.W_key(keys)
        
        ks = k.chunk(self.num_heads, dim=-1)
        k = torch.stack(ks, dim=0)
        
        k = torch.tanh(k)
        
        if values.dim() == 2:
            values = values.unsqueeze(0)
        if values.shape[0] != B:
            values = values.expand(B, -1, -1)
        
        v = self.W_value(values)
        vs = v.chunk(self.num_heads, dim=-1)
        v = torch.stack(vs, dim=0)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = torch.softmax(scores, dim=-1)

        if mask_t is not None:
            attn_mask = (mask_t.unsqueeze(0) == 0)
            attn = attn.masked_fill(attn_mask, 0.0)

        out = torch.matmul(attn, v)

        outs = out.chunk(self.num_heads, dim=0)
        out = torch.cat(outs, dim=-1).squeeze(0)
        
        out = self.out_fc(out)

        if mask_t is not None:
            out = out * mask_t

        return out


class StyleAttention(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int = 2,
                 num_style_tokens: int = 50):
        super().__init__()
        # Explicitly named layers to match ONNX trace
        self.attention1 = StyleAttentionLayer(dim, num_heads)
        self.attention2 = StyleAttentionLayer(dim, num_heads)

        self.style_key = nn.Parameter(
            torch.randn(1, num_style_tokens, dim) * 0.02
        )

        # Norm wrapper matching ONNX path: norm.norm.weight
        self.norm = StyleNorm(dim)

    def forward(self,
                x: torch.Tensor,
                style_values: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        
        # x input is [B, C, L]
        # Transpose to [B, L, C] for attention
        x = x.transpose(1, 2)
        
        mask_t = None
        if mask is not None:
            # mask is [B, 1, L], we need [B, L, 1] for broadcasting against [B, L, C]
            mask_t = mask.transpose(1, 2)

        # Always use baked-in style_key for keys (matches ONNX constant)
        keys = self.style_key

        # Layer 1
        # Residual adds to original x
        # x1 = x + Attn1(x)
        out1 = self.attention1(x, keys, style_values, mask_t=mask_t)
        x1 = x + out1
        
        # Layer 2
        # Query comes from x1 (output of layer 1)
        # Residual adds to original x (input to layer 1)
        # x2 = x + Attn2(x1)
        out2 = self.attention2(x1, keys, style_values, mask_t=mask_t)
        x2 = x + out2

        # Norm + Transpose back to [B, C, T]
        x = self.norm(x2)

        if mask is not None:
            x = x * mask

        return x


# =========================================================
# Text Encoder Main Class
# =========================================================

class TextEncoder(nn.Module):
    """
    Text Encoder for Text-to-Latent model.
    Aligned with ONNX graph structure.
    Vocab size kept at 37 as per configuration.
    """
    def __init__(self,
                 vocab_size: int = 384,
                 d_model: int = 256,
                 n_conv_layers: int = 6,
                 n_attn_layers: int = 4,
                 expansion_factor: int = 4,
                 p_dropout: float = 0.1,
                 kernel_size: int = 5,
                 dilation_lst: list = None,
                 attn_n_heads: int = 4,
                 attn_filter_channels: int = 1024,
                 spte_n_heads: int = 2):
        super().__init__()

        self.d_model = d_model
        hidden_dim = d_model * expansion_factor

        # Wrapper for matching keys: text_embedder.char_embedder
        self.text_embedder = TextEmbedderWrapper(vocab_size, d_model)

        # Wrapper for matching keys: convnext.convnext
        self.convnext = ConvNeXtWrapper(d_model, n_conv_layers, expansion_factor, kernel_size=kernel_size, dilation_lst=dilation_lst)

        self.attn_encoder = AttnEncoder(
            d_model,
            n_heads=attn_n_heads,
            filter_channels=attn_filter_channels,
            n_layers=n_attn_layers,
            p_dropout=p_dropout
        )

        self.speech_prompted_text_encoder = StyleAttention(
            d_model,
            num_heads=spte_n_heads,
            num_style_tokens=50
        )

        self.proj_out = nn.Identity()

    def forward(self,
                text_ids: torch.Tensor,
                style_ttl: torch.Tensor,
                text_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.text_embedder(text_ids)

        lang_emb = x[:, 0:1, :].clone()
        x[:, 1:, :] = x[:, 1:, :] + lang_emb

        x = x.transpose(1, 2)

        if text_mask is not None:
            x = x * text_mask

        x = self.convnext(x, mask=text_mask)
        convnext_output = x

        x = self.attn_encoder(x, mask=text_mask)

        x = x + convnext_output

        x = self.proj_out(x)
        if text_mask is not None:
            x = x * text_mask

        x = self.speech_prompted_text_encoder(
            x,
            style_values=style_ttl,
            mask=text_mask
        )
            
        return x



if __name__ == "__main__":
    batch_size = 2
    text_length = 60
    vocab_size = 384
    d_model = 256

    model = TextEncoder(vocab_size=vocab_size, d_model=d_model)
    model.eval()

    text_ids = torch.randint(0, vocab_size, (batch_size, text_length)).long()
    text_mask = torch.ones(batch_size, 1, text_length)

    # Style values from reference encoder: [B, 50, 256]
    N_ref = 50
    style_ttl = torch.randn(batch_size, N_ref, d_model)
    
    # Test 1: ONNX-style positional call (text_ids, style_ttl, text_mask)
    with torch.no_grad():
        h_text = model(text_ids, style_ttl, text_mask=text_mask)
    print("ONNX-style call:", h_text.shape)

    # Test 2: Training-style keyword call (ref_values, ref_keys, text_mask)
    # ref_values = torch.randn(batch_size, N_ref, d_model)
    # ref_keys = torch.randn(batch_size, N_ref, d_model)  # ignored, uses baked-in
    # with torch.no_grad():
    #     h_text2, style_key2 = model(
    #         text_ids,
    #         ref_keys=ref_keys,
    #         text_mask=text_mask,
    #         ref_values=ref_values,
    #     )
    # print("Training-style call:", h_text2.shape, style_key2.shape)
