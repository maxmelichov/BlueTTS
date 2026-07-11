import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from training.model_utils import register_contiguous_grad_hook


# =========================================================
# LayerNorm with [B, C, L] layout
# =========================================================

class LayerNormWrapper(nn.Module):
    """
    Matches ONNX hierarchy: norm.norm.weight
    Uses nn.LayerNorm internally for torch layer usage.
    """
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        # Trace: Transpose -> LayerNormalization -> Transpose
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


# =========================================================
# ConvNeXt Block (1D)
# =========================================================

class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block with explicit replicate padding to match ONNX trace.
    Hierarchy: convnext.convnext.i/...
    """
    def __init__(self,
                 dim: int,
                 expansion_factor: int = 4,
                 kernel_size: int = 5,
                 dilation: int = 1,
                 layer_scale_init_value: float = 1e-6):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.pad = ((kernel_size - 1) // 2) * dilation
        
        # dwconv child: /text_encoder/convnext/convnext.i/dwconv/Conv
        # ONNX traces explicit edge (replicate) padding (Pad mode='edge') followed by a
        # valid conv (pads=[0,0]). Padding is applied manually in forward to match exactly.
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=0,
                                groups=dim, dilation=dilation)
        
        # norm.norm hierarchy: /text_encoder/convnext/convnext.i/norm/norm/LayerNormalization
        self.norm = LayerNormWrapper(dim, eps=1e-6)
        
        # pwconv1 child: /text_encoder/convnext/convnext.i/pwconv1/Conv
        self.pwconv1 = nn.Conv1d(dim, hidden_dim, kernel_size=1)
        
        # act child (GELU): /text_encoder/convnext/convnext.i/act/...
        self.act = nn.GELU()
        
        # pwconv2 child: /text_encoder/convnext/convnext.i/pwconv2/Conv
        self.pwconv2 = nn.Conv1d(hidden_dim, dim, kernel_size=1)
        
        # gamma parameter: /text_encoder/convnext/convnext.i/Mul_2
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((1, dim, 1)),
            requires_grad=True
        )

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            x = x * mask # /text_encoder/convnext/convnext.i/Mul
        
        residual = x

        # Depthwise conv: explicit replicate padding then valid conv.
        # Matches /text_encoder/convnext/convnext.i/dwconv/Pad (mode='edge') + Conv (pads=[0,0]).
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.dwconv(x)
        
        if mask is not None:
            x = x * mask # /text_encoder/convnext/convnext.i/Mul_1
        
        x = self.norm(x)   # /text_encoder/convnext/convnext.i/norm/norm/LayerNormalization
        x = self.pwconv1(x) # /text_encoder/convnext/convnext.i/pwconv1/Conv
        x = self.act(x)     # /text_encoder/convnext/convnext.i/act/... (GELU)
        x = self.pwconv2(x) # /text_encoder/convnext/convnext.i/pwconv2/Conv
        
        # Scaling: /text_encoder/convnext/convnext.i/Mul_2
        x = self.gamma * x

        # Residual connection: /text_encoder/convnext/convnext.i/Add
        x = residual + x
        
        if mask is not None:
            x = x * mask # /text_encoder/convnext/convnext.i/Mul_3
        return x


class ConvNeXtStack(nn.Module):
    """
    Hierarchy: convnext.convnext.0...
    """
    def __init__(self,
                 dim: int,
                 n_layers: int,
                 expansion_factor: int,
                 kernel_size: int,
                 dilation_lst: list):
        super().__init__()
        self.convnext = nn.ModuleList([
            ConvNeXtBlock(dim, expansion_factor=expansion_factor, kernel_size=kernel_size, dilation=dilation_lst[i])
            for i in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for block in self.convnext:
            x = block(x, mask=mask)
        return x


# =========================================================
# FeedForward
# =========================================================

class FeedForward(nn.Module):
    """
    Hierarchy: ffn.conv_1, ffn.act, ffn.conv_2
    Matches trace hierarchy for transformer FFN.
    """
    def __init__(self,
                 channels: int,
                 filter_channels: int):
        super().__init__()
        self.conv_1 = nn.Conv1d(channels, filter_channels, 1)
        self.act = nn.ReLU()
        self.conv_2 = nn.Conv1d(filter_channels, channels, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            x = x * mask
        x = self.conv_1(x)
        x = self.act(x)
        if mask is not None:
            x = x * mask
        x = self.conv_2(x)
        if mask is not None:
            x = x * mask
        return x


# =========================================================
# Relative Attention Block
# =========================================================

class RelativeAttentionBlock(nn.Module):
    """
    Matches /text_encoder/attn_encoder/attn_layers.i/...
    Includes windowed relative positional encoding and masking.
    """
    def __init__(self,
                 channels: int,
                 n_heads: int,
                 filter_channels: int,
                 window_size: int = 4):
        super().__init__()
        assert channels % n_heads == 0
        self.channels = channels
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.scale = self.head_dim ** 0.5
        self.window_size = window_size

        # Attention projections: /text_encoder/attn_encoder/attn_layers.i/conv_q, k, v
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)

        # Relative embeddings
        self.emb_rel_k = nn.Parameter(torch.randn(1, 2 * window_size + 1, self.head_dim) * 0.02)
        self.emb_rel_v = nn.Parameter(torch.randn(1, 2 * window_size + 1, self.head_dim) * 0.02)
        register_contiguous_grad_hook(self.emb_rel_k)
        register_contiguous_grad_hook(self.emb_rel_v)

        # Output projection
        self.conv_o = nn.Conv1d(channels, channels, 1)

    def _matmul_with_relative_keys(self, x, y):
        # x: [B, H, L, head_dim], y: [1, 2L-1, head_dim] -> [B, H, L, 2L-1]
        return torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))

    def _matmul_with_relative_values(self, x, y):
        # x: [B, H, L, 2L-1], y: [1, 2L-1, head_dim] -> [B, H, L, head_dim]
        return torch.matmul(x, y.unsqueeze(0))

    def _get_relative_embeddings(self, relative_embeddings, length):
        # Slices/pads the (2W+1) windowed embeddings to the (2L-1) span used at this length.
        W = self.window_size
        
        # Use torch operations so they trace dynamically in ONNX
        L_t = torch.tensor(length, device=relative_embeddings.device, dtype=torch.long)
        W_t = torch.tensor(W + 1, device=relative_embeddings.device, dtype=torch.long)
        
        pad_length = torch.clamp(L_t - W_t, min=0)
        slice_start = torch.clamp(W_t - L_t, min=0)
        slice_end = slice_start + 2 * length - 1
        
        # F.pad requires a tuple of pad sizes. PyTorch ONNX export supports 0D tensors inside tuples for padding.
        relative_embeddings = F.pad(relative_embeddings, (0, 0, pad_length, pad_length))
        
        return relative_embeddings[:, slice_start:slice_end].contiguous()

    def _relative_position_to_absolute_position(self, x):
        # x: [B, H, L, 2L-1] -> [B, H, L, L]
        B, H, L, _ = x.shape
        x = F.pad(x, (0, 1))
        x = x.view(B, H, L * 2 * L)
        x = F.pad(x, (0, L - 1))
        x = x.view(B, H, L + 1, 2 * L - 1)[:, :, :L, L - 1:]
        return x

    def _absolute_position_to_relative_position(self, x):
        # x: [B, H, L, L] -> [B, H, L, 2L-1]
        B, H, L, _ = x.shape
        x = F.pad(x, (0, L - 1))
        x = x.view(B, H, L ** 2 + L * (L - 1))
        x = F.pad(x, (L, 0))
        x = x.view(B, H, L, 2 * L)[:, :, :, 1:]
        return x

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, L = x.shape

        # 1. Attention projections
        q_raw = self.conv_q(x)
        k_raw = self.conv_k(x)
        v_raw = self.conv_v(x)

        # ONNX trace: q and v get Reshape+Transpose ([B,H,L,hd]); k is only
        # reshaped to [B,H,hd,L] and consumed by MatMul directly (no Transpose).
        q = q_raw.view(B, self.n_heads, self.head_dim, L).transpose(2, 3)  # [B, H, L, hd]
        k = k_raw.view(B, self.n_heads, self.head_dim, L)                  # [B, H, hd, L]
        v = v_raw.view(B, self.n_heads, self.head_dim, L).transpose(2, 3)  # [B, H, L, hd]

        # Single Div node: the scaled q feeds both the content-score MatMul
        # and the relative-logits MatMul_1.
        q = q / self.scale

        # Content scores
        scores = torch.matmul(q, k)  # [B, H, L, L]

        # Windowed relative positional scores (VITS scheme)
        key_rel = self._get_relative_embeddings(self.emb_rel_k, L)
        rel_logits = self._matmul_with_relative_keys(q, key_rel)  # [B, H, L, 2L-1]
        scores = scores + self._relative_position_to_absolute_position(rel_logits)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e4)

        attn_w = torch.softmax(scores, dim=-1)  # [B, H, L, L]

        # Content context
        out = torch.matmul(attn_w, v)  # [B, H, L, hd]

        # Relative value context
        rel_weights = self._absolute_position_to_relative_position(attn_w)  # [B, H, L, 2L-1]
        value_rel = self._get_relative_embeddings(self.emb_rel_v, L)
        out = out + self._matmul_with_relative_values(rel_weights, value_rel)

        out = out.transpose(2, 3).contiguous().view(B, self.channels, L)
        out = self.conv_o(out)
        return out


class AttnEncoder(nn.Module):
    """
    Hierarchy: attn_encoder.attn_layers.0...
    """
    def __init__(self,
                 channels: int,
                 n_heads: int,
                 filter_channels: int,
                 n_layers: int):
        super().__init__()
        self.attn_layers = nn.ModuleList(
            [RelativeAttentionBlock(channels, n_heads, filter_channels, window_size=4)
             for _ in range(n_layers)]
        )
        self.norm_layers_1 = nn.ModuleList(
            [LayerNormWrapper(channels) for _ in range(n_layers)]
        )
        self.ffn_layers = nn.ModuleList(
            [FeedForward(channels, filter_channels) for _ in range(n_layers)]
        )
        self.norm_layers_2 = nn.ModuleList(
            [LayerNormWrapper(channels) for _ in range(n_layers)]
        )

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            x = x * mask # /text_encoder/attn_encoder/Mul_1

        attn_mask = None
        if mask is not None:
            # 2D attention mask: /text_encoder/attn_encoder/Mul
            attn_mask = mask.unsqueeze(1) * mask.unsqueeze(-1)

        for i in range(len(self.attn_layers)):
            residual = x
            x_attn = self.attn_layers[i](x, mask=mask, attn_mask=attn_mask)
            x = residual + x_attn
            x = self.norm_layers_1[i](x)
            
            residual_ffn = x
            x_ffn = self.ffn_layers[i](x, mask=mask)
            x = residual_ffn + x_ffn
            x = self.norm_layers_2[i](x)
            
        if mask is not None:
            x = x * mask # /text_encoder/attn_encoder/Mul_2
        return x


# =========================================================
# Style Cross-Attention (SPTE)
# =========================================================

class LinearWrapper(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    def forward(self, x):
        return self.linear(x)


class SPTENorm(nn.Module):
    """
    SPTE final norm. Matches /speech_prompted_text_encoder/norm/...

    The SPTE runs in [B, L, C] (channels-last) layout, so the trace applies
    LayerNormalization directly to the last axis and then a SINGLE Transpose
    back to [B, C, L]:
        /speech_prompted_text_encoder/norm/norm/LayerNormalization
        /speech_prompted_text_encoder/norm/Transpose
    The inner nn.LayerNorm gives the norm.norm.weight / norm.norm.bias hierarchy.
    """
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C] -> LayerNormalization -> Transpose -> [B, C, L]
        return self.norm(x).transpose(1, 2)


class StyleAttentionLayer(nn.Module):
    """
    Matches /speech_prompted_text_encoder/attentioni/...

    Cross-attention between text tokens (queries) and the style prompt, operating
    in [B, L, C] layout. Keys come from the learned style-token bank (W_key + tanh),
    values come from the dynamic style embedding (W_value):
        q = W_query(x)
        k = tanh( transpose( heads( W_key(style_key) ) ) )
        v = heads( W_value(style_values) )
        scores = (q @ k) / scale  ->  softmax  ->  mask(after softmax)  ->  @ v
        out = out_fc( merge_heads(...) ) * mask_t
    """
    def __init__(self,
                 text_dim: int,
                 style_dim: int,
                 n_units: int,
                 num_heads: int = 2,
                 num_style_tokens: int = 50):
        super().__init__()
        assert n_units % num_heads == 0
        self.num_heads = num_heads
        self.dim = n_units
        self.head_dim = n_units // num_heads
        # ONNX divides the scores by sqrt(n_units) (Constant_9 = 16.0 for n_units=256),
        # i.e. the full unit dim, not the per-head dim.
        self.scale = self.dim ** 0.5

        # Projections: /speech_prompted_text_encoder/attentioni/W_query|W_key|W_value/linear
        self.W_query = LinearWrapper(text_dim, n_units)
        self.W_key = LinearWrapper(style_dim, n_units)
        self.W_value = LinearWrapper(style_dim, n_units)

        # Output FC hierarchy: out_fc.linear
        self.out_fc = LinearWrapper(n_units, text_dim)

    def forward(self,
                x: torch.Tensor,
                style_key: torch.Tensor,
                style_values: torch.Tensor,
                mask_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, L, text_dim]; style_key: [*, S, style_dim]; style_values: [B, S, style_dim]
        # mask_t: [B, L, 1]

        # Queries: W_query -> Split -> Unsqueeze -> Concat
        q = self.W_query(x)  # [B, L, n_units]
        q = torch.stack(torch.split(q, self.head_dim, dim=-1), dim=0)  # [H, B, L, head_dim]

        # Keys: W_key over the learned style-token bank -> Split_1 -> Concat_1 -> Transpose -> tanh/Tanh
        k = self.W_key(style_key)  # [*, S, n_units]
        k = torch.stack(torch.split(k, self.head_dim, dim=-1), dim=0)  # [H, *, S, head_dim]
        k = k.transpose(-2, -1)  # [H, *, head_dim, S]
        k = torch.tanh(k)

        # Values: W_value over the dynamic style embedding -> Split_2 -> Concat_2
        v = self.W_value(style_values)  # [B, S, n_units]
        v = torch.stack(torch.split(v, self.head_dim, dim=-1), dim=0)  # [H, B, S, head_dim]

        # Scores: MatMul then Div by scale (Constant_9).
        scores = torch.matmul(q, k)  # [H, B, L, S]
        scores = scores / self.scale

        # ONNX masks AFTER softmax: Softmax -> Where(mask==0, 0, attn).
        attn = torch.softmax(scores, dim=-1)
        if mask_t is not None:
            attn = attn.masked_fill(mask_t.unsqueeze(0) == 0.0, 0.0)  # [1, B, L, 1]
        out = torch.matmul(attn, v)  # [H, B, L, head_dim]

        # Merge heads: Split_3 -> Concat_3 -> Squeeze
        out = torch.cat(torch.split(out, 1, dim=0), dim=-1).squeeze(0)  # [B, L, n_units]

        # Output projection and residual masking (out_fc/linear + Mul)
        out = self.out_fc(out)  # [B, L, text_dim]
        if mask_t is not None:
            out = out * mask_t
        return out


class StyleAttention(nn.Module):
    """
    Hierarchy: speech_prompted_text_encoder.attention1, attention2, norm...

    Main SPTE module. Mirrors the ONNX trace exactly: the text features are
    transposed once on entry ([B, C, L] -> [B, L, C]), processed by two stacked
    cross-attention layers in channels-last layout, normed, and transposed back.
    """
    def __init__(self,
                 text_dim: int,
                 style_dim: int,
                 n_units: int,
                 num_heads: int = 2,
                 num_style_tokens: int = 50):
        super().__init__()

        # Learned style-token bank shared by both cross-attention layers (W_key input).
        # In the full production graph this is the external constant
        # /style_encoder/style_token_layer/style_key ([1, S, style_dim]), Expand/Tile'd
        # to the batch and fed to attention1.W_key and attention2.W_key alike.
        self.style_key = nn.Parameter(torch.randn(1, num_style_tokens, style_dim) * 0.02)

        # Cross-attention layers
        self.attention1 = StyleAttentionLayer(text_dim, style_dim, n_units, num_heads, num_style_tokens)
        self.attention2 = StyleAttentionLayer(text_dim, style_dim, n_units, num_heads, num_style_tokens)

        # Final norm: LayerNormalization (on [B, L, C]) followed by a single Transpose.
        self.norm = SPTENorm(text_dim)

    def forward(self,
                x: torch.Tensor,
                style_values: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, C, L]; style_values: [B, S, style_dim]; mask: [B, 1, L]

        B, _, L = x.shape

        # Matches /Expand -> /Tile in ONNX graph
        style_key = self.style_key.repeat(B, 1, 1)

        # /speech_prompted_text_encoder/Transpose : [B, C, L] -> [B, L, C]
        x = x.transpose(1, 2)

        if mask is not None:
            mask_t = mask.transpose(1, 2)  # [B, L, 1]
        else:
            mask_t = None

        # Both attention layers use the ORIGINAL xt as their residual base:
        #   Add   = attention1(xt) + xt         (feeds attention2)
        #   Add_1 = attention2(Add) + xt        (NOT Add) -> norm
        residual = x
        x = residual + self.attention1(x, style_key, style_values, mask_t=mask_t)
        x = residual + self.attention2(x, style_key, style_values, mask_t=mask_t)

        # Final SPTE norm + transpose back to [B, C, L].
        out = self.norm(x)

        # /speech_prompted_text_encoder/Mul : final output masking -> text_emb
        if mask is not None:
            out = out * mask
        return out


# =========================================================
# Text Encoder Hierarchy Wrappers
# =========================================================

class TextEmbedder(nn.Module):
    """Matches ONNX path: /text_encoder/text_embedder/..."""
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.char_embedder = nn.Embedding(vocab_size, d_model)

    def forward(self, text_ids: torch.Tensor, text_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # /text_encoder/text_embedder/char_embedder/Gather & Transpose
        x = self.char_embedder(text_ids).transpose(1, 2)
        if text_mask is not None:
            x = x * text_mask # /text_encoder/text_embedder/Mul
        return x


class ProjOut(nn.Module):
    """Matches ONNX path: /text_encoder/proj_out/Mul.

    The exported graph has no convolution here; proj_out is purely the mask
    multiply applied to the global residual output.
    """
    def __init__(self, d_model: int):
        super().__init__()

    def forward(self, x: torch.Tensor, text_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if text_mask is not None:
            x = x * text_mask # /text_encoder/proj_out/Mul
        return x


class CoreTextEncoder(nn.Module):
    """Matches ONNX path: /text_encoder/..."""
    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 n_conv_layers: int,
                 expansion_factor: int,
                 kernel_size: int,
                 dilation_lst: list,
                 attn_n_heads: int,
                 attn_filter_channels: int,
                 n_attn_layers: int):
        super().__init__()
        # 1. Text Embedder
        self.text_embedder = TextEmbedder(vocab_size, d_model)

        # 2. ConvNeXt Stack
        self.convnext = ConvNeXtStack(
            d_model, n_conv_layers, expansion_factor, kernel_size, dilation_lst
        )

        # 3. Attention Encoder Stack
        self.attn_encoder = AttnEncoder(
            d_model,
            n_heads=attn_n_heads,
            filter_channels=attn_filter_channels,
            n_layers=n_attn_layers
        )

        # 4. Output Projection
        self.proj_out = ProjOut(d_model)

    def forward(self, text_ids: torch.Tensor, text_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. Embed
        x = self.text_embedder(text_ids, text_mask)

        # 2. ConvNeXt
        x_cnx = self.convnext(x, mask=text_mask)

        # 3. Attn Encoder
        x_attn = self.attn_encoder(x_cnx, mask=text_mask)

        # 4. Global residual: /text_encoder/Add
        x = x_attn + x_cnx

        # 5. Output Projection
        x = self.proj_out(x, text_mask)

        return x


# =========================================================
# Text Encoder Main Class
# =========================================================

class TextEncoder(nn.Module):
    """
    Text Encoder strictly aligned with ONNX 1-to-1 production trace hierarchy.
    Exposes /text_encoder/ and /speech_prompted_text_encoder/ as children.
    """
    def __init__(self,
                 vocab_size: int = 8322,
                 d_model: int = 256,
                 n_conv_layers: int = 6,
                 n_attn_layers: int = 4,
                 expansion_factor: int = 4,
                 p_dropout: float = 0.0,
                 kernel_size: int = 5,
                 dilation_lst: list = None,
                 attn_n_heads: int = 4,
                 attn_filter_channels: int = 1024,
                 spte_n_heads: int = 2,
                 spte_text_dim: int = 256,
                 spte_style_dim: int = 256,
                 spte_n_units: int = 256,
                 spte_n_style: int = 50):
        super().__init__()
        # p_dropout is accepted for config/API parity; attn blocks are deterministic.
        self.p_dropout = p_dropout

        if dilation_lst is None:
            # ONNX/config default for the 6-layer ConvNeXt stack.
            dilation_lst = [1, 1, 2, 2, 4, 4]

        # Core Text Encoder wrapper: /text_encoder/...
        self.text_encoder = CoreTextEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_conv_layers=n_conv_layers,
            expansion_factor=expansion_factor,
            kernel_size=kernel_size,
            dilation_lst=dilation_lst,
            attn_n_heads=attn_n_heads,
            attn_filter_channels=attn_filter_channels,
            n_attn_layers=n_attn_layers
        )

        # Speech Prompted Text Encoder (SPTE): /speech_prompted_text_encoder/...
        self.speech_prompted_text_encoder = StyleAttention(
            text_dim=spte_text_dim,
            style_dim=spte_style_dim,
            n_units=spte_n_units,
            num_heads=spte_n_heads,
            num_style_tokens=spte_n_style,
        )

    @staticmethod
    def remap_legacy_state_dict(state_dict: dict) -> dict:
        """Map the pre-ONNX-parity core encoder keys into the nested layout."""
        remapped = {}
        core_prefixes = ("text_embedder.", "convnext.", "attn_encoder.", "proj_out.")
        for key, value in state_dict.items():
            if key.startswith(core_prefixes):
                key = f"text_encoder.{key}"
            remapped[key] = value
        return remapped

    def forward(self,
                text_ids: torch.Tensor,
                style_ttl: torch.Tensor,
                text_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Run core text encoder (/text_encoder/...)
        x = self.text_encoder(text_ids, text_mask)

        # Run style attention (/speech_prompted_text_encoder/...).
        # The final output mask (/speech_prompted_text_encoder/Mul -> text_emb) is applied
        # inside the SPTE, so no extra root-level masking is needed here.
        x = self.speech_prompted_text_encoder(x, style_values=style_ttl, mask=text_mask)

        return x


if __name__ == "__main__":
    # Functional test for shape and hierarchy verification
    batch_size = 2
    text_length = 60
    vocab_size = 8322
    d_model = 256

    model = TextEncoder(vocab_size=vocab_size, d_model=d_model)
    model.eval()

    dummy_ids = torch.randint(0, vocab_size, (batch_size, text_length)).long()
    dummy_mask = torch.ones(batch_size, 1, text_length)
    dummy_style = torch.randn(batch_size, 50, d_model)

    with torch.no_grad():
        out = model(dummy_ids, dummy_style, text_mask=dummy_mask)
    
    print(f"Input text_ids: {dummy_ids.shape}")
    print(f"Output text_emb: {out.shape}")
    assert out.shape == (batch_size, d_model, text_length)
    print("Alignment test passed!")
