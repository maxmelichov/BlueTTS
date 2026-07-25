import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.model_utils import register_contiguous_grad_hook
from training.t2l.models.reference_encoder import ReferenceEncoder

# =========================================================
# LayerNorm with [B, C, L] layout
# Matches Hierarchy: .../norm/norm/LayerNormalization (Transpose -> LN -> Transpose)
# =========================================================

class DPLayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # Production ONNX exports every LayerNormalization with epsilon=1e-6
        # (torch default is 1e-5).
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L] -> Transpose -> LayerNormalization -> Transpose
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


# =========================================================
# ConvNeXt Block (1D)
# Hierarchy: /sentence_encoder/convnext/convnext.i/...
# =========================================================

class DPConvNextBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        expansion_factor: int = 4,
        kernel_size: int = 5,
        dilation: int = 1,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        hidden_dim = dim * expansion_factor
        # ONNX exports this dwconv with an explicit edge(replicate)-padding op
        # (Pad mode='edge') before the Conv, not zero padding.
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding="same",
            padding_mode="replicate",
            groups=dim,
            dilation=dilation,
        )
        self.norm = DPLayerNorm(dim)
        self.pwconv1 = nn.Conv1d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(hidden_dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((1, dim, 1)))

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # /convnext.i/Mul
        if mask is not None:
            x = x * mask
        residual = x

        # /convnext.i/dwconv/Pad -> /Conv
        x = self.dwconv(x)

        # /convnext.i/Mul_1
        if mask is not None:
            x = x * mask

        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)  # GELU: Div/Erf/Add/Mul/Mul_1
        x = self.pwconv2(x)

        # /convnext.i/Mul_2 (gamma)
        x = self.gamma * x
        # /convnext.i/Add (residual)
        x = residual + x
        # /convnext.i/Mul_3
        if mask is not None:
            x = x * mask
        return x


# =========================================================
# Relative-position Multi-Head Attention (VITS-style)
# Hierarchy: /sentence_encoder/attn_encoder/attn_layers.i/...
#   conv_q, conv_k, conv_v, conv_o, emb_rel_k, emb_rel_v
# =========================================================

class DPRelativeAttention(nn.Module):
    def __init__(self, channels: int, n_heads: int = 2, window_size: int = 4):
        super().__init__()
        assert channels % n_heads == 0
        self.channels = channels
        self.n_heads = n_heads
        self.k_channels = channels // n_heads
        self.window_size = window_size

        # /attn_layers.i/conv_q, conv_k, conv_v, conv_o
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, channels, 1)

        # Relative embeddings: [1, 2*W+1, k_channels]
        # tts.dp.sentence_encoder.attn_encoder.attn_layers.i.emb_rel_k / emb_rel_v
        rel_stddev = self.k_channels**-0.5
        self.emb_rel_k = nn.Parameter(
            torch.randn(1, window_size * 2 + 1, self.k_channels) * rel_stddev
        )
        self.emb_rel_v = nn.Parameter(
            torch.randn(1, window_size * 2 + 1, self.k_channels) * rel_stddev
        )
        register_contiguous_grad_hook(self.emb_rel_k)
        register_contiguous_grad_hook(self.emb_rel_v)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        # conv_q/k/v -> attention -> conv_o
        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)
        out = self._attention(q, k, v, mask=attn_mask)
        out = self.conv_o(out)
        return out

    def _attention(self, query, key, value, mask=None):
        b, d, t_t = query.size()
        t_s = key.size(2)

        # /Reshape + /Transpose -> [B, H, L, k]
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        # Content scores: /Div -> /MatMul
        scores = torch.matmul(
            query / math.sqrt(self.k_channels), key.transpose(-2, -1)
        )

        # Relative-key scores: /MatMul_1 -> relative_position_to_absolute
        rel_emb_k = self._get_relative_embeddings(self.emb_rel_k, t_s)
        rel_logits = torch.matmul(
            query / math.sqrt(self.k_channels),
            rel_emb_k.unsqueeze(0).transpose(-2, -1),
        )
        scores_local = self._relative_position_to_absolute_position(rel_logits)
        # /Add_2
        scores = scores + scores_local

        # /Where (masked_fill with -1e4)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)

        # /Softmax
        p_attn = F.softmax(scores, dim=-1)

        # Content context: /MatMul_2
        output = torch.matmul(p_attn, value)

        # Relative-value context: absolute_position_to_relative -> /MatMul_3 -> /Add_4
        rel_weights = self._absolute_position_to_relative_position(p_attn)
        rel_emb_v = self._get_relative_embeddings(self.emb_rel_v, t_s)
        output = output + torch.matmul(rel_weights, rel_emb_v.unsqueeze(0))

        # /Transpose_9 -> /Reshape_19
        output = output.transpose(2, 3).contiguous().view(b, d, t_t)
        return output

    def _get_relative_embeddings(self, relative_embeddings, length):
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start = max((self.window_size + 1) - length, 0)
        slice_end = slice_start + 2 * length - 1
        if pad_length > 0:
            relative_embeddings = F.pad(
                relative_embeddings, (0, 0, pad_length, pad_length, 0, 0)
            )
        return relative_embeddings[:, slice_start:slice_end].contiguous()

    def _relative_position_to_absolute_position(self, x):
        b, h, l, _ = x.size()
        x = F.pad(x, (0, 1))
        x = x.view(b, h, l * 2 * l)
        x = F.pad(x, (0, l - 1))
        x = x.view(b, h, l + 1, 2 * l - 1)[:, :, :l, l - 1 :]
        return x

    def _absolute_position_to_relative_position(self, x):
        b, h, l, _ = x.size()
        x = F.pad(x, (0, l - 1))
        x = x.view(b, h, l * l + l * (l - 1))
        x = F.pad(x, (l, 0))
        x = x.view(b, h, l, 2 * l)[:, :, :, 1:]
        return x


# =========================================================
# Position-wise Feed Forward (ReLU)
# Hierarchy: /attn_encoder/ffn_layers.i/conv_1 -> Relu -> conv_2
# =========================================================

class DPFFN(nn.Module):
    def __init__(self, channels: int, filter_channels: int):
        super().__init__()
        self.conv_1 = nn.Conv1d(channels, filter_channels, 1)
        self.conv_2 = nn.Conv1d(filter_channels, channels, 1)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask


# =========================================================
# Attention Encoder
# Hierarchy: /attn_encoder/{attn_layers, norm_layers_1, ffn_layers, norm_layers_2}.i
# =========================================================

class DPAttnEncoder(nn.Module):
    def __init__(
        self,
        channels: int,
        n_heads: int = 2,
        filter_channels: int = 256,
        n_layers: int = 2,
        window_size: int = 4,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for _ in range(n_layers):
            self.attn_layers.append(
                DPRelativeAttention(channels, n_heads=n_heads, window_size=window_size)
            )
            self.norm_layers_1.append(DPLayerNorm(channels))
            self.ffn_layers.append(DPFFN(channels, filter_channels))
            self.norm_layers_2.append(DPLayerNorm(channels))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        # /attn_encoder/Unsqueeze -> Unsqueeze_1 -> Mul  (attn_mask [B,1,L,L])
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        # /attn_encoder/Mul_1
        x = x * x_mask
        for i in range(self.n_layers):
            y = self.attn_layers[i](x, attn_mask)
            # /attn_encoder/Add -> norm_layers_1.i
            x = self.norm_layers_1[i](x + y)
            y = self.ffn_layers[i](x, x_mask)
            # /attn_encoder/Add_1 -> norm_layers_2.i
            x = self.norm_layers_2[i](x + y)
        # /attn_encoder/Mul_2
        x = x * x_mask
        return x


# =========================================================
# Text embedder
# Hierarchy: /sentence_encoder/text_embedder/char_embedder
# =========================================================

class DPTextEmbedder(nn.Module):
    def __init__(self, vocab_size: int = 163, d_model: int = 64):
        super().__init__()
        self.char_embedder = nn.Embedding(vocab_size, d_model)

    def forward(self, text_ids: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # char_embedder/Gather -> Transpose -> Mul
        x = self.char_embedder(text_ids).transpose(1, 2)
        if mask is not None:
            x = x * mask
        return x


# =========================================================
# Output projection
# Hierarchy: /sentence_encoder/proj_out/net (Conv1d, no bias) -> Mul
# =========================================================

class DPProjOut(nn.Module):
    def __init__(self, channels: int, out_channels: int = None):
        super().__init__()
        self.net = nn.Conv1d(channels, out_channels or channels, 1, bias=False)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None) -> torch.Tensor:
        x = self.net(x)
        if x_mask is not None:
            x = x * x_mask
        return x


# =========================================================
# Sentence Encoder
# Hierarchy: /sentence_encoder/...
# =========================================================

class DPSentenceEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 163,
        d_model: int = 64,
        n_heads: int = 2,
        filter_channels: int = 256,
        n_layers: int = 2,
        window_size: int = 4,
        n_convnext: int = 6,
        convnext_kernel_size: int = 5,
        convnext_expansion: int = 4,
        convnext_dilation_lst: list = None,
        proj_out_dim: int = None,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        # p_dropout is accepted for config/API parity; the production DP graph
        # is deterministic (config value is 0.0).
        self.p_dropout = p_dropout

        # text_embedder.char_embedder
        self.text_embedder = DPTextEmbedder(vocab_size, d_model)

        # convnext.convnext.0..5
        if convnext_dilation_lst is None:
            convnext_dilation_lst = [1] * n_convnext
        self.convnext = nn.Module()
        self.convnext.convnext = nn.ModuleList(
            [
                DPConvNextBlock(
                    d_model,
                    expansion_factor=convnext_expansion,
                    kernel_size=convnext_kernel_size,
                    dilation=convnext_dilation_lst[i],
                )
                for i in range(n_convnext)
            ]
        )

        # sentence_token
        self.sentence_token = nn.Parameter(torch.randn(1, d_model, 1) * 0.02)

        # attn_encoder
        self.attn_encoder = DPAttnEncoder(
            d_model,
            n_heads=n_heads,
            filter_channels=filter_channels,
            n_layers=n_layers,
            window_size=window_size,
        )

        # proj_out
        self.proj_out = DPProjOut(d_model, out_channels=proj_out_dim)

    def forward(self, text_ids: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B = text_ids.shape[0]

        # text_embedder/Gather -> Transpose -> Mul
        x = self.text_embedder(text_ids, mask=mask)

        # /Expand -> /Concat_1 (prepend sentence token)
        token = self.sentence_token.expand(B, -1, -1)
        x = torch.cat([token, x], dim=2)

        # /Concat_2 (prepend token mask)
        if mask is not None:
            token_mask = torch.ones_like(mask[:, :, :1])
            mask = torch.cat([token_mask, mask], dim=2)
        else:
            mask = torch.ones(B, 1, x.shape[2], device=x.device, dtype=x.dtype)

        # convnext stack
        for block in self.convnext.convnext:
            x = block(x, mask=mask)
        convnext_out = x

        # attn_encoder
        attn_out = self.attn_encoder(convnext_out, mask)

        # /sentence_encoder/Add (global residual)
        x = attn_out + convnext_out

        # /sentence_encoder/Slice_1 (take sentence token, position 0)
        x = x[:, :, :1]

        # /sentence_encoder/proj_out/net/Conv -> Mul
        x = self.proj_out(x, mask[:, :, :1])
        return x


# =========================================================
# Predictor (Duration Estimator)
# Hierarchy: /predictor/layers.0 -> activation (PRelu) -> layers.1 -> Exp -> Squeeze
# =========================================================

class DPPredictor(nn.Module):
    def __init__(self, input_dim: int = 192, hidden_dim: int = 128, n_layer: int = 2):
        super().__init__()
        if n_layer < 2:
            raise ValueError(f"DPPredictor needs at least 2 layers, got {n_layer}")
        # layers.0 .. layers.{n_layer-1} (production graph: n_layer=2)
        self.layers = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim)]
            + [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layer - 2)]
            + [nn.Linear(hidden_dim, 1)]
        )
        # activation -> single PRelu op (weight shape [1])
        self.activation = nn.PReLU(num_parameters=1)

    def forward(
        self,
        text_feat: torch.Tensor,
        style_dp: torch.Tensor,
        return_log: bool = False,
    ) -> torch.Tensor:
        B = text_feat.shape[0]

        # /predictor/Reshape  (text feature [B, 64, 1] -> [B, 64])
        text_feat = text_feat.reshape(B, -1)
        # /predictor/Reshape_1 (style [B, 8, 16] -> [B, 128])
        style = style_dp.reshape(B, -1)

        # /predictor/Concat_2 (text then style)
        x = torch.cat([text_feat, style], dim=1)

        # layers.0/Gemm -> activation/PRelu -> layers.1/Gemm
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)  # [B, 1]

        if return_log:
            # log-duration branch (no ONNX counterpart) squeezes the same axis
            return x.squeeze(1)

        # /predictor/Exp -> /predictor/Squeeze (graph order)
        x = torch.exp(x)
        x = x.squeeze(1)
        return x


# =========================================================
# Main TTS Duration Model (== tts.dp)
# Children sentence_encoder and predictor are siblings.
# =========================================================

class TTSDurationModel(nn.Module):
    def __init__(
        self,
        vocab_size=37,
        style_dp=8,
        style_dim=16,
        sentence_encoder_cfg=None,
        style_encoder_cfg=None,
        predictor_cfg=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size

        # Parse configs
        se_cfg = sentence_encoder_cfg or {}
        st_cfg = style_encoder_cfg or {}
        pr_cfg = predictor_cfg or {}

        # DP config resolution (mirrors tts.json dp.* layout):
        se_d_model = se_cfg.get("text_embedder", {}).get(
            "char_emb_dim", se_cfg.get("char_emb_dim", 64)
        )
        if "char_emb_dim" in se_cfg and se_d_model != se_cfg["char_emb_dim"]:
            raise ValueError(
                "dp.sentence_encoder char_emb_dim mismatch: "
                f"text_embedder={se_d_model} vs top-level={se_cfg['char_emb_dim']}"
            )

        # attn params live under dp.sentence_encoder.attn_encoder (legacy
        # top-level keys are honored as fallback).
        se_attn = se_cfg.get("attn_encoder", {})
        se_n_heads = se_attn.get("n_heads", se_cfg.get("n_heads", 2))
        se_filter_channels = se_attn.get(
            "filter_channels", se_cfg.get("filter_channels", 256)
        )
        se_n_layers = se_attn.get("n_layers", se_cfg.get("n_layers", 2))
        se_window_size = se_attn.get("window_size", se_cfg.get("window_size", 4))
        se_p_dropout = se_attn.get("p_dropout", 0.0)
        se_hidden_channels = se_attn.get("hidden_channels", se_d_model)
        if se_hidden_channels != se_d_model:
            raise ValueError(
                "dp.sentence_encoder.attn_encoder.hidden_channels "
                f"({se_hidden_channels}) must equal char_emb_dim ({se_d_model})"
            )

        se_convnext = se_cfg.get("convnext", {})
        se_n_convnext = se_convnext.get("num_layers", 6)
        se_conv_idim = se_convnext.get("idim", se_d_model)
        if se_conv_idim != se_d_model:
            raise ValueError(
                f"dp.sentence_encoder.convnext.idim ({se_conv_idim}) must "
                f"equal char_emb_dim ({se_d_model})"
            )
        se_conv_ksz = se_convnext.get("ksz", 5)
        se_conv_intermediate = se_convnext.get("intermediate_dim", se_d_model * 4)
        if se_conv_intermediate % se_d_model != 0:
            raise ValueError(
                f"dp.sentence_encoder.convnext.intermediate_dim ({se_conv_intermediate}) "
                f"must be a multiple of char_emb_dim ({se_d_model})"
            )
        se_conv_dilations = se_convnext.get("dilation_lst", [1] * se_n_convnext)

        se_proj_out = se_cfg.get("proj_out", {})
        se_proj_idim = se_proj_out.get("idim", se_d_model)
        se_proj_odim = se_proj_out.get("odim", se_d_model)
        if se_proj_idim != se_d_model:
            raise ValueError(
                f"dp.sentence_encoder.proj_out.idim ({se_proj_idim}) must "
                f"equal char_emb_dim ({se_d_model})"
            )

        st_proj = st_cfg.get("proj_in", {})
        st_d_model = st_proj.get("odim", 64)
        st_in_channels = st_proj.get("ldim", 24) * st_proj.get("chunk_compress_factor", 6)

        st_convnext = st_cfg.get("convnext", {})
        st_hidden_dim = st_convnext.get("intermediate_dim", 256)
        st_num_blocks = st_convnext.get("num_layers", 4)
        st_ksz = st_convnext.get("ksz", 5)
        st_dilation = st_convnext.get("dilation_lst", None)
        st_conv_idim = st_convnext.get("idim", st_d_model)
        if st_conv_idim != st_d_model:
            raise ValueError(
                f"dp.style_encoder.convnext.idim ({st_conv_idim}) must "
                f"equal proj_in.odim ({st_d_model})"
            )

        st_token_layer = st_cfg.get("style_token_layer", {})
        st_num_queries = st_token_layer.get("n_style", style_dp)
        st_query_dim = st_token_layer.get("style_value_dim", style_dim)
        st_num_heads = st_token_layer.get("n_heads", 2)
        st_prototype_dim = st_token_layer.get("prototype_dim", st_query_dim)
        st_n_units = st_token_layer.get("n_units", st_query_dim)
        st_input_dim = st_token_layer.get("input_dim", st_d_model)
        if st_input_dim != st_d_model:
            raise ValueError(
                f"dp.style_encoder.style_token_layer.input_dim ({st_input_dim}) "
                f"must equal proj_in.odim ({st_d_model})"
            )
        # style_key_dim=0 means the DP style extractor produces values only
        # (no key bank output); this implementation has no DP key path.
        st_key_dim = st_token_layer.get("style_key_dim", 0)
        if st_key_dim != 0:
            raise NotImplementedError(
                "dp.style_encoder.style_token_layer.style_key_dim != 0 is not "
                f"supported (got {st_key_dim}); the DP has no style-key output"
            )

        pr_text_dim = pr_cfg.get("sentence_dim", se_d_model)
        pr_style_dim = pr_cfg.get("n_style", st_num_queries) * pr_cfg.get(
            "style_dim", st_query_dim
        )
        pr_hidden_dim = pr_cfg.get("hdim", pr_cfg.get("hidden_dim", 128))
        pr_n_layer = pr_cfg.get("n_layer", 2)

        # 1. Text Encoder (Must match ONNX trace exactly)
        self.sentence_encoder = DPSentenceEncoder(
            vocab_size=vocab_size,
            d_model=se_d_model,
            n_heads=se_n_heads,
            filter_channels=se_filter_channels,
            n_layers=se_n_layers,
            window_size=se_window_size,
            n_convnext=se_n_convnext,
            convnext_kernel_size=se_conv_ksz,
            convnext_expansion=se_conv_intermediate // se_d_model,
            convnext_dilation_lst=se_conv_dilations,
            proj_out_dim=se_proj_odim,
            p_dropout=se_p_dropout,
        )

        # 2. Reference Encoder (Used during training to get style_dp from z_ref)
        self.ref_encoder = ReferenceEncoder(
            in_channels=st_in_channels,
            d_model=st_d_model,
            hidden_dim=st_hidden_dim,
            num_blocks=st_num_blocks,
            num_tokens=st_num_queries,
            num_heads=st_num_heads,
            kernel_size=st_ksz,
            dilation_lst=st_dilation,
            prototype_dim=st_prototype_dim,
            n_units=st_n_units,
            style_value_dim=st_query_dim,
        )

        # 3. Predictor (Must match ONNX trace exactly)
        self.predictor = DPPredictor(
            input_dim=pr_text_dim + pr_style_dim,
            hidden_dim=pr_hidden_dim,
            n_layer=pr_n_layer,
        )

    @staticmethod
    def remap_legacy_state_dict(state_dict: dict) -> dict:
        """Map the former flat DP ConvNeXt keys into ``DPSentenceEncoder``."""
        remapped = {}
        prefix = "sentence_encoder.convnext."
        for key, value in state_dict.items():
            if key.startswith(prefix) and not key.startswith(f"{prefix}convnext."):
                key = f"{prefix}convnext.{key[len(prefix):]}"
            remapped[key] = value
        return remapped

    def forward(
        self,
        text_ids,
        z_ref=None,
        text_mask=None,
        ref_mask=None,
        style_dp=None,
        return_log=False,
    ):
        # 1. Text path (Sentence Encoder)
        text_feat = self.sentence_encoder(text_ids, mask=text_mask)

        # 2. Style path
        if style_dp is not None:
            style = style_dp
        elif z_ref is not None:
            style = self.ref_encoder(z_ref, mask=ref_mask)
        else:
            raise ValueError("Either z_ref or style_dp must be provided")

        # 3. Predictor
        duration = self.predictor(text_feat, style, return_log=return_log)
        return duration
