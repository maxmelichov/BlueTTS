import torch
import torch.nn as nn

from models.text2latent.text_encoder import (
    AttnEncoder,
    TextEmbedderWrapper,
    ConvNeXtWrapper,
)


class DPReferenceEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 144,
        d_model: int = 64,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        num_queries: int = 8,
        query_dim: int = 16,
        num_heads: int = 2,
        kernel_size: int = 5,
        dilation_lst: list = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.query_dim = query_dim
        mlp_ratio = hidden_dim // d_model

        self.input_proj = nn.Conv1d(in_channels, d_model, kernel_size=1)
        self.convnext = ConvNeXtWrapper(
            d_model,
            n_layers=num_blocks,
            expansion_factor=mlp_ratio,
            kernel_size=kernel_size,
            dilation_lst=dilation_lst,
        )
        self.ref_keys = nn.Parameter(torch.randn(num_queries, query_dim) * 0.02)
        self.attn1 = nn.MultiheadAttention(
            embed_dim=query_dim, num_heads=num_heads, kdim=d_model, vdim=d_model, batch_first=True
        )
        self.attn2 = nn.MultiheadAttention(
            embed_dim=query_dim, num_heads=num_heads, kdim=d_model, vdim=d_model, batch_first=True
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
        q1, _ = self.attn1(query=q0, key=kv, value=kv, key_padding_mask=key_padding_mask, need_weights=False)
        q2 = q0 + q1
        out, _ = self.attn2(query=q2, key=kv, value=kv, key_padding_mask=key_padding_mask, need_weights=False)
        return out.reshape(B, -1)


class DPTextEncoder(nn.Module):
    def __init__(self, vocab_size=37, d_model=64):
        super().__init__()
        self.d_model = d_model
        self.text_embedder = TextEmbedderWrapper(vocab_size, d_model)
        self.convnext = ConvNeXtWrapper(d_model, n_layers=6, expansion_factor=4)
        self.sentence_token = nn.Parameter(torch.randn(1, d_model, 1) * 0.02)
        self.attn_encoder = AttnEncoder(
            channels=d_model,
            n_heads=2,
            filter_channels=d_model * 4,
            n_layers=2,
        )
        self.proj_out = nn.Sequential()
        self.proj_out.add_module("net", nn.Conv1d(d_model, d_model, 1, bias=False))

    def forward(self, text_ids, mask=None):
        B, T = text_ids.shape
        x = self.text_embedder(text_ids)
        x = x.transpose(1, 2)
        if mask is not None:
            x = x * mask

        u_token = self.sentence_token.expand(B, -1, -1)
        x = torch.cat([u_token, x], dim=2)

        if mask is not None:
            mask_u = torch.ones(B, 1, 1, device=mask.device)
            mask = torch.cat([mask_u, mask], dim=2)

        x = self.convnext(x, mask=mask)
        conv_out = x
        x = self.attn_encoder(x, mask=mask)
        x = x + conv_out

        first_token = x[:, :, :1]
        out = self.proj_out(first_token)
        if mask is not None:
            out = out * mask[:, :, :1]
        return out.squeeze(2)


class DurationEstimator(nn.Module):
    def __init__(self, text_dim=64, style_dim=128):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(text_dim + style_dim, 128),
            nn.Linear(128, 1),
        ])
        self.activation = nn.PReLU()

    def forward(self, text_emb, style_emb, text_mask=None, return_log=False):
        if style_emb.dim() > 2:
            style_emb = style_emb.reshape(style_emb.shape[0], -1)
        x = torch.cat([text_emb, style_emb], dim=1)
        x = self.layers[0](x)
        x = self.activation(x)
        x = self.layers[1](x)
        if return_log:
            return x.squeeze(1)
        return torch.exp(x).squeeze(1)


class TTSDurationModel(nn.Module):
    def __init__(self, vocab_size=37, style_dp=8, style_dim=16, sentence_encoder_cfg=None, style_encoder_cfg=None, predictor_cfg=None):
        super().__init__()
        self.vocab_size = vocab_size

        se_cfg = sentence_encoder_cfg or {}
        st_cfg = style_encoder_cfg or {}
        pr_cfg = predictor_cfg or {}

        se_d_model = se_cfg.get("char_emb_dim", 64)

        st_proj = st_cfg.get("proj_in", {})
        st_d_model = st_proj.get("odim", 64)

        st_convnext = st_cfg.get("convnext", {})
        st_hidden_dim = st_convnext.get("intermediate_dim", 256)
        st_num_blocks = st_convnext.get("num_layers", 4)
        st_dilation = st_convnext.get("dilation_lst", None)

        st_token_layer = st_cfg.get("style_token_layer", {})
        st_num_queries = st_token_layer.get("n_style", style_dp)
        st_query_dim = st_token_layer.get("style_value_dim", style_dim)
        st_num_heads = st_token_layer.get("n_heads", 2)

        pr_text_dim = pr_cfg.get("sentence_dim", 64)
        pr_style_dim = pr_cfg.get("n_style", st_num_queries) * pr_cfg.get("style_dim", st_query_dim)

        self.sentence_encoder = DPTextEncoder(vocab_size=vocab_size, d_model=se_d_model)
        self.ref_encoder = DPReferenceEncoder(
            in_channels=144,
            d_model=st_d_model,
            hidden_dim=st_hidden_dim,
            num_blocks=st_num_blocks,
            num_queries=st_num_queries,
            query_dim=st_query_dim,
            num_heads=st_num_heads,
            dilation_lst=st_dilation,
        )
        self.predictor = DurationEstimator(text_dim=pr_text_dim, style_dim=pr_style_dim)

    def forward(self, text_ids, z_ref=None, text_mask=None, ref_mask=None, style_dp=None, return_log=False):
        text_emb = self.sentence_encoder(text_ids, mask=text_mask)

        if style_dp is not None:
            style_emb = style_dp
        elif z_ref is not None:
            style_emb = self.ref_encoder(z_ref, mask=ref_mask)
        else:
            raise ValueError("Either z_ref or style_dp must be provided")

        return self.predictor(text_emb, style_emb, text_mask=text_mask, return_log=return_log)
