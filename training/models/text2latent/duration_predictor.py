import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from models.text2latent.text_encoder import (
    AttnEncoder,
    TextEmbedderWrapper,
    ConvNeXtWrapper,
    ConvNeXtBlock,
    LayerNorm
)

class DPReferenceEncoder(nn.Module):
    def __init__(self, in_channels=144, d_model=64, hidden_dim=256, num_blocks=4, num_queries=8, query_dim=16):
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, d_model, 1)
        self.convnext = nn.ModuleList([
            ConvNeXtBlock(d_model, expansion_factor=hidden_dim // d_model)
            for _ in range(num_blocks)
        ])
        
        self.num_queries = num_queries
        self.query_dim = query_dim
        self.queries = nn.Parameter(torch.randn(1, self.num_queries, self.query_dim) * 0.02)
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=self.query_dim, num_heads=1, kdim=d_model, vdim=d_model, batch_first=True)
            for _ in range(2)
        ])
        
    def forward(self, z_ref, mask=None):
        B = z_ref.shape[0]
        x = self.input_proj(z_ref)

        if mask is not None:
            x = x * mask

        for blk in self.convnext:
            x = blk(x, mask=mask)

        context = x.transpose(1, 2)
        q = self.queries.expand(B, -1, -1)

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = (mask.squeeze(1) == 0)

        for layer in self.attn_layers:
            out, _ = layer(q, context, context, key_padding_mask=key_padding_mask)
            q = q + out

        out = q.reshape(B, -1)
        return out


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
            n_layers=2
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
            nn.Linear(128, 1)
        ])
        self.activation = nn.PReLU()

    def forward(self, text_emb, style_emb, return_log=False):
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
    def __init__(self, vocab_size=37, style_tokens=8, style_dim=16, sentence_encoder_cfg=None, style_encoder_cfg=None, predictor_cfg=None):
        super().__init__()
        self.vocab_size = vocab_size
        
        se_cfg = sentence_encoder_cfg or {}
        st_cfg = style_encoder_cfg or {}
        pr_cfg = predictor_cfg or {}
        
        se_d_model = se_cfg.get("char_emb_dim", 64)
        
        st_proj = st_cfg.get("proj_in", {})
        st_d_model = st_proj.get("odim", 64)
        
        pr_text_dim = pr_cfg.get("sentence_dim", 64)
        pr_style_dim = pr_cfg.get("n_style", style_tokens) * pr_cfg.get("style_dim", style_dim)
        
        self.sentence_encoder = DPTextEncoder(vocab_size=vocab_size, d_model=se_d_model)
        self.ref_encoder = DPReferenceEncoder(num_queries=style_tokens, query_dim=style_dim, d_model=st_d_model)
        self.predictor = DurationEstimator(text_dim=pr_text_dim, style_dim=pr_style_dim)

    def forward(self, text_ids, z_ref=None, text_mask=None, ref_mask=None, style_tokens=None, return_log=False):
        text_emb = self.sentence_encoder(text_ids, mask=text_mask)

        if style_tokens is not None:
            style_emb = style_tokens
        elif z_ref is not None:
            style_emb = self.ref_encoder(z_ref, mask=ref_mask)
        else:
            raise ValueError("Either z_ref or style_tokens must be provided")

        duration = self.predictor(text_emb, style_emb, return_log=return_log)
        return duration

if __name__ == "__main__":
    model = TTSDurationModel(vocab_size=37, style_tokens=8, style_dim=16)
    model.eval()
    B = 2
    T = 50

    text = torch.randint(0, 37, (B, T))
    style_dp = torch.randn(B, 8, 16)
    text_mask = torch.ones(B, 1, T)
    with torch.no_grad():
        dur_style = model(text, text_mask=text_mask, style_tokens=style_dp)
    print(f"Style path  - Duration output: {dur_style.shape}  values: {dur_style}")

    z_ref = torch.randn(B, 144, 100)
    ref_mask = torch.ones(B, 1, 100)
    with torch.no_grad():
        dur_ref = model(text, z_ref=z_ref, text_mask=text_mask, ref_mask=ref_mask)
    print(f"Ref path    - Duration output: {dur_ref.shape}  values: {dur_ref}")
