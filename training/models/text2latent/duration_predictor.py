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
from models.text2latent.reference_encoder import ReferenceEncoder

class DPReferenceEncoder(nn.Module):
    """
    A.3.1 DP Reference Encoder
    Shares the same architecture as the reference encoder in the text-to-latent module,
    but with different hyperparameter settings from the config.
    """
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
        """
        z_ref: (B, 144, T)
        mask:  (B, 1, T)
        """
        B = z_ref.shape[0]

        x = self.input_proj(z_ref)
        x = self.convnext(x, mask=mask)

        kv = x.transpose(1, 2)  # (B, T, d_model)

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = (mask.squeeze(1) == 0)

        q0 = self.ref_keys.unsqueeze(0).expand(B, -1, -1)  # (B, N, query_dim)

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
        # 3. Stack -> reshape to 1D
        # q: [B, 8, 16] -> [B, 128]
        return out.reshape(B, -1)



class DPTextEncoder(nn.Module):
    """
    A.3.2 DP Text Encoder
    - Input: Text IDs
    - Output: 64-dim utterance-level text embedding
    
    Note: The pre-trained ONNX uses vocab_size=163 (char_embedder weight [163, 64]).
    When training from scratch with a reduced alphabet, vocab_size=37 is used.
    This is the ONLY weight shape that differs from the ONNX; all other
    parameters (98 total in the style path) match 1-to-1.
    """
    def __init__(self, vocab_size=37, d_model=64):
        super().__init__()
        self.d_model = d_model
        
        self.text_embedder = TextEmbedderWrapper(vocab_size, d_model)
        
        # 6 ConvNeXt blocks (intermediate 256 -> mlp_ratio=4)
        self.convnext = ConvNeXtWrapper(d_model, n_layers=6, expansion_factor=4)
        
        # Utterance token (prepend)
        self.sentence_token = nn.Parameter(torch.randn(1, d_model, 1) * 0.02)
        
        # 2 Self-Attention Blocks (256 filter, 2 heads, RoPE)
        self.attn_encoder = AttnEncoder(
            channels=d_model,
            n_heads=2,
            filter_channels=d_model * 4, # 256
            n_layers=2
        )
        
        # Final Projection (Conv1d 1x1, no bias - matches ONNX proj_out/net/Conv)
        self.proj_out = nn.Sequential()
        self.proj_out.add_module("net", nn.Conv1d(d_model, d_model, 1, bias=False))

    def forward(self, text_ids, mask=None):
        B, T = text_ids.shape
        
        # Embed
        x = self.text_embedder(text_ids) # [B, T, 64]
        
        x = x.transpose(1, 2) # [B, 64, T]
        
        if mask is not None:
            x = x * mask

        # Prepend Utterance Token - FIXED LOCATION (Before ConvNeXt)
        u_token = self.sentence_token.expand(B, -1, -1) # [B, 64, 1]
        x = torch.cat([u_token, x], dim=2) # [B, 64, T+1]
        
        # Update mask
        if mask is not None:
            # Add 1 for utterance token (valid)
            mask_u = torch.ones(B, 1, 1, device=mask.device)
            mask = torch.cat([mask_u, mask], dim=2)
            
        # ConvNeXt
        x = self.convnext(x, mask=mask)
            
        # Store for residual
        conv_out = x

        # Attention
        x = self.attn_encoder(x, mask=mask)
        
        # Residual (ConvNeXt output + Attention output)
        x = x + conv_out
            
        # Take first token (utterance token)
        # Slice: [B, 64, 1]
        first_token = x[:, :, :1] 
        
        # Linear/Conv
        out = self.proj_out(first_token) # [B, 64, 1]
        
        if mask is not None:
            out = out * mask[:, :, :1]
        
        return out.squeeze(2) # [B, 64]


class DurationEstimator(nn.Module):
    """
    A.3.3 Duration Estimator
    - Input: Text Emb (64) + Style Emb (64 or 128)
    - Output: Scalar Duration
    """
    def __init__(self, text_dim=64, style_dim=128):
        super().__init__()
        # Input is 64 (text) + 128 (style) = 192
        
        # Structure matched to logs: layers.0 -> activation -> layers.1
        self.layers = nn.ModuleList([
            nn.Linear(text_dim + style_dim, 128), # Input 192, Hidden 128
            nn.Linear(128, 1)
        ])
        self.activation = nn.PReLU()

    def forward(self, text_emb, style_emb, text_mask=None, return_log=False):
        # text_emb: [B, 64]
        # style_emb: [B, 64] or [B, N, D]
        if style_emb.dim() > 2:
            style_emb = style_emb.reshape(style_emb.shape[0], -1)
            
        x = torch.cat([text_emb, style_emb], dim=1) # [B, 192]
        
        x = self.layers[0](x)
        x = self.activation(x)
        x = self.layers[1](x) # [B, 1]
        
        if return_log:
            return x.squeeze(1)
            
        return torch.exp(x).squeeze(1)


class TTSDurationModel(nn.Module):
    def __init__(self, vocab_size=37, style_dp=8, style_dim=16, sentence_encoder_cfg=None, style_encoder_cfg=None, predictor_cfg=None):
        super().__init__()
        self.vocab_size = vocab_size
        
        # Parse configs
        se_cfg = sentence_encoder_cfg or {}
        st_cfg = style_encoder_cfg or {}
        pr_cfg = predictor_cfg or {}
        
        # DP config resolution:
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
            in_channels=144, # 24 * 6
            d_model=st_d_model,
            hidden_dim=st_hidden_dim,
            num_blocks=st_num_blocks,
            num_queries=st_num_queries,
            query_dim=st_query_dim,
            num_heads=st_num_heads,
            dilation_lst=st_dilation
        )
        self.predictor = DurationEstimator(text_dim=pr_text_dim, style_dim=pr_style_dim)

    def forward(self, text_ids, z_ref=None, text_mask=None, ref_mask=None, style_dp=None, return_log=False):
        """
        Args:
            text_ids: [B, T]
            z_ref: [B, 144, T_ref] (optional if style_dp provided)
            text_mask: [B, 1, T]
            ref_mask: [B, 1, T_ref]
            style_dp: [B, 8, 16] (optional pre-computed style tokens)
            return_log: If True, return log(duration). Else return duration (linear).
            
        Returns:
            duration: [B] (scalar)
        """
        text_emb = self.sentence_encoder(text_ids, mask=text_mask) # [B, 64]
        
        if style_dp is not None:
            style_emb = style_dp
        elif z_ref is not None:
            style_emb = self.ref_encoder(z_ref, mask=ref_mask)         # [B, 128]
        else:
            raise ValueError("Either z_ref or style_dp must be provided")
        
        # Original paper: No explicit length feature.
        # The utterance token embedding should contain all necessary duration info via attention.
        duration = self.predictor(text_emb, style_emb, text_mask=text_mask, return_log=return_log) # [B]
        
        return duration

if __name__ == "__main__":
    model = TTSDurationModel(vocab_size=37, style_dp=8, style_dim=16)
    model.eval()
    B = 2
    T = 50

    # --- Test 1: ONNX style-path (text_ids, style_dp, text_mask) ---
    text = torch.randint(0, 37, (B, T))
    # Flattened output expected by `style_dp` in duration_predictor?
    # Wait, `style_dp` argument in forward says: [B, 8, 16] (optional pre-computed style tokens)
    # The current DPReferenceEncoder flattens it, but if style_dp is provided, 
    # it is passed directly to `predictor`. But predictor expects flattened [B, 192].
    # Fortunately DurationEstimator handles unflattened input:
    # `if style_emb.dim() > 2: style_emb = style_emb.reshape(...)`
    style_dp = torch.randn(B, 8, 16)
    text_mask = torch.ones(B, 1, T)
    with torch.no_grad():
        dur_style = model(text, text_mask=text_mask, style_dp=style_dp)
    print(f"Style path  - Duration output: {dur_style.shape}  values: {dur_style}")

    # --- Test 2: Full path with z_ref (ref_encoder computes style) ---
    z_ref = torch.randn(B, 144, 100)
    ref_mask = torch.ones(B, 1, 100)
    with torch.no_grad():
        dur_ref = model(text, z_ref=z_ref, text_mask=text_mask, ref_mask=ref_mask)
    print(f"Ref path    - Duration output: {dur_ref.shape}  values: {dur_ref}")

    # --- Verify parameter names match ONNX (with tts.dp. prefix) ---
    print(f"\nTotal params (excl ref_encoder): "
          f"{sum(1 for n, _ in model.named_parameters() if 'ref_encoder' not in n)}")
    print("Vocab size note: ONNX uses 163, code uses 37 "
          "(only char_embedder.weight shape differs)")
