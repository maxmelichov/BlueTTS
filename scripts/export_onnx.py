#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxruntime as ort

# Import your models
from models.text2latent.text_encoder import TextEncoder
from models.text2latent.vf_estimator import VectorFieldEstimator
from models.autoencoder.latent_decoder import LatentDecoder1D
from models.text2latent.dp_network import DPNetwork
from models.text2latent.reference_encoder import ReferenceEncoder
from models.utils import load_ttl_config


# =====================================================================
# ONNX-safe MultiheadAttention replacement
# nn.MultiheadAttention has known ONNX tracing issues with batch_first,
# key_padding_mask, and cross-attention (different kdim/vdim).
# This manual implementation avoids all those issues.
# =====================================================================

class OnnxSafeMultiheadAttention(nn.Module):
    """
    Drop-in replacement for nn.MultiheadAttention that uses only
    basic PyTorch ops for clean ONNX tracing.
    """
    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, batch_first=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.batch_first = batch_first
        self._qkv_same_embed_dim = (self.kdim == embed_dim and self.vdim == embed_dim)
        self.scale = self.head_dim ** -0.5

        if self._qkv_same_embed_dim:
            self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.q_proj_weight = None
            self.k_proj_weight = None
            self.v_proj_weight = None
        else:
            self.in_proj_weight = None
            self.q_proj_weight = nn.Parameter(torch.empty(embed_dim, embed_dim))
            self.k_proj_weight = nn.Parameter(torch.empty(embed_dim, self.kdim))
            self.v_proj_weight = nn.Parameter(torch.empty(embed_dim, self.vdim))

        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        # Ensure batch_first layout: [B, T, C]
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        B, T_q, _ = query.shape
        T_k = key.shape[1]

        # Project Q, K, V
        if self._qkv_same_embed_dim:
            # Combined projection
            bias_q = self.in_proj_bias[:self.embed_dim]
            bias_k = self.in_proj_bias[self.embed_dim:2*self.embed_dim]
            bias_v = self.in_proj_bias[2*self.embed_dim:]
            w_q = self.in_proj_weight[:self.embed_dim]
            w_k = self.in_proj_weight[self.embed_dim:2*self.embed_dim]
            w_v = self.in_proj_weight[2*self.embed_dim:]
            q = F.linear(query, w_q, bias_q)
            k = F.linear(key, w_k, bias_k)
            v = F.linear(value, w_v, bias_v)
        else:
            # Separate projections
            bias_q = self.in_proj_bias[:self.embed_dim]
            bias_k = self.in_proj_bias[self.embed_dim:2*self.embed_dim]
            bias_v = self.in_proj_bias[2*self.embed_dim:]
            q = F.linear(query, self.q_proj_weight, bias_q)
            k = F.linear(key, self.k_proj_weight, bias_k)
            v = F.linear(value, self.v_proj_weight, bias_v)

        # Reshape to [B, T, H, D] -> [B, H, T, D]
        q = q.view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply key_padding_mask: [B, T_k] where True = ignore
        if key_padding_mask is not None:
            # key_padding_mask: [B, T_k] bool, True means "drop this key"
            # Expand to [B, 1, 1, T_k] for broadcasting with [B, H, T_q, T_k]
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T_k]
            attn_weights = attn_weights.masked_fill(mask, float('-inf'))

        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        attn_weights = torch.softmax(attn_weights, dim=-1)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [B, H, T_q, D]

        # Reshape back: [B, H, T_q, D] -> [B, T_q, H*D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T_q, self.embed_dim)

        # Output projection
        attn_output = self.out_proj(attn_output)

        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        return attn_output, None  # (output, attn_weights=None for compat)


def _replace_mha_with_safe(module: nn.Module) -> None:
    """
    Recursively replace all nn.MultiheadAttention instances in `module`
    with OnnxSafeMultiheadAttention, transferring weights exactly.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.MultiheadAttention):
            safe = OnnxSafeMultiheadAttention(
                embed_dim=child.embed_dim,
                num_heads=child.num_heads,
                kdim=child.kdim,
                vdim=child.vdim,
                batch_first=child.batch_first,
            )
            # Transfer weights
            with torch.no_grad():
                if child._qkv_same_embed_dim:
                    safe.in_proj_weight.copy_(child.in_proj_weight)
                else:
                    safe.q_proj_weight.copy_(child.q_proj_weight)
                    safe.k_proj_weight.copy_(child.k_proj_weight)
                    safe.v_proj_weight.copy_(child.v_proj_weight)
                safe.in_proj_bias.copy_(child.in_proj_bias)
                safe.out_proj.weight.copy_(child.out_proj.weight)
                safe.out_proj.bias.copy_(child.out_proj.bias)
            setattr(module, name, safe)
            print(f"  [FIX] Replaced nn.MultiheadAttention '{name}' with OnnxSafeMultiheadAttention")
        else:
            _replace_mha_with_safe(child)

    # Also handle nn.ModuleList/nn.ModuleDict containing MHA
    if isinstance(module, nn.ModuleList):
        for i, child in enumerate(module):
            if isinstance(child, nn.MultiheadAttention):
                safe = OnnxSafeMultiheadAttention(
                    embed_dim=child.embed_dim,
                    num_heads=child.num_heads,
                    kdim=child.kdim,
                    vdim=child.vdim,
                    batch_first=child.batch_first,
                )
                with torch.no_grad():
                    if child._qkv_same_embed_dim:
                        safe.in_proj_weight.copy_(child.in_proj_weight)
                    else:
                        safe.q_proj_weight.copy_(child.q_proj_weight)
                        safe.k_proj_weight.copy_(child.k_proj_weight)
                        safe.v_proj_weight.copy_(child.v_proj_weight)
                    safe.in_proj_bias.copy_(child.in_proj_bias)
                    safe.out_proj.weight.copy_(child.out_proj.weight)
                    safe.out_proj.bias.copy_(child.out_proj.bias)
                module[i] = safe
                print(f"  [FIX] Replaced nn.MultiheadAttention at index {i} with OnnxSafeMultiheadAttention")
            else:
                _replace_mha_with_safe(child)

    if isinstance(module, nn.ModuleDict):
        for key, child in module.items():
            if isinstance(child, nn.MultiheadAttention):
                safe = OnnxSafeMultiheadAttention(
                    embed_dim=child.embed_dim,
                    num_heads=child.num_heads,
                    kdim=child.kdim,
                    vdim=child.vdim,
                    batch_first=child.batch_first,
                )
                with torch.no_grad():
                    if child._qkv_same_embed_dim:
                        safe.in_proj_weight.copy_(child.in_proj_weight)
                    else:
                        safe.q_proj_weight.copy_(child.q_proj_weight)
                        safe.k_proj_weight.copy_(child.k_proj_weight)
                        safe.v_proj_weight.copy_(child.v_proj_weight)
                    safe.in_proj_bias.copy_(child.in_proj_bias)
                    safe.out_proj.weight.copy_(child.out_proj.weight)
                    safe.out_proj.bias.copy_(child.out_proj.bias)
                module[key] = safe
                print(f"  [FIX] Replaced nn.MultiheadAttention key='{key}' with OnnxSafeMultiheadAttention")
            else:
                _replace_mha_with_safe(child)


# =====================================================================
# VF Estimator ONNX wrappers
# =====================================================================

class VectorFieldEstimatorWrapper(nn.Module):
    """
    ONNX-signature wrapper for VectorFieldEstimator.

    The reference ONNX graph (see `checks/notebook.ipynb`) has EXACT inputs:
      noisy_latent, text_emb, style_ttl, latent_mask, text_mask, current_step, total_step
    and output:
      denoised_latent

    Notably, it does NOT expose `style_keys` as an input; keys are a baked-in
    constant expanded to batch internally.
    """
    def __init__(self, model: VectorFieldEstimator):
        super().__init__()
        self.model = model

    def forward(self, noisy_latent, text_emb, style_ttl, latent_mask, text_mask, current_step, total_step):
        return self.model(
            noisy_latent=noisy_latent,
            text_emb=text_emb,
            style_ttl=style_ttl,
            latent_mask=latent_mask,
            text_mask=text_mask,
            current_step=current_step,
            total_step=total_step,
            style_keys=None,  # force baked-in style key path
        )



# =====================================================================
# Export & Verify helpers
# =====================================================================

def export_one(model, out_path, inputs, input_names, output_names, dynamic_axes):
    model.eval()
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with torch.no_grad():
        torch.onnx.export(
            model,
            inputs,
            out_path,
            opset_version=17,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )
    print(f"[OK] Exported: {out_path}")


def verify_onnx(model, onnx_path, inputs, input_names, label="model", atol=1e-4, rtol=1e-3):
    """
    Compare PyTorch model output vs ONNX Runtime output for numerical parity.
    Returns True if outputs match within tolerance, False otherwise.
    """
    model.eval()
    with torch.no_grad():
        if isinstance(inputs, torch.Tensor):
            pt_inputs = (inputs,)
        else:
            pt_inputs = inputs
        pt_out = model(*pt_inputs)
        if isinstance(pt_out, tuple):
            pt_out = pt_out[0]
        pt_np = pt_out.cpu().numpy()

    # Build ONNX Runtime feed
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    feed = {}
    if isinstance(inputs, torch.Tensor):
        inputs = (inputs,)
    for name, tensor in zip(input_names, inputs):
        feed[name] = tensor.detach().cpu().numpy()
    onnx_out = sess.run(None, feed)[0]

    # Compare
    max_diff = np.max(np.abs(pt_np - onnx_out))
    mean_diff = np.mean(np.abs(pt_np - onnx_out))
    cos_sim = np.dot(pt_np.flatten(), onnx_out.flatten()) / (
        np.linalg.norm(pt_np.flatten()) * np.linalg.norm(onnx_out.flatten()) + 1e-12
    )
    match = np.allclose(pt_np, onnx_out, atol=atol, rtol=rtol)

    status = "PASS" if match else "FAIL"
    print(f"  [{status}] {label}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, cos_sim={cos_sim:.6f}")
    if not match:
        print(f"         PT  range: [{pt_np.min():.4f}, {pt_np.max():.4f}], mean={pt_np.mean():.4f}")
        print(f"         ORT range: [{onnx_out.min():.4f}, {onnx_out.max():.4f}], mean={onnx_out.mean():.4f}")

    return match

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/tts.json", help="Path to tts.json config")
    parser.add_argument("--onnx_dir", type=str, default="onnx_models", help="Output directory for ONNX models")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/text2latent", help="Text2Latent checkpoint dir")
    parser.add_argument("--ttl_ckpt", type=str, default=None, help="Explicit TTL checkpoint file to export (optional)")
    parser.add_argument("--ae_ckpt", type=str, default="checkpoints/ae/ae_latest.pt", help="AE checkpoint")
    parser.add_argument("--dp_ckpt", type=str, default="checkpoints/duration_predictor/duration_predictor_final.pt", help="DP checkpoint")
    parser.add_argument("--no-verify", action="store_true", help="Skip ONNX vs PyTorch verification")
    args = parser.parse_args()

    device = "cpu" # Export on CPU is usually safer/easier for reproducibility

    # Load config – REQUIRED so that all model dimensions and normalizer_scale
    # always come from tts.json (never silently fall back to wrong defaults).
    if not os.path.exists(args.config):
        print(f"[ERROR] Config not found: {args.config}")
        print("       The tts.json config is required. Pass --config <path> or place it at configs/tts.json")
        return

    cfg = load_ttl_config(args.config)
    print(f"[INFO] Loaded config: {args.config} (v{cfg['full_config'].get('tts_version', '?')})")

    onnx_dir = args.onnx_dir
    os.makedirs(onnx_dir, exist_ok=True)

    ckpt_dir = args.ckpt_dir
    ae_ckpt_path = args.ae_ckpt
    
    # Helper to find latest checkpoint
    import glob
    def get_latest_ckpt(dir_path):
        # Prefer ckpt_step_*.pt by step number (matches training/inference convention).
        ckpt_step = glob.glob(os.path.join(dir_path, "ckpt_step_*.pt"))
        if ckpt_step:
            def step_num(p):
                base = os.path.basename(p)
                try:
                    return int(base.split("ckpt_step_")[-1].split(".pt")[0])
                except Exception:
                    return -1
            ckpt_step.sort(key=step_num)
            return ckpt_step[-1]
        # Fallback: newest mtime among .pt
        ckpts = glob.glob(os.path.join(dir_path, "*.pt"))
        if not ckpts:
            return None
        return max(ckpts, key=os.path.getmtime)

    text2latent_ckpt = args.ttl_ckpt if (args.ttl_ckpt and os.path.exists(args.ttl_ckpt)) else get_latest_ckpt(ckpt_dir)
    if text2latent_ckpt is None:
        print(f"[WARN] No text2latent checkpoint found in {ckpt_dir}. Using random weights for demonstration/structure.")
    else:
        print(f"[INFO] Loading text2latent from {text2latent_ckpt}")

    # ---- Load Checkpoints ----
    t2l_state = torch.load(text2latent_ckpt, map_location=device) if text2latent_ckpt else {}
    ae_state = torch.load(ae_ckpt_path, map_location=device) if os.path.exists(ae_ckpt_path) else {}

    # ---- Dimensions from config (all sourced from tts.json) ----
    vocab_size = cfg["vocab_size"]
    compressed_channels = cfg["compressed_channels"]
    latent_dim = cfg["latent_dim"]
    chunk_compress_factor = cfg["chunk_compress_factor"]
    te_d_model = cfg["te_d_model"]
    se_d_model = cfg["se_d_model"]
    se_n_style = cfg["se_n_style"]
    print(f"[INFO] vocab_size={vocab_size}, compressed_channels={compressed_channels}, "
          f"latent_dim={latent_dim}, chunk_compress_factor={chunk_compress_factor}")
    print(f"[INFO] te_d_model={te_d_model}, se_d_model={se_d_model}, se_n_style={se_n_style}")

    # 1. Reference Encoder
    print("Loading ReferenceEncoder...")
    ref_enc = ReferenceEncoder(
        in_channels=compressed_channels,
        d_model=se_d_model,
        hidden_dim=cfg["se_hidden_dim"],
        num_blocks=cfg["se_num_blocks"],
        num_tokens=se_n_style,
        num_heads=cfg["se_n_heads"],
    ).to(device).eval()
    if "reference_encoder" in t2l_state:
        ref_enc.load_state_dict(t2l_state["reference_encoder"], strict=True)
    else:
        print("[WARN] reference_encoder not found in checkpoint!")

    # Replace nn.MultiheadAttention with ONNX-safe manual attention.
    # nn.MHA has known ONNX tracing bugs (batch_first, key_padding_mask,
    # cross-attention with different kdim/vdim).
    print("[FIX] Replacing nn.MultiheadAttention in ReferenceEncoder...")
    _replace_mha_with_safe(ref_enc)

    # 2. Text Encoder
    print("Loading TextEncoder...")
    text_enc = TextEncoder(
        vocab_size=vocab_size,
        d_model=te_d_model,
        n_conv_layers=cfg["te_convnext_layers"],
        n_attn_layers=cfg["te_attn_n_layers"],
        expansion_factor=cfg["te_expansion_factor"],
        p_dropout=cfg["te_attn_p_dropout"],
    ).to(device).eval()
    if "text_encoder" in t2l_state:
        text_enc.load_state_dict(t2l_state["text_encoder"], strict=True)

    # 3. Vector Field Estimator
    print("Loading VectorFieldEstimator...")
    vf = VectorFieldEstimator(
        in_channels=compressed_channels,
        out_channels=compressed_channels,
        hidden_channels=cfg["vf_hidden"],
        text_dim=cfg["vf_text_dim"],
        style_dim=cfg["vf_style_dim"],
        num_style_tokens=se_n_style,
        num_superblocks=cfg["vf_n_blocks"],
        time_embed_dim=cfg["vf_time_dim"],
        rope_gamma=cfg["vf_rotary_scale"],
    ).to(device).eval()
    if "vf_estimator" in t2l_state:
        missing, unexpected = vf.load_state_dict(t2l_state["vf_estimator"], strict=False)
        if missing:
            print(f"[WARN] VF missing keys: {missing}")
        if unexpected:
            print(f"[WARN] VF unexpected keys: {unexpected}")

    # 4. Vocoder (Latent Decoder)
    print("Loading Vocoder...")
    ae_dec_cfg = cfg["ae_dec_cfg"]
    vocoder = LatentDecoder1D(cfg=ae_dec_cfg).to(device).eval()
    if "decoder" in ae_state:
        vocoder.load_state_dict(ae_state["decoder"], strict=True)

    # 5. Duration Predictor (DPNetwork)
    print("Loading DurationPredictor...")
    dp_style_tokens = cfg["dp_style_tokens"]
    dp_style_dim = cfg["dp_style_dim"]
    dp = DPNetwork(
        vocab_size=cfg["dp_vocab_size"],
        style_tokens=dp_style_tokens,
        style_dim=dp_style_dim,
    ).to(device).eval()
    
    dp_ckpt_path = args.dp_ckpt
    if os.path.exists(dp_ckpt_path):
        print(f"[INFO] Loading Duration Predictor from {dp_ckpt_path}")
        dp_state = torch.load(dp_ckpt_path, map_location=device)
        if isinstance(dp_state, dict) and "state_dict" in dp_state:
            dp_state = dp_state["state_dict"]
        dp.load_state_dict(dp_state, strict=False)
    elif "dp_network" in t2l_state:
        print("[INFO] Loading Duration Predictor from text2latent checkpoint (dp_network)")
        dp.load_state_dict(t2l_state["dp_network"], strict=True)
    elif "dp_model" in t2l_state:
        print("[INFO] Loading Duration Predictor from text2latent checkpoint (dp_model)")
        dp.load_state_dict(t2l_state["dp_model"], strict=True)
    else:
        print("[WARN] No Duration Predictor weights found! Exporting random initialization.")

    # Replace nn.MultiheadAttention in DP's reference encoder too
    print("[FIX] Replacing nn.MultiheadAttention in DPNetwork...")
    _replace_mha_with_safe(dp)
    
    # ---- Dummy Inputs (derived from config) ----
    B = 1
    T_text = 32
    T_ref = se_n_style  # matches ReferenceEncoder num_tokens (output)
    T_audio_ref = 256    # Fixed for consistent benchmarking
    T_lat = 100
    C_lat = compressed_channels
    C_dec = latent_dim
    style_dim = se_d_model

    text_ids  = torch.zeros(B, T_text, dtype=torch.long, device=device)
    text_mask = torch.ones(B, 1, T_text, dtype=torch.float32, device=device)

    # Reference Encoder Inputs
    z_ref     = torch.randn(B, C_lat, T_audio_ref, dtype=torch.float32, device=device)
    ref_mask  = torch.ones(B, 1, T_audio_ref, dtype=torch.float32, device=device)

    do_verify = not getattr(args, 'no_verify', False)
    all_pass = True

    # ---------------- Reference Encoder ----------------
    print("Exporting ReferenceEncoder...")
    ref_enc_path = os.path.join(onnx_dir, "reference_encoder.onnx")
    export_one(
        ref_enc,
        ref_enc_path,
        (z_ref, ref_mask),
        input_names=["z_ref", "mask"],
        output_names=["ref_values", "ref_keys"],
        dynamic_axes={
            "z_ref": {2: "T_ref_in"},
            "mask": {2: "T_ref_in"},
        }
    )
    if do_verify:
        print("Verifying ReferenceEncoder...")
        ok = verify_onnx(ref_enc, ref_enc_path, (z_ref, ref_mask),
                         ["z_ref", "mask"], label="ReferenceEncoder")
        all_pass = all_pass and ok
        # Also verify with a different time dimension to test dynamic axes
        z_ref_short = torch.randn(B, C_lat, 64, dtype=torch.float32, device=device)
        ref_mask_short = torch.ones(B, 1, 64, dtype=torch.float32, device=device)
        ok2 = verify_onnx(ref_enc, ref_enc_path, (z_ref_short, ref_mask_short),
                          ["z_ref", "mask"], label="ReferenceEncoder (T=64)")
        all_pass = all_pass and ok2

    # ---------------- Text Encoder ----------------
    print("Exporting TextEncoder...")
    # Matches ONNX 3-input signature: (text_ids, style_ttl, text_mask)
    # style_ttl: [B, 50, 256] - style values from reference encoder
    style_ttl = torch.randn(B, 50, style_dim, dtype=torch.float32, device=device)

    # TextEncoder now matches ONNX signature directly
    te_path = os.path.join(onnx_dir, "text_encoder.onnx")
    export_one(
        text_enc,
        te_path,
        (text_ids, style_ttl, text_mask),
        input_names=["text_ids", "style_ttl", "text_mask"],
        output_names=["text_emb"],
        dynamic_axes={
            "text_ids": {1: "T_text"},
            "style_ttl": {1: "T_ref"},
            "text_mask": {2: "T_text"},
            "text_emb": {2: "T_text"},
        }
    )
    if do_verify:
        print("Verifying TextEncoder...")
        ok = verify_onnx(text_enc, te_path, (text_ids, style_ttl, text_mask),
                         ["text_ids", "style_ttl", "text_mask"], label="TextEncoder")
        all_pass = all_pass and ok

    # ---------------- Vector Field Estimator ----------------
    print("Exporting VectorFieldEstimator...")
    noisy_latent = torch.randn(B, C_lat, T_lat, dtype=torch.float32, device=device)
    latent_mask  = torch.ones(B, 1, T_lat, dtype=torch.float32, device=device)
    text_emb     = torch.randn(B, style_dim, T_text, dtype=torch.float32, device=device)
    # VF expects style_ttl as [B, T, C] or [B, C, T] - model handles it.
    # We use [B, 50, 256] to match Ref output.
    style_ttl    = torch.randn(B, 50, style_dim, dtype=torch.float32, device=device)

    current_step = torch.tensor([0.0], dtype=torch.float32, device=device)
    total_step   = torch.tensor([1.0], dtype=torch.float32, device=device)
    
    # ---- 1-to-1 ONNX parity ----
    # Tie VF's baked-in style key to TextEncoder's baked-in style_key so the
    # exported ONNX constant matches the rest of the stack.
    with torch.no_grad():
        # TextEncoder stores the baked-in style key at:
        #   text_enc.speech_prompted_text_encoder.style_key  (shape [1, 50, 256])
        vf.style_key.copy_(text_enc.speech_prompted_text_encoder.style_key)

    # Wrap VF to expose the EXACT ONNX signature (no style_keys input).
    vf_wrapped = VectorFieldEstimatorWrapper(vf)

    vf_path = os.path.join(onnx_dir, "vector_estimator.onnx")
    vf_inputs = (noisy_latent, text_emb, style_ttl, latent_mask, text_mask, current_step, total_step)
    vf_input_names = [
        "noisy_latent", "text_emb", "style_ttl",
        "latent_mask", "text_mask", "current_step", "total_step"
    ]
    export_one(
        vf_wrapped,
        vf_path,
        vf_inputs,
        input_names=vf_input_names,
        output_names=["denoised_latent"],
        dynamic_axes={
            "noisy_latent": {2: "T_lat"},
            "text_emb": {2: "T_text"},
            "style_ttl": {1: "T_ref"},
            "latent_mask": {2: "T_lat"},
            "text_mask": {2: "T_text"},
            "denoised_latent": {2: "T_lat"},
        }
    )
    if do_verify:
        print("Verifying VectorFieldEstimator...")
        ok = verify_onnx(vf_wrapped, vf_path, vf_inputs,
                         vf_input_names, label="VF", atol=1e-3, rtol=1e-2)
        all_pass = all_pass and ok

    # ---------------- Vocoder ----------------
    print("Exporting Vocoder...")
    # Decoder expects [B, latent_dim, T]. The compressed latent is decompressed
    # from [B, C_lat, T] -> [B, latent_dim, T * chunk_compress_factor].
    latent_dec = torch.randn(B, C_dec, T_lat * chunk_compress_factor, dtype=torch.float32, device=device)
    voc_path = os.path.join(onnx_dir, "vocoder.onnx")
    export_one(
        vocoder,
        voc_path,
        (latent_dec,),
        input_names=["latent"],
        output_names=["waveform"],
        dynamic_axes={
            "latent": {2: "T_dec"},
            "waveform": {2: "T_wav"},
        }
    )
    if do_verify:
        print("Verifying Vocoder...")
        ok = verify_onnx(vocoder, voc_path, (latent_dec,),
                         ["latent"], label="Vocoder")
        all_pass = all_pass and ok

    # ---------------- Duration Predictor (Standard: z_ref path) ----------------
    print("Exporting DurationPredictor (Standard)...")
    # DPNetwork: forward(text_ids, z_ref, text_mask, ref_mask)
    dp_path = os.path.join(onnx_dir, "duration_predictor.onnx")
    dp_inputs = (text_ids, z_ref, text_mask, ref_mask)
    dp_input_names = ["text_ids", "z_ref", "text_mask", "ref_mask"]
    export_one(
        dp,
        dp_path,
        dp_inputs,
        input_names=dp_input_names,
        output_names=["duration"],
        dynamic_axes={
            "text_ids": {1: "T_text"},
            "text_mask": {2: "T_text"},
            "z_ref": {2: "T_ref_audio"},
            "ref_mask": {2: "T_ref_audio"},
        }
    )
    if do_verify:
        print("Verifying DurationPredictor (Standard)...")
        ok = verify_onnx(dp, dp_path, dp_inputs,
                         dp_input_names, label="DurationPredictor")
        all_pass = all_pass and ok

    # ---------------- Duration Predictor (Style: pre-computed style_dp path) ----
    print("Exporting DurationPredictor (Style)...")

    class DPStyleWrapper(nn.Module):
        """Wrap DPNetwork for the style_tokens input path (no z_ref)."""
        def __init__(self, dp_model):
            super().__init__()
            self.dp = dp_model

        def forward(self, text_ids, style_dp, text_mask):
            return self.dp(text_ids, text_mask=text_mask, style_tokens=style_dp)

    dp_style_wrapper = DPStyleWrapper(dp).eval()
    style_dp_dummy = torch.randn(B, dp_style_tokens, dp_style_dim, dtype=torch.float32, device=device)

    dp_style_path = os.path.join(onnx_dir, "duration_predictor_style.onnx")
    dp_style_inputs = (text_ids, style_dp_dummy, text_mask)
    dp_style_names = ["text_ids", "style_dp", "text_mask"]
    export_one(
        dp_style_wrapper,
        dp_style_path,
        dp_style_inputs,
        input_names=dp_style_names,
        output_names=["duration"],
        dynamic_axes={
            "text_ids": {1: "T_text"},
            "text_mask": {2: "T_text"},
        }
    )
    if do_verify:
        print("Verifying DurationPredictor (Style)...")
        ok = verify_onnx(dp_style_wrapper, dp_style_path, dp_style_inputs,
                         dp_style_names, label="DurationPredictor (Style)")
        all_pass = all_pass and ok

    # ---------------- Unconditional Tokens ----------------
    print("Exporting uncond.npz...")
    uncond_data = {}
    if "u_text" in t2l_state:
        uncond_data["u_text"] = t2l_state["u_text"].cpu().numpy()
    if "u_ref" in t2l_state:
        uncond_data["u_ref"] = t2l_state["u_ref"].cpu().numpy()
    
    if uncond_data:
        np.savez(os.path.join(onnx_dir, "uncond.npz"), **uncond_data)
        print(f"[OK] Saved {os.path.join(onnx_dir, 'uncond.npz')}")
    else:
        print("[WARN] No unconditional tokens found in checkpoint.")

    # ---- Summary ----
    if do_verify:
        print("\n" + "="*60)
        if all_pass:
            print("[PASS] All ONNX models match PyTorch output within tolerance!")
        else:
            print("[FAIL] Some ONNX models do NOT match PyTorch. Check logs above.")
        print("="*60)

if __name__ == "__main__":
    main()
