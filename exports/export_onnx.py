#!/usr/bin/env python3
"""Export the 4 self-contained ONNX models.

Produces (in --onnx_dir):
  - text_encoder.onnx
  - vector_estimator.onnx   (u_text / u_ref baked in; CFG done internally)
  - vocoder.onnx            (mean/std/normalizer_scale baked in; unnorm+reshape in-graph)
  - duration_predictor.onnx (style-conditioned: takes style_dp from the voice JSON)
"""
import argparse
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import onnx
import onnxruntime as ort
import onnxslim
from onnxruntime.quantization import quantize_dynamic, QuantType
import torch
import torch.nn as nn
import torch.nn.functional as F

from bluecodec import LatentDecoder1D
from training.dp.models.dp_network import DPNetwork
from training.t2l.models.text_encoder import TextEncoder
from training.t2l.models.vf_estimator import VectorFieldEstimator
from training.t2l.models.reference_encoder import ReferenceEncoder
from training.utils import load_ttl_config


# ── checkpoint helpers ────────────────────────────────────────────────────────

def _load_state(path, device="cpu"):
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path, device=device)
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    return state


def _submodule(state, key):
    if key in state and isinstance(state[key], dict):
        return state[key]
    head = f"{key}."
    sub = {k[len(head):]: v for k, v in state.items() if k.startswith(head)}
    return sub or None


def _load_into(module, state, key):
    sub = _submodule(state, key)
    if sub is None:
        print(f"[WARN] '{key}' not found in checkpoint; random init.")
        return
    module.load_state_dict(sub, strict=False)


# ── ONNX-safe MHA (avoids torch.nn.MultiheadAttention's dynamo-unfriendly ops) ─

class OnnxSafeMultiheadAttention(nn.Module):
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
            self.q_proj_weight = self.k_proj_weight = self.v_proj_weight = None
        else:
            self.in_proj_weight = None
            self.q_proj_weight = nn.Parameter(torch.empty(embed_dim, embed_dim))
            self.k_proj_weight = nn.Parameter(torch.empty(embed_dim, self.kdim))
            self.v_proj_weight = nn.Parameter(torch.empty(embed_dim, self.vdim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        if not self.batch_first:
            query, key, value = (t.transpose(0, 1) for t in (query, key, value))
        B, T_q, _ = query.shape
        T_k = key.shape[1]

        bq = self.in_proj_bias[: self.embed_dim]
        bk = self.in_proj_bias[self.embed_dim : 2 * self.embed_dim]
        bv = self.in_proj_bias[2 * self.embed_dim :]
        if self._qkv_same_embed_dim:
            w = self.in_proj_weight
            q = F.linear(query, w[: self.embed_dim], bq)
            k = F.linear(key, w[self.embed_dim : 2 * self.embed_dim], bk)
            v = F.linear(value, w[2 * self.embed_dim :], bv)
        else:
            q = F.linear(query, self.q_proj_weight, bq)
            k = F.linear(key, self.k_proj_weight, bk)
            v = F.linear(value, self.v_proj_weight, bv)

        q = q.view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T_q, self.embed_dim)
        out = self.out_proj(out)
        if not self.batch_first:
            out = out.transpose(0, 1)
        return out, None


def _replace_mha(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.MultiheadAttention):
            safe = OnnxSafeMultiheadAttention(
                child.embed_dim, child.num_heads,
                kdim=child.kdim, vdim=child.vdim, batch_first=child.batch_first,
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
            setattr(module, name, safe)
        else:
            _replace_mha(child)


# ── graph wrappers that bake runtime conditioning into the ONNX graph ─────────

class VectorFieldEstimatorCFG(nn.Module):
    """Wraps VF and bakes u_text / u_ref so CFG is one ONNX call per diffusion step."""

    def __init__(self, model: VectorFieldEstimator, u_text: torch.Tensor, u_ref: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("u_text", u_text.detach().clone())
        self.register_buffer("u_ref", u_ref.detach().clone())

    def forward(self, noisy_latent, text_emb, style_ttl, latent_mask, text_mask,
                current_step, total_step, cfg_scale):
        B = noisy_latent.shape[0]
        T_text = text_emb.shape[2]
        u_text_b = self.u_text.expand(B, -1, T_text)
        u_ref_b = self.u_ref.expand(B, -1, -1)
        u_mask = torch.ones_like(text_mask)

        den_cond = self.model(noisy_latent=noisy_latent, text_emb=text_emb, style_ttl=style_ttl,
                              latent_mask=latent_mask, text_mask=text_mask,
                              current_step=current_step, total_step=total_step)
        den_uncond = self.model(noisy_latent=noisy_latent, text_emb=u_text_b, style_ttl=u_ref_b,
                                latent_mask=latent_mask, text_mask=u_mask,
                                current_step=current_step, total_step=total_step)
        return den_uncond + cfg_scale * (den_cond - den_uncond)


class VocoderWithStats(nn.Module):
    """Bakes mean/std/normalizer_scale and the (B,C,T) -> (B,latent_dim,T*factor) reshape."""

    def __init__(self, vocoder, mean, std, normalizer_scale, latent_dim, chunk_compress_factor):
        super().__init__()
        self.vocoder = vocoder
        self.register_buffer("mean", mean.to(torch.float32).view(1, -1, 1))
        self.register_buffer("std", std.to(torch.float32).view(1, -1, 1))
        ns = float(normalizer_scale) or 1.0
        self.register_buffer("inv_scale", torch.tensor(1.0 / ns, dtype=torch.float32))
        self.latent_dim = int(latent_dim)
        self.factor = int(chunk_compress_factor)

    def forward(self, latent):
        z = (latent * self.inv_scale) * self.std + self.mean
        B, _, T = z.shape
        z = z.reshape(B, self.latent_dim, self.factor, T).permute(0, 1, 3, 2).reshape(
            B, self.latent_dim, T * self.factor
        )
        return self.vocoder(z)


class DPStyleWrapper(nn.Module):
    def __init__(self, dp):
        super().__init__()
        self.dp = dp

    def forward(self, text_ids, style_dp, text_mask):
        return self.dp(text_ids=text_ids, text_mask=text_mask, style_dp=style_dp)


# ── export + verify ──────────────────────────────────────────────────────────

def export_one(model, out_path, inputs, input_names, output_names, dynamic_axes,
               do_slim=False, do_int8=False):
    model.eval()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model, inputs, out_path,
            opset_version=17, do_constant_folding=True,
            input_names=input_names, output_names=output_names,
            dynamic_axes=dynamic_axes, dynamo=False,
        )
    if do_slim:
        slimmed = onnxslim.slim(onnx.load(out_path))
        if slimmed is not None:
            onnx.save(slimmed, out_path)
    if do_int8:
        # Per-tensor weight-only QUInt8 (per-channel trips ORT's MatMulInteger zero-point check).
        tmp = out_path + ".q"
        quantize_dynamic(
            model_input=out_path, model_output=tmp,
            per_channel=False, reduce_range=False, weight_type=QuantType.QUInt8,
        )
        os.replace(tmp, out_path)
    tag = "int8" if do_int8 else ("slim" if do_slim else "fp32")
    print(f"[OK] [{tag}] {out_path}")


def verify(model, onnx_path, inputs, input_names, label, atol=1e-3, rtol=1e-2):
    with torch.no_grad():
        pt_out = model(*inputs)
        if isinstance(pt_out, tuple):
            pt_out = pt_out[0]
        pt = pt_out.cpu().numpy()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    feed = {n: t.detach().cpu().numpy() for n, t in zip(input_names, inputs)}
    ort_out = sess.run(None, feed)[0]
    max_diff = float(np.max(np.abs(pt - ort_out)))
    cos = float(np.dot(pt.flatten(), ort_out.flatten()) /
                (np.linalg.norm(pt) * np.linalg.norm(ort_out) + 1e-12))
    ok = np.allclose(pt, ort_out, atol=atol, rtol=rtol)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: max_diff={max_diff:.6f}, cos_sim={cos:.6f}")
    return ok


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/tts.json")
    p.add_argument("--onnx_dir", default="onnx_models")
    p.add_argument("--ttl_ckpt", required=True, help="text2latent checkpoint (.pt / .safetensors)")
    p.add_argument("--ae_ckpt", required=True, help="audio codec checkpoint")
    p.add_argument("--dp_ckpt", required=True, help="duration predictor checkpoint")
    p.add_argument("--stats", required=True, help="normalization stats .pt (mean/std)")
    p.add_argument("--slim", action="store_true", help="run onnxslim after export")
    p.add_argument("--int8", action="store_true", help="weight-only QUInt8 quantization")
    p.add_argument("--no-verify", dest="no_verify", action="store_true")
    args = p.parse_args()

    cfg = load_ttl_config(args.config)
    print(f"[INFO] config: {args.config} (v{cfg['full_config'].get('tts_version', '?')})")
    os.makedirs(args.onnx_dir, exist_ok=True)

    # ---- load checkpoints ---------------------------------------------------
    t2l = _load_state(args.ttl_ckpt)
    ae = _load_state(args.ae_ckpt)
    dp_state = _load_state(args.dp_ckpt)

    u_text, u_ref = t2l.get("u_text"), t2l.get("u_ref")
    if u_text is None or u_ref is None:
        raise RuntimeError(f"u_text/u_ref missing in {args.ttl_ckpt}; re-export with uncond params.")
    print(f"[INFO] baking uncond: u_text {tuple(u_text.shape)}, u_ref {tuple(u_ref.shape)}")

    stats = torch.load(args.stats, map_location="cpu", weights_only=False)
    mean = stats["mean"].to(torch.float32)
    std = stats["std"].to(torch.float32)

    # ---- shared dims --------------------------------------------------------
    vocab_size = cfg["vocab_size"]
    dp_vocab_size = cfg["dp_vocab_size"]
    compressed = cfg["compressed_channels"]
    latent_dim = cfg["latent_dim"]
    ccf = cfg["chunk_compress_factor"]
    d_model = cfg["te_d_model"]
    n_style = cfg["se_n_style"]

    # ---- build PyTorch modules ---------------------------------------------
    text_enc = TextEncoder(
        vocab_size=vocab_size, d_model=d_model,
        n_conv_layers=cfg["te_convnext_layers"],
        n_attn_layers=cfg["te_attn_n_layers"],
        expansion_factor=cfg["te_expansion_factor"],
        p_dropout=cfg["te_attn_p_dropout"],
    ).eval()
    _load_into(text_enc, t2l, "text_encoder")

    ref_enc = ReferenceEncoder(
        in_channels=compressed, d_model=cfg["se_d_model"],
        hidden_dim=cfg.get("re_hidden_dim", 1024),
        num_blocks=cfg.get("re_n_blocks", 6),
        num_tokens=n_style,
        num_heads=cfg.get("re_n_heads", 2),
        kernel_size=cfg.get("re_kernel_size", 5),
    ).eval()
    _load_into(ref_enc, t2l, "reference_encoder")
    _replace_mha(ref_enc)

    vf = VectorFieldEstimator(
        in_channels=compressed, out_channels=compressed,
        hidden_channels=cfg["vf_hidden"],
        text_dim=cfg["vf_text_dim"], style_dim=cfg["vf_style_dim"],
        num_style_tokens=n_style, num_superblocks=cfg["vf_n_blocks"],
        time_embed_dim=cfg["vf_time_dim"], rope_gamma=cfg["vf_rotary_scale"],
    ).eval()
    _load_into(vf, t2l, "vf_estimator")

    vocoder = LatentDecoder1D(cfg=cfg["ae_dec_cfg"]).eval()
    _load_into(vocoder, ae, "decoder")

    dp = DPNetwork(
        vocab_size=dp_vocab_size,
        style_dp=cfg["dp_style_tokens"],
        style_dim=cfg["dp_style_dim"],
    ).eval()

    # DP checkpoints may have been trained with a larger char-embedding table;
    # keep the first `vocab_size` rows (IDs beyond are unused at inference).
    emb_key = "sentence_encoder.text_embedder.char_embedder.weight"
    if emb_key in dp_state:
        target = dp.state_dict()[emb_key].shape[0]
        have = dp_state[emb_key].shape[0]
        if have > target:
            print(f"[INFO] slicing DP char_embedder {have} -> {target} rows")
            dp_state[emb_key] = dp_state[emb_key][:target].clone()
        elif have < target:
            raise RuntimeError(f"DP checkpoint vocab ({have}) < model vocab ({target}).")
    dp.load_state_dict(dp_state, strict=False)
    _replace_mha(dp)

    # ---- dummy inputs -------------------------------------------------------
    B, T_text, T_lat = 1, 32, 100
    text_ids = torch.zeros(B, T_text, dtype=torch.long)
    text_mask = torch.ones(B, 1, T_text)
    style_ttl = torch.randn(B, n_style, d_model)
    latent_mask = torch.ones(B, 1, T_lat)
    noisy_latent = torch.randn(B, compressed, T_lat)
    text_emb = torch.randn(B, d_model, T_text)
    cur_step = torch.tensor([0.0])
    tot_step = torch.tensor([1.0])
    cfg_scale = torch.tensor([3.0])
    style_dp_s = torch.randn(B, cfg["dp_style_tokens"], cfg["dp_style_dim"])
    z_pred = torch.randn(B, compressed, T_lat)

    do_verify = not args.no_verify and not args.int8
    if args.int8 and not args.no_verify:
        print("[INFO] --int8 is lossy; skipping numerical verify.")
    all_ok = True

    # ---- 1) text encoder ---------------------------------------------------
    te_path = os.path.join(args.onnx_dir, "text_encoder.onnx")
    te_inputs = (text_ids, style_ttl, text_mask)
    te_names = ["text_ids", "style_ttl", "text_mask"]
    export_one(text_enc, te_path, te_inputs, te_names, ["text_emb"], {
        "text_ids": {1: "T_text"}, "style_ttl": {1: "T_ref"},
        "text_mask": {2: "T_text"}, "text_emb": {2: "T_text"},
    }, do_slim=args.slim, do_int8=args.int8)
    if do_verify:
        all_ok &= verify(text_enc, te_path, te_inputs, te_names, "text_encoder", atol=1e-4, rtol=1e-3)

    # ---- 2) vector field estimator (CFG baked in) --------------------------
    vf_wrapped = VectorFieldEstimatorCFG(vf, u_text, u_ref).eval()
    vf_path = os.path.join(args.onnx_dir, "vector_estimator.onnx")
    vf_inputs = (noisy_latent, text_emb, style_ttl, latent_mask, text_mask, cur_step, tot_step, cfg_scale)
    vf_names = ["noisy_latent", "text_emb", "style_ttl", "latent_mask", "text_mask",
                "current_step", "total_step", "cfg_scale"]
    export_one(vf_wrapped, vf_path, vf_inputs, vf_names, ["denoised_latent"], {
        "noisy_latent": {2: "T_lat"}, "text_emb": {2: "T_text"}, "style_ttl": {1: "T_ref"},
        "latent_mask": {2: "T_lat"}, "text_mask": {2: "T_text"},
        "denoised_latent": {2: "T_lat"},
    }, do_slim=args.slim, do_int8=args.int8)
    if do_verify:
        all_ok &= verify(vf_wrapped, vf_path, vf_inputs, vf_names, "vector_estimator")

    # ---- 3) vocoder (stats+reshape baked in) -------------------------------
    voc_wrapped = VocoderWithStats(
        vocoder, mean, std, cfg.get("normalizer_scale", 1.0),
        latent_dim=latent_dim, chunk_compress_factor=ccf,
    ).eval()
    voc_path = os.path.join(args.onnx_dir, "vocoder.onnx")
    export_one(voc_wrapped, voc_path, (z_pred,), ["latent"], ["waveform"], {
        "latent": {2: "T_lat"}, "waveform": {2: "T_wav"},
    }, do_slim=args.slim, do_int8=args.int8)
    if do_verify:
        all_ok &= verify(voc_wrapped, voc_path, (z_pred,), ["latent"], "vocoder", atol=1e-4, rtol=1e-3)

    # ---- 4) duration predictor (style-conditioned) -------------------------
    dp_wrapped = DPStyleWrapper(dp).eval()
    dp_path = os.path.join(args.onnx_dir, "duration_predictor.onnx")
    dp_inputs = (text_ids, style_dp_s, text_mask)
    dp_names = ["text_ids", "style_dp", "text_mask"]
    export_one(dp_wrapped, dp_path, dp_inputs, dp_names, ["duration"], {
        "text_ids": {1: "T_text"}, "text_mask": {2: "T_text"},
    }, do_slim=args.slim, do_int8=args.int8)
    if do_verify:
        all_ok &= verify(dp_wrapped, dp_path, dp_inputs, dp_names, "duration_predictor", atol=1e-4, rtol=1e-3)

    if do_verify:
        print("\n" + ("[PASS] All match." if all_ok else "[FAIL] Mismatch."))


if __name__ == "__main__":
    main()
