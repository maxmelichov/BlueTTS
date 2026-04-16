#!/usr/bin/env python3
"""Export all TTS models to ONNX (FP32, optionally slimmed).

Supports:
- Flat *.safetensors checkpoints from pt_weights/ (vf_estimator, blue_codec, duration_predictor).
- Legacy nested *.pt checkpoints (training dumps with `text_encoder`, `reference_encoder`, etc. sub-dicts).
- Optional onnxslim pass (--slim).
"""
import argparse
import glob
import os
from functools import partial

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxslim
import torch
import torch.nn as nn
import torch.nn.functional as F

from bluecodec import LatentDecoder1D
from models.text2latent.dp_network import DPNetwork
from models.text2latent.text_encoder import TextEncoder
from models.text2latent.vf_estimator import VectorFieldEstimator
from models.text2latent.reference_encoder import ReferenceEncoder
from models.utils import load_ttl_config


def _load_state_any(path, device="cpu"):
    if not path or not os.path.exists(path):
        return {}
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path, device=device)
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    return state


def _extract_submodule(state, key):
    if not isinstance(state, dict) or not state:
        return None
    if key in state and isinstance(state[key], dict):
        return state[key]
    head = f"{key}."
    sub = {k[len(head):]: v for k, v in state.items() if k.startswith(head)}
    return sub or None


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
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        B, T_q, _ = query.shape
        T_k = key.shape[1]

        bias_q = self.in_proj_bias[: self.embed_dim]
        bias_k = self.in_proj_bias[self.embed_dim : 2 * self.embed_dim]
        bias_v = self.in_proj_bias[2 * self.embed_dim :]
        if self._qkv_same_embed_dim:
            w = self.in_proj_weight
            q = F.linear(query, w[: self.embed_dim], bias_q)
            k = F.linear(key, w[self.embed_dim : 2 * self.embed_dim], bias_k)
            v = F.linear(value, w[2 * self.embed_dim :], bias_v)
        else:
            q = F.linear(query, self.q_proj_weight, bias_q)
            k = F.linear(key, self.k_proj_weight, bias_k)
            v = F.linear(value, self.v_proj_weight, bias_v)

        q = q.view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(mask, float("-inf"))

        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T_q, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        return attn_output, None


def _copy_mha_weights(src: nn.MultiheadAttention, dst: OnnxSafeMultiheadAttention) -> None:
    with torch.no_grad():
        if src._qkv_same_embed_dim:
            dst.in_proj_weight.copy_(src.in_proj_weight)
        else:
            dst.q_proj_weight.copy_(src.q_proj_weight)
            dst.k_proj_weight.copy_(src.k_proj_weight)
            dst.v_proj_weight.copy_(src.v_proj_weight)
        dst.in_proj_bias.copy_(src.in_proj_bias)
        dst.out_proj.weight.copy_(src.out_proj.weight)
        dst.out_proj.bias.copy_(src.out_proj.bias)


def _to_safe(child: nn.MultiheadAttention) -> OnnxSafeMultiheadAttention:
    safe = OnnxSafeMultiheadAttention(
        embed_dim=child.embed_dim,
        num_heads=child.num_heads,
        kdim=child.kdim,
        vdim=child.vdim,
        batch_first=child.batch_first,
    )
    _copy_mha_weights(child, safe)
    return safe


def _replace_mha_with_safe(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.MultiheadAttention):
            setattr(module, name, _to_safe(child))
        else:
            _replace_mha_with_safe(child)

    if isinstance(module, nn.ModuleList):
        for i, child in enumerate(module):
            if isinstance(child, nn.MultiheadAttention):
                module[i] = _to_safe(child)

    if isinstance(module, nn.ModuleDict):
        for key, child in module.items():
            if isinstance(child, nn.MultiheadAttention):
                module[key] = _to_safe(child)


class VectorFieldEstimatorWrapper(nn.Module):
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
            style_keys=None,
        )


def export_one(
    model,
    out_path,
    inputs,
    input_names,
    output_names,
    dynamic_axes,
    *,
    do_slim=False,
    do_int8=False,
):
    model.eval()
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    work = f"{out_path}.~work.onnx"
    qtmp = f"{out_path}.~q.onnx"
    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                inputs,
                work,
                opset_version=17,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                dynamo=False,
            )

        if do_slim:
            loaded = onnx.load(work)
            slimmed = onnxslim.slim(loaded)
            if slimmed is not None:
                onnx.save(slimmed, work)

        if do_int8:
            # Per-tensor weight-only QUInt8. Per-channel produces zero-point tensor shapes
            # that ORT's MatMulInteger kernel rejects on some attention MatMuls
            # (/.../attention*/MatMul_quant: "B zero point is not valid"). Per-tensor avoids
            # that; quality on Conv layers is fine because the vocoder uses Conv1d, which ORT
            # quantizes via QLinearConv (separate code path, no zero-point validation issue).
            quantize_dynamic(
                model_input=work,
                model_output=qtmp,
                per_channel=False,
                reduce_range=False,
                weight_type=QuantType.QUInt8,
            )
            if os.path.isfile(work):
                os.remove(work)
            if os.path.isfile(out_path):
                os.remove(out_path)
            os.replace(qtmp, out_path)
        else:
            if os.path.isfile(out_path):
                os.remove(out_path)
            os.replace(work, out_path)

        tag = "int8" if do_int8 else ("slim" if do_slim else "fp32")
        print(f"[OK] [{tag}] {out_path}")
    finally:
        for p in (work, qtmp):
            if os.path.isfile(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


def verify_onnx(model, onnx_path, inputs, input_names, label="model", atol=1e-4, rtol=1e-3):
    model.eval()
    with torch.no_grad():
        pt_inputs = (inputs,) if isinstance(inputs, torch.Tensor) else inputs
        pt_out = model(*pt_inputs)
        if isinstance(pt_out, tuple):
            pt_out = pt_out[0]
        pt_np = pt_out.cpu().numpy()

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    feed = {}
    if isinstance(inputs, torch.Tensor):
        inputs = (inputs,)
    for name, tensor in zip(input_names, inputs):
        feed[name] = tensor.detach().cpu().numpy()
    onnx_out = sess.run(None, feed)[0]

    pt_flat = pt_np.flatten()
    ort_flat = np.asarray(onnx_out).flatten()
    max_diff = float(np.max(np.abs(pt_flat - ort_flat)))
    mean_diff = float(np.mean(np.abs(pt_flat - ort_flat)))
    denom = float(np.linalg.norm(pt_flat) * np.linalg.norm(ort_flat) + 1e-12)
    cos_sim = float(np.dot(pt_flat, ort_flat) / denom)
    match = np.allclose(pt_np, onnx_out, atol=atol, rtol=rtol)
    status = "PASS" if match else "FAIL"
    print(f"  [{status}] {label}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, cos_sim={cos_sim:.6f}")
    return match


def _maybe_load(module, state, key, strict=True):
    sub = _extract_submodule(state, key)
    if sub is None:
        print(f"[WARN] '{key}' not found in checkpoint; using random init.")
        return
    missing, unexpected = module.load_state_dict(sub, strict=False)
    if strict and (missing or unexpected):
        print(f"[INFO] '{key}': missing={len(missing)}, unexpected={len(unexpected)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/tts.json", help="Path to tts.json config")
    parser.add_argument("--onnx_dir", type=str, default="onnx_models", help="Output directory for ONNX models")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/text2latent", help="Text2Latent checkpoint dir (fallback)")
    parser.add_argument("--ttl_ckpt", type=str, default="pt_weights/vf_estimator.safetensors",
                        help="Combined text2latent checkpoint (.pt or .safetensors)")
    parser.add_argument("--ae_ckpt", type=str, default="pt_weights/blue_codec.safetensors",
                        help="Audio codec checkpoint (.pt or .safetensors)")
    parser.add_argument("--dp_ckpt", type=str, default="pt_weights/duration_predictor.safetensors",
                        help="Duration predictor checkpoint (.pt or .safetensors)")
    parser.add_argument("--no-verify", dest="no_verify", action="store_true", help="Skip ONNX vs PyTorch verification")
    parser.add_argument("--slim", action="store_true", help="Run onnxslim on each model before finalizing")
    parser.add_argument("--int8", action="store_true",
                        help="Per-channel symmetric QInt8 weight-only quantization (after slim if --slim).")
    args = parser.parse_args()

    device = "cpu"
    export = partial(export_one, do_slim=args.slim, do_int8=args.int8)

    if not os.path.exists(args.config):
        print(f"[ERROR] Config not found: {args.config}")
        return

    cfg = load_ttl_config(args.config)
    print(f"[INFO] Loaded config: {args.config} (v{cfg['full_config'].get('tts_version', '?')})")

    onnx_dir = args.onnx_dir
    os.makedirs(onnx_dir, exist_ok=True)

    def get_latest_ckpt(dir_path):
        if not dir_path or not os.path.isdir(dir_path):
            return None
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
        ckpts = glob.glob(os.path.join(dir_path, "*.pt"))
        if not ckpts:
            return None
        return max(ckpts, key=os.path.getmtime)

    text2latent_ckpt = args.ttl_ckpt if (args.ttl_ckpt and os.path.exists(args.ttl_ckpt)) else get_latest_ckpt(args.ckpt_dir)
    if text2latent_ckpt is None:
        print(f"[WARN] No text2latent checkpoint found (tried {args.ttl_ckpt} and {args.ckpt_dir}).")
    else:
        print(f"[INFO] text2latent: {text2latent_ckpt}")

    t2l_state = _load_state_any(text2latent_ckpt, device) if text2latent_ckpt else {}
    ae_state = _load_state_any(args.ae_ckpt, device)

    vocab_size = cfg["vocab_size"]
    compressed_channels = cfg["compressed_channels"]
    latent_dim = cfg["latent_dim"]
    chunk_compress_factor = cfg["chunk_compress_factor"]
    te_d_model = cfg["te_d_model"]
    se_d_model = cfg["se_d_model"]
    se_n_style = cfg["se_n_style"]

    text_enc = TextEncoder(
        vocab_size=vocab_size,
        d_model=te_d_model,
        n_conv_layers=cfg["te_convnext_layers"],
        n_attn_layers=cfg["te_attn_n_layers"],
        expansion_factor=cfg["te_expansion_factor"],
        p_dropout=cfg["te_attn_p_dropout"],
    ).to(device).eval()
    _maybe_load(text_enc, t2l_state, "text_encoder")

    ref_enc = ReferenceEncoder(
        in_channels=compressed_channels,
        d_model=se_d_model,
        hidden_dim=cfg.get("re_hidden_dim", 1024),
        num_blocks=cfg.get("re_n_blocks", 6),
        num_tokens=se_n_style,
        num_heads=cfg.get("re_n_heads", 2),
        kernel_size=cfg.get("re_kernel_size", 5),
    ).to(device).eval()
    _maybe_load(ref_enc, t2l_state, "reference_encoder")
    _replace_mha_with_safe(ref_enc)

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
    _maybe_load(vf, t2l_state, "vf_estimator", strict=False)

    ae_dec_cfg = cfg["ae_dec_cfg"]
    vocoder = LatentDecoder1D(cfg=ae_dec_cfg).to(device).eval()
    _maybe_load(vocoder, ae_state, "decoder")

    dp = DPNetwork(
        vocab_size=cfg["dp_vocab_size"],
        style_tokens=cfg["dp_style_tokens"],
        style_dim=cfg["dp_style_dim"],
    ).to(device).eval()

    dp_state = _load_state_any(args.dp_ckpt, device)
    if dp_state:
        missing, unexpected = dp.load_state_dict(dp_state, strict=False)
        print(f"[INFO] duration_predictor: missing={len(missing)}, unexpected={len(unexpected)}")
    elif _extract_submodule(t2l_state, "dp_network") is not None:
        dp.load_state_dict(_extract_submodule(t2l_state, "dp_network"), strict=False)
    elif _extract_submodule(t2l_state, "dp_model") is not None:
        dp.load_state_dict(_extract_submodule(t2l_state, "dp_model"), strict=False)
    else:
        print("[WARN] No duration predictor weights; random init.")

    _replace_mha_with_safe(dp)

    B = 1
    T_text = 32
    T_audio_ref = 256
    T_lat = 100
    C_lat = compressed_channels
    C_dec = latent_dim
    style_dim = se_d_model

    text_ids = torch.zeros(B, T_text, dtype=torch.long, device=device)
    text_mask = torch.ones(B, 1, T_text, dtype=torch.float32, device=device)
    z_ref = torch.randn(B, C_lat, T_audio_ref, dtype=torch.float32, device=device)
    ref_mask = torch.ones(B, 1, T_audio_ref, dtype=torch.float32, device=device)
    style_ttl_te = torch.randn(B, se_n_style, style_dim, dtype=torch.float32, device=device)

    do_verify = not args.no_verify
    if do_verify and args.int8:
        print("[WARN] --int8 uses lossy weight quantization; skipping strict numerical verify.")
        do_verify = False
    all_pass = True

    te_path = os.path.join(onnx_dir, "text_encoder.onnx")
    export(
        text_enc,
        te_path,
        (text_ids, style_ttl_te, text_mask),
        input_names=["text_ids", "style_ttl", "text_mask"],
        output_names=["text_emb"],
        dynamic_axes={
            "text_ids": {1: "T_text"},
            "style_ttl": {1: "T_ref"},
            "text_mask": {2: "T_text"},
            "text_emb": {2: "T_text"},
        },
    )
    if do_verify:
        all_pass = verify_onnx(text_enc, te_path, (text_ids, style_ttl_te, text_mask),
                               ["text_ids", "style_ttl", "text_mask"], "text_encoder") and all_pass

    noisy_latent = torch.randn(B, C_lat, T_lat, dtype=torch.float32, device=device)
    latent_mask = torch.ones(B, 1, T_lat, dtype=torch.float32, device=device)
    text_emb = torch.randn(B, style_dim, T_text, dtype=torch.float32, device=device)
    style_ttl_vf = torch.randn(B, se_n_style, style_dim, dtype=torch.float32, device=device)
    current_step = torch.tensor([0.0], dtype=torch.float32, device=device)
    total_step = torch.tensor([1.0], dtype=torch.float32, device=device)

    if hasattr(vf, "style_key") and hasattr(text_enc, "speech_prompted_text_encoder"):
        with torch.no_grad():
            vf.style_key.copy_(text_enc.speech_prompted_text_encoder.style_key)

    vf_wrapped = VectorFieldEstimatorWrapper(vf)
    vf_path = os.path.join(onnx_dir, "vector_estimator.onnx")
    vf_inputs = (noisy_latent, text_emb, style_ttl_vf, latent_mask, text_mask, current_step, total_step)
    vf_input_names = [
        "noisy_latent", "text_emb", "style_ttl",
        "latent_mask", "text_mask",
        "current_step", "total_step",
    ]
    export(
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
        },
    )
    if do_verify:
        all_pass = verify_onnx(vf_wrapped, vf_path, vf_inputs, vf_input_names,
                               "vector_estimator", atol=1e-3, rtol=1e-2) and all_pass

    latent_dec = torch.randn(B, C_dec, T_lat * chunk_compress_factor, dtype=torch.float32, device=device)
    voc_path = os.path.join(onnx_dir, "vocoder.onnx")
    export(
        vocoder,
        voc_path,
        (latent_dec,),
        input_names=["latent"],
        output_names=["waveform"],
        dynamic_axes={"latent": {2: "T_dec"}, "waveform": {2: "T_wav"}},
    )
    if do_verify:
        all_pass = verify_onnx(vocoder, voc_path, (latent_dec,), ["latent"], "vocoder") and all_pass

    dp_path = os.path.join(onnx_dir, "duration_predictor.onnx")
    dp_inputs = (text_ids, z_ref, text_mask, ref_mask)
    dp_input_names = ["text_ids", "z_ref", "text_mask", "ref_mask"]
    export(
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
        },
    )
    if do_verify:
        all_pass = verify_onnx(dp, dp_path, dp_inputs, dp_input_names, "duration_predictor") and all_pass

    if do_verify:
        print("\n" + "=" * 60)
        print("[PASS] All match." if all_pass else "[FAIL] Mismatch.")
        print("=" * 60)


if __name__ == "__main__":
    main()
