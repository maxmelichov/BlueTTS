from __future__ import annotations

import glob
import os
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from training.utils import seed_worker

class UncondParams(nn.Module):
    """Learnable unconditional tokens for CFG. Dims from ttl.uncond_masker config."""
    def __init__(self, text_dim=256, n_style=50, style_value_dim=256, init_std=0.1):
        super().__init__()
        self.u_text = nn.Parameter(torch.randn(1, text_dim, 1) * init_std)
        self.u_ref = nn.Parameter(torch.randn(1, n_style, style_value_dim) * init_std)

def _validate_ttl_config(ttl_cfg: dict) -> None:
    """Validate every field in `ttl_cfg`."""
    def _eq(label, got, expected):
        if got != expected:
            raise ValueError(f"Config mismatch [{label}]: got {got!r}, expected {expected!r}")

    latent_dim = ttl_cfg["latent_dim"]
    ccf = ttl_cfg["chunk_compress_factor"]
    compressed = latent_dim * ccf

    te = ttl_cfg["text_encoder"]
    char_emb_dim = te["text_embedder"]["char_emb_dim"]
    _eq("text_encoder.convnext.idim", te["convnext"]["idim"], char_emb_dim)
    _eq("text_encoder.attn_encoder.hidden_channels", te["attn_encoder"]["hidden_channels"], char_emb_dim)
    _eq("text_encoder.proj_out.idim", te["proj_out"]["idim"], char_emb_dim)
    _eq("text_encoder.proj_out.odim", te["proj_out"]["odim"], char_emb_dim)
    _eq("text_encoder.text_embedder.char_dict_path", te["text_embedder"]["char_dict_path"], te["char_dict_path"])
    _eq("text_encoder.convnext.num_layers == len(dilation_lst)", len(te["convnext"]["dilation_lst"]), te["convnext"]["num_layers"])

    se = ttl_cfg["style_encoder"]
    se_in = se["proj_in"]["ldim"] * se["proj_in"]["chunk_compress_factor"]
    _eq("style_encoder.proj_in in_channels", se_in, compressed)
    se_odim = se["proj_in"]["odim"]
    _eq("style_encoder.convnext.idim", se["convnext"]["idim"], se_odim)
    stl = se["style_token_layer"]
    _eq("style_encoder.style_token_layer.input_dim", stl["input_dim"], se_odim)
    _eq("style_encoder.style_token_layer.style_key_dim", stl["style_key_dim"], stl["prototype_dim"])
    _eq("style_encoder.convnext.num_layers == len(dilation_lst)", len(se["convnext"]["dilation_lst"]), se["convnext"]["num_layers"])

    spte = ttl_cfg["speech_prompted_text_encoder"]
    _eq("speech_prompted_text_encoder.text_dim", spte["text_dim"], char_emb_dim)
    _eq("speech_prompted_text_encoder.n_units", spte["n_units"], char_emb_dim)
    _eq("speech_prompted_text_encoder.style_dim", spte["style_dim"], stl["style_value_dim"])

    um = ttl_cfg["uncond_masker"]
    _eq("uncond_masker.text_dim", um["text_dim"], char_emb_dim)
    _eq("uncond_masker.n_style", um["n_style"], stl["n_style"])
    _eq("uncond_masker.style_value_dim", um["style_value_dim"], stl["style_value_dim"])
    _eq("uncond_masker.style_key_dim", um["style_key_dim"], stl["style_key_dim"])

    vf = ttl_cfg["vector_field"]
    vf_in = vf["proj_in"]["ldim"] * vf["proj_in"]["chunk_compress_factor"]
    vf_out = vf["proj_out"]["ldim"] * vf["proj_out"]["chunk_compress_factor"]
    hidden = vf["proj_in"]["odim"]
    _eq("vector_field.proj_in in_channels", vf_in, compressed)
    _eq("vector_field.proj_out out_channels", vf_out, compressed)
    _eq("vector_field.proj_out.idim", vf["proj_out"]["idim"], hidden)

    mb = vf["main_blocks"]
    _eq("main_blocks.time_cond_layer.idim", mb["time_cond_layer"]["idim"], hidden)
    _eq("main_blocks.time_cond_layer.time_dim", mb["time_cond_layer"]["time_dim"], vf["time_encoder"]["time_dim"])
    _eq("main_blocks.style_cond_layer.idim", mb["style_cond_layer"]["idim"], hidden)
    _eq("main_blocks.style_cond_layer.style_dim", mb["style_cond_layer"]["style_dim"], stl["style_value_dim"])
    _eq("main_blocks.text_cond_layer.idim", mb["text_cond_layer"]["idim"], hidden)
    _eq("main_blocks.text_cond_layer.text_dim", mb["text_cond_layer"]["text_dim"], char_emb_dim)

    for name in ("convnext_0", "convnext_1", "convnext_2"):
        sub = mb[name]
        _eq(f"main_blocks.{name}.idim", sub["idim"], hidden)
        _eq(f"main_blocks.{name}.num_layers == len(dilation_lst)", len(sub["dilation_lst"]), sub["num_layers"])
    lc = vf["last_convnext"]
    _eq("last_convnext.idim", lc["idim"], hidden)
    _eq("last_convnext.num_layers == len(dilation_lst)", len(lc["dilation_lst"]), lc["num_layers"])


# --- dataloader / DDP / checkpoint helpers (kept with cfg to avoid extra tiny modules) ---


def unwrap_ddp(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, DDP) else module


def ddp_state_dict(module: nn.Module) -> dict:
    return module.module.state_dict() if isinstance(module, DDP) else module.state_dict()


def _latest_ckpt_in_dir(directory: str) -> Optional[str]:
    """Path to latest ``ckpt_step_*.pt`` in ``directory``, or None."""
    ckpts = glob.glob(os.path.join(directory, "ckpt_step_*.pt"))
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return ckpts[-1]
