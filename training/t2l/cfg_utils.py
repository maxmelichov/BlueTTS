from __future__ import annotations

import glob
import os
from collections import Counter
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from training.utils import seed_worker

class UncondParams(nn.Module):
    """Learnable unconditional tokens for CFG. Dims from ttl.uncond_masker config.

    uncond_key (config: uncond_masker.style_key_uncond) adds u_key, the Supertonic-3 unconditional style-key
    token: the VF style-attention keys of every row whose style is dropped (CFG both-drop and SPFM-dirty rows,
    the SPFM probe and the CFG uncond pass). Off: those rows key on the conditional `tile`."""
    def __init__(self, text_dim=256, n_style=50, style_value_dim=256, init_std=0.1, uncond_key=False, style_key_dim=256):
        super().__init__()
        self.u_text = nn.Parameter(torch.randn(1, text_dim, 1) * init_std)
        self.u_ref = nn.Parameter(torch.randn(1, n_style, style_value_dim) * init_std)
        if uncond_key:
            self.u_key = nn.Parameter(torch.randn(1, n_style, style_key_dim) * init_std)

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


def all_reduce_grads(params) -> None:
    """Average gradients across ranks for parameters used outside a DDP forward.

    u_text / u_ref (and u_key, when configured) are used as raw tensors (the DDP wrapper's
    forward is never called), so DDP never all-reduces their gradients and each rank's copy drifts apart."""
    if not dist.is_initialized():
        return
    for p in params:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)


def build_lr_scheduler(optimizer, base_lr, milestones, global_step, scheduler_state=None, finetune=False):
    """MultiStepLR (gamma 0.5) positioned at ``global_step``; finetune restarts from ``base_lr``.

    On resume the lr is pinned to the schedule's closed form. Resuming exactly at a milestone
    otherwise halves twice: the loaded optimizer already holds the decayed lr and
    MultiStepLR(last_epoch=step - 1) steps onto the milestone and halves it again; a fresh
    optimizer (state not loaded) would restart at ``base_lr`` instead."""
    last_epoch = -1 if finetune else global_step - 1
    if last_epoch != -1:
        for pg in optimizer.param_groups:
            pg.setdefault('initial_lr', pg['lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5, last_epoch=last_epoch)
    if scheduler_state and not finetune:
        try:
            scheduler.load_state_dict(scheduler_state)
        except Exception:
            pass
    scheduler.milestones = Counter(milestones)  # a loaded state carries its own milestones
    if not finetune and global_step > 0:
        lr_now = base_lr * 0.5 ** sum(global_step >= m for m in milestones)
        for pg in optimizer.param_groups:
            pg['lr'] = lr_now
        scheduler._last_lr = [lr_now for _ in optimizer.param_groups]
    return scheduler


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
