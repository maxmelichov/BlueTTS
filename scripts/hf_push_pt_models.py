#!/usr/bin/env python3
"""One-off: convert pt_models/*.pt to .safetensors, wipe Hub repo, upload weights + text files only."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors.torch import save_file
from huggingface_hub import HfApi, logging as hf_logging

REPO = "notmax123/blue-v2"
PT_ROOT = Path(__file__).resolve().parents[1] / "pt_models"


def _flatten_vf_ckpt(ckpt: dict) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for block in ("text_encoder", "vf_estimator", "reference_encoder"):
        sd = ckpt.get(block)
        if not isinstance(sd, (dict, OrderedDict)):
            continue
        for k, v in sd.items():
            if isinstance(v, torch.Tensor):
                out[f"{block}.{k}"] = v
    for name in ("u_text", "u_ref"):
        t = ckpt.get(name)
        if isinstance(t, torch.Tensor):
            out[name] = t
    return out


def _stats_to_tensors(obj: dict) -> dict[str, torch.Tensor]:
    d: dict[str, torch.Tensor] = {}
    for k, v in obj.items():
        if isinstance(v, torch.Tensor):
            d[k] = v
        elif isinstance(v, bool):
            d[k] = torch.tensor(int(v), dtype=torch.int64)
        elif isinstance(v, int):
            d[k] = torch.tensor(v, dtype=torch.int64)
        elif isinstance(v, float):
            d[k] = torch.tensor(v, dtype=torch.float32)
        else:
            raise TypeError(f"stats: unsupported {k!r} type {type(v)}")
    return d


def convert_all(base: Path) -> None:
    vf_pt = base / "vf_estimetor.pt"
    if vf_pt.exists():
        ckpt = torch.load(vf_pt, map_location="cpu", weights_only=True)
        if not isinstance(ckpt, dict):
            raise TypeError("vf_estimetor.pt: expected dict")
        tensors = _flatten_vf_ckpt(ckpt)
        save_file(tensors, str(base / "vf_estimetor.safetensors"))

    dp_pt = base / "duration_predictor_final.pt"
    if dp_pt.exists():
        sd = torch.load(dp_pt, map_location="cpu", weights_only=True)
        if not isinstance(sd, (dict, OrderedDict)):
            raise TypeError("duration_predictor_final.pt: expected state dict")
        to_save = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}
        save_file(to_save, str(base / "duration_predictor_final.safetensors"))

    st_pt = base / "stats_multilingual.pt"
    if st_pt.exists():
        raw = torch.load(st_pt, map_location="cpu", weights_only=True)
        if not isinstance(raw, dict):
            raise TypeError("stats_multilingual.pt: expected dict")
        # ints -> 0-dim int64 tensors (safetensors has no int scalars)
        tmap = _stats_to_tensors({k: raw[k] for k in ("mean", "std", "Kc", "latent_dim") if k in raw})
        # preserve key order: mean, std, Kc, latent_dim
        order = [k for k in ("mean", "std", "Kc", "latent_dim") if k in tmap]
        if set(order) != set(tmap.keys()):
            # include any other tensor keys
            for k in tmap:
                if k not in order:
                    order.append(k)
        ordered = {k: tmap[k] for k in order}
        save_file(ordered, str(base / "stats_multilingual.safetensors"))


def build_staging(base: Path, staging: Path) -> None:
    staging.mkdir(parents=True, exist_ok=True)
    for f in base.iterdir():
        if f.name in (".cache",):
            continue
        if f.suffix == ".pt":
            continue
        if f.is_file() and f.suffix == ".safetensors":
            shutil.copy2(f, staging / f.name)
        elif f.is_file() and f.name in (".gitattributes", "README.md"):
            shutil.copy2(f, staging / f.name)
    for name in (".gitattributes", "README.md"):
        if (base / name).exists() and not (staging / name).exists():
            shutil.copy2(base / name, staging / name)


def main() -> int:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN", file=sys.stderr)
        return 1
    if not PT_ROOT.is_dir():
        print(f"Missing {PT_ROOT}", file=sys.stderr)
        return 1

    hf_logging.set_verbosity_info()
    convert_all(PT_ROOT)

    with tempfile.TemporaryDirectory() as td:
        staging = Path(td)
        build_staging(PT_ROOT, staging)
        to_upload = sorted(staging.iterdir())
        if not to_upload:
            print("Nothing to upload after staging", file=sys.stderr)
            return 1

        api = HfApi(token=token)
        # Clear remote (if repo exists and has files)
        try:
            files = api.list_repo_files(REPO, repo_type="model")
        except Exception as e:  # noqa: BLE001
            print("list_repo_files failed (new repo?)", e, file=sys.stderr)
            files = []
        if files:
            api.delete_files(
                REPO,
                files,
                repo_type="model",
                commit_message="Clear repo before re-uploading safetensors",
                token=token,
            )

        api.upload_folder(
            folder_path=str(staging),
            repo_id=REPO,
            repo_type="model",
            token=token,
            commit_message="Upload pt_models weights (safetensors) + README + .gitattributes",
        )
        for p in to_upload:
            print("uploaded:", p.name)
    print("Done:", REPO)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
