import os
import sys
import glob
import json
import random
import numpy as np
import soundfile as sf
from typing import Optional
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, WeightedRandomSampler, DistributedSampler
from torch.optim import AdamW
from tqdm import tqdm

from training.data.text2latent_dataset import Text2LatentDataset, collate_text2latent
from training.t2l.data_module import Text2LatentDataset, collate_text2latent
from training.data.audio_utils import ensure_sr
from training.data.text_vocab import text_to_indices, text_to_indices_multilang, VOCAB_SIZE, normalize_text
from bluecodec import LatentEncoder, LatentDecoder1D
from training.utils import LinearMelSpectrogram, compress_latents, decompress_latents
from training.t2l.models.text_encoder import TextEncoder
from training.t2l.models.vf_estimator import VectorFieldEstimator
from training.t2l.models.reference_encoder import ReferenceEncoder
from training.dp.models.dp_network import DPNetwork
from training.t2l.builders import build_models

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_reference_only(z_ref_input, valid_z_ref_len, device, max_frames=72):
    B, C, T_ref = z_ref_input.shape
    if max_frames is not None and T_ref > max_frames:
        z_ref_input = z_ref_input[:, :, :max_frames]
        T_ref = max_frames
    arange_T = torch.arange(T_ref, device=device).unsqueeze(0)
    valid_len = valid_z_ref_len.clamp(min=0, max=T_ref).unsqueeze(1)
    ref_mask_left = (arange_T < valid_len).unsqueeze(1).float()
    z_ref_left = z_ref_input * ref_mask_left
    return z_ref_left, ref_mask_left

def build_reference_from_latents(z_1, valid_z_len, z_ref_input, valid_z_ref_len, is_self_ref, device, chunk_compress_factor=6):
    B, C, T = z_1.shape
    _, _, T_ref_in = z_ref_input.shape

    sr = 44100
    hop = 512
    compressed_rate = (sr / hop) / chunk_compress_factor

    min_frames = max(1, int(round(0.2 * compressed_rate)))
    max_frames = int(round(9.0 * compressed_rate))

    z_ref_left = torch.zeros(B, C, T, device=device)
    ref_mask_left = torch.zeros(B, 1, T, device=device)
    target_loss_mask = torch.ones(B, 1, T, device=device)
    train_T_lat = valid_z_len.clone()

    for i in range(B):
        sample_T = int(valid_z_len[i].item())
        ref_T = int(valid_z_ref_len[i].item())
        ref_T = min(ref_T, T_ref_in)

        if is_self_ref[i]:
            half_len = max(1, sample_T // 2)
            upper_bound = min(max_frames, half_len)
            upper_bound = max(1, upper_bound)

            if upper_bound < min_frames:
                 length = int(torch.randint(1, upper_bound + 1, (1,), device=device).item())
            else:
                 length = int(torch.randint(min_frames, upper_bound + 1, (1,), device=device).item())
            length = min(length, sample_T)
            if length < 1: length = 1
            max_start = max(0, sample_T - length)
            start = int(torch.randint(0, max_start + 1, (1,), device=device).item())

            mask_start = start
            mask_end = min(start + length, sample_T)
            target_loss_mask[i, :, mask_start:mask_end] = 0.0
            copy_len = min(length, T)
            z_ref_left[i, :, :copy_len] = z_1[i, :, mask_start:mask_start + copy_len]
            ref_mask_left[i, :, :copy_len] = 1.0

        else:
            half_ref = max(1, ref_T // 2)
            upper_bound = min(max_frames, half_ref)
            upper_bound = max(1, upper_bound)
            if upper_bound < min_frames:
                 length = int(torch.randint(1, upper_bound + 1, (1,), device=device).item())
            else:
                 length = int(torch.randint(min_frames, upper_bound + 1, (1,), device=device).item())
            length = min(length, ref_T)
            if length < 1: length = 1
            max_start = max(0, ref_T - length)
            start = int(torch.randint(0, max_start + 1, (1,), device=device).item())
            copy_len = min(length, T)
            z_ref_left[i, :, :copy_len] = z_ref_input[i, :, start:start+copy_len]
            ref_mask_left[i, :, :copy_len] = 1.0

    return z_ref_left, ref_mask_left, train_T_lat, target_loss_mask

def length_to_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    max_len = max_len or int(lengths.max().item())
    ids = torch.arange(0, max_len, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1)).float()
    return mask.unsqueeze(1)

def get_latent_mask(wav_lengths: torch.Tensor, base_chunk_size: int, chunk_compress_factor: int) -> torch.Tensor:
    latent_size = base_chunk_size * chunk_compress_factor
    latent_lengths = (wav_lengths + latent_size - 1) // latent_size
    latent_mask = length_to_mask(latent_lengths)
    return latent_mask

def sample_noisy_latent(
    duration: torch.Tensor, sample_rate: int, base_chunk_size: int,
    chunk_compress_factor: int, latent_dim: int, device: str | torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz = len(duration)
    wav_len_max = duration.max() * sample_rate
    wav_lengths = (duration * sample_rate).long()
    chunk_size = base_chunk_size * chunk_compress_factor
    latent_len = int((wav_len_max + chunk_size - 1) // chunk_size)
    latent_channels = latent_dim * chunk_compress_factor

    noisy_latent = torch.randn(bsz, latent_channels, latent_len, device=device)
    latent_mask = get_latent_mask(wav_lengths, base_chunk_size, chunk_compress_factor)

    if latent_mask.shape[2] < latent_len:
        latent_mask = torch.nn.functional.pad(latent_mask, (0, latent_len - latent_mask.shape[2]))
    elif latent_mask.shape[2] > latent_len:
        latent_mask = latent_mask[:, :, :latent_len]

    noisy_latent = noisy_latent * latent_mask
    return noisy_latent, latent_mask

@torch.no_grad()
def sample_audio(
    vf_estimator,
    text_encoder,
    reference_encoder,
    ae_decoder,
    text_ids,
    text_mask,
    z_ref,
    ref_enc_mask,
    mean,
    std,
    duration_predictor=None,
    steps=32,
    device: str | torch.device = 'cuda',
    debug_label=None,
    speed=1.0,
    style_ttl=None,
    style_dp=None,
    uncond_params=None,
    cfg_scale=1.0,
    # Config-derived params (from ttl section of tts.json)
    latent_dim=24,
    chunk_compress_factor=6,
    normalizer_scale=1.0,
):
    """
    Plain flow-matching sampling:

      1. Encode reference -> style values h_ref
      2. Encode text (+ style) -> h_text
      3. Choose target length T (from duration predictor or ref length)
      4. Sample x_0 ~ N(0, I)
      5. Integrate dx/dt = v_theta(x_t, cond, t) from t=0..1
      6. Decode latents to waveform

    VF estimator uses its own internal self.tile (ONNX /Tile_output_0) for timing (not passed explicitly).
    """
    if debug_label:
        print(f"[{debug_label}] Starting sampling...")

    B = text_ids.shape[0]
    C = latent_dim * chunk_compress_factor  # e.g. 24 * 6 = 144

    # -------------------------
    # 1. Encode style / reference
    # -------------------------
    # ref_values: [B, 50, 256] (speaker-specific style values)
    if style_ttl is not None:
        ref_values = style_ttl
    else:
        ref_values = reference_encoder(z_ref, mask=ref_enc_mask)

    # -------------------------
    # 2. Duration / target length
    # -------------------------
    if duration_predictor is not None:
        dur_pred = duration_predictor(
            text_ids,
            z_ref=z_ref,
            text_mask=text_mask,
            ref_mask=ref_enc_mask,
            style_dp=style_dp,
            return_log=True,
        )

        duration = torch.exp(dur_pred) / speed
        
        sample_rate = 44100
        base_chunk_size = 512
        
        xt, latent_mask = sample_noisy_latent(
            duration, sample_rate, base_chunk_size, chunk_compress_factor, latent_dim, device
        )
        
        if debug_label:
             print(f"[{debug_label}] DP latent_lengths max: {latent_mask.shape[2]}")

        T = latent_mask.shape[2]
    else:
        dur_pred = None
        if z_ref is not None:
            T = z_ref.shape[2]
        else:
            T = 200
        latent_mask = torch.ones(B, 1, T, device=device)
        xt = torch.randn(B, C, T, device=device) * latent_mask

    print("[DBG] text_len =", text_mask.sum(dim=(1,2)).detach().cpu().tolist())
    if z_ref is not None:
        print("[DBG] z_ref_T  =", z_ref.shape[2])
    if duration_predictor is None:
        print("[DBG] duration_predictor is None")
    else:
        print("[DBG] duration_predictor is set")
        print("[DBG] dur_pred  =", dur_pred.detach().cpu().numpy() if dur_pred is not None else None)
        print("[DBG] latent_lengths    =", latent_mask.sum(dim=2).squeeze(1).detach().cpu().numpy())

    print("[DBG] T chosen =", T)
    print("[DBG] latent_mask sum =", latent_mask.sum().item())

    # -------------------------
    # 3. Encode text conditioned on style
    # -------------------------
    # TextEncoder returns:
    #   h_text:    [B, 256, T_txt]
    #   style_key: [B, 50, 256] (baked-in constant) - accessed via property
    h_text = text_encoder(
        text_ids,
        ref_values,
        text_mask=text_mask,
    )
    
    T_txt = h_text.shape[2]

    # -------------------------
    # 4. Sampler init
    # -------------------------
    x = xt
    dt = 1.0 / steps

    # -------------------------
    # 5. Euler integration
    # -------------------------
    for i in range(steps):
        t_val = i / steps
        t = torch.full((B,), t_val, device=device)

        # Conditional velocity
        x_in = x * latent_mask # Zero out padding before forward for hygiene
        v_cond = vf_estimator(
            noisy_latent=x_in,
            text_emb=h_text,      # [B, 256, T_txt]
            style_ttl=ref_values, # [B, 50, 256] (values)
            latent_mask=latent_mask,
            text_mask=text_mask,
            current_step=t,
        )

        if cfg_scale > 1.0 and uncond_params is not None:
            # Unconditional velocity
            if hasattr(uncond_params, 'module'):
                u_text = uncond_params.module.u_text.expand(B, -1, T_txt)
                u_ref = uncond_params.module.u_ref.expand(B, -1, -1)
            else:
                u_text = uncond_params.u_text.expand(B, -1, T_txt)
                u_ref = uncond_params.u_ref.expand(B, -1, -1)
                
            v_uncond = vf_estimator(
                noisy_latent=x_in,
                text_emb=u_text,
                style_ttl=u_ref,
                latent_mask=latent_mask,
                text_mask=text_mask,
                current_step=t,
            )
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = v_cond
        
        # Stabilize boundaries: apply mask to velocity
        v = v * latent_mask


        x = x + v * dt

    
        x = x * latent_mask  # Zero out padded frames per-sample
    # -------------------------
    # 6. Decode to waveform
    # -------------------------
    # Un-normalize: reverse z_1 = ((z - mean) / std) * normalizer_scale
    if normalizer_scale != 1.0 and normalizer_scale != 0.0:
        z_pred = (x / normalizer_scale) * std + mean
    else:
        z_pred = x * std + mean
    
    print("[DBG] x shape pre-decode:", x.shape)
    print("[DBG] z_pred shape pre-decompress:", z_pred.shape)
    
    z_pred = decompress_latents(
        z_pred, 
        factor=chunk_compress_factor, 
        target_channels=latent_dim
    )                                             # [B, latent_dim, T_dec]
    
    print("[DBG] z_pred shape post-decompress:", z_pred.shape)
    
    wav_pred = ae_decoder(z_pred)                 # [B, 1, T_wav]
    
    print("[DBG] wav_pred shape:", wav_pred.shape)

    # 7. Enforce exact length contract
    # Contract: 1 latent frame = hop_length * chunk_compress_factor samples
    frame_len = 512 * chunk_compress_factor
    # Truncate to exact expected frames
    wav_pred = wav_pred[..., frame_len:-frame_len]
    

    return wav_pred

def seed_worker(worker_id):
    np.random.seed(worker_id)
    random.seed(worker_id)
    torch.manual_seed(worker_id)
    torch.cuda.manual_seed(worker_id)
    torch.cuda.manual_seed_all(worker_id)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

class UncondParams(nn.Module):
    """Learnable unconditional tokens for CFG. Dims from ttl.uncond_masker config."""
    def __init__(self, text_dim=256, n_style=50, style_value_dim=256, init_std=0.1):
        super().__init__()
        self.u_text = nn.Parameter(torch.randn(1, text_dim, 1) * init_std)
        self.u_ref = nn.Parameter(torch.randn(1, n_style, style_value_dim) * init_std)

def _latest_ckpt_in_dir(directory):
    """Return path to highest ckpt_step_*.pt in ``directory``, or None."""
    ckpts = glob.glob(os.path.join(directory, "ckpt_step_*.pt"))
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return ckpts[-1]


def _validate_ttl_config(ttl_cfg: dict) -> None:
    """Validate every field in `ttl_cfg`.

    Every setting declared in `configs/tts.json` is consumed below: either
    directly to instantiate a module, or as a cross-check that downstream
    architecture assumptions still hold. If a user tweaks a number in the
    config that would silently desynchronize the model, this raises instead.
    """
    def _eq(label, got, expected):
        if got != expected:
            raise ValueError(
                f"Config mismatch [{label}]: got {got!r}, expected {expected!r}"
            )

    latent_dim = ttl_cfg["latent_dim"]
    ccf = ttl_cfg["chunk_compress_factor"]
    compressed = latent_dim * ccf

    # text_encoder: convnext idim / attn hidden_channels / proj_out must all equal
    # the char_emb_dim that feeds the stack.
    te = ttl_cfg["text_encoder"]
    char_emb_dim = te["text_embedder"]["char_emb_dim"]
    _eq("text_encoder.convnext.idim", te["convnext"]["idim"], char_emb_dim)
    _eq("text_encoder.attn_encoder.hidden_channels",
        te["attn_encoder"]["hidden_channels"], char_emb_dim)
    _eq("text_encoder.proj_out.idim", te["proj_out"]["idim"], char_emb_dim)
    _eq("text_encoder.proj_out.odim", te["proj_out"]["odim"], char_emb_dim)
    _eq("text_encoder.text_embedder.char_dict_path",
        te["text_embedder"]["char_dict_path"], te["char_dict_path"])
    _eq("text_encoder.convnext.num_layers == len(dilation_lst)",
        len(te["convnext"]["dilation_lst"]), te["convnext"]["num_layers"])

    # style_encoder: proj_in in-channels come from (ldim * chunk_compress_factor)
    # and must match the global compressed_channels; style_token_layer widths
    # must agree with the conv trunk.
    se = ttl_cfg["style_encoder"]
    se_in = se["proj_in"]["ldim"] * se["proj_in"]["chunk_compress_factor"]
    _eq("style_encoder.proj_in in_channels", se_in, compressed)
    se_odim = se["proj_in"]["odim"]
    _eq("style_encoder.convnext.idim", se["convnext"]["idim"], se_odim)
    stl = se["style_token_layer"]
    _eq("style_encoder.style_token_layer.input_dim", stl["input_dim"], se_odim)
    _eq("style_encoder.style_token_layer.style_key_dim",
        stl["style_key_dim"], stl["prototype_dim"])
    _eq("style_encoder.convnext.num_layers == len(dilation_lst)",
        len(se["convnext"]["dilation_lst"]), se["convnext"]["num_layers"])

    # speech_prompted_text_encoder: must match TextEncoder d_model + style values.
    spte = ttl_cfg["speech_prompted_text_encoder"]
    _eq("speech_prompted_text_encoder.text_dim", spte["text_dim"], char_emb_dim)
    _eq("speech_prompted_text_encoder.n_units", spte["n_units"], char_emb_dim)
    _eq("speech_prompted_text_encoder.style_dim",
        spte["style_dim"], stl["style_value_dim"])

    # uncond_masker: unconditional tokens mirror the text / style dims.
    um = ttl_cfg["uncond_masker"]
    _eq("uncond_masker.text_dim", um["text_dim"], char_emb_dim)
    _eq("uncond_masker.n_style", um["n_style"], stl["n_style"])
    _eq("uncond_masker.style_value_dim",
        um["style_value_dim"], stl["style_value_dim"])
    _eq("uncond_masker.style_key_dim",
        um["style_key_dim"], stl["style_key_dim"])

    # vector_field: proj_in/out channels + every sub-block idim must be
    # consistent with the hidden width.
    vf = ttl_cfg["vector_field"]
    vf_in = vf["proj_in"]["ldim"] * vf["proj_in"]["chunk_compress_factor"]
    vf_out = vf["proj_out"]["ldim"] * vf["proj_out"]["chunk_compress_factor"]
    hidden = vf["proj_in"]["odim"]
    _eq("vector_field.proj_in in_channels", vf_in, compressed)
    _eq("vector_field.proj_out out_channels", vf_out, compressed)
    _eq("vector_field.proj_out.idim", vf["proj_out"]["idim"], hidden)

    mb = vf["main_blocks"]
    _eq("main_blocks.time_cond_layer.idim",
        mb["time_cond_layer"]["idim"], hidden)
    _eq("main_blocks.time_cond_layer.time_dim",
        mb["time_cond_layer"]["time_dim"], vf["time_encoder"]["time_dim"])
    _eq("main_blocks.style_cond_layer.idim",
        mb["style_cond_layer"]["idim"], hidden)
    _eq("main_blocks.style_cond_layer.style_dim",
        mb["style_cond_layer"]["style_dim"], stl["style_value_dim"])
    _eq("main_blocks.text_cond_layer.idim",
        mb["text_cond_layer"]["idim"], hidden)
    _eq("main_blocks.text_cond_layer.text_dim",
        mb["text_cond_layer"]["text_dim"], char_emb_dim)

    for name in ("convnext_0", "convnext_1", "convnext_2"):
        sub = mb[name]
        _eq(f"main_blocks.{name}.idim", sub["idim"], hidden)
        _eq(f"main_blocks.{name}.num_layers == len(dilation_lst)",
            len(sub["dilation_lst"]), sub["num_layers"])
    lc = vf["last_convnext"]
    _eq("last_convnext.idim", lc["idim"], hidden)
    _eq("last_convnext.num_layers == len(dilation_lst)",
        len(lc["dilation_lst"]), lc["num_layers"])


def train(
    metadata_path="generated_audio/combined_dataset_cleaned_real_data.csv",
    checkpoint_dir="checkpoints/text2latent",
    ae_checkpoint="checkpoints/ae/ae_latest.pt",
    stats_path="stats_multilingual.pt",
    config_path="configs/tts.json",
    epochs=1000,
    batch_size=14,
    lr=5e-4,
    Ke=None,
    puncond=None,
    device="cuda:1" if torch.cuda.is_available() else "cpu",
    finetune=False,
    accumulation_steps=1,
    resume_from=None,
    inference_ref_wav: Optional[str] = None,
):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        if rank == 0:
            print(f"Initialized DDP on Rank {rank}")
    else:
        rank = 0
        local_rank = 0
        if isinstance(device, str):
            device = torch.device(device)
        print(f"Running on single device {device} (No DDP env found)")

    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        log_dir = os.path.join(checkpoint_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        print(f"Initializing training on {device}...")
    else:
        log_dir = os.path.join(checkpoint_dir, "logs")

    spfm_start_override = 10_000 if finetune else None
    if finetune: lr = 2.5e-4

    with open(config_path, 'r') as f:
        full_config = json.load(f)
    ttl_cfg = full_config["ttl"]
    ae_cfg_json = full_config.get("ae", {})

    _validate_ttl_config(ttl_cfg)

    latent_dim = ttl_cfg["latent_dim"]
    chunk_compress_factor = ttl_cfg["chunk_compress_factor"]
    cfg_Ke = ttl_cfg["batch_expander"]["n_batch_expand"]
    Ke = Ke if Ke is not None else cfg_Ke

    normalizer_scale = ttl_cfg["normalizer"]["scale"]
    sigma_min = ttl_cfg["flow_matching"]["sig_min"]

    um_cfg = ttl_cfg["uncond_masker"]
    prob_both_uncond = um_cfg["prob_both_uncond"]
    prob_text_uncond = um_cfg["prob_text_uncond"]
    if puncond is None:
        puncond = prob_both_uncond + prob_text_uncond

    stats = torch.load(stats_path, map_location=device)
    if "mean" in stats and stats["mean"].dim() == 3:
        mean, std = stats["mean"].to(device), stats["std"].to(device)
    else:
        mean, std = stats['mean'].to(device).view(1, -1, 1), stats['std'].to(device).view(1, -1, 1)

    # Optional reference audio for multi-language logging inference (Voice 1). Set
    # --inference_ref_wav or the T2L_INFERENCE_REF_WAV environment variable.
    ref_wav_path_v1 = (inference_ref_wav or os.environ.get("T2L_INFERENCE_REF_WAV") or "").strip()
    ref_wav_torch_v1 = None
    if ref_wav_path_v1:
        if os.path.exists(ref_wav_path_v1):
            print(f"Loading inference reference for Voice 1 from {ref_wav_path_v1}")
            ref_wav_np, sr = sf.read(ref_wav_path_v1)
            ref_wav_torch_v1 = torch.from_numpy(ref_wav_np).float().to(device)
            if ref_wav_torch_v1.dim() > 1:
                ref_wav_torch_v1 = ref_wav_torch_v1.mean(dim=1)  # mono

            # Resample to 44.1kHz using high-quality resampler
            if sr != 44100:
                ref_wav_torch_v1 = ensure_sr(ref_wav_torch_v1, sr, 44100, device=device)
            else:
                ref_wav_torch_v1 = ref_wav_torch_v1.unsqueeze(0)  # [1, T]

            # enforce [1, T]
            if ref_wav_torch_v1.dim() == 2 and ref_wav_torch_v1.size(0) != 1:
                ref_wav_torch_v1 = ref_wav_torch_v1.mean(dim=0, keepdim=True)
            elif ref_wav_torch_v1.dim() == 1:
                ref_wav_torch_v1 = ref_wav_torch_v1.unsqueeze(0)
        else:
            print(f"Warning: Inference reference for Voice 1 {ref_wav_path_v1} not found.")
            ref_wav_torch_v1 = None

    ae_sample_rate = ae_cfg_json.get('sample_rate', 44100)

    text_encoder, reference_encoder, vf_estimator, uncond_params, dp_model, ae_encoder, ae_decoder, mel_spec, hop_length = build_models(
        ttl_cfg, ae_cfg_json, ae_sample_rate, device
    )

    if os.path.exists(ae_checkpoint):
        ckpt = torch.load(ae_checkpoint, map_location='cpu')
        ae_encoder.load_state_dict(ckpt.get('encoder', ckpt.get('state_dict', ckpt)), strict=False)
        if 'decoder' in ckpt: ae_decoder.load_state_dict(ckpt['decoder'])

    ae_encoder.eval().requires_grad_(False)
    ae_decoder.eval().requires_grad_(False)
    mel_spec.eval()

    u_text, u_ref = uncond_params.u_text, uncond_params.u_ref

    params = list(text_encoder.parameters()) + list(reference_encoder.parameters()) + \
             list(vf_estimator.parameters()) + list(uncond_params.parameters())
    optimizer = AdamW(params, lr=lr)

    global_step = 0
    scheduler_state = None
    latest_ckpt = _latest_ckpt_in_dir(checkpoint_dir)
    if latest_ckpt is None and resume_from:
        latest_ckpt = _latest_ckpt_in_dir(resume_from)

    if latest_ckpt is not None:
        print(f"Resuming from {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        shapes_changed = False
        for mod, name in [(vf_estimator, 'vf_estimator'), (text_encoder, 'text_encoder')]:
            if name in checkpoint:
                model_state, ckpt_state = mod.state_dict(), checkpoint[name]
                filtered_state = {}
                for k, v in ckpt_state.items():
                    if k in model_state and v.shape != model_state[k].shape:
                        if "char_embedder.weight" in k and v.shape[0] > model_state[k].shape[0]:
                            filtered_state[k] = v[:model_state[k].shape[0], :]
                        else:
                            shapes_changed = True
                        continue
                    filtered_state[k] = v
                mod.load_state_dict(filtered_state, strict=False)
        if 'reference_encoder' in checkpoint: reference_encoder.load_state_dict(checkpoint['reference_encoder'], strict=False)
        if 'u_text' in checkpoint: u_text.data = checkpoint['u_text']
        if 'u_ref' in checkpoint: u_ref.data = checkpoint['u_ref']

        optimizer = AdamW(params, lr=lr)
        if 'optimizer' in checkpoint and not (finetune or shapes_changed):
            try: optimizer.load_state_dict(checkpoint['optimizer'])
            except: pass
        if 'global_step' in checkpoint:
            global_step = checkpoint['global_step']
            if finetune: spfm_start_override = global_step + spfm_start_override
        if 'scheduler' in checkpoint and not shapes_changed: scheduler_state = checkpoint['scheduler']

    scheduler_last_epoch = -1 if finetune else (global_step - 1)
    if scheduler_last_epoch != -1:
        for pg in optimizer.param_groups: pg.setdefault('initial_lr', pg['lr'])

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300_000, 600_000], gamma=0.5, last_epoch=scheduler_last_epoch)
    if scheduler_state and not finetune:
        try: scheduler.load_state_dict(scheduler_state)
        except: pass

    if dist.is_initialized():
        text_encoder = DDP(text_encoder, device_ids=[local_rank], find_unused_parameters=True)
        reference_encoder = DDP(reference_encoder, device_ids=[local_rank], find_unused_parameters=True)
        vf_estimator = DDP(vf_estimator, device_ids=[local_rank], find_unused_parameters=True)
        uncond_params = DDP(uncond_params, device_ids=[local_rank], find_unused_parameters=True)

    val_z_ref = None
    val_ref_enc_mask = None
    val_wavs = None
    val_text_ids = None
    val_text_masks = None
    val_ref_wavs = None
    val_ref_lengths = None
    
    # Dataset
    dataset = Text2LatentDataset(
        metadata_path, 
        sample_rate=ae_sample_rate,
        max_wav_len=ae_sample_rate * 20,
        max_text_len=300,
        cross_ref_prob=0.0,  # 0% cross-ref for zero-shot speaker generalization
    )
    if rank == 0:
        print(f"Dataset loaded with {len(dataset)} samples.")

    # Sampler Setup
    if dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=True)
    else:
        # Calculate inverse-frequency weights for balanced speaker sampling
        speaker_ids = dataset.speaker_ids
        unique_speakers, counts = np.unique(speaker_ids, return_counts=True)
        freq = dict(zip(unique_speakers, counts))
        print(f"Speaker counts: {freq}")
        
        sample_weights = np.array([1.0 / freq[sid] for sid in speaker_ids])
        sample_weights = sample_weights / sample_weights.sum()
        weights = sample_weights.tolist()
        sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler, 
        num_workers=4, 
        collate_fn=collate_text2latent,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        worker_init_fn=seed_worker
    )
    
    # Validation Batch
    try:
        val_batch = next(iter(dataloader))
        # Handle unpacking dynamically for validation too
        if len(val_batch) == 9:
            val_wavs, val_text_ids, val_text_masks, val_lengths, _, val_ref_wavs, val_ref_lengths, val_is_self, _ = val_batch
        else:
            val_wavs, val_text_ids, val_text_masks, val_lengths, _, val_ref_wavs, val_ref_lengths, val_is_self = val_batch

        val_wavs = val_wavs[:4].to(device)
        val_text_ids = val_text_ids[:4].to(device)
        val_text_masks = val_text_masks[:4].to(device)
        val_ref_wavs = val_ref_wavs[:4].to(device)
        val_ref_lengths = val_ref_lengths[:4].to(device)
        val_is_self = val_is_self[:4].to(device)
        
        with torch.no_grad():
            val_mel = mel_spec(val_wavs.squeeze(1))
            val_z = ae_encoder(val_mel)
            val_z = compress_latents(val_z, factor=chunk_compress_factor)
            val_z_1 = ((val_z - mean) / std) * normalizer_scale
            
            B_val, C, T_val = val_z_1.shape
            
            # Use valid lengths for validation to match training
            valid_mel_len_val = val_lengths[:4].to(device).float() / hop_length
            valid_z_len_val = (valid_mel_len_val / chunk_compress_factor).ceil().long().clamp(min=1, max=T_val)

            # Encode Ref
            val_mel_ref = mel_spec(val_ref_wavs.squeeze(1))
            val_z_ref_full_enc = ae_encoder(val_mel_ref)
            val_z_ref_full_enc = compress_latents(val_z_ref_full_enc, factor=chunk_compress_factor)
            val_z_ref_full = ((val_z_ref_full_enc - mean) / std) * normalizer_scale

            valid_mel_len_ref = val_ref_lengths[:4].to(device).float() / hop_length
            valid_z_len_ref = (valid_mel_len_ref / chunk_compress_factor).ceil().long().clamp(min=1, max=val_z_ref_full.shape[2])
            
            # Use build_reference_only for correct inference behavior
            val_z_ref, val_ref_enc_mask = build_reference_only(
                val_z_ref_full, valid_z_len_ref, device, max_frames=256
            )
            
    except Exception as e:
        if rank == 0:
            print(f"Validation batch init failed: {e}")
        val_batch = None

    max_steps = 1_000_000
    epoch = 0

    while global_step < max_steps:
        if dist.is_initialized() and hasattr(sampler, 'set_epoch'): sampler.set_epoch(epoch)
        epoch += 1

        text_encoder.train()
        reference_encoder.train()
        vf_estimator.train()

        progress_bar = tqdm(dataloader, desc=f"{'[FT] ' if finetune else ''}Step {global_step}")
        epoch_loss = 0.0
        num_batches = 0
        spfm_dirty_total = 0
        spfm_total_samples = 0
        spfm_score_sum = 0.0
        spfm_call_batches = 0

        for batch_idx, batch in enumerate(progress_bar):
            if global_step >= max_steps: break

            if len(batch) == 9:
                wavs, text_ids, text_masks, lengths, speaker_ids, ref_wavs, ref_lengths, is_self_ref, ref_speaker_ids = batch
            else:
                wavs, text_ids, text_masks, lengths, speaker_ids, ref_wavs, ref_lengths, is_self_ref = batch
                ref_speaker_ids = speaker_ids
            
            wavs = wavs.to(device)
            text_ids = text_ids.to(device)
            text_masks = text_masks.to(device)
            ref_wavs = ref_wavs.to(device)
            ref_lengths = ref_lengths.to(device)
            is_self_ref = is_self_ref.to(device)
            speaker_ids = speaker_ids.to(device)
            if ref_speaker_ids is None:
                ref_speaker_ids = speaker_ids
            
            # Sanity Check Logging (every 100 steps)
            if global_step % 100 == 0:
                same_speaker = (speaker_ids == ref_speaker_ids).float().mean().item()
                self_ref_ratio = is_self_ref.float().mean().item()
                
                # Check for "Self Ref but Diff Indices" (dataset logic error)
                # Currently we don't have indices in batch, but is_self_ref implies we used wav.clone().
                # So if is_self_ref is true, the content is identical by definition in __getitem__.
                # But let's log if same_speaker is low.
                
                if same_speaker < 0.99:
                     print(f"WARNING: Speaker Mismatch! Same-speaker ratio: {same_speaker:.2f}")
                
                # Check if is_self_ref is consistent (only for same speaker)
                # We can't strictly check "same utterance" without indices, but we trust dataset logic.
                
                if global_step % 1000 == 0:
                     cross_ref_ratio = 1.0 - self_ref_ratio
                     print(f"[Ref Check] Step {global_step} | Self-Ref: {self_ref_ratio:.2f} | Cross-Ref: {cross_ref_ratio:.2f} | Same-Spk: {same_speaker:.2f}")

            B = wavs.shape[0]

            with torch.no_grad():
                mel = mel_spec(wavs.squeeze(1))
                z = ae_encoder(mel)
                z = compress_latents(z, factor=chunk_compress_factor)
                z_1 = ((z - mean) / std) * normalizer_scale

                mel_ref = mel_spec(ref_wavs.squeeze(1))
                z_ref_full_enc = ae_encoder(mel_ref)
                z_ref_full_enc = compress_latents(z_ref_full_enc, factor=chunk_compress_factor)
                z_ref_full = ((z_ref_full_enc - mean) / std) * normalizer_scale

            _, C, T = z_1.shape
            valid_mel_len = lengths.to(device).float() / hop_length
            valid_z_len = (valid_mel_len / chunk_compress_factor).ceil().long().clamp(min=1, max=T)

            latent_mask = (torch.arange(T, device=device).expand(B, T) < valid_z_len.unsqueeze(1)).unsqueeze(1).float()
            z_1 = z_1 * latent_mask

            valid_mel_len_ref = ref_lengths.to(device).float() / hop_length
            valid_z_len_ref = (valid_mel_len_ref / chunk_compress_factor).ceil().long().clamp(min=1, max=z_ref_full.shape[2])

            T_ref_in = z_ref_full.shape[2]
            ref_full_mask = (torch.arange(T_ref_in, device=device).expand(B, T_ref_in) < valid_z_len_ref.unsqueeze(1)).unsqueeze(1).float()
            z_ref_full = z_ref_full * ref_full_mask

            z_ref, ref_enc_mask, train_T_lat, target_loss_mask = build_reference_from_latents(
                z_1, valid_z_len, z_ref_full, valid_z_len_ref, is_self_ref, device,
                chunk_compress_factor=chunk_compress_factor
            )

            ref_values = reference_encoder(z_ref, mask=ref_enc_mask)
            h_text = text_encoder(text_ids, ref_values, text_mask=text_masks)
            
            # Compute valid/padding mask ONCE for base batch (used by SPFM + FM)
            # (latent_mask already computed above)
            valid_len_mask = latent_mask 
            # latent_mask = valid_len_mask  # [B, 1, T]
            
            # Zero out padded regions in z_1 for stability
            # z_1 = z_1 * latent_mask # ALREADY DONE ABOVE

            _, D_text, T_txt = h_text.shape 

            # ---------------------------------------------
            # SPFM: Self-Purifying Flow Matching (paper-aligned)
            # ---------------------------------------------
            spfm_mask = torch.ones(B, 1, 1, device=device)  # keep all by default
            spfm_start = spfm_start_override if spfm_start_override is not None else 40_000
            end_spfm = max_steps
            # spfm_every = 4        # Run periodically for speed and stability

            if global_step >= spfm_start and global_step <= end_spfm:
                # Optimized SPFM: Use existing tensors (cheaper)
                # No eval() switch, no re-computation
                
                with torch.no_grad():
                    # Reuse computed conditions
                    h_text_spfm = h_text
                    ref_values_spfm = ref_values

                    _, D_text_spfm, T_txt_spfm = h_text_spfm.shape

                    # Probe time t' = Fixed 0.5 (Paper recommendation)
                    t_spfm = torch.full((B,), 0.5, device=device)
                    t_b = t_spfm.view(B, 1, 1)

                    # Fresh noise x0
                    x0 = torch.randn(B, C, T, device=device)

                    # Same interpolation rule (sigma_min from config: ttl.flow_matching.sig_min)
                    x_t = (1 - (1 - sigma_min) * t_b) * x0 + t_b * z_1
                    v_target_spfm = z_1 - (1 - sigma_min) * x0

                    # Mask x_t before VF to avoid padding noise affecting the decision
                    x_t_in = x_t * latent_mask

                    v_cond = vf_estimator(
                        noisy_latent=x_t_in,
                        text_emb=h_text_spfm,
                        style_ttl=ref_values_spfm,
                        latent_mask=latent_mask,
                        text_mask=text_masks,
                        current_step=t_spfm,
                    )

                    # Unconditional tensors
                    u_text_spfm = u_text.expand(B, D_text_spfm, 1)
                    u_ref_spfm  = u_ref.expand(B, -1, -1)
                    u_text_mask_spfm = torch.ones(B, 1, 1, device=device)

                    v_uncond = vf_estimator(
                        noisy_latent=x_t_in,
                        text_emb=u_text_spfm,
                        style_ttl=u_ref_spfm,
                        latent_mask=latent_mask,
                        text_mask=u_text_mask_spfm,
                        current_step=t_spfm,
                    )

                    # Loss mask: padding AND self-ref hole
                    final_mask_spfm = latent_mask * target_loss_mask          # [B,1,T]
                    mask_ct = final_mask_spfm.expand(-1, C, -1)               # [B,C,T]
                    denom = (final_mask_spfm.sum(dim=(1,2)) * C).clamp_min(1) # [B]

                    # MSE for Decision (Paper-aligned)
                    err_c2 = (v_cond   - v_target_spfm).pow(2)
                    err_u2 = (v_uncond - v_target_spfm).pow(2)

                    L_cond   = (err_c2 * mask_ct).sum(dim=(1,2)) / denom
                    L_uncond = (err_u2 * mask_ct).sum(dim=(1,2)) / denom

                    is_dirty_candidate = (L_cond > L_uncond)
                    spfm_score = L_cond - L_uncond

                    # No Top-K filtering on detection (Pure "honest" detection)
                    spfm_mask = torch.ones(B, 1, 1, device=device)
                    dirty_indices = torch.where(is_dirty_candidate)[0]
                    
                    if dirty_indices.numel() > 0:
                         spfm_mask[dirty_indices] = 0.0

                    # Log raw dirty count for visibility
                    raw_dirty_count = dirty_indices.numel()
                    
                    if global_step % 1000 == 0:
                        print(f"[SPFM] Detected Dirty: {raw_dirty_count}/{B}")

                # 4) Diagnostics: use TEXT length and LATENT length (not waveform length)
                dirty_bool = (spfm_mask.squeeze(-1).squeeze(-1) < 0.5)
                dirty_count = dirty_bool.sum().item()
                
                spfm_dirty_total += dirty_count
                spfm_total_samples += B
                spfm_score_sum += spfm_score.mean().item()
                spfm_call_batches += 1

                if global_step % 1000 == 0 and rank == 0:
                    clean_bool = ~dirty_bool

                    txt_len = text_masks.sum(dim=(1, 2)).float()   # tokens
                    lat_len = valid_z_len.float()                  # latent frames

                    avg_txt_clean = txt_len[clean_bool].mean().item() if clean_bool.any() else 0.0
                    avg_txt_dirty = txt_len[dirty_bool].mean().item() if dirty_bool.any() else 0.0
                    avg_lat_clean = lat_len[clean_bool].mean().item() if clean_bool.any() else 0.0
                    avg_lat_dirty = lat_len[dirty_bool].mean().item() if dirty_bool.any() else 0.0
                    
                    print(
                        f"\n[SPFM Diag] Step {global_step} | Dirty: {dirty_count}/{B} ({dirty_count/B:.1%}) | "
                        f"Score mean: {spfm_score.mean().item():.3f} | "
                        f"TxtLen clean/dirty: {avg_txt_clean:.1f}/{avg_txt_dirty:.1f} | "
                        f"LatLen clean/dirty: {avg_lat_clean:.1f}/{avg_lat_dirty:.1f}"
                    )

            z_1_exp = z_1.repeat_interleave(Ke, dim=0)
            h_text_exp = h_text.repeat_interleave(Ke, dim=0)
            ref_values_exp = ref_values.repeat_interleave(Ke, dim=0)
            text_masks_base_exp = text_masks.repeat_interleave(Ke, dim=0)
            latent_mask_exp = latent_mask.repeat_interleave(Ke, dim=0)
            target_loss_mask_exp = target_loss_mask.repeat_interleave(Ke, dim=0)
            spfm_mask_exp = spfm_mask.repeat_interleave(Ke, dim=0)

            B_eff = B * Ke
            t = torch.rand(B_eff, device=device)
            with torch.no_grad():
                x_0 = torch.randn(B_eff, C, T, device=device)
                t_broad = t.view(B_eff, 1, 1)
                x_t = (1 - (1 - sigma_min) * t_broad) * x_0 + t_broad * z_1_exp
                v_target = z_1_exp - (1 - sigma_min) * x_0

            cfg_rand = torch.rand(B_eff, device=device)
            force_text_uncond = cfg_rand < puncond
            force_style_uncond = cfg_rand < prob_both_uncond
            if spfm_mask_exp is not None:
                is_dirty = (spfm_mask_exp.view(B_eff) < 0.5)
                force_text_uncond = force_text_uncond | is_dirty
                force_style_uncond = force_style_uncond | is_dirty

            mask_text_uncond = force_text_uncond.view(-1, 1, 1).float()
            mask_text_cond = 1.0 - mask_text_uncond
            mask_style_uncond = force_style_uncond.view(-1, 1, 1).float()
            mask_style_cond = 1.0 - mask_style_uncond

            u_text_batch = F.pad(u_text, (0, T_txt - 1)).expand(B_eff, -1, -1)
            h_context = h_text_exp * mask_text_cond + u_text_batch * mask_text_uncond

            mask_uncond_valid = torch.zeros_like(text_masks_base_exp)
            mask_uncond_valid[:, :, 0] = 1.0
            text_mask_final = text_masks_base_exp * mask_text_cond + mask_uncond_valid * mask_text_uncond

            u_ref_batch = u_ref.expand(B_eff, -1, -1)
            ref_values_final = ref_values_exp * mask_style_cond + u_ref_batch * mask_style_uncond

            v_pred = vf_estimator(
                noisy_latent=x_t * latent_mask_exp, text_emb=h_context, style_ttl=ref_values_final,
                latent_mask=latent_mask_exp, text_mask=text_mask_final, current_step=t
            )

            final_mask = latent_mask_exp * target_loss_mask_exp
            loss_raw = F.l1_loss(v_pred, v_target, reduction='none')
            mask_ct = final_mask.expand(-1, C, -1)
            loss = (loss_raw * mask_ct).sum() / (mask_ct.sum() + 1e-8)
            loss = loss / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(params, 10.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            epoch_loss += loss.item() * accumulation_steps
            num_batches += 1
            progress_bar.set_postfix(loss=epoch_loss / num_batches, step=global_step, lr=scheduler.get_last_lr()[0])

            if global_step % 1000 == 0 and rank == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"ckpt_step_{global_step}.pt")
                torch.save({
                    'vf_estimator': vf_estimator.module.state_dict() if isinstance(vf_estimator, DDP) else vf_estimator.state_dict(),
                    'text_encoder': text_encoder.module.state_dict() if isinstance(text_encoder, DDP) else text_encoder.state_dict(),
                    'reference_encoder': reference_encoder.module.state_dict() if isinstance(reference_encoder, DDP) else reference_encoder.state_dict(),
                    'u_text': u_text.data, 'u_ref': u_ref.data,
                    'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'global_step': global_step
                }, ckpt_path)

                # Unwrap for inference to avoid DDP sync issues on single rank
                vf_infer = vf_estimator.module if isinstance(vf_estimator, DDP) else vf_estimator
                te_infer = text_encoder.module if isinstance(text_encoder, DDP) else text_encoder
                re_infer = reference_encoder.module if isinstance(reference_encoder, DDP) else reference_encoder
                
                vf_infer.eval()
                te_infer.eval()
                re_infer.eval()
                
                try:
                    # Hebrew sentences (pre-computed IPA)
                    hebrew_sentences = [
                        "ʃalˈom janˈon kˈaχa niʃmˈa hamˈodel heχadˈaʃ mˈa daʔtχˈa ? lifʔamˈim tsaʁˈiχ baχajˈim lelatˈeʃ ʁaʔjˈon ʃˈuv vaʃˈuv ʔˈad ʃehˈu matslˈiaχ"
                    ]

                    # English sentences — phonemize with espeak at inference time
                    english_sentences_raw = [
                        "Hello, how does the new model sound to you? Sometimes in life you need to push an idea again and again until it succeeds."
                    ]
                    try:
                        from phonemizer.backend import EspeakBackend
                        from phonemizer.separator import Separator
                        _sep = Separator(phone='', word=' ', syllable='')
                        _en_backend = EspeakBackend(
                            'en-us', preserve_punctuation=True,
                            with_stress=True, language_switch='remove-flags',
                        )
                        english_sentences = [
                            normalize_text(s)
                            for s in _en_backend.phonemize(english_sentences_raw, separator=_sep)
                        ]
                    except Exception as _e:
                        print(f"[Inference] English phonemization failed: {_e}")
                        english_sentences = []

                    # German sentences
                    german_sentences_raw = [
                        "Hallo, wie klingt das neue Modell für dich? Manchmal muss man eine Idee immer wieder versuchen, bis sie endlich funktioniert."
                    ]
                    try:
                        from phonemizer.backend import EspeakBackend
                        from phonemizer.separator import Separator
                        _de_backend = EspeakBackend(
                            'de', preserve_punctuation=True,
                            with_stress=True, language_switch='remove-flags',
                        )
                        german_sentences = [
                            normalize_text(s)
                            for s in _de_backend.phonemize(german_sentences_raw, separator=Separator(phone='', word=' ', syllable=''))
                        ]
                    except Exception as _e:
                        print(f"[Inference] German phonemization failed: {_e}")
                        german_sentences = []

                    # Italian sentences
                    italian_sentences_raw = [
                        "Ciao, come suona il nuovo modello per te? A volte nella vita bisogna insistere su un'idea ancora e ancora finché non riesce."
                    ]
                    try:
                        from phonemizer.backend import EspeakBackend
                        from phonemizer.separator import Separator
                        _it_backend = EspeakBackend(
                            'it', preserve_punctuation=True,
                            with_stress=True, language_switch='remove-flags',
                        )
                        italian_sentences = [
                            normalize_text(s)
                            for s in _it_backend.phonemize(italian_sentences_raw, separator=Separator(phone='', word=' ', syllable=''))
                        ]
                    except Exception as _e:
                        print(f"[Inference] Italian phonemization failed: {_e}")
                        italian_sentences = []

                    # Spanish sentences
                    spanish_sentences_raw = [
                        "Hola, ¿cómo suena el nuevo modelo para ti? A veces en la vida hay que insistir en una idea una y otra vez hasta que funciona."
                    ]
                    try:
                        from phonemizer.backend import EspeakBackend
                        from phonemizer.separator import Separator
                        _es_backend = EspeakBackend(
                            'es', preserve_punctuation=True,
                            with_stress=True, language_switch='remove-flags',
                        )
                        spanish_sentences = [
                            normalize_text(s)
                            for s in _es_backend.phonemize(spanish_sentences_raw, separator=Separator(phone='', word=' ', syllable=''))
                        ]
                    except Exception as _e:
                        print(f"[Inference] Spanish phonemization failed: {_e}")
                        spanish_sentences = []

                    # Computes style_ttl + style_dp once from the reference wav,
                    # then reuses them across all sentences — matches ONNX inference pattern.
                    def run_inference_for_ref(ref_wav_torch, suffix, sentences, lang, label):
                        if ref_wav_torch is None or not sentences:
                            return

                        with torch.no_grad():
                            ref_mel = mel_spec(ref_wav_torch)
                            ref_z_enc = ae_encoder(ref_mel)
                            ref_z_enc = compress_latents(ref_z_enc, factor=chunk_compress_factor)
                            ref_z_norm = ((ref_z_enc - mean) / std) * normalizer_scale

                            B_ref, _, T_ref = ref_z_norm.shape
                            valid_z_len_ref = torch.tensor([T_ref], device=device)
                            ref_z, ref_mask = build_reference_only(ref_z_norm, valid_z_len_ref, device, max_frames=256)

                            # Extract style tokens once — reused for all sentences
                            style_ttl = re_infer(ref_z, mask=ref_mask)      # [1, 50, 256]
                            style_dp = None
                            if dp_model is not None:
                                style_dp = dp_model.ref_encoder(ref_z, mask=ref_mask)  # [1, 128]
                                style_dp = style_dp.reshape(B_ref, 8, 16)              # [1, 8, 16]

                        for i, text in enumerate(sentences):
                            ids = text_to_indices_multilang(text, base_lang=lang)
                            txt_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
                            txt_mask = torch.ones(1, 1, txt_ids.shape[1], device=device)

                            wav_out = sample_audio(
                                vf_infer, te_infer, re_infer, ae_decoder,
                                txt_ids, txt_mask,
                                z_ref=None, ref_enc_mask=None,
                                mean=mean, std=std,
                                duration_predictor=dp_model,
                                steps=16,
                                device=str(device) if isinstance(device, torch.device) else device,
                                debug_label=f"{label}_{suffix}",
                                latent_dim=latent_dim,
                                chunk_compress_factor=chunk_compress_factor,
                                normalizer_scale=normalizer_scale,
                                style_ttl=style_ttl,
                                style_dp=style_dp,
                                uncond_params=uncond_params,
                                cfg_scale=3.0,
                            )
                            wav = wav_out.squeeze().cpu().numpy()
                            sf.write(os.path.join(log_dir, f"step_{global_step}_{label}_{i+1}_{suffix}.wav"), wav, ae_sample_rate)

                    # Run for Voice 1
                    if 'ref_wav_torch_v1' in locals():
                        run_inference_for_ref(ref_wav_torch_v1, "voice1", hebrew_sentences, "he", "hebrew")
                        run_inference_for_ref(ref_wav_torch_v1, "voice1", english_sentences, "en", "english")
                        run_inference_for_ref(ref_wav_torch_v1, "voice1", german_sentences, "de", "german")
                        run_inference_for_ref(ref_wav_torch_v1, "voice1", italian_sentences, "it", "italian")
                        run_inference_for_ref(ref_wav_torch_v1, "voice1", spanish_sentences, "es", "spanish")

                    # Run for Validation Batch
                    if val_batch is not None and val_wavs is not None and val_text_ids is not None and val_text_masks is not None and val_ref_wavs is not None and val_ref_lengths is not None and val_z_ref is not None and val_ref_enc_mask is not None:
                        with torch.no_grad():
                            ref_z_val = val_z_ref[0:1] if val_z_ref is not None else None
                            ref_mask_val = val_ref_enc_mask[0:1] if val_ref_enc_mask is not None else None
                            val_style_ttl = re_infer(ref_z_val, mask=ref_mask_val) if ref_z_val is not None else None  # [1, 50, 256]
                            val_style_dp = None
                            if dp_model is not None and ref_z_val is not None:
                                val_style_dp = dp_model.ref_encoder(ref_z_val, mask=ref_mask_val)  # [1, 128]
                                val_style_dp = val_style_dp.reshape(1, 8, 16)

                        for lang, sentences, label in [
                            ("he", hebrew_sentences, "hebrew"),
                            ("en", english_sentences, "english"),
                            ("de", german_sentences, "german"),
                            ("it", italian_sentences, "italian"),
                            ("es", spanish_sentences, "spanish"),
                        ]:
                            for i, text in enumerate(sentences):
                                ids = text_to_indices_multilang(text, base_lang=lang)
                                txt_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
                                txt_mask = torch.ones(1, 1, txt_ids.shape[1], device=device)

                                wav_out = sample_audio(
                                    vf_infer, te_infer, re_infer, ae_decoder,
                                    txt_ids, txt_mask,
                                    z_ref=None, ref_enc_mask=None,
                                    mean=mean, std=std,
                                    duration_predictor=dp_model,
                                    steps=16,
                                    device=str(device) if isinstance(device, torch.device) else device,
                                    debug_label=f"{label}_val_sample",
                                    latent_dim=latent_dim,
                                    chunk_compress_factor=chunk_compress_factor,
                                normalizer_scale=normalizer_scale,
                                style_ttl=val_style_ttl,
                                style_dp=val_style_dp,
                                uncond_params=uncond_params,
                                cfg_scale=3.0,
                            )
                                wav = wav_out.squeeze().cpu().numpy()
                                sf.write(os.path.join(log_dir, f"step_{global_step}_{label}_{i+1}_val_sample.wav"), wav, ae_sample_rate)

                    # VC Check: val_batch[0] content → reference.wav speaker
                    # Saves source audio + converted output so content preservation can be verified.
                    vc_ref_path = "reference.wav"
                    if val_batch is not None and os.path.exists(vc_ref_path):
                        try:
                            # Save source for comparison
                            sf.write(
                                os.path.join(log_dir, f"step_{global_step}_vc_source.wav"),
                                val_wavs[0].squeeze().cpu().numpy() if val_wavs is not None else None,
                                ae_sample_rate,
                            )

                            # Load reference.wav
                            vc_ref_np, vc_ref_sr = sf.read(vc_ref_path)
                            vc_ref_wav = torch.from_numpy(vc_ref_np).float()
                            if vc_ref_wav.dim() > 1:
                                vc_ref_wav = vc_ref_wav.mean(dim=-1)
                            vc_ref_wav = ensure_sr(vc_ref_wav, vc_ref_sr, ae_sample_rate, device=device)
                            if vc_ref_wav.dim() == 1:
                                vc_ref_wav = vc_ref_wav.unsqueeze(0)  # [1, T]

                            # Build reference latent
                            with torch.no_grad():
                                vc_ref_mel = mel_spec(vc_ref_wav)
                                vc_ref_z = ae_encoder(vc_ref_mel)
                                vc_ref_z = compress_latents(vc_ref_z, factor=chunk_compress_factor)
                                vc_ref_z_norm = ((vc_ref_z - mean) / std) * normalizer_scale
                                vc_ref_z_built, vc_ref_mask = build_reference_only(
                                    vc_ref_z_norm,
                                    torch.tensor([vc_ref_z_norm.shape[2]], device=device),
                                    device,
                                    max_frames=256
                                )

                            # Pre-extract target speaker style values
                            with torch.no_grad():
                                vc_style_ttl = re_infer(vc_ref_z_built, mask=vc_ref_mask)
                                vc_style_dp = None
                                if dp_model is not None:
                                    vc_style_dp = dp_model.ref_encoder(vc_ref_z_built, mask=vc_ref_mask)
                                    vc_style_dp = vc_style_dp.reshape(1, 8, 16)

                            wav_vc = sample_audio(
                                vf_infer, te_infer, re_infer, ae_decoder,
                                val_text_ids[0:1] if val_text_ids is not None else None, 
                                val_text_masks[0:1] if val_text_masks is not None else None,
                                z_ref=None, ref_enc_mask=None,
                                mean=mean, std=std,
                                duration_predictor=dp_model,
                                steps=16,
                                device=str(device) if isinstance(device, torch.device) else device, debug_label="vc",
                                latent_dim=latent_dim,
                                chunk_compress_factor=chunk_compress_factor,
                                normalizer_scale=normalizer_scale,
                                style_ttl=vc_style_ttl,
                                style_dp=vc_style_dp,
                                uncond_params=uncond_params,
                                cfg_scale=3.0,
                            )
                            sf.write(
                                os.path.join(log_dir, f"step_{global_step}_vc_output.wav"),
                                wav_vc.squeeze().cpu().numpy(),
                                ae_sample_rate,
                            )
                        except Exception as _vc_e:
                            print(f"[Inference] VC check failed: {_vc_e}")

                except Exception as e:
                    print(f"Inference failed: {e}")
                    import traceback
                    traceback.print_exc()
                
                vf_estimator.train()
                text_encoder.train()
                reference_encoder.train()

        # Flush remaining gradients if dataloader length is not divisible by accumulation_steps
        if num_batches % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        if spfm_call_batches > 0:
            if dist.is_initialized():
                # Aggregate SPFM stats across all GPUs
                spfm_stats = torch.tensor([
                    spfm_dirty_total, 
                    spfm_total_samples, 
                    spfm_score_sum, 
                    spfm_call_batches
                ], dtype=torch.float64, device=device)
                dist.all_reduce(spfm_stats, op=dist.ReduceOp.SUM)
                spfm_dirty_total = spfm_stats[0].item()
                spfm_total_samples = spfm_stats[1].item()
                spfm_score_sum = spfm_stats[2].item()
                spfm_call_batches = spfm_stats[3].item()

        if spfm_call_batches > 0 and rank == 0:
            epoch_dirty_rate = spfm_dirty_total / max(spfm_total_samples, 1)
            epoch_score_mean = spfm_score_sum / spfm_call_batches
            print(
                f"[SPFM Epoch] step={global_step} "
                f"dirty_rate={epoch_dirty_rate:.3f} "
                f"score_mean={epoch_score_mean:.3f} "
                f"batches={spfm_call_batches}"
            )

    if rank == 0:
        print("Training complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune", action="store_true", 
                        help="Finetune mode: lr=5e-4, SPFM starts after warm-up")
    parser.add_argument("--config", type=str, default="configs/tts.json",
                        help="Path to tts.json config file (default: configs/tts.json)")
    parser.add_argument("--Ke", type=int, default=None,
                        help="Override batch expansion factor (default: from config)")
    parser.add_argument("--accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps (default: 1)")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/text2latent",
        help="Directory for ckpt_step_*.pt and logs (default: checkpoints/text2latent)",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="If set and checkpoint_dir has no ckpt_step_*.pt, load latest from this directory",
    )
    parser.add_argument(
        "--unicode_indexer",
        type=str,
        default=None,
        help="Path to unicode_indexer.json. If provided, switches to character-level mode.",
    )
    parser.add_argument(
        "--inference_ref_wav",
        type=str,
        default=None,
        help="WAV for Voice 1 validation inference logs (or set T2L_INFERENCE_REF_WAV).",
    )
    args = parser.parse_args()

    set_seed(42)
    train(
        finetune=args.finetune,
        config_path=args.config,
        Ke=args.Ke,
        accumulation_steps=args.accumulation_steps,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
        inference_ref_wav=args.inference_ref_wav,
    )
