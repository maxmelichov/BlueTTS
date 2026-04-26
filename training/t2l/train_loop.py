"""Main T2L `train` loop: data, SPFM, flow-matching loss, checkpointing, periodic inference."""

from __future__ import annotations

import os
import json
import numpy as np
import soundfile as sf
from typing import Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, WeightedRandomSampler, DistributedSampler
from torch.optim import AdamW
from tqdm import tqdm

from training.t2l.data_module import Text2LatentDataset, collate_text2latent
from training.data.audio_utils import ensure_sr
from training.data.text_vocab import normalize_text, text_to_indices_multilang
from training.utils import compress_latents
from training.t2l.builders import build_models
from training.t2l.cfg_utils import (
    _latest_ckpt_in_dir,
    _validate_ttl_config,
    ddp_state_dict,
    seed_worker,
    unwrap_ddp,
)
from training.t2l.sampling import build_reference_only, build_reference_from_latents, sample_audio
from training.t2l.spfm import spfm_forward_mask

# --- periodic checkpoint log phrases (Hebrew IPA + espeak for other langs) ---

_PHRASE_ROWS: list[tuple[str, str | None, str, list[str]]] = [
    (
        "he",
        None,
        "hebrew",
        [
            "ʃalˈom janˈon kˈaχa niʃmˈa hamˈodel heχadˈaʃ mˈa daʔtχˈa ? "
            "lifʔamˈim tsaʁˈiχ baχajˈim lelatˈeʃ ʁaʔjˈon ʃˈuv vaʃˈuv ʔˈad ʃehˈu matslˈiaχ"
        ],
    ),
    (
        "en",
        "en-us",
        "english",
        [
            "Hello, how does the new model sound to you? Sometimes in life you need to "
            "push an idea again and again until it succeeds."
        ],
    ),
    (
        "de",
        "de",
        "german",
        [
            "Hallo, wie klingt das neue Modell für dich? Manchmal muss man eine Idee "
            "immer wieder versuchen, bis sie endlich funktioniert."
        ],
    ),
    (
        "it",
        "it",
        "italian",
        [
            "Ciao, come suona il nuovo modello per te? A volte nella vita bisogna "
            "insistere su un'idea ancora e ancora finché non riesce."
        ],
    ),
    (
        "es",
        "es",
        "spanish",
        [
            "Hola, ¿cómo suena el nuevo modelo para ti? A veces en la vida hay que "
            "insistir en una idea una y otra vez hasta que funciona."
        ],
    ),
]


def _phonemize_eval_with_espeak(espeak_lang: str, label: str, lines: list[str]) -> list[str]:
    try:
        from phonemizer.backend import EspeakBackend
        from phonemizer.separator import Separator
        sep = Separator(phone="", word=" ", syllable="")
        backend = EspeakBackend(
            espeak_lang,
            preserve_punctuation=True,
            with_stress=True,
            language_switch="remove-flags",
        )
        return [normalize_text(s) for s in backend.phonemize(lines, separator=sep)]
    except Exception as e:  # noqa: BLE001
        print(f"[Inference] {label} phonemization failed: {e}")
        return []


def build_periodic_eval_phrases() -> list[tuple[str, str, list[str]]]:
    out: list[tuple[str, str, list[str]]] = []
    for lang, espeak, label, raw_lines in _PHRASE_ROWS:
        if espeak is None:
            out.append((lang, label, list(raw_lines)))
        else:
            out.append((lang, label, _phonemize_eval_with_espeak(espeak, label, raw_lines)))
    return out


def train(
    metadata_path="generated_audio/combined_dataset_cleaned_real_data.csv",
    checkpoint_dir="checkpoints/text2latent",
    ae_checkpoint="checkpoints/ae/ae_latest.pt",
    stats_path="stats_multilingual.pt",
    config_path="configs/tts.json",
    epochs=1000,
    max_steps=1_000_000,
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
        print(f"checkpoint_dir={checkpoint_dir}")
        if resume_from:
            print(f"resume_from={resume_from} (used if no ckpt in checkpoint_dir)")
    else:
        log_dir = os.path.join(checkpoint_dir, "logs")

    if finetune:
        lr = 2.5e-4
        spfm_start_override = 10_000
        if rank == 0:
            print(f"[Finetune Mode] lr={lr}, SPFM warm-up offset={spfm_start_override} steps (after resume)")
    else:
        spfm_start_override = None

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

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

    if rank == 0:
        print(
            f"\n{'=' * 60}\nTTL: config={config_path!r} | latent_dim={latent_dim} | "
            f"ccf={chunk_compress_factor} | Ke={Ke} (config {cfg_Ke}) | "
            f"normalizer={normalizer_scale} | sig_min={sigma_min}\n"
            f"  uncond: prob_both={prob_both_uncond} prob_text={prob_text_uncond} total_puncond={puncond}\n"
            f"{'=' * 60}\n"
        )

    if not os.path.exists(stats_path):
        if rank == 0:
            print(f"Error: Stats file {stats_path} not found. Run latent stats (e.g. compute_latent_stats) first.")
        return

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
        if ae_checkpoint.endswith(".safetensors"):
            from safetensors.torch import load_file
            ckpt = load_file(ae_checkpoint, device="cpu")
        else:
            ckpt = torch.load(ae_checkpoint, map_location='cpu')
        if "encoder" in ckpt:
            ae_encoder.load_state_dict(ckpt["encoder"], strict=False)
        elif "state_dict" in ckpt:
            ae_encoder.load_state_dict(ckpt["state_dict"], strict=False)
        elif any(k.startswith("encoder.") for k in ckpt.keys()):
            ae_encoder.load_state_dict(
                {k.removeprefix("encoder."): v for k, v in ckpt.items() if k.startswith("encoder.")},
                strict=False,
            )
        else:
            ae_encoder.load_state_dict(ckpt, strict=False)
        if "decoder" in ckpt:
            ae_decoder.load_state_dict(ckpt["decoder"], strict=False)
        elif any(k.startswith("decoder.") for k in ckpt.keys()):
            ae_decoder.load_state_dict(
                {k.removeprefix("decoder."): v for k, v in ckpt.items() if k.startswith("decoder.")},
                strict=False,
            )

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
        if latest_ckpt is not None and rank == 0:
            print(
                f"No ckpt in checkpoint_dir; loading weights from resume_from: {latest_ckpt}"
            )

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
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except Exception as e:
                if rank == 0:
                    print(f"Warning: Failed to load optimizer state: {e}")
        elif 'optimizer' in checkpoint and finetune and rank == 0:
            print(f"Finetune mode: Skipping optimizer state load to use fresh lr={lr}")
        elif 'optimizer' in checkpoint and shapes_changed and rank == 0:
            print("Warning: Model shapes changed. Skipping optimizer state load.")
        if 'global_step' in checkpoint:
            global_step = checkpoint['global_step']
            if finetune and spfm_start_override is not None:
                spfm_start_override = global_step + spfm_start_override
                if rank == 0:
                    print(
                        f"Finetune: SPFM from global_step >= {spfm_start_override} "
                        f"(resume step was {global_step})"
                    )
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
        weights = torch.from_numpy(sample_weights).double()
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

    if rank == 0:
        print("Starting training loop...")

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
                (
                    wavs, text_ids, text_masks, lengths, speaker_ids,
                    ref_wavs, ref_lengths, is_self_ref, ref_speaker_ids,
                ) = batch
                ref_speaker_ids = ref_speaker_ids.to(device)
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

            if global_step % 1000 == 0:
                with torch.no_grad():
                    z_ref_noise = z_ref.clone()
                    inv_mask = 1.0 - ref_enc_mask
                    z_ref_noise = z_ref_noise + inv_mask * torch.randn_like(z_ref) * 10.0
                    ref_vals_noise = reference_encoder(z_ref_noise, mask=ref_enc_mask)
                    diff = (ref_vals_noise - ref_values).abs().max().item()
                    if diff > 1e-5:
                        print(
                            f"WARNING: ReferenceEncoder is sensitive to padded values! Max Diff: {diff}"
                        )

            h_text = text_encoder(text_ids, ref_values, text_mask=text_masks)
            
            # (latent_mask already computed above; z_1 masked)
            T_txt = h_text.shape[2]

            spfm_out = spfm_forward_mask(
                global_step=global_step,
                max_steps=max_steps,
                spfm_start_override=spfm_start_override,
                z_1=z_1,
                h_text=h_text,
                ref_values=ref_values,
                latent_mask=latent_mask,
                target_loss_mask=target_loss_mask,
                text_masks=text_masks,
                valid_z_len=valid_z_len,
                vf_estimator=vf_estimator,
                u_text=u_text,
                u_ref=u_ref,
                sigma_min=sigma_min,
                device=device,
                B=B,
                rank=rank,
            )
            spfm_mask = spfm_out.spfm_mask
            if spfm_out.ran_spfm:
                spfm_dirty_total += spfm_out.dirty_count
                spfm_total_samples += B
                spfm_score_sum += spfm_out.spfm_score_mean()
                spfm_call_batches += 1

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

            # CFG & SPFM routing: prob_both drops both; (prob_both, puncond) drops text only
            cfg_rand = torch.rand(B_eff, device=device)
            drop_both = cfg_rand < prob_both_uncond
            drop_text_only = (cfg_rand >= prob_both_uncond) & (cfg_rand < puncond)
            force_text_uncond = drop_both | drop_text_only
            force_style_uncond = drop_both
            is_dirty = (spfm_mask_exp.view(B_eff) < 0.5)
            force_text_uncond = force_text_uncond | is_dirty
            force_style_uncond = force_style_uncond | is_dirty

            mask_text_uncond = force_text_uncond.view(-1, 1, 1).float()
            mask_text_cond = 1.0 - mask_text_uncond
            mask_style_uncond = force_style_uncond.view(-1, 1, 1).float()
            mask_style_cond = 1.0 - mask_style_uncond

            u_text_padded = F.pad(u_text, (0, T_txt - 1))
            u_text_batch = u_text_padded.expand(B_eff, -1, -1)
            h_context = h_text_exp * mask_text_cond + u_text_batch * mask_text_uncond

            mask_uncond_valid = torch.zeros_like(text_masks_base_exp)
            mask_uncond_valid[:, :, 0] = 1.0
            text_mask_final = text_masks_base_exp * mask_text_cond + mask_uncond_valid * mask_text_uncond

            u_ref_batch = u_ref.expand(B_eff, -1, -1)
            ref_values_final = ref_values_exp * mask_style_cond + u_ref_batch * mask_style_uncond

            x_t_in = x_t * latent_mask_exp
            v_pred = vf_estimator(
                noisy_latent=x_t_in,
                text_emb=h_context,
                style_ttl=ref_values_final,
                latent_mask=latent_mask_exp,
                text_mask=text_mask_final,
                current_step=t,
            )

            final_mask = latent_mask_exp * target_loss_mask_exp
            loss_raw = F.l1_loss(v_pred, v_target, reduction='none')
            mask_ct = final_mask.expand(-1, C, -1)
            loss = (loss_raw * mask_ct).sum() / (mask_ct.sum() + 1e-8)

            if global_step % 1000 == 0 and rank == 0:
                with torch.no_grad():
                    dirty_rate = (spfm_mask_exp < 0.5).float().mean().item()
                    p_text_uncond_eff = force_text_uncond.float().mean().item()
                    p_style_uncond_eff = force_style_uncond.float().mean().item()
                    print(
                        f"\nStep {global_step} (pre-step) training debug: "
                        f"z1 std={z_1_exp.std().item():.3f} | x0 std={x_0.std().item():.3f} | "
                        f"v_target std={v_target.std().item():.3f} | v_pred std={v_pred.std().item():.3f}\n"
                        f"  final_mask_mean={final_mask.mean().item():.3f} | spfm dirty_rate={dirty_rate:.3f} | "
                        f"eff_text_uncond={p_text_uncond_eff:.3f} | eff_style_uncond={p_style_uncond_eff:.3f}"
                    )

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
            postfix = dict(
                loss=epoch_loss / num_batches,
                step=global_step,
                lr=scheduler.get_last_lr()[0],
            )
            if finetune:
                postfix["mode"] = "FT"
            progress_bar.set_postfix(**postfix)

            if global_step % 1000 == 0 and rank == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"ckpt_step_{global_step}.pt")
                torch.save({
                    "vf_estimator": ddp_state_dict(vf_estimator),
                    "text_encoder": ddp_state_dict(text_encoder),
                    "reference_encoder": ddp_state_dict(reference_encoder),
                    "u_text": u_text.data,
                    "u_ref": u_ref.data,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "global_step": global_step,
                }, ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")
                print("Running inference (periodic log)...")

                # Unwrap for inference to avoid DDP sync issues on single rank
                vf_infer = unwrap_ddp(vf_estimator)
                te_infer = unwrap_ddp(text_encoder)
                re_infer = unwrap_ddp(reference_encoder)

                vf_infer.eval()
                te_infer.eval()
                re_infer.eval()
                dev = str(device) if isinstance(device, torch.device) else device

                try:
                    eval_phrase_rows = build_periodic_eval_phrases()

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
                                device=dev,
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

                    if ref_wav_torch_v1 is not None:
                        for _lang, _label, sents in eval_phrase_rows:
                            run_inference_for_ref(ref_wav_torch_v1, "voice1", sents, _lang, _label)

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

                        for lang, label, sentences in eval_phrase_rows:
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
                                    device=dev,
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
                                device=dev, debug_label="vc",
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
