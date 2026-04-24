import os
import json
import random
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from training.dp.data_module import get_dp_dataloader
from training.dp.models.dp_network import DPNetwork
from training.data.text_vocab import VOCAB_SIZE
from training.utils import LinearMelSpectrogram, compress_latents
from bluecodec import LatentEncoder


def _load_dp_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        full = json.load(f)
    dp_cfg = full.get("dp")
    if isinstance(dp_cfg, dict) and dp_cfg:
        return {"_full": full, **dp_cfg}
    ttl_cfg = full.get("ttl", {}) if isinstance(full.get("ttl"), dict) else {}
    return {
        "_full": full,
        "latent_dim": ttl_cfg.get("latent_dim", 24),
        "chunk_compress_factor": ttl_cfg.get("chunk_compress_factor", 6),
        "normalizer": {"scale": ttl_cfg.get("normalizer", {}).get("scale", 1.0)},
        "style_encoder": {"style_token_layer": {"n_style": 8, "style_value_dim": 16}},
    }


def _build_dp_models(
    full_cfg,
    style_dp,
    style_dim,
    sentence_encoder_cfg,
    style_encoder_cfg,
    predictor_cfg,
    ae_checkpoint,
    device,
):
    ae_enc_arch = full_cfg["ae"]["encoder"]
    ae_spec_cfg = ae_enc_arch.get("spec_processor", {})
    mel_spec = LinearMelSpectrogram(
        sample_rate=ae_spec_cfg.get("sample_rate", 44100),
        n_fft=ae_spec_cfg.get("n_fft", 2048),
        win_length=ae_spec_cfg.get("win_length", ae_spec_cfg.get("n_fft", 2048)),
        hop_length=ae_spec_cfg.get("hop_length", 512),
        n_mels=ae_spec_cfg.get("n_mels", 228),
    ).to(device)
    ae_encoder = LatentEncoder(cfg=ae_enc_arch).to(device)

    if os.path.exists(ae_checkpoint):
        print(f"Loading AE checkpoint from {ae_checkpoint}")
        if ae_checkpoint.endswith(".safetensors"):
            from safetensors.torch import load_file
            ckpt = load_file(ae_checkpoint)
        else:
            ckpt = torch.load(ae_checkpoint, map_location="cpu")

        if "encoder" in ckpt:
            ae_encoder.load_state_dict(ckpt["encoder"])
        elif "state_dict" in ckpt:
            ae_encoder.load_state_dict(ckpt["state_dict"], strict=False)
        elif any(k.startswith("encoder.") for k in ckpt.keys()):
            enc_dict = {k.replace("encoder.", ""): v for k, v in ckpt.items() if k.startswith("encoder.")}
            ae_encoder.load_state_dict(enc_dict)
        else:
            try:
                ae_encoder.load_state_dict(ckpt)
            except Exception as e:
                print(f"Could not load AE encoder: {e}")
    else:
        print("Warning: AE checkpoint not found. Latent quality may be poor.")

    ae_encoder.eval().requires_grad_(False)
    mel_spec.eval()

    model = DPNetwork(
        vocab_size=VOCAB_SIZE,
        style_dp=style_dp,
        style_dim=style_dim,
        sentence_encoder_cfg=sentence_encoder_cfg,
        style_encoder_cfg=style_encoder_cfg,
        predictor_cfg=predictor_cfg,
    ).to(device)

    return mel_spec, ae_encoder, model

def train_duration_predictor(
    metadata_path: str = "generated_audio/combined_dataset_cleaned_real_data.csv",
    checkpoint_dir: str = "checkpoints/duration_predictor",
    ae_checkpoint: str = "checkpoints/ae/blue_codec.safetensors",
    stats_path: str = "stats_multilingual.pt",
    config_path: str = "configs/tts.json",
    max_steps: int = 1000,
    batch_size: int = 64,
    lr: float = 1e-4,
    device: str = "cuda:1" if torch.cuda.is_available() else "cpu",
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Initializing Duration Predictor training on {device}...")
    dp_cfg = _load_dp_config(config_path)
    full_cfg = dp_cfg.get("_full", {})
    latent_dim = int(dp_cfg.get("latent_dim", 24))
    chunk_compress_factor = int(dp_cfg.get("chunk_compress_factor", 6))
    normalizer_scale = float(dp_cfg.get("normalizer", {}).get("scale", 1.0))
    stl = dp_cfg.get("style_encoder", {}).get("style_token_layer", {})
    style_dp = int(stl.get("n_style", 8))
    style_dim = int(stl.get("style_value_dim", 16))
    sentence_encoder_cfg = dp_cfg.get("sentence_encoder", {})
    style_encoder_cfg = dp_cfg.get("style_encoder", {})
    predictor_cfg = dp_cfg.get("predictor", {})
    compressed_channels = latent_dim * chunk_compress_factor
    
    print(f"\n{'='*60}")
    print(f"DP Config loaded from: {config_path}")
    print(f"  Version: {full_cfg.get('tts_version', 'unknown')}")
    print(f"  Split: {full_cfg.get('split', 'unknown')}")
    print(f"  latent_dim={latent_dim}, chunk_compress_factor={chunk_compress_factor}")
    print(f"  compressed_channels={compressed_channels}")
    print(f"  normalizer_scale={normalizer_scale}")
    print(f"  style_dp={style_dp}, style_dim={style_dim}")
    print(f"{'='*60}\n")
    
    mel_spec, ae_encoder, model = _build_dp_models(
        full_cfg, style_dp, style_dim, sentence_encoder_cfg, style_encoder_cfg, predictor_cfg, ae_checkpoint, device
    )
    
    optimizer = AdamW(model.parameters(), lr=lr)
    dataloader = get_dp_dataloader(metadata_path, batch_size)
    
    if not os.path.exists(stats_path):
        print(f"Error: Stats file {stats_path} not found.")
        return
    stats = torch.load(stats_path, map_location=device)
    if "mean" in stats and hasattr(stats["mean"], "dim") and stats["mean"].dim() == 3:
        mean = stats["mean"].to(device)
        std = stats["std"].to(device)
    else:
        mean = stats["mean"].to(device).view(1, -1, 1)
        std = stats["std"].to(device).view(1, -1, 1)
    if mean.shape[1] != compressed_channels:
        print(f"Warning: stats channels ({mean.shape[1]}) != expected compressed_channels ({compressed_channels}).")
        
    global_step = 0
    mean_loss = 0.0
    print("Starting DP training loop...")
    while global_step < max_steps:
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Step {global_step}/{max_steps}")
        for batch in progress_bar:
            if global_step >= max_steps:
                break
            wavs, text_ids, text_masks, lengths, speaker_ids = batch
            wavs = wavs.to(device)          # [B, 1, T_wav]
            text_ids = text_ids.to(device)  # [B, T_text]
            text_masks = text_masks.to(device)  # [B, 1, T_text]
            speaker_ids = speaker_ids.to(device) # [B]
            
            B = wavs.shape[0]
            # -------------------------------------------------
            # 1) Latent extraction + normalization
            # -------------------------------------------------
            with torch.no_grad():
                mel = mel_spec(wavs.squeeze(1))       # [B, 228, T_mel]
                z = ae_encoder(mel)                   # [B, 24, T_lat]
                z = compress_latents(z, factor=chunk_compress_factor)  # [B, Cc, T_lat_c]
                # normalized compressed latents (+ optional extra normalizer scale)
                z = ((z - mean) / std) * normalizer_scale

            B, C, T_lat = z.shape

            # Compute valid latent length from waveform length
            valid_mel_len = lengths.to(device).float() / 512
            valid_z_len = (valid_mel_len / chunk_compress_factor).ceil().long().clamp(min=1, max=T_lat)

            # -------------------------------------------------
            # 2) Build reference segments (5% to 95% of speech)
            # -------------------------------------------------
            # Move to CPU/numpy once to avoid per-sample GPU sync
            vz_np = valid_z_len.cpu().numpy()

            ref_list = []
            ref_len_list = []
            for i in range(B):
                L_i = int(vz_np[i])
                start_min = int(L_i * 0.05)
                start_max = int(L_i * 0.95)
                if start_max <= start_min:
                    start = 0
                    end = L_i
                else:
                    start = random.randint(start_min, start_max - 1)
                    max_end = max(start + 1, int(L_i * 0.95))
                    seg_len = random.randint(1, max_end - start)
                    end = start + seg_len
                ref_list.append((i, start, end))
                ref_len_list.append(end - start)
            max_ref_len = max(ref_len_list)
            z_ref = torch.zeros(B, C, max_ref_len, device=device)
            ref_mask = torch.zeros(B, 1, max_ref_len, device=device)

            for i, s, e in ref_list:
                L_ref = e - s
                z_ref[i, :, :L_ref] = z[i, :, s:e]
                ref_mask[i, :, :L_ref] = 1.0

            # -------------------------------------------------
            # 3) Forward DPNetwork (in LOG domain)
            # -------------------------------------------------
            log_pred = model(
                text_ids=text_ids,
                z_ref=z_ref,
                text_mask=text_masks,
                ref_mask=ref_mask,
                return_log=True,
            )
            if global_step == 0:
                print(f"[Sanity] Pred shape: {log_pred.shape}") # Should be [B]
                print("raw speaker examples:", speaker_ids[:10])
                print("mapped speaker examples:", speaker_ids[:10])
                print("unique mapped:", speaker_ids.unique())

            # -------------------------------------------------
            # 4) Target and loss (in LOG domain)
            # -------------------------------------------------
            dur_gt = valid_z_len.float() * (512 * 6) / 44100.0  # [B] (seconds)
            log_gt = torch.log(dur_gt.clamp(min=1e-5))

            # L1 loss on log duration
            loss = F.l1_loss(log_pred, log_gt)

            if global_step % 20 == 0:
                # Sanity check print (convert back to linear for display)
                pred_linear = torch.exp(log_pred[:4]).detach()
                gt_linear = dur_gt[:4].detach()
                print(f"\n[Step {global_step}]")
                print(f"  Pred (Lin): {pred_linear.cpu().numpy()}")
                print(f"  Target:     {gt_linear.cpu().numpy()}")
                print(f"  Loss (Log): {loss.item():.4f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            mean_loss += loss.item()
            progress_bar.set_postfix(loss=mean_loss / (global_step + 1), global_step=global_step)
            if global_step % 500 == 0:
                save_path = os.path.join(checkpoint_dir, f"duration_predictor_{global_step}.pt")
                torch.save(model.state_dict(), save_path)
                print(f"Saved DP checkpoint to {save_path}")
                
    final_path = os.path.join(checkpoint_dir, "duration_predictor_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Duration Predictor training complete. Saved to {final_path}")
