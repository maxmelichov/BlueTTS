import os
import sys
import random
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from tqdm import tqdm
from functools import partial

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.text2latent_dataset import Text2LatentDataset, collate_text2latent
from data.text_vocab import CHAR_TO_ID, VOCAB_SIZE
from models.utils import LinearMelSpectrogram, compress_latents
from bluecodec.src.bluecodec.autoencoder import LatentEncoder
from models.text2latent.dp_network import DPNetwork

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

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

def collate_dp(batch, spk2idx=None, unknown_spk=0):
    wavs = [b[0].reshape(-1) for b in batch]
    texts = [b[1] for b in batch]
    speaker_ids_raw = [b[2] for b in batch]
    speaker_ids = []
    for s in speaker_ids_raw:
        if spk2idx is not None and s in spk2idx:
            speaker_ids.append(int(spk2idx[s]))
        elif spk2idx is not None:
            try:
                s_int = int(s)
                speaker_ids.append(int(spk2idx.get(s_int, unknown_spk)))
            except (ValueError, TypeError):
                speaker_ids.append(unknown_spk)
        else:
            speaker_ids.append(int(s))
    B = len(wavs)
    max_wav_len = max(w.numel() for w in wavs)
    max_text_len = max(t.numel() for t in texts)
    wavs_padded = torch.zeros(B, 1, max_wav_len, dtype=wavs[0].dtype)
    wav_lengths = torch.empty(B, dtype=torch.long)
    texts_padded = torch.zeros(B, max_text_len, dtype=texts[0].dtype)
    text_masks = torch.zeros(B, 1, max_text_len, dtype=torch.float32)
    for i, (w, t) in enumerate(zip(wavs, texts)):
        wl = w.numel()
        tl = t.numel()
        wavs_padded[i, 0, :wl] = w
        wav_lengths[i] = wl
        texts_padded[i, :tl] = t
        text_masks[i, 0, :tl] = 1.0
    speaker_ids_tensor = torch.tensor(speaker_ids, dtype=torch.long)
    return wavs_padded, texts_padded, text_masks, wav_lengths, speaker_ids_tensor

def train_duration_predictor(
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
    style_tokens = int(stl.get("n_style", 8))
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
    print(f"  style_tokens={style_tokens}, style_dim={style_dim}")
    print(f"{'='*60}\n")
    ae_enc_arch = full_cfg['ae']['encoder']
    ae_spec_cfg = ae_enc_arch.get('spec_processor', {})
    mel_spec = LinearMelSpectrogram(
        sample_rate=ae_spec_cfg.get('sample_rate', 44100),
        n_fft=ae_spec_cfg.get('n_fft', 2048),
        win_length=ae_spec_cfg.get('win_length', ae_spec_cfg.get('n_fft', 2048)),
        hop_length=ae_spec_cfg.get('hop_length', 512),
        n_mels=ae_spec_cfg.get('n_mels', 228),
    ).to(device)
    ae_encoder = LatentEncoder(cfg=ae_enc_arch).to(device)
    if os.path.exists(ae_checkpoint):
        print(f"Loading AE checkpoint from {ae_checkpoint}")
        ckpt = torch.load(ae_checkpoint, map_location="cpu")
        if "encoder" in ckpt:
            ae_encoder.load_state_dict(ckpt["encoder"])
        elif "state_dict" in ckpt:
            ae_encoder.load_state_dict(ckpt["state_dict"], strict=False)
        else:
            ae_encoder.load_state_dict(ckpt)
    else:
        print("Warning: AE checkpoint not found. Latent quality may be poor.")
    ae_encoder.eval()
    ae_encoder.requires_grad_(False)
    mel_spec.eval()
    model = DPNetwork(
        vocab_size=VOCAB_SIZE,
        style_tokens=style_tokens,
        style_dim=style_dim,
        sentence_encoder_cfg=sentence_encoder_cfg,
        style_encoder_cfg=style_encoder_cfg,
        predictor_cfg=predictor_cfg,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    metadata_path = "generated_audio/combined_dataset_cleaned_real_data.csv"
    dataset = Text2LatentDataset(
        metadata_path,
        sample_rate=44100,
        max_wav_len=44100 * 20,
        max_text_len=800,
    )
    speaker_ids = dataset.speaker_ids
    unique_speakers, counts = np.unique(speaker_ids, return_counts=True)
    freq = dict(zip(unique_speakers, counts))
    print(f"Speaker counts: {freq}")
    try:
        spk_raw = np.array(speaker_ids, dtype=np.int64)
        uniq = np.unique(spk_raw)
        spk2idx = {int(s): int(i) for i, s in enumerate(uniq)}
    except Exception as e:
        print(f"Warning: Could not cast speaker_ids to int64 ({e}). Using raw values.")
        uniq = np.unique(speaker_ids)
        spk2idx = {s: int(i) for i, s in enumerate(uniq)}
    num_speakers = len(uniq)
    print("num_speakers mapped:", num_speakers)
    weights = np.array([1.0 / freq[s] for s in speaker_ids], dtype=np.float32)
    weights = torch.from_numpy(weights)
    sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
    collate_fn = partial(collate_dp, spk2idx=spk2idx, unknown_spk=0)
    def worker_init_fn(worker_id):
        np.random.seed(42 + worker_id)
        random.seed(42 + worker_id)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=16,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        worker_init_fn=worker_init_fn
    )
    print(f"Dataset loaded with {len(dataset)} samples.")
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
        print(f"Warning: stats channels ({mean.shape[1]}) != expected compressed_channels ({compressed_channels} = {latent_dim}*{chunk_compress_factor}).")
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
            wavs = wavs.to(device)
            text_ids = text_ids.to(device)
            text_masks = text_masks.to(device)
            speaker_ids = speaker_ids.to(device)
            B = wavs.shape[0]
            with torch.no_grad():
                mel = mel_spec(wavs.squeeze(1))
                z = ae_encoder(mel)
                z = compress_latents(z, factor=chunk_compress_factor)
                z = ((z - mean) / std) * normalizer_scale
            B, C, T_lat = z.shape
            valid_mel_len = lengths.to(device).float() / 512
            valid_z_len = (valid_mel_len / chunk_compress_factor).ceil().long().clamp(min=1, max=T_lat)
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
            log_pred = model(
                text_ids=text_ids,
                z_ref=z_ref,
                text_mask=text_masks,
                ref_mask=ref_mask,
                return_log=True,
            )
            if global_step == 0:
                print(f"[Sanity] Pred shape: {log_pred.shape}")
                print("raw speaker examples:", dataset.speaker_ids[:10])
                print("mapped speaker examples:", speaker_ids[:10])
                print("unique mapped:", speaker_ids.unique())
            dur_gt = valid_z_len.float()
            log_gt = torch.log(dur_gt.clamp(min=1e-5))
            loss = F.l1_loss(log_pred, log_gt)
            if global_step % 20 == 0:
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

if __name__ == "__main__":
    set_seed(42)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/tts.json")
    parser.add_argument("--max_steps", type=int, default=9181)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/duration_predictor")
    parser.add_argument("--ae_checkpoint", type=str, default="checkpoints/ae/blue_codec.safetensors")
    parser.add_argument("--stats_path", type=str, default="stats_multilingual.pt")
    args = parser.parse_args()
    train_duration_predictor(
        config_path=args.config,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=(args.device if args.device is not None else ("cuda:1" if torch.cuda.is_available() else "cpu")),
        checkpoint_dir=args.checkpoint_dir,
        ae_checkpoint=args.ae_checkpoint,
        stats_path=args.stats_path,
    )
