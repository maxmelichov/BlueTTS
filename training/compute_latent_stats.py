import os
import sys
import json
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.text2latent_dataset import Text2LatentDataset, collate_text2latent
from bluecodec import LatentEncoder
from bluecodec.utils import LinearMelSpectrogram, compress_latents

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tts-json", default=os.path.join(os.path.dirname(__file__), "configs", "tts.json"), help="Path to the TTS config json")
    args = parser.parse_args()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    tts_json_path = args.tts_json
    if not os.path.exists(tts_json_path):
        print(f"ERROR: TTS config not found: {tts_json_path}")
        return
    with open(tts_json_path, "r", encoding="utf-8") as f:
        tts_cfg = json.load(f)
    if "ae" not in tts_cfg or "encoder" not in tts_cfg["ae"] or "spec_processor" not in tts_cfg["ae"]["encoder"]:
        print(f"ERROR: Unexpected tts.json format: {tts_json_path}")
        return
    spec_cfg = tts_cfg["ae"]["encoder"]["spec_processor"]
    ae_enc_cfg = tts_cfg["ae"]["encoder"]
    compression_factor = int(tts_cfg.get("ttl", {}).get("chunk_compress_factor", 6))
    sample_rate = int(tts_cfg.get("ae", {}).get("sample_rate", spec_cfg.get("sample_rate", 44100)))
    hop_length = int(spec_cfg.get("hop_length", 512))
    print(f"Using TTS config: {tts_json_path}")
    metadata_candidates = ["generated_audio/combined_dataset_cleaned_real_data.csv"]
    metadata_path = next((p for p in metadata_candidates if os.path.exists(p)), None)
    if metadata_path is None:
        print("ERROR: No metadata CSV found.")
        return
    print(f"Using metadata: {metadata_path}")
    checkpoint_path = tts_cfg.get("ae_ckpt_path", "checkpoints/ae/ae_latest_newer.pt")
    if not checkpoint_path or checkpoint_path == "unknown.pt":
        checkpoint_path = "checkpoints/ae/ae_latest_newer.pt"
    output_path = "stats_real_data.pt"
    mel_args = {
        "sample_rate": int(spec_cfg.get("sample_rate", sample_rate)),
        "n_fft": int(spec_cfg.get("n_fft", 2048)),
        "hop_length": hop_length,
        "n_mels": int(spec_cfg.get("n_mels", 228)),
    }
    ae_cfg = {
        "ksz": int(ae_enc_cfg.get("ksz", 7)),
        "hdim": int(ae_enc_cfg.get("hdim", 512)),
        "intermediate_dim": int(ae_enc_cfg.get("intermediate_dim", 2048)),
        "dilation_lst": ae_enc_cfg.get("dilation_lst", [1] * int(ae_enc_cfg.get("num_layers", 10))),
        "odim": int(ae_enc_cfg.get("odim", 24)),
        "idim": int(ae_enc_cfg.get("idim", 1253)),
    }
    latent_dim = int(ae_cfg["odim"] * compression_factor)
    print(f"AE/mel settings: sr={sample_rate}, hop={hop_length}, n_mels={mel_args['n_mels']}, odim={ae_cfg['odim']}, Kc={compression_factor}, latent_dim={latent_dim}")
    dataset = Text2LatentDataset(metadata_path=metadata_path, sample_rate=sample_rate, max_wav_len=sample_rate * 20, max_text_len=None)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8, collate_fn=collate_text2latent)
    mel_spec = LinearMelSpectrogram(**mel_args).to(device)
    encoder = LatentEncoder(cfg=ae_cfg).to(device)
    if os.path.exists(checkpoint_path):
        print(f"Loading AE checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if "encoder" in ckpt:
            encoder.load_state_dict(ckpt["encoder"])
        elif "state_dict" in ckpt:
            encoder.load_state_dict(ckpt["state_dict"], strict=False)
        else:
            try:
                encoder.load_state_dict(ckpt)
            except Exception as e:
                print(f"Could not load AE encoder: {e}")
    else:
        print(f"WARNING: Checkpoint {checkpoint_path} not found.")
    encoder.eval()
    mel_spec.eval()
    total_sum, total_sq_sum, total_frames = torch.zeros(latent_dim, device=device), torch.zeros(latent_dim, device=device), 0
    print("Computing latent stats...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            wavs, lengths = batch[0].to(device), batch[3]
            mel = mel_spec(wavs.squeeze(1))
            z = compress_latents(encoder(mel), factor=compression_factor)
            B, C, T_z = z.shape

            valid_z_len = (
                lengths.to(device).float() / float(hop_length) / compression_factor
            ).ceil().long().clamp(min=1, max=T_z)

            mask    = torch.arange(T_z, device=device).expand(B, T_z) < valid_z_len.unsqueeze(1)
            valid_z = z.permute(0, 2, 1).contiguous()[mask]

            if valid_z.numel() == 0:
                continue

            total_sum    += valid_z.sum(dim=0)
            total_sq_sum += (valid_z ** 2).sum(dim=0)
            total_frames += valid_z.shape[0]
    if total_frames == 0:
        print("ERROR: No valid frames found.")
        return
    mean = total_sum / total_frames
    std = torch.sqrt(torch.clamp((total_sq_sum / total_frames) - mean ** 2, min=1e-8))
    print(f"Global mean: {mean.mean().item():.4f}, Global std: {std.mean().item():.4f}")
    torch.save({"mean": mean.cpu(), "std": std.cpu(), "Kc": compression_factor, "latent_dim": latent_dim}, output_path)
    print(f"Saved latent stats to {output_path}")

if __name__ == "__main__":
    main()