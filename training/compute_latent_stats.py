import os
import sys
import json
import argparse
from collections.abc import Mapping

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.text2latent_dataset import Text2LatentDataset, collate_text2latent
from bluecodec import LatentEncoder
from bluecodec.utils import LinearMelSpectrogram, compress_latents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tts-json",
        default=os.path.join(os.path.dirname(__file__), "configs", "tts.json"),
        help="Path to the TTS config json",
    )
    parser.add_argument(
        "--ae-ckpt",
        default=None,
        help="Path to the AE checkpoint",
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="Combined/cleaned metadata CSV produced by combine_datasets.py",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Where to write stats_multilingual.pt for this dataset",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing --output")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda:1" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # ---- Load config ----
    tts_json_path = args.tts_json
    if not os.path.exists(tts_json_path):
        print(f"ERROR: TTS config not found: {tts_json_path}")
        return

    with open(tts_json_path, "r", encoding="utf-8") as f:
        tts_cfg = json.load(f)

    ae_cfg_root = tts_cfg.get("ae", {})
    if "encoder" not in ae_cfg_root or "spec_processor" not in ae_cfg_root["encoder"]:
        print(f"ERROR: Unexpected tts.json format: {tts_json_path}")
        return

    spec_cfg   = ae_cfg_root["encoder"]["spec_processor"]
    ae_enc_cfg = ae_cfg_root["encoder"]

    compression_factor = int(tts_cfg.get("ttl", {}).get("chunk_compress_factor", 6))
    sample_rate        = int(ae_cfg_root.get("sample_rate", spec_cfg.get("sample_rate", 44100)))
    hop_length         = int(spec_cfg.get("hop_length", 512))

    print(f"Using TTS config: {tts_json_path}")

    metadata_path = args.metadata
    if not os.path.exists(metadata_path):
        print(f"ERROR: Metadata CSV not found: {metadata_path}")
        return
    print(f"Using metadata: {metadata_path}")

    # ---- Resolve AE checkpoint ----
    if args.ae_ckpt:
        checkpoint_path = args.ae_ckpt
    else:
        checkpoint_path = tts_cfg.get("ae_ckpt_path", "checkpoints/ae/ae_latest_newer.pt")
        if not checkpoint_path or checkpoint_path == "unknown.pt":
            checkpoint_path = "checkpoints/ae/ae_latest_newer.pt"

    output_path = args.output
    if os.path.exists(output_path) and not args.overwrite:
        print(f"ERROR: Refusing to overwrite existing stats file: {output_path}")
        print("Pass --overwrite or choose a new --output path.")
        return
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # ---- Build model configs ----
    mel_args = {
        "sample_rate": int(spec_cfg.get("sample_rate", sample_rate)),
        "n_fft":       int(spec_cfg.get("n_fft", 2048)),
        "hop_length":  hop_length,
        "n_mels":      int(spec_cfg.get("n_mels", 228)),
    }
    ae_cfg = {
        "ksz":              int(ae_enc_cfg.get("ksz", 7)),
        "hdim":             int(ae_enc_cfg.get("hdim", 512)),
        "intermediate_dim": int(ae_enc_cfg.get("intermediate_dim", 2048)),
        "dilation_lst":     ae_enc_cfg.get("dilation_lst", [1] * int(ae_enc_cfg.get("num_layers", 10))),
        "odim":             int(ae_enc_cfg.get("odim", 24)),
        "idim":             int(ae_enc_cfg.get("idim", 1253)),
    }

    latent_dim = int(ae_cfg["odim"] * compression_factor)
    print(
        f"AE/mel settings: sr={sample_rate}, hop={hop_length}, "
        f"n_mels={mel_args['n_mels']}, odim={ae_cfg['odim']}, "
        f"Kc={compression_factor}, latent_dim={latent_dim}"
    )

    # ---- Dataset & DataLoader ----
    dataset = Text2LatentDataset(
        metadata_path=metadata_path,
        sample_rate=sample_rate,
        max_wav_len=sample_rate * 20,
        max_text_len=None,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_text2latent,
    )

    # ---- Build models ----
    mel_spec = LinearMelSpectrogram(**mel_args).to(device)
    encoder  = LatentEncoder(cfg=ae_cfg).to(device)

    if os.path.exists(checkpoint_path):
        print(f"Loading AE checkpoint from {checkpoint_path}")
        if checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            ckpt = load_file(checkpoint_path)
        else:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
        
        encoder_state = ckpt.get("encoder")
        state_dict = ckpt.get("state_dict")
        if isinstance(encoder_state, Mapping):
            encoder.load_state_dict(dict(encoder_state))
        elif isinstance(state_dict, Mapping):
            encoder.load_state_dict(dict(state_dict), strict=False)
        else:
            # Check if keys start with 'encoder.'
            if any(k.startswith("encoder.") for k in ckpt.keys()):
                enc_dict = {k.replace("encoder.", ""): v for k, v in ckpt.items() if k.startswith("encoder.")}
                encoder.load_state_dict(enc_dict)
            else:
                try:
                    encoder.load_state_dict(ckpt)
                except Exception as e:
                    print(f"Could not load AE encoder: {e}")
    else:
        print(f"WARNING: Checkpoint {checkpoint_path} not found.")

    encoder.eval()
    mel_spec.eval()

    # ---- Accumulate per-channel statistics ----
    total_sum    = torch.zeros(latent_dim, device=device)
    total_sq_sum = torch.zeros(latent_dim, device=device)
    total_frames = 0

    print("Computing latent stats...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            wavs, lengths = batch[0].to(device), batch[3]

            mel = mel_spec(wavs.squeeze(1))
            z   = compress_latents(encoder(mel), factor=compression_factor)
            B, C, T_z = z.shape

            # Mask out padding frames based on original audio lengths
            valid_z_len = (
                lengths.to(device).float() / float(hop_length) / compression_factor
            ).ceil().long().clamp(min=1, max=T_z)

            mask    = torch.arange(T_z, device=device).expand(B, T_z) < valid_z_len.unsqueeze(1)
            valid_z = z.permute(0, 2, 1).contiguous()[mask]  # [N_valid, C]

            if valid_z.numel() == 0:
                continue

            total_sum    += valid_z.sum(dim=0)
            total_sq_sum += (valid_z ** 2).sum(dim=0)
            total_frames += valid_z.shape[0]

    if total_frames == 0:
        print("ERROR: No valid frames found.")
        return

    # ---- Compute and save stats ----
    mean = total_sum / total_frames
    std  = torch.sqrt(torch.clamp((total_sq_sum / total_frames) - mean ** 2, min=1e-8))

    print(f"Global mean: {mean.mean().item():.4f}, Global std: {std.mean().item():.4f}")

    torch.save(
        {
            "mean":      mean.cpu(),
            "std":       std.cpu(),
            "Kc":        compression_factor,
            "latent_dim": latent_dim,
        },
        output_path,
    )
    print(f"Saved latent stats to {output_path}")


if __name__ == "__main__":
    main()
