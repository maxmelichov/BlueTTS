import argparse
import torch

from training.dp.trainer import train_duration_predictor
from training.utils import seed_all

def main():
    seed_all(42, deterministic=False)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/tts.json")
    parser.add_argument("--data", type=str, default="generated_audio/combined_dataset_cleaned_real_data.csv")
    parser.add_argument("--max_steps", type=int, default=9181)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out", type=str, default="checkpoints/duration_predictor")
    parser.add_argument("--ae_checkpoint", type=str, default="checkpoints/ae/blue_codec.safetensors")
    parser.add_argument("--stats_path", type=str, default="stats_multilingual.pt")
    args = parser.parse_args()
    
    train_duration_predictor(
        metadata_path=args.data,
        config_path=args.config,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=(args.device if args.device is not None else ("cuda:1" if torch.cuda.is_available() else "cpu")),
        checkpoint_dir=args.out,
        ae_checkpoint=args.ae_checkpoint,
        stats_path=args.stats_path,
    )

if __name__ == "__main__":
    main()
