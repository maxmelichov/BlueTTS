import argparse
import torch

from training.dp.trainer import train_duration_predictor
from training.utils import seed_all


def main():
    seed_all(42, deterministic=False)
    parser = argparse.ArgumentParser(
        description="Train the paper-aligned utterance duration predictor"
    )
    parser.add_argument("--config", type=str, default="config/tts.json")
    parser.add_argument(
        "--data",
        type=str,
        default="generated_audio/combined_dataset_cleaned_real_data.csv",
    )
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--loss",
        choices=("linear", "log"),
        default="linear",
        help="L1 loss domain: 'linear' (paper, seconds) or 'log' (relative error)",
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Flatten language and text-length skew with weighted sampling",
    )
    parser.add_argument(
        "--max_wav_sec",
        type=float,
        default=30.0,
        help="Keep natural utterances up to this duration; <=0 disables the cap",
    )
    parser.add_argument("--max_text_len", type=int, default=800)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--style_check_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out", type=str, default="checkpoints/duration_predictor")
    parser.add_argument(
        "--ae_checkpoint",
        type=str,
        default=None,
        help="Override ae_ckpt_path from --config",
    )
    parser.add_argument("--stats_path", type=str, default="stats_multilingual.pt")
    args = parser.parse_args()

    train_duration_predictor(
        metadata_path=args.data,
        config_path=args.config,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        loss_domain=args.loss,
        balance_sampling=args.balance,
        max_wav_sec=args.max_wav_sec,
        max_text_len=args.max_text_len,
        num_workers=args.num_workers,
        save_every=args.save_every,
        log_every=args.log_every,
        style_check_every=args.style_check_every,
        seed=args.seed,
        device=(
            args.device
            if args.device is not None
            else ("cuda:0" if torch.cuda.is_available() else "cpu")
        ),
        checkpoint_dir=args.out,
        ae_checkpoint=args.ae_checkpoint,
        stats_path=args.stats_path,
    )


if __name__ == "__main__":
    main()
