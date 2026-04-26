import argparse
import sys
import os
import torch

from training.t2l.trainer import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune", action="store_true",
                        help="Finetune mode: lr=5e-4, SPFM starts after warm-up")
    parser.add_argument("--config", type=str, default="configs/tts.json",
                        help="Path to tts.json config file (default: configs/tts.json)")
    parser.add_argument("--data", type=str, default="generated_audio/combined_dataset_cleaned_real_data.csv",
                        help="Path to training metadata CSV")
    parser.add_argument("--out", type=str, default="checkpoints/text2latent",
                        help="Directory for ckpt_step_*.pt and logs (default: checkpoints/text2latent)")
    parser.add_argument("--ae_checkpoint", type=str, default="checkpoints/ae/ae_latest.pt",
                        help="Path to AE checkpoint")
    parser.add_argument("--stats_path", type=str, default="stats_multilingual.pt",
                        help="Path to latent stats .pt file")
    parser.add_argument("--max_steps", type=int, default=1_000_000,
                        help="Maximum optimization steps")
    parser.add_argument("--batch_size", type=int, default=14,
                        help="Training batch size")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--Ke", type=int, default=None,
                        help="Override batch expansion factor (default: from config)")
    parser.add_argument("--accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps (default: 1)")
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="If set and checkpoint_dir has no ckpt_step_*.pt, load latest from this directory",
    )
    parser.add_argument(
        "--inference_ref_wav",
        type=str,
        default=None,
        help="WAV for Voice 1 validation inference logs (or set T2L_INFERENCE_REF_WAV).",
    )
    args = parser.parse_args()

    train(
        finetune=args.finetune,
        config_path=args.config,
        metadata_path=args.data,
        ae_checkpoint=args.ae_checkpoint,
        stats_path=args.stats_path,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        device=args.device or ("cuda:1" if torch.cuda.is_available() else "cpu"),
        Ke=args.Ke,
        accumulation_steps=args.accumulation_steps,
        checkpoint_dir=args.out,
        resume_from=args.resume_from,
        inference_ref_wav=args.inference_ref_wav,
    )

if __name__ == "__main__":
    main()
