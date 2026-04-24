import argparse
import sys
import os

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
        Ke=args.Ke,
        accumulation_steps=args.accumulation_steps,
        checkpoint_dir=args.out,
        resume_from=args.resume_from,
        inference_ref_wav=args.inference_ref_wav,
    )

if __name__ == "__main__":
    main()
