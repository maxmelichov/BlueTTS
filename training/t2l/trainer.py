"""T2L training entry — re-exports `train` / `set_seed` from `train_loop` and `sampling`."""

from training.t2l.sampling import set_seed
from training.t2l.train_loop import train

__all__ = ["train", "set_seed"]


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
