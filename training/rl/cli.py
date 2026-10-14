import argparse

import torch

from training.rl.train_grpo import train_grpo


def main():
    p = argparse.ArgumentParser(description="FlowTTS-GRPO post-training of the T2L vector field (arXiv 2606.23190)")
    p.add_argument("--ckpt", required=True,
                   help="T2L checkpoint to start from (ckpt_step_*.pt). Pass the same one on --resume: it is the KL "
                        "reference and the base of every saved checkpoint")
    p.add_argument("--config", required=True, help="tts.json the checkpoint was trained with")
    p.add_argument("--data", required=True, help="training metadata CSV: prompt texts and reference clips")
    p.add_argument("--ae_checkpoint", required=True)
    p.add_argument("--stats_path", required=True, help="latent stats the checkpoint was trained with")
    p.add_argument("--dp_ckpt", required=True, help="duration predictor checkpoint")
    p.add_argument("--out", required=True, help="directory for ckpt_step_*.pt, train_state.pt and grpo_log.jsonl")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--score_device", default=None, help="device of the reward models (default: --device)")
    p.add_argument("--resume", default=None, help="train_state.pt of an earlier run (pass that run's original --ckpt)")
    p.add_argument("--seed", type=int, default=1234)
    # iteration (paper Sec 3.3)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--prompts_per_iter", type=int, default=16)
    p.add_argument("--G", type=int, default=8, help="samples per prompt (the GRPO group)")
    p.add_argument("--updates_per_iter", type=int, default=8, help="optimizer steps per iteration")
    p.add_argument("--steps_min", type=int, default=5)
    p.add_argument("--steps_max", type=int, default=8)
    p.add_argument("--smin_lo", type=int, default=1, help="earliest SDE window start (0-based step)")
    p.add_argument("--smin_hi", type=int, default=3)
    p.add_argument("--sde_steps", type=int, default=2, help="SDE window size")
    p.add_argument("--noise_level", type=float, default=0.5, help="a in sigma_t = a sqrt((1 - t) / t)")
    # optimisation
    p.add_argument("--lr", type=float, default=1e-5, help="linearly decayed to 0 over the run")
    p.add_argument("--clip", type=float, default=1e-4)
    p.add_argument("--beta", type=float, default=0.04, help="KL weight")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--ema", type=float, default=0.99, help="EMA of the VF weights saved in ckpt_step_*.pt; 0 = off")
    p.add_argument("--micro_batch", type=int, default=4, help="samples per backward (memory only)")
    p.add_argument("--decode_bs", type=int, default=2, help="latents per AE decode (memory only)")
    # rewards (Sec 2.3, Eq. 12)
    p.add_argument("--w_sim", type=float, default=1.0)
    p.add_argument("--w_asr", type=float, default=1.0)
    p.add_argument("--w_mos", type=float, default=0.4)
    p.add_argument("--asr_model", default="openai/whisper-large-v3")
    p.add_argument("--asr_text_col", default="transcript", help="CSV column with the words the ASR reward scores")
    # prompts and references (Sec 2.5)
    p.add_argument("--hard_frac", type=float, default=0.3, help="share of hard-case prompts")
    p.add_argument("--n_hard", type=int, default=400)
    p.add_argument("--max_tokens", type=int, default=200)
    p.add_argument("--ref_pool", type=int, default=400, help="reference clips (3-10 s)")
    p.add_argument("--ref_pool_per_spk", type=int, default=2)
    p.add_argument("--ref_max_frames", type=int, default=144, help="reference crop in latent frames")
    p.add_argument("--save_every", type=int, default=250)
    train_grpo(p.parse_args())


if __name__ == "__main__":
    main()
