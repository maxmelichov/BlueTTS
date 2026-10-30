"""FlowTTS-GRPO (arXiv 2606.23190) for the T2L vector field.

A few Euler ODE steps become a marginal-preserving SDE, each a Gaussian policy with a PPO-clipped ratio; the
advantage comes from a group of G samples of the same prompt. Time as in the T2L trainer (and the paper's Eq. 7):
t = 0 noise, t = 1 data, x_t = (1 - t) x0 + t x1, v = x1 - x0.
"""

import math

import numpy as np
import torch


def sde_mean_std(x, v, k, dt, noise_level):
    """One SDE step from t = k * dt (Eq. 7):

        x_mean = x + [v + sigma_t^2 / (2 (1 - t)) * (-x + t v)] dt,   sigma_t = a * sqrt((1 - t) / t)
        x'     = x_mean + sigma_t * sqrt(dt) * eps

    The extra drift is sigma_t^2 / 2 times the score of x_t, so the marginals stay the ODE's. At t = 0 the
    denominator takes the next grid point (Flow-GRPO). Returns (x_mean, sigma_t sqrt(dt)).
    """
    t = k * dt
    s = noise_level * math.sqrt((1.0 - t) / max(t, dt))
    mean = x + v * dt - (s * s * dt / (2.0 * (1.0 - t))) * (x - t * v)
    return mean, s * math.sqrt(dt)


def gauss_logp(x_next, mean, sd, latent_mask):
    """Eq. 11, log N(x_next; mean, sd^2), averaged (as Flow-GRPO) over each sample's valid elements. Returns [B]."""
    lp = -((x_next - mean) ** 2) / (2 * sd * sd) - math.log(sd) - 0.5 * math.log(2 * math.pi)
    m = latent_mask.expand_as(lp)
    return (lp * m).sum((1, 2)) / m.sum((1, 2))


def gaussian_kl(mean, mean_ref, sd, latent_mask):
    """KL(N(mean, sd^2) || N(mean_ref, sd^2)) = ||mean - mean_ref||^2 / (2 sd^2) (shared sd), element then batch mean."""
    m = latent_mask.expand_as(mean)
    return ((((mean - mean_ref) ** 2) * m).sum((1, 2)) / m.sum((1, 2))).mean() / (2 * sd * sd)


def clipped_surrogate(logp, old_logp, adv, clip):
    """Eq. 10 without the KL: -min(r A, clip(r, 1 - eps, 1 + eps) A), r = exp(logp - old_logp). Returns (loss, r)."""
    ratio = torch.exp(logp - old_logp)
    loss = torch.maximum(-adv * ratio, -adv * torch.clamp(ratio, 1 - clip, 1 + clip)).mean()
    return loss, ratio


def vf_cond(vf, x, h, style_ttl, latent_mask, text_mask, t):
    """Conditional velocity for a group of latents that share one prompt (no CFG in rollouts or log-probs)."""
    B = x.shape[0]
    return vf(noisy_latent=x, text_emb=h.expand(B, -1, -1), style_ttl=style_ttl.expand(B, -1, -1),
              latent_mask=latent_mask.expand(B, -1, -1), text_mask=text_mask.expand(B, -1, -1),
              current_step=torch.full((B,), t, device=x.device))


def sample_window(rng, steps_min, steps_max, smin_lo, smin_hi, sde_steps):
    """Window training (Sec 2.2.2): n_steps in [steps_min, steps_max], the SDE window starting at a 0-based step in
    [smin_lo, smin_hi]. Returns (n_steps, SDE step indices)."""
    n_steps = rng.randint(steps_min, steps_max)
    s_min = min(rng.randint(smin_lo, smin_hi), n_steps - 1)
    return n_steps, range(s_min, min(n_steps, s_min + sde_steps))


@torch.no_grad()
def rollout(vf, h, style_ttl, latent_mask, text_mask, group_size, channels, n_steps, window, noise_level, generator):
    """G latents for one prompt with the inference Euler loop; steps in `window` are SDE steps whose transitions are
    stored for the update. Returns (x1, transitions)."""
    dt = 1.0 / n_steps
    x = torch.randn(group_size, channels, latent_mask.shape[2], device=latent_mask.device,
                    generator=generator) * latent_mask
    traj = []
    for k in range(n_steps):
        v = vf_cond(vf, x, h, style_ttl, latent_mask, text_mask, k * dt)
        if k in window:
            mean, sd = sde_mean_std(x, v, k, dt, noise_level)
            x_next = (mean + sd * torch.randn(x.shape, device=x.device, generator=generator)) * latent_mask
            traj.append(dict(k=k, dt=dt, x=x, x_next=x_next,
                             logp=gauss_logp(x_next, mean * latent_mask, sd, latent_mask)))
            x = x_next
        else:
            x = (x + dt * v) * latent_mask
    return x, traj


def advantages(sim, err, mos, w_sim=1.0, w_err=1.0, w_mos=0.4):
    """sim / err / mos: [n_prompts, G] (speaker cosine, ASR error rate, DNSMOS OVRL).
    Eq. 12: R = sum of weighted rewards, each divided by its std over the iteration. Eq. 9: A = (R - mean(R)) / std(R)
    within each prompt's group; a zero-std group gets A = 0. Returns (A, R)."""
    parts = [(w_sim, sim), (w_err, -err), (w_mos, mos)]
    r = sum(w * x / max(float(x.std()), 1e-6) for w, x in parts if w)
    sd = r.std(1, keepdims=True)
    a = np.where(sd > 1e-6, (r - r.mean(1, keepdims=True)) / np.maximum(sd, 1e-6), 0.0)
    return a, r


def ppo_backward(vf, vf_ref, h, style_ttl, latent_mask, text_mask, traj, adv, clip=1e-4, beta=0.04, noise_level=0.5,
                 micro_batch=4, weight=1.0, h_ref=None):
    """Backward of Eq. 10 for one prompt: clipped surrogate + beta * KL to the frozen reference `vf_ref` on `h_ref`
    (default `h`), averaged over the SDE steps and the group, scaled by `weight`, one micro-batch graph at a time.
    Returns the mean loss, KL, clip fraction and ratio."""
    h_ref = h if h_ref is None else h_ref
    A_all = torch.as_tensor(adv, device=latent_mask.device, dtype=torch.float32)
    G = A_all.shape[0]
    mb = max(1, min(micro_batch, G))
    n = len(traj)
    tot, kl_tot, cf, ratio_sum = 0.0, 0.0, 0.0, 0.0
    for st in traj:
        k, dt = st["k"], st["dt"]
        for i0 in range(0, G, mb):
            sl = slice(i0, i0 + mb)
            A, x = A_all[sl], st["x"][sl]
            frac = A.shape[0] / G
            v = vf_cond(vf, x, h, style_ttl, latent_mask, text_mask, k * dt)
            mean, sd = sde_mean_std(x, v, k, dt, noise_level)
            logp = gauss_logp(st["x_next"][sl], mean * latent_mask, sd, latent_mask)
            loss, ratio = clipped_surrogate(logp, st["logp"][sl], A, clip)
            if beta > 0:
                with torch.no_grad():
                    v_ref = vf_cond(vf_ref, x, h_ref, style_ttl, latent_mask, text_mask, k * dt)
                    mean_ref, _ = sde_mean_std(x, v_ref, k, dt, noise_level)
                kl = gaussian_kl(mean, mean_ref, sd, latent_mask)
                loss = loss + beta * kl
                kl_tot += float(kl.detach()) * frac
            (loss * (frac * weight / n)).backward()
            tot += float(loss.detach()) * frac
            cf += float(((ratio.detach() - 1).abs() > clip).float().mean()) * frac
            ratio_sum += float(ratio.detach().mean()) * frac
    return dict(loss=tot / n, kl=kl_tot / n, clipfrac=cf / n, ratio=ratio_sum / n)
