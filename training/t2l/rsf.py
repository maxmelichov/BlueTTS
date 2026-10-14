"""RobustSpeechFlow contrastive flow matching (Yang et al., arXiv 2605.22083).

    L = L_pos - lambda_rand * L_rand - lambda_aug * L_aug            (paper Eq. 7, Alg. 1 lines 11-14)

L_pos regresses the true velocity (Eq. 2), L_rand the velocity toward another utterance of the
batch (Eq. 3) and L_aug the velocity toward a repeat/skip corruption of the same utterance
(Eq. 6, Sec. 3.3). Both negatives reuse the row's noise eps, time t and x_t (Alg. 1 lines 3-9);
only the regression target changes, so the vector field runs once:

    v(x, eps) = x - (1 - sigma_min) * eps,   x_t = (1 - (1 - sigma_min) t) eps + t x

All three terms are L1 (the SupertonicTTS Eq. 1 loss this trainer already uses), not the paper's
squared L2. In squared L2 the sum collapses to (1 - lambda_rand - lambda_aug) * ||u - v~||^2 + const
with v~ = v + [lambda_rand (x - x_rand) + lambda_aug (x - x_aug)] / (1 - lambda_rand - lambda_aug):
plain flow matching toward a stretched target (see tests/test_rsf.py). In L1 the negatives stay a
bounded contrast; lambda_rand + lambda_aug < 1 keeps the loss bounded below.

Negatives apply to every row, including CFG-dropped and SPFM-dirty ones (Alg. 1).
"""
from __future__ import annotations

import random

import torch

from training.utils import compress_latents

FPS = 44100.0 / 512.0 / 6.0   # compressed latent frames per second (sample_rate / hop / chunk_compress_factor)


@torch.no_grad()
def silence_latent(ae_encoder, mel_spec, mean, std, normalizer_scale, chunk_compress_factor,
                   sample_rate=44100, device="cpu") -> torch.Tensor:
    """x_sil (Sec. 3.3.3): 3 s of zeros through the AE encoder, normalized, averaged over the
    interior frames (the two edge frames see the conv padding). Returns [C]."""
    z = compress_latents(ae_encoder(mel_spec(torch.zeros(1, int(3 * sample_rate), device=device))),
                         factor=chunk_compress_factor)
    return (((z - mean) / std) * normalizer_scale)[0, :, 1:-1].mean(dim=1)


def rsf_augment(z: torch.Tensor, valid_len: torch.Tensor, z_sil: torch.Tensor, fps: float = FPS) -> torch.Tensor:
    """Same-length failure-mode negatives (Sec. 3.3) on the compressed latents.

    One mode per utterance, repeat or skip with probability 0.5 each. Spans are uniform over
    [0.1 s, 5 s] in frames.
      repeat (Eq. 4): x[k:k+l] <- x[s:s+l], s != k, source always the original x; coverage
          kappa ~ U(0.2, 0.4); edits repeat until the next one would exceed kappa * n.
      skip (Eq. 5): x[s1:n-l] <- x[s1+l:n], tail [n-l, n) <- x_sil; kappa ~ U(0.4, 0.8).
          One local skip whose start is s1 = n - kappa * n, so the frames it changes (shifted
          plus padded) equal the coverage budget.
    The first edit always happens, so a negative never equals its positive. Rows shorter than
    2 * lo + 1 frames and frames past valid_len are left unchanged.

    z: [B, C, T]; valid_len: [B] frames; z_sil: [C]. Returns a new [B, C, T].
    """
    B, C, _ = z.shape
    out = z.clone()
    lo, hi = max(1, round(0.1 * fps)), max(1, round(5.0 * fps))
    for b in range(B):
        n = int(valid_len[b])
        if n < 2 * lo + 1:
            continue
        rep = random.random() < 0.5
        kappa = random.uniform(0.2, 0.4) if rep else random.uniform(0.4, 0.8)
        budget = round(kappa * n)
        cov = 0
        while True:
            l = random.randint(lo, min(hi, n - 1))
            if cov + l > budget:
                if cov > 0:
                    break
                l = max(lo, min(budget, n - 1))
            if not rep:
                s1 = max(0, min(n - budget, n - lo - 1))
                l = random.randint(lo, min(hi, n - s1 - 1))
                out[b, :, s1:n - l] = z[b, :, s1 + l:n]
                out[b, :, n - l:n] = z_sil.view(C, 1)
                break
            s = random.randint(0, n - l)
            k = random.randint(0, n - l - 1)   # uniform over {0..n-l} \ {s}
            if k >= s:
                k += 1
            out[b, :, k:k + l] = z[b, :, s:s + l]
            cov += l
    return out


def len_matched_donors(valid_len: torch.Tensor, Ke: int) -> torch.Tensor:
    """Donor row of each expanded row for the random negative (Alg. 1 line 6, RollBatch).

    The paper batches by length (Sec. 4.3), so a rolled donor is about as long as the row. With
    randomly composed batches the donor is instead, per utterance i, the j != i with the smallest
    length >= n_i (ties at random), or the longest j != i when none is that long. Copy k of row i
    takes copy k of its donor. valid_len is per expanded row ([B * Ke], repeat_interleave'd); B >= 2.
    """
    B = valid_len.shape[0] // Ke
    nb = [int(x) for x in valid_len[::Ke].tolist()]
    perm = []
    for i in range(B):
        others = [j for j in range(B) if j != i]
        longer = [j for j in others if nb[j] >= nb[i]]
        if longer:
            best = min(nb[j] - nb[i] for j in longer)
            ties = [j for j in longer if nb[j] - nb[i] == best]
        else:
            best = max(nb[j] for j in others)
            ties = [j for j in others if nb[j] == best]
        perm.append(ties[random.randrange(len(ties))])
    perm = torch.tensor(perm, dtype=torch.long, device=valid_len.device)
    ar = torch.arange(B * Ke, device=valid_len.device)
    return perm.repeat_interleave(Ke) * Ke + ar % Ke


@torch.no_grad()
def rsf_negative_targets(z_1: torch.Tensor, latent_mask: torch.Tensor, x_0: torch.Tensor,
                         valid_len: torch.Tensor, z_sil: torch.Tensor, Ke: int, sigma_min: float):
    """Velocity targets of the two negatives with the row's own noise x_0 (Alg. 1 lines 6-9).

    All inputs are the Ke-expanded batch: z_1 / x_0 [B*Ke, C, T], latent_mask [B*Ke, 1, T],
    valid_len [B*Ke]. Returns (v_rand, v_aug, m_rand); m_rand [B*Ke, 1, T] is the donor's
    valid-frame mask. A donor shorter than the row has no content past its end, so those
    frames are dropped from L_rand. v_rand and m_rand are None for a single-utterance batch,
    where every donor would be the row itself.
    """
    v_rand = m_rand = None
    if z_1.shape[0] > Ke:
        idx = len_matched_donors(valid_len, Ke)
        m_rand = latent_mask[idx]
        v_rand = z_1[idx] * m_rand - (1 - sigma_min) * x_0
    z_aug = rsf_augment(z_1, valid_len, z_sil) * latent_mask
    v_aug = z_aug - (1 - sigma_min) * x_0
    return v_rand, v_aug, m_rand


def rsf_loss(v_pred: torch.Tensor, v_target: torch.Tensor, mask_ct: torch.Tensor,
             v_rand: torch.Tensor | None, v_aug: torch.Tensor, m_rand: torch.Tensor | None,
             lambda_rand: float, lambda_aug: float) -> tuple[torch.Tensor, float, float]:
    """L_pos - lambda_rand * L_rand - lambda_aug * L_aug, each an L1 mean over mask_ct
    ([B, C, T]: padding + reference hole). With both lambdas 0 this is the plain masked L1
    flow-matching loss. Returns (loss, L_rand, L_aug), the last two detached for logging."""
    assert lambda_rand + lambda_aug < 1.0, "RSF needs lambda_rand + lambda_aug < 1"
    l_pos = (v_pred - v_target).abs()
    l_aug = (v_pred - v_aug).abs()
    loss_raw = l_pos - lambda_aug * l_aug
    l_rand = None
    if v_rand is not None:
        l_rand = (v_pred - v_rand).abs() * m_rand
        loss_raw = loss_raw - lambda_rand * l_rand
    denom = mask_ct.sum() + 1e-8
    loss = (loss_raw * mask_ct).sum() / denom
    with torch.no_grad():
        la = float((l_aug * mask_ct).sum() / denom)
        lr = float((l_rand * mask_ct).sum() / denom) if l_rand is not None else 0.0
    return loss, lr, la
