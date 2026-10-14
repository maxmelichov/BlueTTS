"""FlowTTS-GRPO math (arXiv 2606.23190): SDE step, Gaussian log-prob, KL, advantages, clipped ratio."""

from __future__ import annotations

import math
import random
import unittest

import numpy as np

try:
    import torch
except ImportError:  # the inference install has no torch
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class TestFlowGRPO(unittest.TestCase):
    def test_sde_at_zero_noise_is_the_euler_step(self):
        from training.rl.flow_grpo import sde_mean_std

        x, v = torch.randn(2, 6, 5), torch.randn(2, 6, 5)
        for k in range(4):
            mean, sd = sde_mean_std(x, v, k, 0.25, 0.0)
            self.assertEqual(sd, 0.0)
            self.assertTrue(torch.equal(mean, x + v * 0.25))

    def test_sde_keeps_the_ode_marginals(self):
        # Eq. 7 with a = 0.5 on 1-D Gaussian data x1 ~ N(m, s^2), where E[v | x_t] is exact: the SDE ends at the
        # data distribution like the ODE; with the drift correction's sign flipped it does not.
        from training.rl.flow_grpo import sde_mean_std

        m, s, n = 1.5, 0.5, 100

        def v_star(x, t):
            var = (1 - t) ** 2 + t * t * s * s
            return m + (t * s * s - (1 - t)) / var * (x - t * m)

        def run(a, flip=False):
            g = torch.Generator().manual_seed(0)
            x = torch.randn(100_000, dtype=torch.float64, generator=g)
            for k in range(n):
                v = v_star(x, k / n)
                mean, sd = sde_mean_std(x, v, k, 1 / n, a)
                if flip:
                    mean = 2 * (x + v / n) - mean
                x = mean + sd * torch.randn(x.shape, dtype=torch.float64, generator=g)
            return x.mean().item(), x.std().item()

        for mu, sd in (run(0.0), run(0.5), run(0.7)):
            self.assertAlmostEqual(mu, m, delta=0.01)
            self.assertAlmostEqual(sd, s, delta=0.01)
        self.assertGreater(run(0.5, flip=True)[1], 1.0)

    def test_logp_of_a_sampled_transition(self):
        from training.rl.flow_grpo import gauss_logp

        mean, sd = torch.randn(3, 6, 8, dtype=torch.float64), 0.3
        x = mean + sd * torch.randn(3, 6, 8, dtype=torch.float64)
        mask = torch.ones(3, 1, 8, dtype=torch.float64)
        mask[1, :, 5:] = 0
        ref = torch.distributions.Normal(mean, sd).log_prob(x)
        for b, L in enumerate((8, 5, 8)):
            self.assertAlmostEqual(gauss_logp(x, mean, sd, mask)[b].item(), ref[b, :, :L].mean().item(), places=10)

    def test_kl_closed_form_vs_monte_carlo(self):
        from training.rl.flow_grpo import gaussian_kl

        g = torch.Generator().manual_seed(0)
        mu, sd = torch.randn(2, 4, 3, generator=g, dtype=torch.float64), 0.2
        mu_ref = mu + 0.1 * torch.randn(2, 4, 3, generator=g, dtype=torch.float64)
        mask = torch.ones(2, 1, 3, dtype=torch.float64)
        p, q = torch.distributions.Normal(mu, sd), torch.distributions.Normal(mu_ref, sd)
        x = p.sample((200_000,))
        mc = (p.log_prob(x) - q.log_prob(x)).mean(0).mean()
        self.assertAlmostEqual(gaussian_kl(mu, mu_ref, sd, mask).item(), mc.item(), delta=0.01)
        self.assertAlmostEqual(gaussian_kl(mu, mu_ref, sd, mask).item(),
                               torch.distributions.kl_divergence(p, q).mean().item(), places=10)

    def test_advantages(self):
        from training.rl.flow_grpo import advantages

        rng = np.random.default_rng(0)
        sim, err, mos = rng.random((4, 8)), rng.random((4, 8)), 3 + rng.random((4, 8))
        err[2] = 0.3
        sim[2] = 0.5
        mos[2] = 3.1                                          # a group whose rewards are all equal
        a, _ = advantages(sim, err, mos)
        ok = [0, 1, 3]
        np.testing.assert_allclose(a[ok].mean(1), 0, atol=1e-12)
        np.testing.assert_allclose(a[ok].std(1), 1, atol=1e-12)
        self.assertTrue(np.all(a[2] == 0))
        # Eq. 12: each reward is normalised by its own std, so rescaling one reward does not move A
        np.testing.assert_allclose(advantages(10 * sim, err, mos)[0], a, atol=1e-12)
        # the ASR term enters with a minus sign: lower error, higher advantage
        a_err, _ = advantages(sim, err, mos, w_sim=0, w_mos=0)
        self.assertTrue(np.all(np.argsort(a_err[0]) == np.argsort(-err[0])))

    def test_window(self):
        from training.rl.flow_grpo import sample_window

        rng = random.Random(0)
        for _ in range(200):
            n, w = sample_window(rng, 5, 8, 1, 3, 2)
            self.assertTrue(5 <= n <= 8 and 1 <= w.start <= 3 and len(w) == 2 and w.stop <= n)

    def test_on_policy_ratio_is_one_and_clip_is_inert(self):
        from training.rl.flow_grpo import ppo_backward, rollout

        class TinyVF(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Conv1d(6, 6, 1)

            def forward(self, noisy_latent, text_emb, style_ttl, latent_mask, text_mask, current_step):
                return self.w(noisy_latent) + text_emb.mean() + style_ttl.mean() + current_step[:, None, None]

        torch.manual_seed(0)
        vf = TinyVF()
        h, style, tmask = torch.randn(1, 4, 3), torch.randn(1, 2, 4), torch.ones(1, 1, 3)
        lmask = torch.ones(1, 1, 7)
        lmask[..., 5:] = 0
        g = torch.Generator().manual_seed(0)
        _, traj = rollout(vf, h, style, lmask, tmask, 4, 6, 6, range(1, 3), 0.5, g)
        adv = np.array([1.0, -0.5, 0.25, -0.75])
        grads = []
        for clip in (1e-4, 1e9):
            vf.zero_grad()
            st = ppo_backward(vf, vf, h, style, lmask, tmask, traj, adv, clip=clip, beta=0.04, micro_batch=3)
            self.assertAlmostEqual(st["ratio"], 1.0, places=6)
            self.assertEqual(st["clipfrac"], 0.0)
            self.assertAlmostEqual(st["kl"], 0.0, places=12)
            grads.append(torch.cat([p.grad.flatten() for p in vf.parameters()]))
        self.assertTrue(torch.equal(grads[0], grads[1]))
        self.assertGreater(grads[0].abs().max().item(), 0)
        # the reference policy runs on h_ref (the frozen text encoder's output), not on the policy's h
        st = ppo_backward(vf, vf, h, style, lmask, tmask, traj, adv, h_ref=h + 0.1, micro_batch=3)
        self.assertGreater(st["kl"], 0.0)


if __name__ == "__main__":
    unittest.main()
