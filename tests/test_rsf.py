"""Unit tests for the RobustSpeechFlow loss (training/t2l/rsf.py, arXiv 2605.22083)."""

from __future__ import annotations

import random
import unittest

import torch

from training.t2l.rsf import len_matched_donors, rsf_augment, rsf_loss
from training.t2l.rsf import rsf_negative_targets as negatives

FPS = 44100 / 512 / 6
LO, HI = round(0.1 * FPS), round(5.0 * FPS)   # span bounds in frames: 1 and 72


class TestLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.shape = (4, 8, 30)
        self.u, self.v, self.v_rand, self.v_aug = (torch.randn(self.shape) for _ in range(4))
        self.m_rand = (torch.rand(4, 1, 30) > 0.3).float()
        self.mask = (torch.rand(4, 1, 30) > 0.2).float().expand(self.shape)

    def loss(self, u, lr=0.2, la=0.3):
        return rsf_loss(u, self.v, self.mask, self.v_rand, self.v_aug, self.m_rand, lr, la)

    def test_eq7_l1(self):
        """L = L_pos - lambda_rand L_rand - lambda_aug L_aug (Eq. 7, Alg. 1 lines 11-14), L1 terms;
        donor padding (m_rand = 0) is out of L_rand; lambdas 0 give the plain masked L1, bit for bit."""
        def mean(x):
            return (x * self.mask).sum() / self.mask.sum()
        l_rand = mean((self.u - self.v_rand).abs() * self.m_rand)
        l_aug = mean((self.u - self.v_aug).abs())
        loss, lr, la = self.loss(self.u)
        torch.testing.assert_close(loss, mean((self.u - self.v).abs()) - 0.2 * l_rand - 0.3 * l_aug)
        self.assertAlmostEqual(lr, float(l_rand), places=5)
        self.assertAlmostEqual(la, float(l_aug), places=5)
        self.assertAlmostEqual(self.loss(self.u + 5.0 * (1 - self.m_rand))[1], lr, places=6)
        plain = ((self.u - self.v).abs() * self.mask).sum() / (self.mask.sum() + 1e-8)
        self.assertTrue(torch.equal(self.loss(self.u, 0.0, 0.0)[0], plain))
        with self.assertRaises(AssertionError):
            self.loss(self.u, 0.5, 0.5)

    def test_squared_l2_collapses_to_plain_fm(self):
        """Why L1: in squared L2 (Eqs. 2, 3, 6) Eq. 7 is (1 - lr - la) ||u - v~||^2 + c(not u), plain FM
        toward the stretched target v~ = v + [lr (x - x_rand) + la (x - x_aug)] / (1 - lr - la)."""
        lr, la = 0.2, 0.2
        v, v_rand, v_aug = self.v.double(), self.v_rand.double(), self.v_aug.double()
        eps = torch.randn(self.shape, dtype=torch.float64)   # v(x, eps) = x - eps for all three
        x, x_rand, x_aug = v + eps, v_rand + eps, v_aug + eps
        v_tilde = v + (lr * (x - x_rand) + la * (x - x_aug)) / (1 - lr - la)

        def const(u):   # paper loss minus the stretched-target loss
            paper = (u - v) ** 2 - lr * (u - v_rand) ** 2 - la * (u - v_aug) ** 2
            return paper - (1 - lr - la) * (u - v_tilde) ** 2
        torch.testing.assert_close(const(torch.randn_like(v)), const(10 * torch.randn_like(v)))


class TestNegatives(unittest.TestCase):
    def test_velocity_shares_noise(self):
        """v(x, eps) = x - (1 - sigma_min) eps with the row's own eps (Eq. 1, Alg. 1 lines 5-9)."""
        torch.manual_seed(0)
        Ke, C, T, sig = 2, 6, 60, 1e-8
        n = torch.tensor([60, 45, 30]).repeat_interleave(Ke)
        mask = (torch.arange(T)[None] < n[:, None]).float().unsqueeze(1)
        z, x0 = torch.randn(6, C, T) * mask, torch.randn(6, C, T)
        random.seed(1)
        v_rand, v_aug, m_rand = negatives(z, mask, x0, n, torch.zeros(C), Ke, sig, FPS)
        random.seed(1)
        idx = len_matched_donors(n, Ke)
        z_aug = rsf_augment(z, n, torch.zeros(C), FPS) * mask
        v = z - (1 - sig) * x0
        torch.testing.assert_close(v - v_aug, z - z_aug)
        torch.testing.assert_close(v - v_rand, z - z[idx] * mask[idx])
        self.assertTrue(torch.equal(m_rand, mask[idx]))
        # a single utterance has no random negative
        self.assertIsNone(negatives(z[:Ke], mask[:Ke], x0[:Ke], n[:Ke], torch.zeros(C), Ke, sig, FPS)[0])

    def test_len_matched_donors(self):
        """Donor = closest length >= own, never self; else the longest other; copy k -> copy k."""
        random.seed(0)
        Ke, lens = 3, [50, 20, 35, 35, 80]
        for _ in range(20):
            idx = len_matched_donors(torch.tensor(lens).repeat_interleave(Ke), Ke)
            for r, d in enumerate(idx.tolist()):
                i, others = r // Ke, [lens[j] for j in range(len(lens)) if j != r // Ke]
                self.assertNotEqual(i, d // Ke)
                self.assertEqual(d % Ke, r % Ke)
                longer = [m for m in others if m >= lens[i]]
                self.assertEqual(lens[d // Ke], min(longer) if longer else max(others))

    def test_repeat_and_skip(self):
        """Repeat: Eq. 4, kappa ~ U(0.2, 0.4). Skip: Eq. 5 with an x_sil tail, one local skip whose
        changed frames equal the budget kappa ~ U(0.4, 0.8). Modes 50/50; padding untouched."""
        random.seed(0)
        torch.manual_seed(0)
        B, C, T = 400, 4, 200
        z = torch.randn(B, C, T)
        z[:, 0, :] = torch.arange(T, dtype=torch.float32)   # channel 0 = frame index
        n = torch.randint(2 * LO + 1, T + 1, (B,))
        n[:2] = 2 * LO
        z_sil = torch.full((C,), -7.0)
        out = rsf_augment(z, n, z_sil, FPS)
        self.assertTrue(torch.equal(out[:2], z[:2]))   # too short to edit
        n_skip = 0
        for b in range(2, B):
            nb = int(n[b])
            self.assertTrue(torch.equal(out[b, :, nb:], z[b, :, nb:]))
            changed = (out[b, :, :nb] != z[b, :, :nb]).any(dim=0)
            self.assertGreater(int(changed.sum()), 0)
            is_sil = (out[b, :, :nb] == z_sil[:, None]).all(dim=0)
            if is_sil.any():   # skip
                n_skip += 1
                l, s1 = int(is_sil.sum()), int(changed.nonzero()[0])
                self.assertTrue(is_sil[nb - l:].all() and LO <= l <= HI)
                self.assertTrue(torch.equal(out[b, :, s1:nb - l], z[b, :, s1 + l:nb]))
                self.assertLessEqual(abs((nb - s1) / nb - 0.6), 0.2 + (1.5 + LO) / nb)
            else:              # repeat: every frame is an original frame of the same row
                self.assertTrue(torch.equal(out[b, :, :nb], z[b, :, out[b, 0, :nb].long()]))
                self.assertLessEqual(int(changed.sum()), max(round(0.4 * nb), LO))
        self.assertLess(abs(n_skip / (B - 2) - 0.5), 0.1)


if __name__ == "__main__":
    unittest.main()
