"""Unit tests for the RobustSpeechFlow loss (training/t2l/rsf.py, arXiv 2605.22083)."""

from __future__ import annotations

import random
import unittest

import torch

from training.t2l.rsf import (
    FPS,
    len_matched_donors,
    rsf_augment,
    rsf_loss,
    rsf_negative_targets,
)

LO, HI = round(0.1 * FPS), round(5.0 * FPS)   # span bounds in frames: 1 and 72


def _frame_ids(B, C, T):
    """Latents whose channel 0 holds the frame index, so every frame is identifiable."""
    z = torch.randn(B, C, T)
    z[:, 0, :] = torch.arange(T, dtype=torch.float32)
    return z


class TestLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.shape = (4, 8, 30)
        self.v_pred, self.v, self.v_rand, self.v_aug = (torch.randn(self.shape) for _ in range(4))
        self.m_rand = (torch.rand(4, 1, 30) > 0.3).float()
        self.mask = (torch.rand(4, 1, 30) > 0.2).float().expand(self.shape)

    def test_eq7_l1(self):
        """L = L_pos - lambda_rand L_rand - lambda_aug L_aug (Eq. 7, Alg. 1 lines 11-14), L1 terms."""
        loss, lr, la = rsf_loss(self.v_pred, self.v, self.mask, self.v_rand, self.v_aug, self.m_rand, 0.2, 0.3)
        mean = lambda x: (x * self.mask).sum() / self.mask.sum()
        l_pos = mean((self.v_pred - self.v).abs())
        l_rand = mean((self.v_pred - self.v_rand).abs() * self.m_rand)
        l_aug = mean((self.v_pred - self.v_aug).abs())
        torch.testing.assert_close(loss, l_pos - 0.2 * l_rand - 0.3 * l_aug)
        self.assertAlmostEqual(lr, float(l_rand), places=5)
        self.assertAlmostEqual(la, float(l_aug), places=5)

    def test_zero_lambda_is_plain_l1(self):
        """With both lambdas 0 the loss is the trainer's masked L1 flow-matching loss, bit for bit."""
        loss, _, _ = rsf_loss(self.v_pred, self.v, self.mask, self.v_rand, self.v_aug, self.m_rand, 0.0, 0.0)
        plain = ((self.v_pred - self.v).abs() * self.mask).sum() / (self.mask.sum() + 1e-8)
        self.assertTrue(torch.equal(loss, plain))

    def test_donor_padding_masked(self):
        """Frames past the donor's end do not reach L_rand."""
        v_pred = self.v_pred.clone()
        v_pred2 = v_pred.clone()
        v_pred2[self.m_rand.expand(self.shape) == 0] += 5.0
        _, lr1, _ = rsf_loss(v_pred, self.v, self.mask, self.v_rand, self.v_aug, self.m_rand, 0.2, 0.2)
        _, lr2, _ = rsf_loss(v_pred2, self.v, self.mask, self.v_rand, self.v_aug, self.m_rand, 0.2, 0.2)
        self.assertAlmostEqual(lr1, lr2, places=6)

    def test_lambda_bound(self):
        with self.assertRaises(AssertionError):
            rsf_loss(self.v_pred, self.v, self.mask, self.v_rand, self.v_aug, self.m_rand, 0.5, 0.5)

    def test_squared_l2_collapses_to_plain_fm(self):
        """Why the terms are L1: in squared L2 (Eqs. 2, 3, 6) Eq. 7 equals
        (1 - lr - la) ||u - v~||^2 + c(v, v_rand, v_aug) with v~ = (v - lr v_rand - la v_aug) / (1 - lr - la),
        i.e. plain FM toward a stretched target; c does not depend on u."""
        lr, la = 0.2, 0.2
        v, v_rand, v_aug = self.v.double(), self.v_rand.double(), self.v_aug.double()
        v_tilde = (v - lr * v_rand - la * v_aug) / (1 - lr - la)
        # v~ = v + [lr (x - x_rand) + la (x - x_aug)] / (1 - lr - la), since all velocities share eps
        x, x_rand, x_aug = v + 0.7, v_rand + 0.7, v_aug + 0.7   # any common eps term cancels
        torch.testing.assert_close(v_tilde, v + (lr * (x - x_rand) + la * (x - x_aug)) / (1 - lr - la))
        consts = []
        for seed in range(3):
            u = torch.randn(self.shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
            paper = (u - v) ** 2 - lr * (u - v_rand) ** 2 - la * (u - v_aug) ** 2
            stretched = (1 - lr - la) * (u - v_tilde) ** 2
            consts.append(paper - stretched)
        torch.testing.assert_close(consts[0], consts[1])
        torch.testing.assert_close(consts[0], consts[2])


class TestNegatives(unittest.TestCase):
    def test_velocity_shares_noise(self):
        """v(x, eps) = x - (1 - sigma_min) eps with the row's own eps (Eq. 1, Alg. 1 lines 5-9):
        v_pos - v_neg is x - x_neg, independent of eps."""
        random.seed(0)
        torch.manual_seed(0)
        B, Ke, C, T, sig = 3, 2, 6, 60, 1e-8
        n = torch.tensor([60, 45, 30]).repeat_interleave(Ke)
        mask = (torch.arange(T)[None] < n[:, None]).float().unsqueeze(1)
        z = torch.randn(B * Ke, C, T) * mask
        x0 = torch.randn(B * Ke, C, T)
        random.seed(1)
        v_rand, v_aug, m_rand = rsf_negative_targets(z, mask, x0, n, torch.zeros(C), Ke, sig)
        random.seed(1)
        idx = len_matched_donors(n, Ke)
        z_aug = rsf_augment(z, n, torch.zeros(C)) * mask
        v = z - (1 - sig) * x0
        torch.testing.assert_close(v - v_aug, z - z_aug)
        torch.testing.assert_close(v - v_rand, z - z[idx] * mask[idx])
        self.assertTrue(torch.equal(m_rand, mask[idx]))

    def test_single_utterance_has_no_random_negative(self):
        n = torch.full((4,), 40)
        mask = torch.ones(4, 1, 40)
        v_rand, _, m_rand = rsf_negative_targets(torch.randn(4, 3, 40), mask, torch.randn(4, 3, 40), n,
                                                 torch.zeros(3), 4, 0.0)
        self.assertIsNone(v_rand)
        self.assertIsNone(m_rand)

    def test_len_matched_donors(self):
        """Donor = closest length >= own, never self; else the longest other; copy k -> copy k."""
        random.seed(0)
        Ke = 3
        lens = [50, 20, 35, 35, 80]
        n = torch.tensor(lens).repeat_interleave(Ke)
        for _ in range(20):
            idx = len_matched_donors(n, Ke)
            for r in range(len(lens) * Ke):
                i, d = r // Ke, int(idx[r]) // Ke
                self.assertNotEqual(i, d)
                self.assertEqual(int(idx[r]) % Ke, r % Ke)
                longer = [lens[j] for j in range(len(lens)) if j != i and lens[j] >= lens[i]]
                want = min(longer) if longer else max(lens[j] for j in range(len(lens)) if j != i)
                self.assertEqual(lens[d], want)


class TestAugment(unittest.TestCase):
    def test_repeat_and_skip(self):
        """Repeat: Eq. 4, kappa ~ U(0.2, 0.4). Skip: Eq. 5 with an x_sil tail, one local skip whose
        changed frames equal the budget kappa ~ U(0.4, 0.8). Modes 50/50; padding untouched."""
        random.seed(0)
        torch.manual_seed(0)
        B, C, T = 400, 4, 200
        z = _frame_ids(B, C, T)
        n = torch.randint(2 * LO + 1, T + 1, (B,))
        z_sil = torch.full((C,), -7.0)
        out = rsf_augment(z, n, z_sil)
        n_skip = 0
        for b in range(B):
            nb = int(n[b])
            self.assertTrue(torch.equal(out[b, :, nb:], z[b, :, nb:]))
            changed = (out[b, :, :nb] != z[b, :, :nb]).any(dim=0)
            self.assertGreater(int(changed.sum()), 0)
            ids = out[b, 0, :nb]
            is_sil = (out[b, :, :nb] == z_sil[:, None]).all(dim=0)
            if is_sil.any():   # skip
                n_skip += 1
                l = int(is_sil.sum())
                self.assertTrue(is_sil[nb - l:].all())
                self.assertTrue(LO <= l <= HI)
                s1 = int(changed.nonzero()[0])
                self.assertTrue(torch.equal(out[b, :, s1:nb - l], z[b, :, s1 + l:nb]))
                self.assertTrue(torch.equal(out[b, :, :s1], z[b, :, :s1]))
                self.assertLessEqual(abs((nb - s1) / nb - 0.6), 0.2 + 1.5 / nb + LO / nb)
            else:              # repeat: every frame is some original frame of the same row
                src = ids.long()
                self.assertTrue(torch.equal(out[b, :, :nb], z[b, :, src]))
                self.assertLessEqual(int(changed.sum()), max(round(0.4 * nb), LO))
        self.assertLess(abs(n_skip / B - 0.5), 0.1)

    def test_short_rows_untouched(self):
        z = torch.randn(2, 3, 10)
        out = rsf_augment(z, torch.tensor([2 * LO, 1]), torch.zeros(3))
        self.assertTrue(torch.equal(out, z))


if __name__ == "__main__":
    unittest.main()
