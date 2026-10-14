"""DistributedWeightedSampler (needs torch)."""

from __future__ import annotations

import unittest

import torch

from training.t2l.cfg_utils import DistributedWeightedSampler


def _draw(weights, world_size, epoch=0):
    samplers = [DistributedWeightedSampler(weights, num_replicas=world_size, rank=r) for r in range(world_size)]
    for s in samplers:
        s.set_epoch(epoch)
    return [list(s) for s in samplers]


class TestDistributedWeightedSampler(unittest.TestCase):
    def test_ranks_take_disjoint_strides_of_one_draw(self):
        w = torch.rand(1000, dtype=torch.float64)
        single = _draw(w, 1)[0]
        r0, r1 = _draw(w, 2)
        self.assertEqual(len(r0), 500)
        self.assertEqual(len(r1), 500)
        interleaved = [i for pair in zip(r0, r1) for i in pair]
        self.assertEqual(interleaved, single)  # same draw as one process, split by position

    def test_frequencies_follow_weights(self):
        w = torch.tensor([0.0, 1.0, 2.0, 7.0], dtype=torch.float64)
        counts = torch.zeros(4)
        for epoch in range(2000):  # 2000 epochs x 4 draws
            for idx in _draw(w, 2, epoch):
                counts += torch.bincount(torch.tensor(idx), minlength=4)
        freq = counts / counts.sum()
        self.assertEqual(counts[0].item(), 0)
        torch.testing.assert_close(freq, (w / w.sum()).float(), atol=0.015, rtol=0)

    def test_fewer_rows_than_ranks_is_an_error(self):
        with self.assertRaises(ValueError):
            DistributedWeightedSampler(torch.ones(1, dtype=torch.float64), num_replicas=2)

    def test_epoch_reshuffles_and_is_reproducible(self):
        w = torch.ones(64, dtype=torch.float64)
        self.assertEqual(_draw(w, 2, epoch=3), _draw(w, 2, epoch=3))
        self.assertNotEqual(_draw(w, 2, epoch=3), _draw(w, 2, epoch=4))


if __name__ == "__main__":
    unittest.main()
