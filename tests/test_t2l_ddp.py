"""T2L trainer DDP / accumulation fixes (CPU, gloo; needs torch)."""

from __future__ import annotations

import os
import socket
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from training.t2l.cfg_utils import UncondParams, all_reduce_grads, seed_worker


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ddp_worker(rank, world_size, port, out_dir):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.manual_seed(0)
    net = nn.Linear(4, 4)
    uncond = UncondParams(text_dim=4, n_style=2, style_value_dim=4)
    u_text = uncond.u_text  # taken before wrapping, as in train_loop
    net = DDP(net)
    uncond = DDP(uncond, find_unused_parameters=True)

    # Different data per rank, as with a DistributedSampler.
    x = torch.full((2, 4), float(rank + 1))
    loss = (net(x) + u_text.view(1, 4) * x).pow(2).mean()
    loss.backward()

    out = {"net_raw": net.module.weight.grad.clone(), "u_raw": u_text.grad.clone()}
    all_reduce_grads([u_text])
    out["u_synced"] = u_text.grad.clone()
    torch.save(out, os.path.join(out_dir, f"rank{rank}.pt"))
    dist.destroy_process_group()


class TestUncondGradSync(unittest.TestCase):
    def test_ddp_does_not_sync_raw_u_tokens_and_all_reduce_does(self):
        with tempfile.TemporaryDirectory() as d:
            mp.spawn(_ddp_worker, args=(2, _free_port(), d), nprocs=2)
            r0 = torch.load(os.path.join(d, "rank0.pt"))
            r1 = torch.load(os.path.join(d, "rank1.pt"))
        # DDP averages the gradients of modules whose forward it runs ...
        torch.testing.assert_close(r0["net_raw"], r1["net_raw"], rtol=0, atol=0)
        # ... but not of u_text, used as a raw tensor: the two ranks would step apart.
        self.assertGreater((r0["u_raw"] - r1["u_raw"]).abs().max().item(), 1e-3)
        # all_reduce_grads gives both ranks the mean gradient.
        mean = (r0["u_raw"] + r1["u_raw"]) / 2
        torch.testing.assert_close(r0["u_synced"], mean)
        torch.testing.assert_close(r1["u_synced"], mean)

    def test_no_op_without_process_group(self):
        p = nn.Parameter(torch.ones(3))
        p.grad = torch.full((3,), 2.0)
        all_reduce_grads([p])
        self.assertTrue(torch.equal(p.grad, torch.full((3,), 2.0)))


class TestTailFlushRescale(unittest.TestCase):
    def test_partial_group_uses_mean_of_its_micro_batches(self):
        accumulation_steps, k = 4, 3
        torch.manual_seed(0)
        w = nn.Parameter(torch.randn(5))
        xs = [torch.randn(5) for _ in range(k)]
        for x in xs:
            ((w * x).sum() ** 2 / accumulation_steps).backward()
        w.grad.mul_(accumulation_steps / k)  # as at the epoch tail in train_loop
        ref = torch.autograd.grad(sum((w * x).sum() ** 2 for x in xs) / k, w)[0]
        torch.testing.assert_close(w.grad, ref)


class TestSeedWorker(unittest.TestCase):
    def _draw(self, worker_id, **kw):
        seed_worker(worker_id, **kw)
        return torch.rand(3)

    def test_rank_and_resume_step_change_the_stream(self):
        base = self._draw(0)
        self.assertTrue(torch.equal(base, self._draw(0, rank=0, start_step=0)))
        self.assertFalse(torch.equal(base, self._draw(0, rank=1)))
        self.assertFalse(torch.equal(base, self._draw(0, start_step=1000)))
        self.assertFalse(torch.equal(base, self._draw(1)))

    def test_base_seed_keeps_the_fixed_scheme(self):
        torch.manual_seed(42 + 3)
        expected = torch.rand(3)
        self.assertTrue(torch.equal(self._draw(3, base_seed=42), expected))


if __name__ == "__main__":
    unittest.main()
