"""T2L lr schedule on resume (needs torch)."""

from __future__ import annotations

import unittest

import torch
from torch.optim import AdamW

from training.t2l.cfg_utils import build_lr_scheduler

BASE_LR = 5e-4
MILESTONES = [3, 6]


def _closed_form(step: int) -> float:
    return BASE_LR * 0.5 ** sum(step >= m for m in MILESTONES)


def _train(n_steps: int):
    """Run n_steps optimizer steps from scratch; return (optimizer, scheduler, lr after each step)."""
    p = torch.nn.Parameter(torch.zeros(2))
    opt = AdamW([p], lr=BASE_LR)
    sched = build_lr_scheduler(opt, BASE_LR, MILESTONES, global_step=0)
    lrs = []
    for _ in range(n_steps):
        p.grad = torch.ones(2)
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    return opt, sched, lrs


def _resume(step: int, load_optimizer: bool = True, milestones=MILESTONES):
    opt0, sched0, _ = _train(step)
    p = torch.nn.Parameter(torch.zeros(2))
    opt = AdamW([p], lr=BASE_LR)
    if load_optimizer:
        opt.load_state_dict(opt0.state_dict())
    sched = build_lr_scheduler(opt, BASE_LR, milestones, step, sched0.state_dict())
    return opt, sched


class TestResumeLr(unittest.TestCase):
    def test_uninterrupted_run_follows_closed_form(self):
        _, _, lrs = _train(8)
        self.assertEqual(lrs, [_closed_form(s) for s in range(1, 9)])

    def test_resume_at_every_step_matches_uninterrupted_run(self):
        _, _, ref = _train(9)
        for step in range(1, 8):  # includes resuming exactly at milestones 3 and 6
            for load_optimizer in (True, False):
                opt, sched = _resume(step, load_optimizer)
                self.assertEqual(opt.param_groups[0]["lr"], ref[step - 1], (step, load_optimizer))
                self.assertEqual(sched.get_last_lr()[0], ref[step - 1])
                got = []
                for _ in range(2):
                    sched.step()
                    got.append(opt.param_groups[0]["lr"])
                self.assertEqual(got, ref[step:step + 2], (step, load_optimizer))

    def test_without_pin_a_milestone_resume_halves_twice(self):
        # What the trainer did before: MultiStepLR(last_epoch=step-1) on the loaded optimizer.
        opt0, sched0, _ = _train(3)
        p = torch.nn.Parameter(torch.zeros(2))
        opt = AdamW([p], lr=BASE_LR)
        opt.load_state_dict(opt0.state_dict())
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, MILESTONES, gamma=0.5, last_epoch=2)
        sched.load_state_dict(sched0.state_dict())
        self.assertEqual(opt.param_groups[0]["lr"], BASE_LR / 4)
        self.assertEqual(_resume(3)[0].param_groups[0]["lr"], BASE_LR / 2)

    def test_configured_milestones_override_the_saved_ones(self):
        opt, sched = _resume(4, milestones=[5, 7])
        self.assertEqual(opt.param_groups[0]["lr"], BASE_LR)
        sched.step()
        self.assertEqual(opt.param_groups[0]["lr"], BASE_LR / 2)

    def test_finetune_restarts_at_base_lr(self):
        opt0, sched0, _ = _train(7)
        p = torch.nn.Parameter(torch.zeros(2))
        opt = AdamW([p], lr=BASE_LR)
        sched = build_lr_scheduler(opt, BASE_LR, MILESTONES, 7, sched0.state_dict(), finetune=True)
        self.assertEqual(opt.param_groups[0]["lr"], BASE_LR)
        self.assertEqual(sched.last_epoch, 0)


if __name__ == "__main__":
    unittest.main()
