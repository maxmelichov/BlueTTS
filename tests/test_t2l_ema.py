"""T2L weight EMA (needs torch)."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from training.t2l.ema import ModelEMA, ema_checkpoint
from training.t2l.cfg_utils import UncondParams


class TestModelEMA(unittest.TestCase):
    def test_update_matches_hand_computation(self):
        lin = nn.Linear(1, 1, bias=False).double()
        values = [2.0, -1.0, 4.0, 0.5]
        with torch.no_grad():
            lin.weight.fill_(1.0)
        ema = ModelEMA([("m", lin)], decay=0.9999)
        shadow = 1.0
        for n, v in enumerate(values):
            with torch.no_grad():
                lin.weight.fill_(v)
            ema.update()
            d = min(0.9999, (1 + n) / (10 + n))
            shadow = d * shadow + (1 - d) * v
        self.assertAlmostEqual(ema.state_dicts()["m"]["weight"].item(), shadow, places=12)
        self.assertEqual(ema.updates, len(values))

    def test_warmup_reaches_the_configured_decay(self):
        ema = ModelEMA([("m", nn.Linear(1, 1))], decay=0.9999)
        ema.updates = 89_989
        self.assertLess(ema.current_decay(), 0.9999)
        ema.updates = 89_991
        self.assertEqual(ema.current_decay(), 0.9999)

    def test_tied_parameter_has_one_shadow(self):
        a, b = nn.Linear(2, 2), nn.Linear(2, 2)
        b.weight = a.weight
        ema = ModelEMA([("a", a), ("b", b)], decay=0.5)
        self.assertEqual(len(ema.params), 3)  # a.weight (shared), a.bias, b.bias
        with torch.no_grad():
            a.weight.add_(1.0)
        ema.update()
        sd = ema.state_dicts()
        self.assertTrue(torch.equal(sd["a"]["weight"], sd["b"]["weight"]))

    def test_frozen_parameter_and_buffers_follow_the_model(self):
        m = nn.BatchNorm1d(3)
        m.weight.requires_grad_(False)
        ema = ModelEMA([("m", m)], decay=0.99)
        with torch.no_grad():
            m.weight.fill_(3.0)
            m.bias.fill_(3.0)
            m.running_mean.fill_(7.0)
        ema.update()
        sd = ema.state_dicts()["m"]
        self.assertTrue(torch.equal(sd["weight"], torch.full((3,), 3.0)))  # copied, not averaged
        self.assertTrue(torch.allclose(sd["bias"], torch.full((3,), 0.1 * 0 + 0.9 * 3.0)))
        self.assertTrue(torch.equal(sd["running_mean"], torch.full((3,), 7.0)))  # buffers: current value

    def test_state_round_trip(self):
        torch.manual_seed(0)
        lin = nn.Linear(4, 4)
        ema = ModelEMA([("m", lin)], decay=0.9)
        for _ in range(3):
            with torch.no_grad():
                lin.weight.add_(torch.randn(4, 4))
            ema.update()
        saved = ema.state_dicts()
        other = ModelEMA([("m", nn.Linear(4, 4))], decay=0.9)
        self.assertEqual(other.load_state_dicts(saved), 2)
        self.assertEqual(other.updates, 3)
        for k, v in saved["m"].items():
            self.assertTrue(torch.equal(other.state_dicts()["m"][k], v))

    def test_ema_checkpoint_saves_the_averaged_cfg_tokens(self):
        torch.manual_seed(0)
        lin, unc = nn.Linear(2, 2), UncondParams(text_dim=4, n_style=3, style_value_dim=4, uncond_key=True,
                                                 style_key_dim=4)
        ema = ModelEMA([("vf_estimator", lin), ("text_encoder", nn.Linear(1, 1)),
                        ("reference_encoder", nn.Linear(1, 1)), ("uncond", unc)], decay=0.5)
        start = {k: v.detach().clone() for k, v in unc.state_dict().items()}
        with torch.no_grad():
            for p in unc.parameters():
                p.add_(1.0)
        ema.update()  # warm-up decay min(0.5, 1/10) = 0.1: shadow = 0.1 * start + 0.9 * live
        ckpt = {"u_text": unc.u_text.data, "u_ref": unc.u_ref.data, "u_key": unc.u_key.data,
                "optimizer": {}, "scheduler": {}, "global_step": 1, "ema": ema.state_dicts()}
        out = ema_checkpoint(ckpt)
        for k in ("u_text", "u_ref", "u_key"):
            want = 0.1 * start[k] + 0.9 * (start[k] + 1.0)
            self.assertTrue(torch.allclose(out[k], want))
            self.assertFalse(torch.equal(out[k], getattr(unc, k).data))  # not the live token
        self.assertNotIn("optimizer", out)
        self.assertNotIn("ema", out)
        self.assertEqual(out["ema_updates"], 1)


if __name__ == "__main__":
    unittest.main()
