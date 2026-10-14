"""Vector-field options of the 64M layout: ConvNeXt width from intermediate_dim, text-attention width, time table."""

from __future__ import annotations

import math
import unittest

try:
    import torch
except ImportError:  # the inference install has no torch
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class TestVFConfig(unittest.TestCase):
    def _vf(self, **kw):
        from training.t2l.models.vf_estimator import VectorFieldEstimator

        return VectorFieldEstimator(in_channels=12, out_channels=12, hidden_channels=32, text_dim=16, style_dim=16,
                                    num_style_tokens=5, num_superblocks=1, time_hdim=32, **kw)

    def test_convnext_width_from_intermediate_dim(self):
        default = self._vf()
        self.assertEqual(default.main_blocks[0].convnext[0].pwconv1.out_channels, 64)  # 2x hidden
        cfg = {k: {"intermediate_dim": 128} for k in ("convnext_0", "convnext_1", "convnext_2")}
        wide = self._vf(main_blocks_cfg=cfg, last_convnext_cfg={"intermediate_dim": 128})
        for stack in (wide.main_blocks[0], wide.main_blocks[2], wide.main_blocks[4], wide.last_convnext):
            self.assertEqual(stack.convnext[0].pwconv1.out_channels, 128)
        with self.assertRaises(ValueError):
            self._vf(main_blocks_cfg={"convnext_0": {"intermediate_dim": 100}})

    def test_text_attention_units(self):
        vf = self._vf(text_n_heads=8, text_n_units=512)
        attn = vf.main_blocks[3].attn
        self.assertEqual((attn.num_heads, attn.head_dim, attn.attn_scale), (8, 64, 16.0))
        self.assertEqual(attn.W_query.linear.out_features, 512)

    def test_supertonic3_time_table(self):
        from training.t2l.models.vf_estimator import SUPERTONIC3_TIME_FREQS, SinusoidalPosEmb

        table = torch.tensor(SUPERTONIC3_TIME_FREQS, dtype=torch.float32)
        formula = torch.exp(torch.arange(32) * -(math.log(10000) / 31))
        self.assertEqual(table[0].item(), 1.0)
        self.assertLess((table - formula).abs().max().item(), 1e-6)   # the same frequencies ...
        self.assertFalse(torch.equal(table, formula))                  # ... but not the same float32 values on CPU

        t = torch.tensor([0.0, 0.3, 0.875])
        x = (t * 1000.0)[:, None]
        on = SinusoidalPosEmb(64, supertonic3_freqs=True)
        off = SinusoidalPosEmb(64)
        self.assertTrue(torch.equal(on(t), torch.cat(((x * table).sin(), (x * table).cos()), dim=-1)))
        self.assertTrue(torch.equal(off(t), torch.cat(((x * formula).sin(), (x * formula).cos()), dim=-1)))
        self.assertNotIn("freqs", on.state_dict())  # no new checkpoint keys
        with self.assertRaises(ValueError):
            SinusoidalPosEmb(32, supertonic3_freqs=True)


if __name__ == "__main__":
    unittest.main()
