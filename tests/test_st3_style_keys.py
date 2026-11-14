"""Projected (Supertonic-3) style keys: same math as the baked key constant they fold into."""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # the inference install has no torch
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class TestStyleKeys(unittest.TestCase):
    def test_projected_keys_fold_into_baked_constant(self):
        from training.t2l.models.text_encoder import TextEncoder

        torch.manual_seed(0)
        kw = dict(vocab_size=32, d_model=64, n_conv_layers=1, n_attn_layers=1, attn_n_heads=2, attn_filter_channels=64,
                  p_dropout=0.0, spte_n_heads=2, spte_text_dim=64, spte_style_dim=64, spte_n_units=64, spte_n_style=5)
        proj = TextEncoder(spte_key_mode="projected", **kw).eval()
        baked = TextEncoder(**kw).eval()
        sd = {k: v for k, v in proj.state_dict().items() if ".W_key." not in k and not k.endswith(".style_key")}
        spte = proj.speech_prompted_text_encoder
        for name in ("attention1", "attention2"):
            layer = getattr(spte, name)
            # what the ONNX export constant-folds: tanh(W_key(style_key)) -> [H, 1, head_dim, n_style]
            k = layer.W_key(spte.style_key)
            sd[f"speech_prompted_text_encoder.{name}.key_const"] = torch.tanh(
                torch.stack(k.chunk(2, dim=-1), dim=0).transpose(-2, -1)).detach()
        baked.load_state_dict(sd, strict=True)
        ids = torch.randint(3, 32, (2, 9))
        mask = torch.ones(2, 1, 9)
        mask[1, :, 6:] = 0
        style = torch.randn(2, 5, 64)
        with torch.no_grad():
            self.assertTrue(torch.allclose(proj(ids, style, mask), baked(ids, style, mask), atol=1e-6))

    def test_vf_default_style_keys_are_tile(self):
        from training.t2l.models.vf_estimator import VectorFieldEstimator

        torch.manual_seed(0)
        vf = VectorFieldEstimator(in_channels=12, out_channels=12, hidden_channels=32, text_dim=16, style_dim=16,
                                  num_style_tokens=5, num_superblocks=1, time_hdim=32).eval()
        args = (torch.randn(2, 12, 7), torch.randn(2, 16, 4), torch.randn(2, 5, 16), torch.ones(2, 1, 7), torch.ones(2, 1, 4),
                torch.tensor([0.2, 0.7]))
        with torch.no_grad():
            self.assertTrue(torch.equal(vf(*args), vf(*args, style_keys=vf.tile.expand(2, -1, -1))))
            self.assertFalse(torch.equal(vf(*args), vf(*args, style_keys=torch.randn(2, 5, 16))))

    def test_uncond_key_token(self):
        from training.t2l.cfg_utils import UncondParams

        self.assertFalse(hasattr(UncondParams(), "u_key"))
        self.assertEqual(tuple(UncondParams(uncond_key=True).u_key.shape), (1, 50, 256))


if __name__ == "__main__":
    unittest.main()
