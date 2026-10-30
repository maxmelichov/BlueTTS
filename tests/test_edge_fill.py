"""edge_fill: a padded row of a ConvNeXt batch computes what the same row computes alone."""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # the inference install has no torch
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class TestEdgeFill(unittest.TestCase):
    def setUp(self):
        from training.t2l.models.text_encoder import ConvNeXtWrapper
        from training.t2l.models.vf_estimator import ConvNeXtStack

        torch.manual_seed(0)
        # gamma 0.1 so the blocks change their input (at 1e-6 the effect is invisible)
        self.stacks = {
            fill: (
                ConvNeXtWrapper(16, n_layers=3, expansion_factor=2, dilation_lst=[1, 2, 4], layer_scale_init=0.1, edge_fill=fill),
                ConvNeXtStack(16, kernel_size=5, dilations=[1, 2], expansion=2, layer_scale_init=0.1, edge_fill=fill),
            )
            for fill in (True, False)
        }
        for a, b in zip(self.stacks[True], self.stacks[False]):
            b.load_state_dict(a.state_dict())
        self.lengths = [20, 13, 6]
        self.x = torch.randn(3, 16, 20)
        self.mask = (torch.arange(20)[None] < torch.tensor(self.lengths)[:, None]).float()[:, None]

    def _row_gap(self, stack):
        with torch.no_grad():
            batch = stack(self.x * self.mask, mask=self.mask)
            gap = 0.0
            for b, n in enumerate(self.lengths):
                alone = stack(self.x[b:b + 1, :, :n], mask=torch.ones(1, 1, n))
                gap = max(gap, (batch[b:b + 1, :, :n] - alone).abs().max().item())
        return gap

    def test_fill_tail_repeats_last_valid_column(self):
        from training.t2l.models.text_encoder import fill_tail

        x = torch.arange(10.0).view(1, 2, 5)
        mask = torch.tensor([[[1.0, 1.0, 1.0, 0.0, 0.0]]])
        out = fill_tail(x * mask, mask)
        self.assertTrue(torch.equal(out, torch.tensor([[[0.0, 1.0, 2.0, 2.0, 2.0], [5.0, 6.0, 7.0, 7.0, 7.0]]])))

    def test_padded_row_matches_exact_length_row(self):
        for stack in self.stacks[True]:
            self.assertLess(self._row_gap(stack), 1e-5)

    def test_without_fill_padded_rows_differ(self):
        for stack in self.stacks[False]:
            self.assertGreater(self._row_gap(stack), 1e-3)

    def test_noop_on_full_mask(self):
        full = torch.ones(3, 1, 20)
        with torch.no_grad():
            for on, off in zip(self.stacks[True], self.stacks[False]):
                self.assertTrue(torch.equal(on(self.x, mask=full), off(self.x, mask=full)))


if __name__ == "__main__":
    unittest.main()
