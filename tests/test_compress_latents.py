"""compress_latents tail padding (needs torch; skipped in the ONNX-only environment)."""

from __future__ import annotations

import unittest

try:
    import torch

    from training.utils import compress_latents, decompress_latents
except ImportError:  # inference-only install
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class CompressLatentsTest(unittest.TestCase):
    def test_tail_repeats_last_frame(self):
        z = torch.randn(2, 24, 13)
        out = decompress_latents(compress_latents(z, factor=6), factor=6, target_channels=24)
        self.assertEqual(out.shape[-1], 18)
        self.assertTrue(torch.equal(out[..., :13], z))
        self.assertTrue(torch.equal(out[..., 13:], z[..., -1:].expand(-1, -1, 5)))

    def test_whole_chunks_unchanged(self):
        z = torch.randn(2, 24, 12)
        expected = z.view(2, 24, 2, 6).permute(0, 1, 3, 2).flatten(1, 2)
        self.assertTrue(torch.equal(compress_latents(z, factor=6), expected))


if __name__ == "__main__":
    unittest.main()
