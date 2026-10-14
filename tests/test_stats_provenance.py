"""Latent-stats provenance check (needs torch; skipped in the ONNX-only environment)."""

from __future__ import annotations

import unittest

try:
    import torch

    from training.utils import check_stats_provenance, encoder_fingerprint
except ImportError:  # inference-only install
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class StatsProvenanceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.encoder = torch.nn.Sequential(torch.nn.Conv1d(4, 4, 3), torch.nn.BatchNorm1d(4))
        self.stats = {"provenance": {"enc_sha256": encoder_fingerprint(self.encoder.state_dict())}}

    def test_match(self):
        self.assertIn("matches", check_stats_provenance(self.stats, self.encoder.state_dict()))

    def test_bn_buffer_change_is_refused(self):
        self.encoder[1].running_mean += 1e-3
        with self.assertRaises(ValueError):
            check_stats_provenance(self.stats, self.encoder.state_dict())

    def test_no_provenance_only_warns(self):
        self.assertIn("WARNING", check_stats_provenance({}, self.encoder.state_dict()))


if __name__ == "__main__":
    unittest.main()
