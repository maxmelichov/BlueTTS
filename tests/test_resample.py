"""The numpy port of the training resampler used for voice cloning references."""
import unittest

import numpy as np

from src.blue_onnx.style import _resample_training


class ResampleTrainingTest(unittest.TestCase):
    def test_length_and_tone(self):
        sr, new = 24000, 44100
        t = np.arange(sr) / sr
        x = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        y = _resample_training(x, sr, new)
        self.assertEqual(len(y), new)
        self.assertEqual(y.dtype, np.float32)
        mid = y[4410:-4410]
        ref = 0.5 * np.sin(2 * np.pi * 1000 * np.arange(new) / new)[4410:-4410]
        self.assertLess(float(np.abs(mid - ref).max()), 1e-3)

    def test_same_rate_is_identity(self):
        x = np.linspace(-1, 1, 1000, dtype=np.float32)
        np.testing.assert_array_equal(_resample_training(x, 44100, 44100), x)

    def test_downsample_length(self):
        x = np.zeros(48000, dtype=np.float32)
        self.assertEqual(len(_resample_training(x, 48000, 44100)), 44100)


if __name__ == "__main__":
    unittest.main()
