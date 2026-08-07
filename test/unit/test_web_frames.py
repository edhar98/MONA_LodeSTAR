from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "web"))

from services.frames import normalize_tdms_frame


class WebFrameTests(unittest.TestCase):
    def test_non_normalized_int32_camera_frame_uses_16_bit_display(self):
        frame = np.array([[2159, 11153], [13587, 27101]], dtype=np.int32)
        expected = np.right_shift(frame, 8).astype(np.uint8)

        converted = normalize_tdms_frame(frame, False)

        np.testing.assert_array_equal(converted, expected)
        self.assertGreater(int(np.ptp(converted)), 0)

    def test_normalized_int32_camera_frame_uses_min_max_display(self):
        frame = np.array([[2159, 11153], [13587, 27101]], dtype=np.int32)
        expected = ((frame.astype(np.float32) - 2159) / (27101 - 2159 + 1e-8) * 255).astype(np.uint8)

        converted = normalize_tdms_frame(frame, True)

        np.testing.assert_array_equal(converted, expected)


if __name__ == "__main__":
    unittest.main()
