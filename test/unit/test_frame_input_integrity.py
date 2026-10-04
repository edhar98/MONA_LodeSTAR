"""Frame ranges must describe current inputs, not registration-time metadata."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "web"))
from services import frames


class FrameInputIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def tdms(self, name, old_count):
        path = self.root / name
        path.touch()
        return dict(path=str(path), type="tdms", frame_count=old_count,
                    width=99, height=99, tdms_settings={"normalize": False})

    def test_refreshes_stale_lengths_before_global_ranges(self):
        first, second = self.tdms("a.tdms", 8), self.tdms("b.tdms", 2)
        data = {first["path"]: np.full((2, 3, 4), 11, dtype=np.uint8),
                second["path"]: np.full((4, 5, 6), 22, dtype=np.uint8)}
        with patch.object(frames, "get_images", side_effect=data.get):
            getter = frames.build_session_frame_getter([first, second], {3})
        self.assertEqual(getter.total_frames, 6)
        self.assertEqual(getter.loaded_files, 1)
        self.assertEqual((first["frame_count"], first["width"], first["height"]), (2, 4, 3))
        self.assertEqual((second["frame_count"], second["width"], second["height"]), (4, 6, 5))
        np.testing.assert_array_equal(getter(3), data[second["path"]][1])
        self.assertIsNone(getter(0))
        self.assertIsNone(getter(6))

    def test_deleted_skipped_prefix_is_error(self):
        first, second = self.tdms("a.tdms", 8), self.tdms("b.tdms", 2)
        Path(first["path"]).unlink()
        with self.assertRaisesRegex(ValueError, "missing or was deleted"):
            frames.build_session_frame_getter([first, second], {9})

    def test_change_during_load_is_rejected(self):
        info = self.tdms("changing.tdms", 4)
        def mutate(_):
            replacement = self.root / "replacement.tdms"
            replacement.touch()
            replacement.replace(info["path"])
            return np.zeros((2, 2, 2), dtype=np.uint8)
        with patch.object(frames, "get_images", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "changed while loading"):
                frames.load_file_stack(info)
        self.assertEqual(info["frame_count"], 4)

    def test_prefix_change_after_snapshot_is_rejected(self):
        first, second = self.tdms("a.tdms", 2), self.tdms("b.tdms", 2)
        with patch.object(frames, "get_images", return_value=np.zeros((2, 2, 2), dtype=np.uint8)):
            getter = frames.build_session_frame_getter([first, second], {3})
        Path(first["path"]).unlink()
        with self.assertRaisesRegex(ValueError, "missing or was deleted"):
            getter(3)

    def test_image_is_loaded_detached_and_metadata_refreshed(self):
        path = self.root / "image.png"
        Image.new("L", (4, 3), 21).save(path)
        info = dict(path=str(path), type="image", frame_count=90, error="old")
        image, count = frames.extract_frame(path, info, 0)
        path.unlink()
        self.assertEqual(image.getpixel((0, 0)), 21)
        self.assertEqual(count, 1)
        self.assertEqual((info["width"], info["height"]), (4, 3))
        self.assertNotIn("error", info)
        self.assertEqual(info["source_version"]["resolved_path"], str(path))
        self.assertIsInstance(info["source_version"]["mtime_ns"], int)

    def test_metadata_error_recovery(self):
        info = self.tdms("a.tdms", 2)
        path = Path(info["path"])
        path.unlink()
        frames.parse_tdms_info(path, info)
        self.assertIn("missing or was deleted", info["error"])
        self.assertNotIn("frame_count", info)
        path.touch()
        with patch.object(frames, "get_images", return_value=np.zeros((3, 2, 4), dtype=np.uint8)):
            frames.parse_tdms_info(path, info)
        self.assertNotIn("error", info)
        self.assertEqual(info["frame_count"], 3)


if __name__ == "__main__":
    unittest.main()
