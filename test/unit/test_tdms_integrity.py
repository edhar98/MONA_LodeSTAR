"""TDMS exports and mutable external sources: synthetic, isolated regressions."""
import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "web"))
from routers import tdms_explorer as router
from services import tdms_cache as cache


class TdmsIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "source.tdms"
        self.source.write_bytes(b"a")
        cache.invalidate()
        self.addCleanup(cache.invalidate)

    def request(self, **kwargs):
        return router.TdmsExportRequest(username="test", file_id="file", output_name="same", **kwargs)

    def export(self, request):
        return asyncio.run(router.export_tdms(request))

    def export_context(self):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch.object(router, "_tdms_file", return_value={"path": str(self.source), "filename": "sample.tdms"}))
        stack.enter_context(patch.object(router.state, "get_user_dir", return_value=self.root))
        stack.enter_context(patch.object(router, "get_images", return_value=np.arange(48, dtype=np.uint16).reshape(3, 4, 4)))
        return stack

    def test_shrinking_export_and_concurrent_names_are_isolated(self):
        with self.export_context():
            first = self.export(self.request())
            second = self.export(self.request(end_frame=1))
            with ThreadPoolExecutor(max_workers=2) as pool:
                saved = list(pool.map(self.export, [self.request(save_to_server=True)] * 2))
        for response, count in ((first, 3), (second, 1)):
            with zipfile.ZipFile(BytesIO(base64.b64decode(response["data"]))) as archive:
                self.assertEqual(len(archive.namelist()), count)
                self.assertEqual(archive.namelist(), [f"same_{i+1:03d}.png" for i in range(count)])
        self.assertNotEqual(saved[0]["path"], saved[1]["path"])
        self.assertEqual(len(list((self.root / "results").iterdir())), 4)

    def test_failed_export_leaves_earlier_output_untouched(self):
        with self.export_context():
            old = Path(self.export(self.request(save_to_server=True))["path"])
            before = {p.name: p.read_bytes() for p in old.iterdir()}
            with patch.object(router.Image.Image, "save", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    self.export(self.request(save_to_server=True))
        self.assertEqual(before, {p.name: p.read_bytes() for p in old.iterdir()})
        self.assertEqual(list((self.root / "results").iterdir()), [old])

    def test_mp4_streams_and_publishes_only_after_close(self):
        frames = []
        root = self.root
        class Writer:
            def __init__(self, path, **kwargs):
                self.path = Path(path)
            def __enter__(self):
                return self
            def append_data(self, frame):
                frames.append(frame.copy())
                self.assert_staging()
            def assert_staging(self):
                assert all(p.name.startswith(".tdms-export-") for p in (root / "results").iterdir())
            def __exit__(self, *args):
                self.path.write_bytes(b"video")
        with self.export_context(), patch.object(router.imageio, "get_writer", Writer):
            result = self.export(self.request(output_format="mp4", save_to_server=True))
        self.assertEqual(len(frames), 3)
        self.assertEqual(Path(result["path"]).read_bytes(), b"video")

    def test_invalid_export_options_and_missing_file_are_clear(self):
        with self.export_context():
            for kwargs in ({"fps": float("nan")}, {"fps": 0}, {"dtype": "bad"}, {"output_format": "bad"}, {"start_frame": 3}):
                with self.assertRaises(HTTPException) as error:
                    self.export(self.request(**kwargs))
                self.assertEqual(error.exception.status_code, 400)
            with patch.object(router, "get_images", side_effect=ValueError("TDMS source is missing or was deleted")):
                with self.assertRaisesRegex(HTTPException, "missing or was deleted"):
                    self.export(self.request())

    def test_cache_refreshes_same_size_mtime_edit_and_replacement(self):
        class Explorer:
            def __init__(self, path):
                self.path = Path(path)
            def extract_images(self):
                return np.array([self.path.read_bytes()[0]])
        with patch.object(cache, "TDMSFileExplorer", Explorer):
            old = cache.get_images(str(self.source))
            self.assertIs(old, cache.get_images(str(self.source)))
            initial = self.source.stat()
            # Metadata-based invalidation cannot distinguish byte changes whose
            # entire stat fingerprint is identical within one filesystem tick.
            time.sleep(0.02)
            self.source.write_bytes(b"b")
            os.utime(self.source, ns=(initial.st_atime_ns, initial.st_mtime_ns))
            self.assertEqual(cache.get_images(str(self.source))[0], ord("b"))
            other = self.root / "replacement"
            other.write_bytes(b"c")
            os.utime(other, ns=(initial.st_atime_ns, initial.st_mtime_ns))
            other.replace(self.source)
            self.assertEqual(cache.get_images(str(self.source))[0], ord("c"))
            self.assertEqual(len(cache._images), 1)
            self.source.unlink()
            with self.assertRaisesRegex(ValueError, "missing or was deleted"):
                cache.get_images(str(self.source))

    def test_mutation_during_extraction_is_not_cached(self):
        source = self.source
        class Explorer:
            def __init__(self, path):
                pass
            def extract_images(self):
                source.write_bytes(b"changed")
                return np.zeros((1, 2, 2))
        with patch.object(cache, "TDMSFileExplorer", Explorer):
            with self.assertRaisesRegex(ValueError, "changed while reading"):
                cache.get_images(str(source))
        self.assertFalse(cache._images)
        self.assertFalse(cache._explorers)


if __name__ == "__main__":
    unittest.main()
