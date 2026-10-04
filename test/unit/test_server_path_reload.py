"""Reload server-linked inputs without changing IDs or deleting old references."""
import asyncio
from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "web"))
import state
from routers import files
from services import frames


class ServerPathReloadTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.object(state, "DATA_DIR", self.root / "state"))
        self.stack.enter_context(patch.object(state, "JUPYTER_MODE", False))
        self.stack.enter_context(patch.dict(state.users, {"user": {}}, clear=True))
        self.stack.enter_context(patch.dict(state.sessions, {"user": {"files": {}}}, clear=True))
        self.path = self.root / "input.png"
        Image.new("L", (2, 3), 10).save(self.path)

    def load(self, path=None, normalize=True):
        return asyncio.run(files.load_from_path(files.PathLoadRequest(
            username="user", path=str(path or self.path), normalize=normalize)))

    def test_repeat_refreshes_metadata_preserving_id_order_and_other_entries(self):
        first = self.load()["files"][0]
        bucket = state.sessions["user"]["files"]
        bucket[first["id"]]["note"] = "keep custom metadata"
        bucket["other"] = {"id": "other", "path": "elsewhere", "server_path": False}
        order = list(bucket)
        Image.new("L", (7, 4), 50).save(self.path)
        result = self.load(normalize=False)
        current = result["files"][0]
        self.assertEqual((result["count"], result["added"], result["updated"]), (1, 0, 1))
        self.assertEqual(current["id"], first["id"])
        self.assertEqual(list(bucket), order)
        self.assertEqual((current["width"], current["height"]), (7, 4))
        self.assertEqual(current["note"], "keep custom metadata")
        self.assertFalse(current["tdms_settings"]["normalize"])
        self.assertEqual(current["source_version"]["resolved_path"], str(self.path))
        self.assertEqual(bucket["other"]["path"], "elsewhere")

    def test_aliases_and_repeated_glob_results_use_one_id(self):
        alias = self.root / "alias.png"
        alias.symlink_to(self.path)
        first = self.load(alias)["files"][0]
        with patch.object(files._glob, "glob", return_value=[str(alias), str(self.path), str(alias)]):
            result = self.load(self.root / "*.png")
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["updated"], 1)
        self.assertEqual(result["files"][0]["id"], first["id"])
        self.assertEqual(len(state.sessions["user"]["files"]), 1)

    def test_legacy_duplicate_ids_remain_and_upload_entries_are_not_reused(self):
        bucket = state.sessions["user"]["files"]
        bucket["uploaded"] = dict(id="uploaded", path=str(self.path), server_path=False)
        first = self.load()["files"][0]
        bucket["duplicate"] = dict(first, id="duplicate", note="old job reference")
        order = list(bucket)
        result = self.load()
        self.assertEqual(result["files"][0]["id"], first["id"])
        self.assertEqual(list(bucket), order)
        self.assertEqual(bucket["duplicate"]["note"], "old job reference")
        self.assertFalse(bucket["uploaded"]["server_path"])
        self.assertTrue(self.path.exists())

    def test_tdms_reload_refreshes_live_counts_and_clears_old_error(self):
        tdms = self.root / "data.tdms"
        tdms.write_bytes(b"fixture")
        with patch.object(frames, "get_images", return_value=np.zeros((2, 3, 4), dtype=np.uint8)):
            first = self.load(tdms)["files"][0]
        state.sessions["user"]["files"][first["id"]]["error"] = "old error"
        tdms.write_bytes(b"edited input")
        with patch.object(frames, "get_images", return_value=np.zeros((5, 6, 7), dtype=np.uint8)):
            second = self.load(tdms)["files"][0]
        self.assertEqual(second["id"], first["id"])
        self.assertEqual((second["frame_count"], second["width"], second["height"]), (5, 7, 6))
        self.assertNotIn("error", second)

    def test_invalid_reload_drops_stale_dimensions_without_deleting_source(self):
        first = self.load()["files"][0]
        self.path.write_bytes(b"incomplete external edit")
        result = self.load()["files"][0]
        self.assertEqual(result["id"], first["id"])
        self.assertIn("error", result)
        for key in ("frame_count", "width", "height", "source_version"):
            self.assertNotIn(key, result)
        self.assertEqual(self.path.read_bytes(), b"incomplete external edit")


if __name__ == "__main__":
    unittest.main()
