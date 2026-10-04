"""Small synthetic regressions for deployment review findings."""
import importlib.util
import asyncio
import json
from pathlib import Path
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "web"))
sys.path.insert(0, str(ROOT / "src"))
from src.analysis.analyze_tracks import compute_msd, compute_angular_msd, fit_msd
from src.tracking.track_particles import link_tracks
import state


class ReviewRegressions(unittest.TestCase):
    def tracks(self, frames):
        return pd.DataFrame(dict(track_id=0, frame=frames, x=frames, y=0.,
                                 phi=np.arange(len(frames)) * .1, ncc=1., is_interpolated=False))

    def test_link_expiry_and_inclusive_missing_frame_boundary(self):
        for frames, expected in [([0, 2], 1), ([0, 3], 2), ([0, 100], 2)]:
            self.assertEqual(link_tracks(self.tracks(frames), 1000, 1).track_id.nunique(), expected)
        self.assertEqual(link_tracks(self.tracks([0, 1]), 1000, 0).track_id.nunique(), 1)
        self.assertEqual(link_tracks(self.tracks([0, 2]), 1000, 0).track_id.nunique(), 2)
        mixed = self.tracks([0, 1, 2, 3])
        mixed["x"] = [0, 100, 100, 0]
        linked = link_tracks(mixed, 5, 1)
        self.assertNotEqual(linked.track_id.iloc[0], linked.track_id.iloc[-1])

    def test_sparse_msd_keeps_frame_time(self):
        result = compute_msd(self.tracks([0, 2, 3]), 3, 1)
        np.testing.assert_array_equal(result.n_samples, [1, 1, 1])
        np.testing.assert_allclose(result.msd, [1, 4, 9])
        dense = self.tracks([0, 1, 2, 3])
        dense.loc[1, "is_interpolated"] = True
        np.testing.assert_allclose(compute_msd(dense, 3, 1, True).msd, [1, 4, 9])
        np.testing.assert_array_equal(compute_msd(dense, 3, 1).n_samples, [1, 1, 1])

    def test_angular_winding_does_not_bridge_missing_frames(self):
        tracks = self.tracks([0, 1, 3, 4])
        tracks["phi"] = [3.1, -3.1, 1., 1.1]
        result = compute_angular_msd(tracks, 3, 1)
        np.testing.assert_array_equal(result.n_samples, [2, 0, 0])
        self.assertAlmostEqual(result.amsd.iloc[0], ((2*np.pi-6.2)**2 + .1**2)/2)
        dense = self.tracks([0, 1, 2, 3])
        dense.loc[1, "phi"] = np.nan
        np.testing.assert_array_equal(compute_angular_msd(dense, 3, 1).n_samples, [1, 0, 0])

    def test_empty_fit_and_invalid_time(self):
        result = compute_msd(self.tracks([0, 100]), 3, 1)
        self.assertIsNone(fit_msd(result, 1/30))
        with self.assertRaises(ValueError):
            fit_msd(result, 0)

    def test_state_reload_identity_and_interruption(self):
        for field, loader, filename in [("users", state.load_users, "USERS_FILE"),
                                        ("training_jobs", state.load_training_jobs, "JOBS_FILE"),
                                        ("background_jobs", state.load_background_jobs, "BG_JOBS_FILE")]:
            target = getattr(state, field)
            previous = dict(target)
            try:
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "state.json"
                    path.write_text(json.dumps({"a": {"status": "queued"}, "b": {"status": "running"}}))
                    with patch.object(state, filename, path):
                        loader()
                    self.assertIs(target, getattr(state, field))
                    if field != "users":
                        self.assertEqual({v["status"] for v in target.values()}, {"interrupted"})
                        self.assertEqual(json.loads(path.read_text()), target)
                        target["new"] = {"status": "completed"}
                        with patch.object(state, filename, path):
                            getattr(state, "save_" + field)()
                        self.assertIn("new", json.loads(path.read_text()))
            finally:
                target.clear()
                target.update(previous)

    def test_paths_reject_traversal_and_symlink(self):
        from fastapi import HTTPException
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            root.mkdir()
            for name in ["../x", "/tmp/x", "a/b", "a\\b", "..", "C:x", ""]:
                with self.assertRaises(HTTPException):
                    state.contained_path(root, name)
            (root / "escape").symlink_to(Path(directory))
            with self.assertRaises(HTTPException):
                state.contained_path(root, "escape")
            with patch.object(state, "JUPYTER_MODE", True), patch.object(state, "resolve_identity", return_value="alice"):
                with self.assertRaises(HTTPException):
                    state.get_user_dir("bob")

    def test_atomic_failed_write_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            state._save_json(path, {"old": True})
            with patch.object(state.os, "replace", side_effect=OSError("simulated")):
                with self.assertRaises(OSError):
                    state._save_json(path, {"new": True})
            self.assertEqual(json.loads(path.read_text()), {"old": True})
            self.assertEqual(list(Path(directory).iterdir()), [path])
            with ThreadPoolExecutor(max_workers=4) as executor:
                list(executor.map(lambda i: state._save_json(path, {"value": i}), range(20)))
            self.assertIn(json.loads(path.read_text())["value"], range(20))
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_elab_dispatch_forwards_subcommand_and_help(self):
        spec = importlib.util.spec_from_file_location("elab_wrapper", ROOT / "tools/elab_cli.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for selector, target in [("simple", "simple_cli_main"), ("full", "full_cli_main")]:
            with patch.object(module, target, return_value=0) as dispatch:
                self.assertEqual(module.main([selector, "--help"]), 0)
                dispatch.assert_called_once_with(["--help"])

    def test_web_models_only_and_legacy_shared_deletion(self):
        import web.app as app
        from fastapi import HTTPException
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "weights.pth"
            weights.touch()
            models = [{"id": key, "path": str(weights), "config": {"lodestar_version": "custom"}}
                      for key in ("a", "b")]
            legacy = [dict(id="cli:particle:run", path=str(weights)),
                      dict(id="old-shared", path=str(weights), source="cli")]
            with patch.dict(app.sessions, {"review-user": {"models": models + legacy}}), \
                    patch.object(app, "save_user_session"), \
                    patch.object(app, "require_user", side_effect=lambda user: user), \
                    patch.object(app.utils, "load_yaml", side_effect=AssertionError("Web must not read CLI summary/config")):
                listed = asyncio.run(app.get_models("review-user"))["models"]
                self.assertEqual([model["id"] for model in listed], ["a", "b"])
                self.assertEqual(app._get_model_info("review-user", "a")["config"]["lodestar_version"], "custom")
                for model_id in ["cli:not-in-session:run", *[model["id"] for model in legacy]]:
                    with self.assertRaises(HTTPException) as rejected:
                        app.load_model("review-user", model_id)
                    self.assertIn(rejected.exception.status_code, (403, 404))
                    with self.assertRaises(HTTPException):
                        asyncio.run(app.delete_model("review-user", model_id))
                    with self.assertRaises(HTTPException):
                        asyncio.run(app.rename_model("review-user", model_id, app.RenameModelRequest(new_name="changed")))
                asyncio.run(app.delete_model("review-user", "a"))
                self.assertTrue(weights.exists())
                self.assertEqual([m["id"] for m in app.sessions["review-user"]["models"]],
                                 ["b", "cli:particle:run", "old-shared"])

    def test_hub_model_ownership_checked_even_for_cached_foreign_session(self):
        import web.app as app
        from fastapi import HTTPException
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "foreign.pth"
            weights.touch()
            foreign = {"models": [{"id": "foreign-model", "path": str(weights), "source": "web"}]}
            with patch.dict(app.sessions, {"foreign": foreign}), \
                    patch.object(app, "JUPYTER_MODE", True), \
                    patch.object(state, "JUPYTER_MODE", True), \
                    patch.object(state, "resolve_identity", return_value="owner"), \
                    patch.object(app, "save_user_session") as save:
                calls = [
                    lambda: app._get_model_info("foreign", "foreign-model"),
                    lambda: app.load_model("foreign", "foreign-model"),
                    lambda: asyncio.run(app.get_models("foreign")),
                    lambda: asyncio.run(app.delete_model("foreign", "foreign-model")),
                    lambda: asyncio.run(app.rename_model("foreign", "foreign-model", app.RenameModelRequest(new_name="changed"))),
                ]
                for call in calls:
                    with self.assertRaises(HTTPException) as caught:
                        call()
                    self.assertEqual(caught.exception.status_code, 403)
                save.assert_not_called()
                self.assertTrue(weights.exists())
                self.assertEqual(foreign["models"][0]["path"], str(weights))

    def test_route_paths_block_before_plot_or_job(self):
        import web.app as app
        from fastapi import HTTPException
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            results.mkdir()
            tracks = self.tracks([0, 1, 2, 3])
            tracks.to_csv(results / "tracks.csv", index=False)
            outside = root / "outside.png"
            outside.write_bytes(b"original")
            (results / "tracks_abp_msd.png").symlink_to(outside)
            (results / "tracks_video.mp4").symlink_to(outside)
            with patch.object(app, "get_user_dir", return_value=root), patch.object(app, "plot_msd") as plot:
                with self.assertRaises(HTTPException):
                    asyncio.run(app.analyze_abp(app.AbpRequest(username="review-user", csv_name="tracks.csv", min_track=1)))
                plot.assert_not_called()
                self.assertEqual(outside.read_bytes(), b"original")
                with patch.object(app.threading, "Thread") as worker:
                    with self.assertRaises(HTTPException):
                        asyncio.run(app.start_visualize_video(app.TrackVisualizeRequest(username="review-user", tracks_csv="tracks.csv")))
                    worker.assert_not_called()

    def test_csv_upload_rejects_escape(self):
        from io import BytesIO
        from starlette.datastructures import UploadFile
        from routers.files import upload_csv
        from fastapi import HTTPException
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(state, "require_user", return_value="review-user"), patch.object(state, "get_user_dir", return_value=Path(directory)):
                with self.assertRaises(HTTPException):
                    asyncio.run(upload_csv("review-user", UploadFile(filename="../escape.csv", file=BytesIO(b"x,y"))))


if __name__ == "__main__":
    unittest.main()
