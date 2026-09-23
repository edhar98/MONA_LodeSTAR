"""Synthetic, non-writing composite web regressions."""
import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
from fastapi import HTTPException
import web.app as app


class CompositeWebTests(unittest.TestCase):
    def test_merge_coordinates_winner_and_empty(self):
        detector = app.CompositeDetector([("a", "A", object()), ("b", "B", object())], 3)
        image = Image.fromarray(np.zeros((20, 30), dtype=np.uint8))
        outputs = [(np.array([[10., 5.]]), np.ones((20, 30)), None, None),
                   (np.array([[12., 5.]]), np.full((20, 30), 2.), None, None)]
        with patch.object(app, "_detect_arrays", side_effect=outputs):
            result = app.run_detection_on_image(detector, image, {"detection_mode": "standard"}, False)
        self.assertEqual(result["detections"], [[11., 5.]])
        self.assertEqual(result["labels"], ["B"])
        self.assertEqual(result["confidence"], [2.])
        self.assertNotIn("phi", result)
        with patch.object(app, "_detect_arrays", return_value=(np.empty((0, 2)), np.ones((20, 30)), None, None)):
            self.assertEqual(app.run_detection_on_image(detector, image, {}, False)["count"], 0)

    def test_nonfinite_member_fails_whole_composite(self):
        detector = app.CompositeDetector([("a", "A", object()), ("b", "B", object())], 20)
        with self.assertRaises(ValueError):
            detector.detect(np.zeros((2, 2)), {}, lambda *args: (np.empty((0, 2)), np.full((2, 2), np.nan), None, None))

    def test_selection_validation_before_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "weights.pth"
            weights.touch()
            def info(user, key):
                if key == "missing":
                    raise HTTPException(404, "missing")
                return {"particle_name": key, "path": str(weights), "config": {}}
            with patch.object(app, "require_user", side_effect=lambda user: user), patch.object(app, "_get_model_info", side_effect=info):
                for ids, mode, distance in [(["a"], "standard", 20), (["a", "a"], "standard", 20),
                                            (["a", "b"], "template", 20), (["a", "b"], "standard", float("nan")),
                                            (["a", "missing"], "standard", 20)]:
                    with self.assertRaises(HTTPException):
                        app._validate_model_selection("u", None, ids, {"detection_mode": mode, "composite_distance": distance})
                params = app._validate_model_selection("u", None, "a,b", {})
                self.assertEqual(params["model_ids"], ["a", "b"])
                with patch.object(app.threading, "Thread") as worker, patch.object(app, "save_background_jobs") as save:
                    with self.assertRaises(HTTPException):
                        asyncio.run(app.detect_batch(app.BatchDetectRequest(username="u", model_ids=["a", "missing"], file_ids=["f"])))
                    worker.assert_not_called()
                    save.assert_not_called()

    def test_class_tracking_does_not_suppress_or_link_other_class(self):
        df = pd.DataFrame({"frame": [0, 1, 0, 1], "x": [10., 11., 10.2, 11.2], "y": [5.]*4,
                           "phi": [np.nan]*4, "ncc": [np.nan]*4, "particle_type": ["A", "A", "B", "B"],
                           "confidence": [1., 1., 2., 2.], "model_id": ["a", "a", "b", "b"]})
        tracks, count = app._track_detection_groups(df, {"min_dist": 20, "max_link": 30, "max_gap": 1, "min_track": 1})
        self.assertEqual(count, 4)
        self.assertEqual(tracks.track_id.nunique(), 2)
        self.assertTrue((tracks.groupby("track_id").particle_type.nunique() == 1).all())
        self.assertTrue(tracks.ncc.isna().all())
        self.assertEqual(set(tracks.confidence), {1., 2.})
        df["ncc"] = [0.4, 0.5, 0.6, 0.7]
        tracks, _ = app._track_detection_groups(df, {"min_dist": 20, "max_link": 30, "max_gap": 1, "min_track": 1})
        self.assertEqual(sorted(tracks.ncc), [0.4, 0.5, 0.6, 0.7])

    def test_batch_preserves_class_fields_and_global_order(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "detections.csv"
            result = {"detections": [[7., 3.]], "labels": ["A"], "confidence": [0.7], "model_ids": ["a"]}
            files = [{"filename": "second.png", "frame_count": 1}, {"filename": "first.png", "frame_count": 1}]
            with patch.dict(app.background_jobs, {"test-composite": {}}), patch.object(app, "save_background_jobs"), \
                 patch.object(app, "_load_detector"), patch.object(app, "_build_template_bank"), \
                 patch.object(app, "_iter_detection_frames", side_effect=lambda f: [(0, None)]), \
                 patch.object(app, "run_detection_on_image", return_value=result):
                app.run_batch_detection("test-composite", "not-a-session", files, None, {"model_ids": ["a", "b"]}, output)
                self.assertEqual(app.background_jobs["test-composite"]["status"], "completed")
            df = pd.read_csv(output, index_col=0)
            self.assertEqual(df.frame.tolist(), [0, 1])
            self.assertEqual(df.source_file.tolist(), ["second.png", "first.png"])
            self.assertEqual(df.particle_type.tolist(), ["A", "A"])
            self.assertEqual(df.confidence.tolist(), [0.7, 0.7])
            self.assertTrue(df.phi.isna().all())


if __name__ == "__main__":
    unittest.main()
