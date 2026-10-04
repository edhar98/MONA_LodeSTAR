"""Bounded synthetic inference regressions; no training or real outputs."""
import sys
import unittest
import tempfile
import json
import asyncio
import hashlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / "web", ROOT / "src", ROOT / "src/tracking"):
    sys.path.insert(0, str(directory))
from services.trajectory_models import infer, validate_tracks, run_inference, catalog, resolve_model
from lstm_gap_filler import BiLSTMGapFiller, INPUT_COLUMNS, QUERY_COLUMNS, TARGET_COLUMNS, PREPROCESSING_VERSION


def norm(size):
    return dict(mean=[0.] * size, std=[1.] * size)


class TrajectoryModelsTests(unittest.TestCase):
    def frame(self, n=8):
        return pd.DataFrame(dict(track_id=np.zeros(n, dtype=int), frame=np.arange(n),
            x=np.arange(n, dtype=float), y=np.zeros(n), phi=np.zeros(n), ncc=np.ones(n),
            is_interpolated=np.zeros(n, dtype=bool), particle_class=["Janus"] * n))

    def checkpoint(self):
        ck = dict(hidden_size=4, layers=1, dropout=0., preprocessing_version=PREPROCESSING_VERSION)
        model = BiLSTMGapFiller(len(INPUT_COLUMNS), len(QUERY_COLUMNS), 4, 1, 0., len(TARGET_COLUMNS))
        ck.update(context_len=2, input_columns=INPUT_COLUMNS, query_columns=QUERY_COLUMNS,
            target_columns=TARGET_COLUMNS, input_normalizer=norm(len(INPUT_COLUMNS)),
            query_normalizer=norm(len(QUERY_COLUMNS)), target_normalizer=norm(len(TARGET_COLUMNS)))
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)
        ck["model_state"] = model.state_dict()
        return ck

    def test_validation(self):
        for column, value in [("phi", np.nan), ("frame", .5), ("x", np.inf), ("is_interpolated", "maybe")]:
            df = self.frame()
            df.loc[0, column] = value
            with self.assertRaises(ValueError):
                validate_tracks(df, "bilstm_gap")
        with self.assertRaises(ValueError):
            validate_tracks(pd.concat([self.frame(), self.frame()]), "bilstm_gap")
        with self.assertRaises(ValueError):
            validate_tracks(self.frame().assign(x_raw=0), "bilstm_gap")

    def test_gap_refinement_preserves_measured_rows_and_labels(self):
        df = self.frame()
        df.loc[3, "is_interpolated"] = True
        original = df.copy(deep=True)
        ck = self.checkpoint()
        ck["model_state"]["head.4.bias"][0] = 2.
        out, counts = infer(validate_tracks(df, "bilstm_gap"), "bilstm_gap", ck)
        pd.testing.assert_frame_equal(df, original)
        self.assertEqual(len(out), len(df))
        self.assertEqual(out.particle_class.tolist(), df.particle_class.tolist())
        self.assertEqual(counts["eligible_rows"], 1)
        self.assertEqual(counts["interpolated_rows"], 1)
        np.testing.assert_array_equal(out.x_raw, df.x)
        np.testing.assert_allclose(out.refinement_shift_px, np.hypot(out.x - out.x_raw, out.y - out.y_raw))
        real = ~df.is_interpolated
        np.testing.assert_array_equal(out.loc[real, ["x", "y", "phi"]], df.loc[real, ["x", "y", "phi"]])
        self.assertFalse(out.loc[real, "is_model_refined"].any())
        self.assertGreater(out.loc[3, "refinement_shift_px"], 0.)

    def test_removed_methods_rejected_before_checkpoint_loading(self):
        for method in ("causal_prediction", "supervised_correction", "unknown"):
            with self.assertRaisesRegex(ValueError, "Unsupported trajectory method"):
                infer(self.frame(), method, {})
            with self.assertRaisesRegex(ValueError, "Unsupported trajectory method"):
                validate_tracks(self.frame(), method)

    def test_unversioned_checkpoint_rejected_even_without_eligible_rows(self):
        checkpoint = self.checkpoint()
        del checkpoint["preprocessing_version"]
        for df in (self.frame(), self.frame(0)):
            with self.assertRaisesRegex(ValueError, "[Rr]etrain"):
                infer(df, "bilstm_gap", checkpoint)

    def test_short_and_sparse_are_noop(self):
        for df in (self.frame(1), self.frame(0), self.frame(3).assign(frame=[0, 3, 6])):
            out, counts = infer(validate_tracks(df, "bilstm_gap"), "bilstm_gap", self.checkpoint())
            self.assertEqual(counts["eligible_rows"], 0)
            self.assertEqual(len(out), len(df))

    def test_gaps_without_clean_context_remain_linear(self):
        df = self.frame()
        df.loc[[0, 2, 3, 4, 7], "is_interpolated"] = True
        out, counts = infer(validate_tracks(df, "bilstm_gap"), "bilstm_gap", self.checkpoint())
        self.assertEqual(counts["refined_rows"], 0)
        np.testing.assert_array_equal(out[["x", "y", "phi"]], df[["x", "y", "phi"]])

    def test_multiple_tracks_preserve_gap_mapping(self):
        df = pd.concat([self.frame(), self.frame().assign(track_id=1, x=100.)], ignore_index=True)
        df.loc[[3, 11], "is_interpolated"] = True
        out, counts = infer(df, "bilstm_gap", self.checkpoint())
        self.assertEqual(counts["refined_rows"], 2)
        self.assertEqual(out.index[out.is_model_refined].tolist(), [3, 11])
        self.assertEqual(out.loc[11, "x"], 100.)

    def test_nonfinite_model_rejected(self):
        ck = self.checkpoint()
        ck["model_state"]["head.4.bias"].fill_(float("nan"))
        df = self.frame()
        df.loc[3, "is_interpolated"] = True
        with self.assertRaises(ValueError):
            infer(df, "bilstm_gap", ck)

    def test_safe_checkpoint_artifacts_and_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "lstm_outputs").mkdir()
            model = root / "lstm_outputs/lstm_gap_filler_tiny.pt"
            torch.save(self.checkpoint(), model)
            legacy_gap = root / "lstm_outputs/lstm_gap_filler_legacy.pt"
            legacy_ck = self.checkpoint()
            del legacy_ck["preprocessing_version"]
            torch.save(legacy_ck, legacy_gap)
            legacy_bytes = legacy_gap.read_bytes()
            # Old checkpoints remain on disk but are not exposed or executable.
            old_paths = ["lstm_outputs/lstm_track_predictor_old.pt",
                         "supervised_correction_outputs/run/supervised_lodestar_to_reference_lstm.pt"]
            for relative in old_paths:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({}, path)
            source = root / "baseline.csv"
            self.frame().to_csv(source, index=False)
            original = source.read_bytes()
            with patch("services.trajectory_models.ROOT", root):
                items = catalog()
                self.assertEqual(len(items), 2)
                legacy_item = next(item for item in items if not item["compatible"])
                available = next(item for item in items if item["compatible"])
                self.assertRegex(legacy_item["compatibility_error"], "[Rr]etrain")
                with self.assertRaisesRegex(ValueError, "[Rr]etrain"):
                    resolve_model(legacy_item["id"])
                with self.assertRaisesRegex(ValueError, "[Rr]etrain"):
                    run_inference(source, legacy_item["id"], root / "legacy.csv", root / "legacy.json")
                self.assertFalse((root / "legacy.csv").exists())
                self.assertEqual(legacy_gap.read_bytes(), legacy_bytes)
                self.assertEqual(available["method"], "bilstm_gap")
                for relative in old_paths:
                    stale_id = hashlib.sha256(relative.encode()).hexdigest()[:24]
                    with self.assertRaises(ValueError):
                        resolve_model(stale_id)
                    with self.assertRaises(ValueError):
                        run_inference(source, stale_id, root / "stale.csv", root / "stale.json")
                    self.assertTrue((root / relative).is_file())
                with self.assertRaises(ValueError):
                    resolve_model("../../unsafe.pt")
                info = run_inference(source, available["id"], root / "result.csv", root / "manifest.json")
                self.assertEqual(info["preprocessing_version"], PREPROCESSING_VERSION)
                self.assertEqual(info["output_tracks_csv"], "result.csv")
                self.assertEqual(info, json.loads((root / "manifest.json").read_text()))
                self.assertEqual(len(info["checkpoint_sha256"]), 64)
                self.assertEqual(source.read_bytes(), original)
                # Exclusive publication must not delete a preexisting artifact.
                with self.assertRaises(FileExistsError):
                    run_inference(source, available["id"], source, root / "unused.json")
                self.assertEqual(source.read_bytes(), original)

    def test_api_paths_and_owner(self):
        from fastapi import HTTPException
        from routers.trajectory_models import start_inference, job_status, TrajectoryRequest
        with tempfile.TemporaryDirectory() as directory:
            with patch("routers.trajectory_models.state.require_user"), patch("routers.trajectory_models.state.get_user_dir", return_value=Path(directory)), patch("routers.trajectory_models.resolve_model", return_value=({"method": "bilstm_gap"}, Path(directory))), patch("routers.trajectory_models.state.background_jobs", {"job": {"username": "other", "type": "trajectory_model"}}):
                with self.assertRaises(HTTPException) as caught:
                    asyncio.run(start_inference(TrajectoryRequest(username="owner", tracks_csv="../escape.csv", model_id="x")))
                self.assertEqual(caught.exception.status_code, 400)
                with self.assertRaises(HTTPException) as caught:
                    asyncio.run(job_status("owner", "job"))
                self.assertEqual(caught.exception.status_code, 404)

    def test_compatibility_cache_refresh_and_execution_revalidation(self):
        from services import trajectory_models as service
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "lstm_outputs").mkdir()
            checkpoint = root / "lstm_outputs/lstm_gap_filler_changing.pt"
            legacy = self.checkpoint()
            del legacy["preprocessing_version"]
            torch.save(legacy, checkpoint)
            source = root / "tracks.csv"
            self.frame().to_csv(source, index=False)
            with patch.object(service, "ROOT", root):
                self.assertFalse(catalog()[0]["compatible"])
                torch.save(self.checkpoint(), checkpoint)
                entry = catalog()[0]
                self.assertTrue(entry["compatible"])
                # A checkpoint replaced after catalog/resolve still cannot run.
                torch.save(legacy, checkpoint)
                with patch.object(service, "resolve_model", return_value=(entry, checkpoint)):
                    with self.assertRaisesRegex(ValueError, "[Rr]etrain"):
                        run_inference(source, entry["id"], root / "out.csv", root / "out.json")
                self.assertFalse((root / "out.csv").exists())
                self.assertFalse((root / "out.json").exists())


if __name__ == "__main__":
    unittest.main()
