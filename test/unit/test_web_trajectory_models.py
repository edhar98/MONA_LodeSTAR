"""Bounded synthetic inference regressions; no training or real outputs."""
import sys
import unittest
import tempfile
import json
import asyncio
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / "web", ROOT / "src", ROOT / "src/tracking"):
    sys.path.insert(0, str(directory))
from services.trajectory_models import infer, validate_tracks, run_inference, catalog, resolve_model
from lstm_track_predictor import LSTMTrackPredictor, STATE_COLUMNS
from train_supervised_correction_lstm import ResidualCorrectionLSTM
from lstm_gap_filler import BiLSTMGapFiller, INPUT_COLUMNS, QUERY_COLUMNS, TARGET_COLUMNS


def norm(size):
    return dict(mean=[0.] * size, std=[1.] * size)


class TrajectoryModelsTests(unittest.TestCase):
    def frame(self, n=8):
        return pd.DataFrame(dict(track_id=np.zeros(n, dtype=int), frame=np.arange(n),
            x=np.arange(n, dtype=float), y=np.zeros(n), phi=np.zeros(n), ncc=np.ones(n),
            is_interpolated=np.zeros(n, dtype=bool), particle_class=["Janus"] * n))

    def checkpoint(self, method):
        ck = dict(hidden_size=4, layers=1, dropout=0.)
        if method == "causal_prediction":
            model = LSTMTrackPredictor(4, 4, 1, 0., 4)
            ck.update(seq_len=2, input_columns=STATE_COLUMNS, x_normalizer=norm(4), y_normalizer=norm(4), target_mode="residual")
        elif method == "supervised_correction":
            model = ResidualCorrectionLSTM(2, 4, 1, 0.)
            ck.update(seq_len=2, input_size=2, feature_cols=["lode_x", "lode_y"], input_normalizer=norm(2), target_normalizer=norm(2))
        else:
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

    def test_all_methods_preserve_baseline_and_labels(self):
        for method in ("causal_prediction", "bilstm_gap", "supervised_correction"):
            with self.subTest(method=method):
                df = self.frame()
                df.loc[3, "is_interpolated"] = True
                original = df.copy(deep=True)
                out, counts = infer(validate_tracks(df, method), method, self.checkpoint(method))
                pd.testing.assert_frame_equal(df, original)
                self.assertEqual(len(out), len(df))
                self.assertEqual(out.particle_class.tolist(), df.particle_class.tolist())
                self.assertGreater(counts["eligible_rows"], 0)
                if method != "causal_prediction":
                    np.testing.assert_array_equal(out.x_raw, df.x)
                    np.testing.assert_allclose(out.refinement_shift_px, np.hypot(out.x - out.x_raw, out.y - out.y_raw))
                    self.assertTrue(np.isfinite(counts["p95_shift_px"]))
                else:
                    np.testing.assert_array_equal(out.x, df.x)
                    self.assertFalse(out.loc[3, "has_prediction"])
                    self.assertFalse(out.loc[4, "has_prediction"])

    def test_short_and_sparse_are_noop(self):
        for method in ("causal_prediction", "bilstm_gap", "supervised_correction"):
            for df in (self.frame(1), self.frame(0), self.frame(3).assign(frame=[0, 3, 6])):
                out, counts = infer(validate_tracks(df, method), method, self.checkpoint(method))
                self.assertEqual(counts["eligible_rows"], 0)
                self.assertEqual(len(out), len(df))

    def test_supervised_preserves_interpolated_rows(self):
        df = self.frame()
        df.loc[3, "is_interpolated"] = True
        df.loc[3, "ncc"] = np.nan
        out, _ = infer(validate_tracks(df, "supervised_correction"), "supervised_correction", self.checkpoint("supervised_correction"))
        self.assertFalse(out.loc[3, "is_model_refined"])
        self.assertFalse(out.loc[4, "is_model_refined"])
        self.assertTrue(out.loc[5, "is_model_refined"])

    def test_multiple_tracks_preserve_prediction_mapping(self):
        df = pd.concat([self.frame(4), self.frame(4).assign(track_id=1, x=100.)], ignore_index=True)
        out, counts = infer(df, "causal_prediction", self.checkpoint("causal_prediction"))
        self.assertEqual(counts["prediction_rows"], 4)
        self.assertEqual(out.loc[6, "pred_x"], 100.)
        self.assertFalse(out.loc[4, "has_prediction"])

    def test_nonfinite_model_rejected(self):
        ck = self.checkpoint("causal_prediction")
        ck["model_state"]["head.2.bias"].fill_(float("nan"))
        with self.assertRaises(ValueError):
            infer(self.frame(), "causal_prediction", ck)

    def test_safe_checkpoint_artifacts_and_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "lstm_outputs").mkdir()
            model = root / "lstm_outputs/lstm_track_predictor_tiny.pt"
            torch.save(self.checkpoint("causal_prediction"), model)
            source = root / "baseline.csv"
            self.frame().to_csv(source, index=False)
            original = source.read_bytes()
            with patch("services.trajectory_models.ROOT", root):
                items = catalog()
                self.assertEqual(len(items), 1)
                with self.assertRaises(ValueError):
                    resolve_model("../../unsafe.pt")
                info = run_inference(source, items[0]["id"], root / "result.csv", root / "manifest.json")
                self.assertIsNone(info["output_tracks_csv"])
                self.assertEqual(info, json.loads((root / "manifest.json").read_text()))
                self.assertEqual(len(info["checkpoint_sha256"]), 64)
                self.assertEqual(source.read_bytes(), original)
                # Exclusive publication must not delete a preexisting artifact.
                with self.assertRaises(FileExistsError):
                    run_inference(source, items[0]["id"], source, root / "unused.json")
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


if __name__ == "__main__":
    unittest.main()
