"""Corrected gap variants cannot mutate baselines or publish failed runs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/tracking"))
import build_track_variants as variants
from lstm_gap_filler import (BiLSTMGapFiller, INPUT_COLUMNS, QUERY_COLUMNS,
                             TARGET_COLUMNS, PREPROCESSING_VERSION)


def checkpoint():
    model = BiLSTMGapFiller(hidden_size=4, num_layers=1, dropout=0.)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    model.head[-1].bias.data[0] = 2.
    norm = lambda n: {"mean": [0.] * n, "std": [1.] * n}
    return dict(preprocessing_version=PREPROCESSING_VERSION, model_state=model.state_dict(),
                input_columns=INPUT_COLUMNS, query_columns=QUERY_COLUMNS, target_columns=TARGET_COLUMNS,
                context_len=2, hidden_size=4, layers=1, dropout=0.,
                input_normalizer=norm(len(INPUT_COLUMNS)), query_normalizer=norm(len(QUERY_COLUMNS)),
                target_normalizer=norm(len(TARGET_COLUMNS)))


class GapVariantIntegrityTests(unittest.TestCase):
    def fixture(self, root):
        frame = pd.DataFrame(dict(track_id=0, frame=np.arange(8), x=np.arange(8, dtype=float),
                                  y=0., phi=0., ncc=1., is_interpolated=False))
        frame.loc[3, "is_interpolated"] = True
        source, model = root / "tracks.csv", root / "gap.pt"
        frame.to_csv(source, index=False)
        torch.save(checkpoint(), model)
        return frame, argparse.Namespace(tracks=str(source), model=str(model),
                                         output_dir=str(root / "outputs"), device="cpu")

    def test_legacy_checkpoint_rejected_before_any_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, args = self.fixture(root)
            old = checkpoint()
            old.pop("preprocessing_version")
            torch.save(old, args.model)
            original = Path(args.model).read_bytes()
            with self.assertRaisesRegex(ValueError, "(?i)legacy|retrain|preprocessing"):
                variants.build_variants(args)
            self.assertFalse(Path(args.output_dir).exists())
            self.assertEqual(Path(args.model).read_bytes(), original)
            Path(args.output_dir).mkdir()
            prior = Path(args.output_dir) / "tracks_linear.csv"
            prior.write_bytes(b"prior result")
            with self.assertRaises(ValueError):
                variants.build_variants(args)
            self.assertEqual(prior.read_bytes(), b"prior result")
            self.assertEqual(list(Path(args.output_dir).iterdir()), [prior])

    def test_unique_runs_preserve_source_measured_rows_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            frame, args = self.fixture(Path(directory))
            original = Path(args.tracks).read_bytes()
            first = variants.build_variants(args)
            second = variants.build_variants(args)
            self.assertNotEqual(first["manifest_path"], second["manifest_path"])
            self.assertEqual(first["preprocessing_version"], PREPROCESSING_VERSION)
            self.assertEqual(first["source_sha256"], hashlib.sha256(original).hexdigest())
            self.assertEqual(Path(args.tracks).read_bytes(), original)
            self.assertEqual(Path(first["variants"]["linear"]["path"]).read_bytes(), original)
            result = pd.read_csv(first["variants"]["bilstm_refined"]["path"])
            np.testing.assert_allclose(result.loc[~frame.is_interpolated, ["x", "y", "phi"]],
                                       frame.loc[~frame.is_interpolated, ["x", "y", "phi"]])
            self.assertEqual(result.loc[3, "x"], 5.)
            self.assertEqual(int(result.variant_refined.sum()), 1)
            saved = json.loads(Path(first["manifest_path"]).read_text())
            self.assertEqual(saved, first)

    def test_failed_inference_publishes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            _, args = self.fixture(Path(directory))
            with patch.object(variants, "predict_gap", return_value=[(np.nan, 0., 0.)]):
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    variants.build_variants(args)
            self.assertEqual(list(Path(args.output_dir).iterdir()), [])
