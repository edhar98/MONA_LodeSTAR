"""Leakage invariants and bounded train/held-out benchmark for gap v2."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/tracking"))
import lstm_gap_filler as gap


def tracks(n=20, ids=(0, 1, 2, 3, 4)):
    return pd.concat([pd.DataFrame(dict(track_id=tid, frame=np.arange(n), x=np.arange(n)**2*.01+tid,
        y=np.arange(n)*.2, phi=np.arange(n)*.1, is_interpolated=False)) for tid in ids], ignore_index=True)


class GapCoreV2Tests(unittest.TestCase):
    def test_publication_collision_rolls_back_only_new_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, second = root / "first.csv", root / "second.csv"
            second.write_bytes(b"previous successful result")
            writers = [(str(first), lambda path: Path(path).write_bytes(b"new first")),
                       (str(second), lambda path: Path(path).write_bytes(b"new second"))]
            with self.assertRaises(FileExistsError):
                gap._publish_output_set(writers)
            self.assertFalse(first.exists())
            self.assertEqual(second.read_bytes(), b"previous successful result")
            self.assertEqual(list(root.iterdir()), [second])

    def test_hidden_target_and_precomputed_features_never_enter_inputs(self):
        for size in (1, 2, 10):
            original = tracks(4+size, (0,))
            original[["dx_prev", "dy_prev", "dt", "sin_phi", "cos_phi"]] = 777.
            changed = original.copy()
            changed.loc[2:1+size, ["x", "y", "phi"]] += 100
            left = gap.build_gap_dataset(original, 2, [size], None, 0)
            right = gap.build_gap_dataset(changed, 2, [size], None, 0)
            for key in ("past", "future", "query", "linear_xy"):
                np.testing.assert_array_equal(getattr(left, key), getattr(right, key))
            expected = gap.build_context_features(original.iloc[:2], original.iloc[2+size:])
            np.testing.assert_array_equal(left.past[0], expected[0])
            np.testing.assert_array_equal(left.future[0], expected[1])
            np.testing.assert_array_equal(left.future[0, -1, 4:], [0, 0, 1])
            self.assertFalse(np.array_equal(left.target, right.target))
        extended = tracks(9, (0,))
        original = gap.build_gap_dataset(extended, 2, [1], None, 0)
        extended.loc[0, ["x", "y", "phi"]] += 500
        changed = gap.build_gap_dataset(extended, 2, [1], None, 0)
        # A window starting at frame1 must ignore frame0 motion/orientation.
        np.testing.assert_array_equal(original.past[1], changed.past[1])

    def test_strict_loader_split_and_cpu_state(self):
        data = tracks()
        for col, value in (("frame", .5), ("track_id", -1), ("is_interpolated", "unknown"), ("phi", np.nan)):
            changed = data.copy().astype({col: object})
            changed.loc[0, col] = value
            with self.assertRaises(ValueError):
                gap.load_gap_tracks(io.StringIO(changed.to_csv(index=False)))
        with self.assertRaises(ValueError):
            gap.load_gap_tracks(io.StringIO(pd.concat([data, data.iloc[:1]]).to_csv(index=False)))
        splits = gap.split_track_ids(data, .2, .2, 0)
        self.assertFalse(set(splits["train"]) & set(splits["validation"]))
        self.assertFalse(set(splits["test"]) & set(splits["train"] + splits["validation"]))
        with self.assertRaises(ValueError):
            gap.split_track_ids(tracks(ids=(0,)), .2, .2, 0)
        model = torch.nn.Linear(2, 2)
        snapshot = gap.clone_cpu_state(model)
        old = snapshot["weight"].clone()
        with torch.no_grad():
            model.weight.add_(10)
        torch.testing.assert_close(snapshot["weight"], old)
        with self.assertRaisesRegex(ValueError, "retrain"):
            gap.validate_checkpoint({})

    def test_training_normalization_holdout_benchmark_and_no_clobber(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, checkpoint = root / "tracks.csv", root / "gap.pt"
            data = tracks()
            data.to_csv(source, index=False)
            args = gap.build_parser().parse_args(["train", "--tracks", str(source), "--model-out", str(checkpoint),
                "--context-len", "2", "--gap-lengths", "1,2", "--max-samples", "20", "--epochs", "1",
                "--hidden-size", "4", "--layers", "1", "--batch-size", "8", "--device", "cpu"])
            self.assertEqual(gap.train(args), 0)
            ck = torch.load(checkpoint, weights_only=True)
            training = gap.build_gap_dataset(data[data.track_id.isin(ck["split_track_ids"]["train"])], 2, [1,2], 20, 0)
            norm = gap.fit_normalizer([training.past, training.future])
            np.testing.assert_allclose(norm.mean, ck["input_normalizer"]["mean"])
            self.assertFalse(set(training.meta.track_id) & set(ck["split_track_ids"]["validation"] + ck["split_track_ids"]["test"]))
            with self.assertRaises(FileExistsError):
                gap.train(args)
            model, ck, xn, qn, yn = gap.load_model(io.BytesIO(checkpoint.read_bytes()), torch.device("cpu"))
            sample = tracks(6, (0,))
            captured = []
            hook = model.register_forward_pre_hook(lambda module, values: captured.append(tuple(v.detach().cpu().numpy().copy() for v in values)))
            a = gap.predict_gap(model, ck, xn, qn, yn, sample.iloc[:2], sample.iloc[4:], sample.iloc[2:4], None, None, torch.device("cpu"))
            hook.remove()
            arrays = gap.build_gap_dataset(sample, 2, [2], None, 0)
            np.testing.assert_array_equal(captured[0][0][0], xn.transform(arrays.past[0]))
            np.testing.assert_array_equal(captured[0][1][0], xn.transform(arrays.future[0]))
            sample.loc[2:3, ["x", "y", "phi"]] += 100
            b = gap.predict_gap(model, ck, xn, qn, yn, sample.iloc[:2], sample.iloc[4:], sample.iloc[2:4], None, None, torch.device("cpu"))
            np.testing.assert_array_equal(a, b)
            output = root / "bench.csv"
            bench = gap.build_parser().parse_args(["benchmark", "--tracks", str(source), "--model", str(checkpoint),
                "--output", str(output), "--gap-lengths", "1,2", "--samples", "2", "--device", "cpu"])
            original_hash = gap.file_sha256(checkpoint)
            original_predict = gap.predict_gap
            def replace_during_inference(model, *args, **kwargs):
                for key, tensor in model.state_dict().items():
                    torch.testing.assert_close(tensor, ck["model_state"][key])
                checkpoint.write_bytes(b"concurrently replaced checkpoint")
                return original_predict(model, *args, **kwargs)
            with patch.object(gap, "predict_gap", side_effect=replace_during_inference):
                self.assertEqual(gap.benchmark(bench), 0)
            result = pd.read_csv(output)
            self.assertTrue(set(result.track_id) <= set(ck["split_track_ids"]["test"]))
            metadata = json.loads(Path(str(output)+".metadata.json").read_text())
            self.assertEqual(metadata["evaluation_split"], "test")
            self.assertEqual(metadata["source_sha256"], gap.file_sha256(source))
            self.assertEqual(metadata["checkpoint_sha256"], original_hash)
            self.assertNotEqual(metadata["checkpoint_sha256"], gap.file_sha256(checkpoint))
            with self.assertRaises(FileExistsError):
                gap.benchmark(bench)
            bench.output = str(root / "overlap.csv")
            ck["split_track_ids"]["test"] = ck["split_track_ids"]["train"]
            torch.save(ck, checkpoint)
            with self.assertRaisesRegex(ValueError, "overlapping"):
                gap.benchmark(bench)
            self.assertFalse(Path(bench.output).exists())


if __name__ == "__main__":
    unittest.main()
