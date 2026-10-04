#!/usr/bin/env python3
"""Opt-in bounded v2 training/held-out benchmark, not model qualification.

Uses a small copy of existing real tracks and only temporary outputs. Historical
checkpoints and reports are never changed. Run from the repository root.
"""
import json
import os
from pathlib import Path
import signal
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "detection_results/JP_FE/wf_2_40/JP_Fe_wf_2_40_5m4rtzfx/04/tracks/JP_Fe_wf_2_40_slm075_tracks.csv"


def main():
    def timeout(*_):
        print("FAIL: gap training smoke exceeded 180 seconds", flush=True)
        os._exit(124)
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(180)
    sys.path.insert(0, str(ROOT / "src/tracking"))
    import numpy as np
    import pandas as pd
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    from lstm_gap_filler import build_parser, train, benchmark, file_sha256, PREPROCESSING_VERSION

    source_hash = file_sha256(SOURCE)
    original = pd.read_csv(SOURCE)
    longest = original.groupby("track_id").size().nlargest(8).index
    subset = original[original.track_id.isin(longest)].sort_values(["track_id", "frame"]).groupby("track_id").head(96)
    with tempfile.TemporaryDirectory(prefix="mona-gap-v2-smoke-") as directory:
        work = Path(directory)
        source = work / "tracks.csv"
        model = work / "gap_v2.pt"
        output = work / "benchmark.csv"
        subset.to_csv(source, index=False)
        parser = build_parser()
        args = parser.parse_args(["train", "--tracks", str(source), "--model-out", str(model),
            "--context-len", "3", "--gap-lengths", "1,2,3", "--max-samples", "256",
            "--epochs", "2", "--batch-size", "64", "--hidden-size", "8", "--layers", "1",
            "--dropout", "0", "--seed", "7", "--device", "cpu"])
        assert train(args) == 0
        ck = torch.load(model, map_location="cpu", weights_only=True)
        assert ck["preprocessing_version"] == PREPROCESSING_VERSION
        splits = {name: set(ids) for name, ids in ck["split_track_ids"].items()}
        assert all(splits.values())
        assert not (splits["train"] & splits["validation"] or splits["train"] & splits["test"] or splits["validation"] & splits["test"])
        model_hash = file_sha256(model)
        try:
            train(args)
        except FileExistsError:
            pass
        else:
            raise AssertionError("Training overwrote an existing checkpoint")
        assert file_sha256(model) == model_hash
        bench = parser.parse_args(["benchmark", "--tracks", str(source), "--model", str(model),
            "--output", str(output), "--summary-output", str(work / "summary.csv"),
            "--samples", "24", "--gap-lengths", "1,2,3", "--device", "cpu"])
        assert benchmark(bench) == 0
        result = pd.read_csv(output)
        assert set(result.track_id).issubset(splits["test"])
        assert np.isfinite(result.position_error_px).all()
        counts = result.groupby("method").size()
        assert counts.nunique() == 1 and len(counts) == 5
        assert Path(str(output) + ".metadata.json").is_file()
        assert file_sha256(SOURCE) == source_hash
        print(json.dumps({"status": "PASS", "preprocessing_version": PREPROCESSING_VERSION,
            "training_rows": len(subset), "tracks": len(longest),
            "benchmark_rows_per_method": int(counts.iloc[0]),
            "test_tracks": sorted(splits["test"]),
            "limitation": "Two-epoch plumbing check only; no deployment checkpoint or superiority claim"}, indent=2))
    signal.alarm(0)


if __name__ == "__main__":
    main()
