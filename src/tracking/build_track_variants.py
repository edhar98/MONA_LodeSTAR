#!/usr/bin/env python3
"""
Build comparable track CSV variants for downstream physics decision gates.

Variants:
  - linear: original track CSV copied unchanged
  - real_only: only non-interpolated detections
  - bilstm_refined: interpolated gaps replaced with BiLSTM gap-filler predictions
  - kalman_refined: interpolated gaps replaced with the existing Kalman smoother

The refined variants preserve the original linear coordinates in extra columns
while keeping the core analyze_tracks.py schema intact.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mona_matplotlib"))

import numpy as np
import pandas as pd
import torch

from lstm_gap_filler import _kalman_smoother_xy, _linear_state, load_model, predict_gap
from lstm_track_predictor import _add_angle_features, _add_motion_features, load_tracks


DEFAULT_TRACKS = (
    "detection_results/JP_FE/wf_2_40/JP_Fe_wf_2_40_5m4rtzfx/04/tracks/"
    "JP_Fe_wf_2_40_slm075_tracks.csv"
)
DEFAULT_MODEL = "lstm_outputs/lstm_gap_filler_jp_fe_wf_2_40_slm075.pt"
DEFAULT_OUTPUT_DIR = "analysis_outputs/track_variants"
CORE_COLUMNS = ["track_id", "frame", "x", "y", "phi", "ncc", "is_interpolated"]


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def _variant_stats(df: pd.DataFrame) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "rows": int(len(df)),
        "tracks": int(df["track_id"].nunique()) if "track_id" in df else 0,
        "interpolated_rows": int(df["is_interpolated"].sum()) if "is_interpolated" in df else 0,
    }
    if "variant_refined" in df:
        stats["refined_rows"] = int(df["variant_refined"].sum())
    else:
        stats["refined_rows"] = 0
    if "variant_has_usable_context" in df:
        stats["usable_context_rows"] = int(df["variant_has_usable_context"].sum())
    return stats


def _copy_core_schema_first(df: pd.DataFrame) -> pd.DataFrame:
    leading = [c for c in CORE_COLUMNS if c in df.columns]
    trailing = [c for c in df.columns if c not in leading]
    return df[leading + trailing]


def _gap_runs(group: pd.DataFrame) -> list[tuple[int, int]]:
    mask = group["is_interpolated"].to_numpy(dtype=bool)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, is_interp in enumerate(mask):
        if is_interp and start is None:
            start = idx
        elif not is_interp and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _prepare_refined_frame(tracks: pd.DataFrame) -> pd.DataFrame:
    df = tracks.copy()
    df["original_x"] = df["x"]
    df["original_y"] = df["y"]
    df["original_phi"] = df["phi"]
    df["variant_refined"] = False
    df["variant_has_usable_context"] = False
    df["variant_refinement_method"] = ""
    return df


def _valid_context(past: pd.DataFrame, gap: pd.DataFrame, future: pd.DataFrame, context_len: int) -> bool:
    if len(past) != context_len or len(future) != context_len or gap.empty:
        return False
    if past["is_interpolated"].any() or future["is_interpolated"].any():
        return False
    if not (past["frame"].diff().dropna() == 1).all():
        return False
    if not (future["frame"].diff().dropna() == 1).all():
        return False
    expected_gap = np.arange(int(gap.iloc[0].frame), int(gap.iloc[-1].frame) + 1)
    if not np.array_equal(gap["frame"].to_numpy(dtype=int), expected_gap):
        return False
    if int(gap.iloc[0].frame) - int(past.iloc[-1].frame) != 1:
        return False
    if int(future.iloc[0].frame) - int(gap.iloc[-1].frame) != 1:
        return False
    return True


def _iter_usable_gaps(tracks: pd.DataFrame, context_len: int):
    feature_tracks = _add_angle_features(tracks)
    feature_tracks = pd.concat(
        [_add_motion_features(group) for _, group in feature_tracks.groupby("track_id", sort=False)],
        ignore_index=True,
    )
    feature_tracks = feature_tracks.sort_values(["track_id", "frame"]).reset_index(drop=True)

    for _, group in feature_tracks.groupby("track_id", sort=False):
        group = group.sort_values("frame").reset_index()
        for start, stop in _gap_runs(group):
            past = group.iloc[start - context_len:start].copy() if start >= context_len else group.iloc[0:0].copy()
            gap = group.iloc[start:stop].copy()
            future = group.iloc[stop:stop + context_len].copy()
            if not _valid_context(past, gap, future, context_len):
                continue
            yield (
                past.drop(columns=["index"]).reset_index(drop=True),
                gap.drop(columns=["index"]).reset_index(drop=True),
                future.drop(columns=["index"]).reset_index(drop=True),
                gap["index"].to_numpy(dtype=int),
            )


def build_variants(args: argparse.Namespace) -> dict[str, Any]:
    tracks_path = Path(args.tracks)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_tracks = pd.read_csv(tracks_path)
    tracks = load_tracks(str(tracks_path))
    tracks_for_output = raw_tracks.sort_values(["track_id", "frame"]).reset_index(drop=True)
    if "is_interpolated" in tracks_for_output and tracks_for_output["is_interpolated"].dtype == object:
        tracks_for_output["is_interpolated"] = (
            tracks_for_output["is_interpolated"].astype(str).str.lower().isin(["true", "1", "yes"])
        )

    base = tracks_path.stem
    linear_path = output_dir / f"{base}_linear.csv"
    real_only_path = output_dir / f"{base}_real_only.csv"
    bilstm_path = output_dir / f"{base}_bilstm_refined.csv"
    kalman_path = output_dir / f"{base}_kalman_refined.csv"
    manifest_path = output_dir / f"{base}_manifest.json"

    shutil.copyfile(tracks_path, linear_path)
    real_only = tracks_for_output[~tracks_for_output["is_interpolated"]].copy()
    real_only.to_csv(real_only_path, index=False)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, checkpoint, input_norm, query_norm, target_norm = load_model(str(args.model), device)
    context_len = int(checkpoint["context_len"])

    bilstm = _prepare_refined_frame(tracks_for_output)
    kalman = _prepare_refined_frame(tracks_for_output)
    gap_count = 0
    gap_rows = 0
    bilstm_refined_rows = 0
    kalman_refined_rows = 0

    for past, gap, future, row_indices in _iter_usable_gaps(tracks, context_len):
        gap_count += 1
        gap_rows += len(gap)
        before = past.iloc[-1]
        after = future.iloc[0]

        bilstm.loc[row_indices, "variant_has_usable_context"] = True
        kalman.loc[row_indices, "variant_has_usable_context"] = True

        predictions = predict_gap(
            model,
            checkpoint,
            input_norm,
            query_norm,
            target_norm,
            past,
            future,
            gap,
            before,
            after,
            device,
        )
        for row_idx, (pred_x, pred_y, pred_phi) in zip(row_indices, predictions):
            bilstm.loc[row_idx, ["x", "y", "phi"]] = [pred_x, pred_y, pred_phi]
            bilstm.loc[row_idx, "variant_refined"] = True
            bilstm.loc[row_idx, "variant_refinement_method"] = "bilstm_gap"
            bilstm_refined_rows += 1

        kalman_xy = _kalman_smoother_xy(past, gap, future)
        for offset, row_idx in enumerate(row_indices):
            _, _, kalman_phi = _linear_state(before, after, gap.iloc[offset])
            kalman.loc[row_idx, ["x", "y", "phi"]] = [kalman_xy[offset, 0], kalman_xy[offset, 1], kalman_phi]
            kalman.loc[row_idx, "variant_refined"] = True
            kalman.loc[row_idx, "variant_refinement_method"] = "kalman_smoother"
            kalman_refined_rows += 1

    bilstm = _copy_core_schema_first(bilstm)
    kalman = _copy_core_schema_first(kalman)
    bilstm.to_csv(bilstm_path, index=False)
    kalman.to_csv(kalman_path, index=False)

    manifest = {
        "source_tracks": str(tracks_path),
        "bilstm_model": str(args.model),
        "context_len": context_len,
        "device": str(device),
        "usable_gap_runs": gap_count,
        "usable_gap_rows": gap_rows,
        "variants": {
            "linear": {"path": str(linear_path), **_variant_stats(tracks_for_output)},
            "real_only": {"path": str(real_only_path), **_variant_stats(real_only)},
            "bilstm_refined": {
                "path": str(bilstm_path),
                **_variant_stats(bilstm),
                "refined_rows": bilstm_refined_rows,
            },
            "kalman_refined": {
                "path": str(kalman_path),
                **_variant_stats(kalman),
                "refined_rows": kalman_refined_rows,
            },
        },
    }
    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build linear, real-only, BiLSTM, and Kalman track CSV variants.")
    parser.add_argument("--tracks", default=DEFAULT_TRACKS, help="Input track CSV")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="BiLSTM gap-filler checkpoint")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for variant CSVs and manifest")
    parser.add_argument("--device", default=None, help="Torch device, default cuda if available else cpu")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = build_variants(args)
    print(json.dumps(_jsonable(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
