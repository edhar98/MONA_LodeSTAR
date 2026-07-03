#!/usr/bin/env python3
"""
Build paired LodeSTAR -> reference correction data.

The intended use is supervised trajectory correction:

    input  = LodeSTAR tracked state and confidence features
    target = reference/YOLO position residual relative to LodeSTAR

Reference CSVs are expected in the JP_FE per-video format:

    x, y, phi, max_inensity, summed_inensity, frame

where ``frame`` is local to the video. The video/stack id is parsed from the
filename suffix, for example ``JP_Fe_wf_2_40_slm075_574_video.csv``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


DEFAULT_LODESTAR_TRACKS = (
    "detection_results/JP_FE/wf_2_40/JP_Fe_wf_2_40_5m4rtzfx/04/tracks/"
    "JP_Fe_wf_2_40_slm075_tracks.csv"
)
DEFAULT_REFERENCE_GLOB = "data/JP_FE/wf_2_40/04/csv/*_video.csv"
DEFAULT_OUTPUT_DIR = "supervised_correction_outputs"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def parse_stack_id(path: Path) -> int:
    match = re.search(r"_(\d+)_video$", path.stem)
    if not match:
        raise ValueError(f"Could not parse stack/video id from {path.name}")
    return int(match.group(1))


def circular_delta(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray | float:
    return ((a - b + np.pi) % (2 * np.pi)) - np.pi


def load_lodestar_tracks(path: Path, include_interpolated: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"track_id", "frame", "x", "y"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    df = df.copy()
    if "phi" not in df.columns:
        df["phi"] = np.nan
    if "ncc" not in df.columns:
        df["ncc"] = np.nan
    if "is_interpolated" not in df.columns:
        df["is_interpolated"] = False
    if df["is_interpolated"].dtype == object:
        df["is_interpolated"] = df["is_interpolated"].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        )
    else:
        df["is_interpolated"] = df["is_interpolated"].fillna(False).astype(bool)

    if not include_interpolated:
        df = df[~df["is_interpolated"]].copy()

    for col in ["track_id", "frame"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["x", "y", "phi", "ncc"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["track_id", "frame", "x", "y"]).copy()
    df["track_id"] = df["track_id"].astype(int)
    df["frame"] = df["frame"].astype(int)
    return df.sort_values(["frame", "track_id"]).reset_index(drop=True)


def load_reference_csvs(
    reference_glob: str,
    first_stack: int,
    frames_per_stack: int,
    stack_offset: int,
    frame_mode: str,
) -> pd.DataFrame:
    paths = sorted(Path().glob(reference_glob))
    if not paths:
        raise FileNotFoundError(f"No reference CSVs matched: {reference_glob}")

    frames = []
    for path in paths:
        stack_id = parse_stack_id(path)
        df = pd.read_csv(path)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        required = {"x", "y", "frame"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        df = df.copy()
        if "phi" not in df.columns:
            df["phi"] = np.nan
        local_or_global_frame = pd.to_numeric(df["frame"], errors="coerce").astype(int)
        if frame_mode == "auto":
            this_frame_mode = "global" if local_or_global_frame.max() >= frames_per_stack else "local"
        else:
            this_frame_mode = frame_mode
        df["reference_csv"] = str(path)
        df["reference_stack"] = int(stack_id)
        df["reference_input_frame"] = local_or_global_frame
        df["reference_frame_mode"] = this_frame_mode
        if this_frame_mode == "global":
            df["frame"] = local_or_global_frame
        elif this_frame_mode == "local":
            df["frame"] = (
                (stack_id - first_stack + stack_offset) * frames_per_stack
                + local_or_global_frame
            )
        else:
            raise ValueError(f"Invalid frame mode: {this_frame_mode}")
        for col in ["x", "y", "phi"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["frame", "x", "y"]).copy()
        frames.append(df)

    ref = pd.concat(frames, ignore_index=True)
    ref["reference_id"] = np.arange(len(ref), dtype=int)
    return ref.sort_values(["frame", "reference_id"]).reset_index(drop=True)


def match_frame(lode: pd.DataFrame, ref: pd.DataFrame, max_distance: float) -> pd.DataFrame:
    if lode.empty or ref.empty:
        return pd.DataFrame()

    lxy = lode[["x", "y"]].to_numpy(dtype=float)
    rxy = ref[["x", "y"]].to_numpy(dtype=float)
    cost = np.linalg.norm(lxy[:, None, :] - rxy[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    keep = cost[rows, cols] <= max_distance
    rows = rows[keep]
    cols = cols[keep]

    if len(rows) == 0:
        return pd.DataFrame()

    left = lode.iloc[rows].reset_index(drop=True).add_prefix("lode_")
    right = ref.iloc[cols].reset_index(drop=True).add_prefix("ref_")
    out = pd.concat([left, right], axis=1)
    out["match_distance_px"] = cost[rows, cols]
    return out


def build_pairs(lodestar: pd.DataFrame, reference: pd.DataFrame, max_distance: float) -> pd.DataFrame:
    matched = []
    ref_by_frame = {int(frame): group for frame, group in reference.groupby("frame")}
    for frame, lode_frame in lodestar.groupby("frame", sort=True):
        ref_frame = ref_by_frame.get(int(frame))
        if ref_frame is None:
            continue
        frame_matches = match_frame(lode_frame, ref_frame, max_distance)
        if not frame_matches.empty:
            matched.append(frame_matches)
    if not matched:
        return pd.DataFrame()

    pairs = pd.concat(matched, ignore_index=True)
    pairs["target_dx"] = pairs["ref_x"] - pairs["lode_x"]
    pairs["target_dy"] = pairs["ref_y"] - pairs["lode_y"]
    pairs["target_distance_px"] = np.hypot(pairs["target_dx"], pairs["target_dy"])
    pairs["target_dphi"] = np.nan
    phi_ok = pairs["ref_phi"].notna() & pairs["lode_phi"].notna()
    pairs.loc[phi_ok, "target_dphi"] = circular_delta(
        pairs.loc[phi_ok, "ref_phi"].to_numpy(dtype=float),
        pairs.loc[phi_ok, "lode_phi"].to_numpy(dtype=float),
    )
    pairs["x_refined_supervised"] = pairs["lode_x"] + pairs["target_dx"]
    pairs["y_refined_supervised"] = pairs["lode_y"] + pairs["target_dy"]
    return pairs


def add_sequence_features(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pairs
    pairs = pairs.sort_values(["lode_track_id", "lode_frame"]).reset_index(drop=True)
    pieces = []
    for _, group in pairs.groupby("lode_track_id", sort=False):
        group = group.copy()
        group["dt_prev"] = group["lode_frame"].diff().fillna(1).clip(lower=1)
        group["dx_prev"] = group["lode_x"].diff().fillna(0.0)
        group["dy_prev"] = group["lode_y"].diff().fillna(0.0)
        group["speed_prev"] = np.hypot(group["dx_prev"], group["dy_prev"]) / group["dt_prev"]
        group["sin_phi"] = np.sin(group["lode_phi"].fillna(0.0))
        group["cos_phi"] = np.cos(group["lode_phi"].fillna(0.0))
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def write_windows_npz(pairs: pd.DataFrame, output_path: Path, seq_len: int) -> dict[str, int]:
    feature_cols = [
        "lode_x",
        "lode_y",
        "sin_phi",
        "cos_phi",
        "lode_ncc",
        "dx_prev",
        "dy_prev",
        "dt_prev",
        "speed_prev",
    ]
    target_cols = ["target_dx", "target_dy"]

    windows = []
    targets = []
    meta = []
    for tid, group in pairs.groupby("lode_track_id", sort=False):
        group = group.sort_values("lode_frame").reset_index(drop=True)
        consecutive = group["lode_frame"].diff().fillna(1).eq(1)
        segment_ids = (~consecutive).cumsum()
        for _, segment in group.groupby(segment_ids):
            if len(segment) < seq_len:
                continue
            values = segment[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
            target_values = segment[target_cols].to_numpy(dtype=np.float32)
            frames = segment["lode_frame"].to_numpy(dtype=np.int64)
            for start in range(0, len(segment) - seq_len + 1):
                stop = start + seq_len
                target_idx = stop - 1
                windows.append(values[start:stop])
                targets.append(target_values[target_idx])
                meta.append((int(tid), int(frames[target_idx])))

    if windows:
        x = np.stack(windows)
        y = np.stack(targets)
        meta_arr = np.asarray(meta, dtype=np.int64)
    else:
        x = np.empty((0, seq_len, len(feature_cols)), dtype=np.float32)
        y = np.empty((0, len(target_cols)), dtype=np.float32)
        meta_arr = np.empty((0, 2), dtype=np.int64)

    np.savez_compressed(
        output_path,
        x=x,
        y=y,
        meta=meta_arr,
        feature_cols=np.asarray(feature_cols),
        target_cols=np.asarray(target_cols),
    )
    return {"windows": int(len(x)), "seq_len": int(seq_len), "features": len(feature_cols)}


def summarize(pairs: pd.DataFrame, lodestar: pd.DataFrame, reference: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "lodestar_rows": len(lodestar),
        "lodestar_tracks": lodestar["track_id"].nunique() if not lodestar.empty else 0,
        "reference_rows": len(reference),
        "reference_frames": reference["frame"].nunique() if not reference.empty else 0,
        "matched_pairs": len(pairs),
        "matched_frames": pairs["lode_frame"].nunique() if not pairs.empty else 0,
        "matched_lodestar_tracks": pairs["lode_track_id"].nunique() if not pairs.empty else 0,
    }
    if not pairs.empty:
        summary.update(
            {
                "mean_match_distance_px": pairs["match_distance_px"].mean(),
                "median_match_distance_px": pairs["match_distance_px"].median(),
                "p95_match_distance_px": pairs["match_distance_px"].quantile(0.95),
                "mean_abs_dx_px": pairs["target_dx"].abs().mean(),
                "mean_abs_dy_px": pairs["target_dy"].abs().mean(),
                "mean_residual_distance_px": pairs["target_distance_px"].mean(),
                "median_residual_distance_px": pairs["target_distance_px"].median(),
                "p95_residual_distance_px": pairs["target_distance_px"].quantile(0.95),
            }
        )
    return {key: _jsonable(value) for key, value in summary.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build supervised LodeSTAR-to-reference correction data")
    parser.add_argument("--lodestar-tracks", default=DEFAULT_LODESTAR_TRACKS)
    parser.add_argument("--reference-glob", default=DEFAULT_REFERENCE_GLOB)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--first-stack", type=int, default=574)
    parser.add_argument("--frames-per-stack", type=int, default=100)
    parser.add_argument("--stack-offset", type=int, default=0)
    parser.add_argument("--reference-frame-mode", choices=["auto", "local", "global"], default="auto",
                        help="Whether reference CSV frame values are local to each stack or already global")
    parser.add_argument("--max-match-distance", type=float, default=20.0)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--include-interpolated", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lodestar = load_lodestar_tracks(Path(args.lodestar_tracks), args.include_interpolated)
    reference = load_reference_csvs(
        args.reference_glob,
        first_stack=args.first_stack,
        frames_per_stack=args.frames_per_stack,
        stack_offset=args.stack_offset,
        frame_mode=args.reference_frame_mode,
    )
    pairs = build_pairs(lodestar, reference, args.max_match_distance)
    pairs = add_sequence_features(pairs)

    base = Path(args.lodestar_tracks).stem
    pairs_path = output_dir / f"{base}_lodestar_reference_pairs.csv"
    windows_path = output_dir / f"{base}_correction_windows_seq{args.seq_len}.npz"
    manifest_path = output_dir / f"{base}_supervised_correction_manifest.json"

    pairs.to_csv(pairs_path, index=False)
    window_stats = write_windows_npz(pairs, windows_path, args.seq_len)
    manifest = {
        "lodestar_tracks": args.lodestar_tracks,
        "reference_glob": args.reference_glob,
        "first_stack": args.first_stack,
        "frames_per_stack": args.frames_per_stack,
        "stack_offset": args.stack_offset,
        "reference_frame_mode": args.reference_frame_mode,
        "max_match_distance": args.max_match_distance,
        "include_interpolated": args.include_interpolated,
        "pairs_csv": str(pairs_path),
        "windows_npz": str(windows_path),
        "summary": summarize(pairs, lodestar, reference),
        "window_stats": window_stats,
    }
    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2))

    print(json.dumps(_jsonable(manifest["summary"]), indent=2))
    print(f"Wrote pairs: {pairs_path}")
    print(f"Wrote windows: {windows_path}")
    print(f"Wrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
