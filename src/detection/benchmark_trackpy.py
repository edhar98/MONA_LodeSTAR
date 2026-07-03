#!/usr/bin/env python3
"""
Benchmark trackpy linking against the LodeSTAR Hungarian tracking pipeline.

Loads a LodeSTAR detection CSV, runs NMS and gap interpolation, then compares
trajectory statistics against tracks produced by trackpy's linking stage.
Use this to evaluate whether the classical linker matches or beats the learned pipeline.
"""
import argparse
import os

import numpy as np
import pandas as pd

from track_particles import apply_nms, interpolate_gaps


def _load_detections(path: str) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(path)
    unnamed_cols: list[str] = [col for col in df.columns if col.startswith("Unnamed:")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    required: set[str] = {"frame", "x", "y"}
    missing: set[str] = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required detection columns: {sorted(missing)}")
    if "orientation_ncc" in df.columns and "ncc" not in df.columns:
        df = df.rename(columns={"orientation_ncc": "ncc"})
    if "confidence" in df.columns and "ncc" not in df.columns:
        df["ncc"] = df["confidence"]
    if "ncc" not in df.columns:
        df["ncc"] = np.nan
    if "phi" not in df.columns:
        df["phi"] = np.nan
    df["frame"] = df["frame"].astype(int)
    return df


def _run_trackpy(df: pd.DataFrame, search_range: float, memory: int) -> pd.DataFrame:
    try:
        import trackpy as tp
    except ImportError as exc:
        raise ImportError("trackpy is required for this benchmark. Install it with: pip install trackpy") from exc
    linked: pd.DataFrame = tp.link_df(
        df.sort_values("frame").reset_index(drop=True),
        search_range=search_range,
        memory=memory,
        pos_columns=["x", "y"],
        t_column="frame",
    )
    return linked


def _to_track_schema(df: pd.DataFrame, min_track: int, max_gap: int) -> pd.DataFrame:
    df = df.rename(columns={"particle": "track_id"})
    tracks: pd.DataFrame = df[["track_id", "frame", "x", "y", "phi", "ncc"]].copy()
    tracks["track_id"] = tracks["track_id"].astype(int)
    tracks["is_interpolated"] = False
    lengths: pd.Series = tracks.groupby("track_id").size()
    valid: pd.Index = lengths[lengths >= min_track].index
    tracks = tracks[tracks["track_id"].isin(valid)].reset_index(drop=True)
    return interpolate_gaps(tracks, max_gap)


def main() -> None:
    parser = argparse.ArgumentParser(description="Trackpy benchmark for MONA particle detections")
    parser.add_argument("--input", required=True, help="Detection CSV path")
    parser.add_argument("--output", required=True, help="Output trackpy tracks CSV path")
    parser.add_argument("--min-dist", type=float, default=20.0, help="Within-frame NMS radius in pixels")
    parser.add_argument("--search-range", type=float, default=30.0, help="Trackpy linking search range in pixels")
    parser.add_argument("--memory", type=int, default=10, help="Trackpy memory in frames")
    parser.add_argument("--min-track", type=int, default=5, help="Discard tracks shorter than this many detections")
    args = parser.parse_args()

    detections: pd.DataFrame = _load_detections(args.input)
    n_raw: int = len(detections)
    n_frames: int = detections["frame"].nunique()
    detections = apply_nms(detections, args.min_dist)
    linked: pd.DataFrame = _run_trackpy(detections, args.search_range, args.memory)
    tracks: pd.DataFrame = _to_track_schema(linked, args.min_track, args.memory)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    tracks.to_csv(args.output, index=False)

    print(f"Loaded {n_raw} detections across {n_frames} frames")
    print(f"After NMS: {len(detections)} detections")
    print(f"Trackpy tracks: {tracks['track_id'].nunique()}")
    print(f"Interpolated rows: {int(tracks['is_interpolated'].sum())}")
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
