#!/usr/bin/env python3
"""
Particle tracking pipeline: within-frame NMS → cross-frame Hungarian linking → gap interpolation.

Usage:
    python src/track_particles.py \
        --input  detection_results/.../csv/..._detections.csv \
        --output detection_results/.../tracks/..._tracks.csv \
        [--min-dist 20] [--max-link 30] [--min-track 5] [--max-gap 10]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Stage 1: within-frame NMS
# ---------------------------------------------------------------------------

def _nms_indices(xy: np.ndarray, scores: np.ndarray, min_dist: float) -> np.ndarray:
    """Return indices of detections surviving NMS (highest-score first)."""
    if len(xy) == 0:
        return np.array([], dtype=int)
    order = np.argsort(scores)[::-1]
    suppressed = np.zeros(len(xy), dtype=bool)
    kept = []
    tree = cKDTree(xy)
    for idx in order:
        if suppressed[idx]:
            continue
        kept.append(idx)
        for nb in tree.query_ball_point(xy[idx], r=min_dist):
            if nb != idx:
                suppressed[nb] = True
    return np.array(kept, dtype=int)


def apply_nms(df: pd.DataFrame, min_dist: float) -> pd.DataFrame:
    """Apply per-frame NMS; returns filtered DataFrame (original index preserved)."""
    kept_rows = []
    for frame_id, group in df.groupby("frame"):
        xy = group[["x", "y"]].values
        scores = group["ncc"].fillna(0.0).values
        idx = _nms_indices(xy, scores, min_dist)
        kept_rows.append(group.iloc[idx])
    return pd.concat(kept_rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Stage 2: Hungarian frame-to-frame linking
# ---------------------------------------------------------------------------

def _match_hungarian(
    track_xy: np.ndarray,
    det_xy: np.ndarray,
    max_dist: float,
) -> tuple[list[int], list[int], list[int]]:
    """
    Returns:
        matched_track_rows  — indices into track_xy
        matched_det_rows    — indices into det_xy (parallel to above)
        unmatched_det_rows  — det indices with no track assigned
    """
    if len(track_xy) == 0:
        return [], [], list(range(len(det_xy)))
    if len(det_xy) == 0:
        return [], [], []

    cost = np.linalg.norm(track_xy[:, None, :] - det_xy[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)

    matched_t, matched_d = [], []
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= max_dist:
            matched_t.append(int(r))
            matched_d.append(int(c))

    unmatched_d = sorted(set(range(len(det_xy))) - set(matched_d))
    return matched_t, matched_d, unmatched_d


def link_tracks(
    df: pd.DataFrame,
    max_link: float,
    max_gap: int,
) -> pd.DataFrame:
    """
    Assign a persistent track_id to every detection row.

    active_tracks: track_id -> {x, y, frame, gap}
    """
    next_id = 0
    active: dict[int, dict] = {}   # track_id -> state
    rows: list[dict] = []

    for frame in sorted(df["frame"].unique()):
        frame_df = df[df["frame"] == frame].reset_index(drop=True)
        det_xy = frame_df[["x", "y"]].values

        # Build arrays from active tracks sorted by id for determinism
        active_ids = sorted(active.keys())
        if active_ids:
            track_xy = np.array([[active[tid]["x"], active[tid]["y"]]
                                  for tid in active_ids])
        else:
            track_xy = np.empty((0, 2))

        matched_t, matched_d, unmatched_d = _match_hungarian(
            track_xy, det_xy, max_link
        )

        updated_ids: set[int] = set()

        # Matched: update existing tracks
        for ti, di in zip(matched_t, matched_d):
            tid = active_ids[ti]
            row = frame_df.iloc[di]
            active[tid] = {"x": row.x, "y": row.y, "frame": frame, "gap": 0}
            rows.append(_make_row(tid, frame, row, interpolated=False))
            updated_ids.add(tid)

        # Unmatched detections → new tracks
        for di in unmatched_d:
            row = frame_df.iloc[di]
            tid = next_id
            next_id += 1
            active[tid] = {"x": row.x, "y": row.y, "frame": frame, "gap": 0}
            rows.append(_make_row(tid, frame, row, interpolated=False))
            updated_ids.add(tid)

        # Unmatched active tracks → increment gap; terminate if too long
        dead = []
        for tid in active_ids:
            if tid not in updated_ids:
                active[tid]["gap"] += 1
                if active[tid]["gap"] > max_gap:
                    dead.append(tid)
        for tid in dead:
            del active[tid]

    return pd.DataFrame(rows)


def _make_row(tid: int, frame: int, det_row, interpolated: bool) -> dict:
    return {
        "track_id": tid,
        "frame": frame,
        "x": det_row.x,
        "y": det_row.y,
        "phi": det_row.phi if pd.notna(det_row.phi) else np.nan,
        "ncc": det_row.ncc if pd.notna(det_row.ncc) else np.nan,
        "is_interpolated": interpolated,
    }


# ---------------------------------------------------------------------------
# Stage 3: gap interpolation
# ---------------------------------------------------------------------------

def interpolate_gaps(df: pd.DataFrame, max_gap: int) -> pd.DataFrame:
    """Linear interpolation for position; circular for phi."""
    interp_rows = []

    for tid, group in df.groupby("track_id"):
        group = group.sort_values("frame").reset_index(drop=True)
        frames = group["frame"].values

        for i in range(len(frames) - 1):
            gap = int(frames[i + 1]) - int(frames[i]) - 1
            if gap <= 0 or gap > max_gap:
                continue

            a = group.iloc[i]
            b = group.iloc[i + 1]

            for g in range(1, gap + 1):
                t = g / (gap + 1)
                x_i = float(a.x) + t * (float(b.x) - float(a.x))
                y_i = float(a.y) + t * (float(b.y) - float(a.y))

                phi_a = float(a.phi) if pd.notna(a.phi) else np.nan
                phi_b = float(b.phi) if pd.notna(b.phi) else np.nan
                if np.isnan(phi_a) or np.isnan(phi_b):
                    phi_i = np.nan
                else:
                    dphi = ((phi_b - phi_a + np.pi) % (2 * np.pi)) - np.pi
                    phi_i = phi_a + t * dphi

                interp_rows.append({
                    "track_id": tid,
                    "frame": int(frames[i]) + g,
                    "x": x_i,
                    "y": y_i,
                    "phi": phi_i,
                    "ncc": np.nan,
                    "is_interpolated": True,
                })

    if interp_rows:
        df = pd.concat([df, pd.DataFrame(interp_rows)], ignore_index=True)

    return df.sort_values(["track_id", "frame"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Particle tracking pipeline")
    parser.add_argument("--input",     required=True, help="Detection CSV path")
    parser.add_argument("--output",    required=True, help="Output tracks CSV path")
    parser.add_argument("--min-dist",  type=float, default=20,
                        help="Within-frame NMS radius in pixels (default: 20)")
    parser.add_argument("--max-link",  type=float, default=30,
                        help="Max cross-frame linking distance in pixels (default: 30)")
    parser.add_argument("--min-track", type=int,   default=5,
                        help="Discard tracks shorter than this many frames (default: 5)")
    parser.add_argument("--max-gap",   type=int,   default=10,
                        help="Max gap length to interpolate in frames (default: 10)")
    args = parser.parse_args()

    # Load
    df = pd.read_csv(args.input, index_col=0)

    # Normalise column names
    if "orientation_ncc" in df.columns and "ncc" not in df.columns:
        df = df.rename(columns={"orientation_ncc": "ncc"})
    if "ncc" not in df.columns:
        df["ncc"] = np.nan
    if "phi" not in df.columns:
        df["phi"] = np.nan

    df["frame"] = df["frame"].astype(int)

    n_raw = len(df)
    n_frames = df["frame"].nunique()
    print(f"Loaded {n_raw} detections across {n_frames} frames")

    # Stage 1: within-frame NMS
    df = apply_nms(df, args.min_dist)
    print(f"After NMS (min_dist={args.min_dist}px): {len(df)} detections")

    # Stage 2: track linking
    tracks = link_tracks(df, args.max_link, args.max_gap)
    n_before_filter = tracks["track_id"].nunique()
    print(f"After linking (max_link={args.max_link}px): {n_before_filter} raw tracks")

    # Filter short tracks
    lengths = tracks.groupby("track_id").size()
    valid = lengths[lengths >= args.min_track].index
    tracks = tracks[tracks["track_id"].isin(valid)].reset_index(drop=True)
    print(f"After min_track={args.min_track}: {tracks['track_id'].nunique()} tracks")

    # Stage 3: gap interpolation
    tracks = interpolate_gaps(tracks, args.max_gap)
    n_interp = int(tracks["is_interpolated"].sum())
    print(f"After gap interpolation (max_gap={args.max_gap}): "
          f"+{n_interp} interpolated rows, {len(tracks)} total rows")

    # Write output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    tracks.to_csv(args.output, index=False)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
