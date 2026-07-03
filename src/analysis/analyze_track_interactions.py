#!/usr/bin/env python3
"""
Analyze particle-particle close approaches in track data.

The script computes nearest-neighbor distances for each particle state, flags
close-neighbor states at configurable pixel thresholds, expands those states
into per-track close-approach windows, and compares short-lag motion statistics
for isolated vs near-neighbor states.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mona_mplconfig")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - scipy is available in the project env.
    cKDTree = None


REQUIRED_COLUMNS = {"track_id", "frame", "x", "y"}


def parse_thresholds(raw: str) -> list[float]:
    thresholds = sorted({float(part.strip()) for part in raw.split(",") if part.strip()})
    if not thresholds:
        raise argparse.ArgumentTypeError("At least one threshold is required.")
    if any(value <= 0 for value in thresholds):
        raise argparse.ArgumentTypeError("Thresholds must be positive pixel values.")
    return thresholds


def fmt_threshold(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def load_tracks(path: Path, include_interpolated: bool) -> pd.DataFrame:
    tracks = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(tracks.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    tracks = tracks.copy()
    tracks["track_id"] = tracks["track_id"].astype(int)
    tracks["frame"] = tracks["frame"].astype(int)
    tracks["x"] = pd.to_numeric(tracks["x"], errors="coerce")
    tracks["y"] = pd.to_numeric(tracks["y"], errors="coerce")
    tracks = tracks.dropna(subset=["x", "y"])

    if "is_interpolated" not in tracks.columns:
        tracks["is_interpolated"] = False
    else:
        tracks["is_interpolated"] = tracks["is_interpolated"].astype(bool)

    if not include_interpolated:
        tracks = tracks[~tracks["is_interpolated"]].copy()

    tracks = tracks.sort_values(["frame", "track_id"]).reset_index(drop=True)
    tracks["state_id"] = np.arange(len(tracks), dtype=np.int64)
    return tracks


def _nearest_neighbor_frame(group: pd.DataFrame) -> pd.DataFrame:
    coords = group[["x", "y"]].to_numpy(dtype=float)
    track_ids = group["track_id"].to_numpy()
    state_ids = group["state_id"].to_numpy()
    n_states = len(group)

    if n_states < 2:
        return pd.DataFrame(
            {
                "state_id": state_ids,
                "nn_track_id": np.full(n_states, np.nan),
                "nn_dist_px": np.full(n_states, np.nan),
            }
        )

    if cKDTree is not None:
        distances, indices = cKDTree(coords).query(coords, k=2)
        nn_dist = distances[:, 1]
        nn_track_id = track_ids[indices[:, 1]]
    else:
        delta = coords[:, None, :] - coords[None, :, :]
        distances = np.sqrt(np.sum(delta * delta, axis=2))
        np.fill_diagonal(distances, np.inf)
        indices = np.argmin(distances, axis=1)
        nn_dist = distances[np.arange(n_states), indices]
        nn_track_id = track_ids[indices]

    return pd.DataFrame(
        {"state_id": state_ids, "nn_track_id": nn_track_id, "nn_dist_px": nn_dist}
    )


def compute_nearest_neighbors(tracks: pd.DataFrame, thresholds: Iterable[float]) -> pd.DataFrame:
    nn_parts = [_nearest_neighbor_frame(group) for _, group in tracks.groupby("frame", sort=True)]
    nn = pd.concat(nn_parts, ignore_index=True)
    states = tracks.merge(nn, on="state_id", how="left")

    for threshold in thresholds:
        label = fmt_threshold(threshold)
        states[f"near_{label}px"] = states["nn_dist_px"] <= threshold

    return states


def add_close_windows(states: pd.DataFrame, thresholds: Iterable[float], padding: int) -> pd.DataFrame:
    states = states.sort_values(["track_id", "frame"]).copy()
    for threshold in thresholds:
        label = fmt_threshold(threshold)
        near_col = f"near_{label}px"
        window_col = f"near_window_{label}px"
        states[window_col] = False

        marked_indices: list[np.ndarray] = []
        for _, group in states.groupby("track_id", sort=False):
            frames = group["frame"].to_numpy()
            near_frames = group.loc[group[near_col], "frame"].to_numpy()
            if len(near_frames) == 0:
                continue

            near_frames = np.sort(near_frames)
            insertion = np.searchsorted(near_frames, frames)
            previous_idx = np.clip(insertion - 1, 0, len(near_frames) - 1)
            next_idx = np.clip(insertion, 0, len(near_frames) - 1)
            nearest_delta = np.minimum(
                np.abs(frames - near_frames[previous_idx]),
                np.abs(frames - near_frames[next_idx]),
            )
            in_window = nearest_delta <= padding
            if in_window.any():
                marked_indices.append(group.index.to_numpy()[in_window])

        if marked_indices:
            states.loc[np.concatenate(marked_indices), window_col] = True

    return states


def compute_motion_metrics(states: pd.DataFrame, thresholds: Iterable[float], dt: float) -> pd.DataFrame:
    rows: list[dict[str, float | int | bool]] = []
    row_by_state_id: dict[int, dict[str, float | int | bool]] = {}

    threshold_labels = [fmt_threshold(t) for t in thresholds]
    state_lookup_cols = (
        ["track_id", "frame", "state_id", "nn_dist_px"]
        + [f"near_{label}px" for label in threshold_labels]
        + [f"near_window_{label}px" for label in threshold_labels]
    )

    for track_id, group in states[state_lookup_cols + ["x", "y"]].groupby("track_id", sort=False):
        group = group.sort_values("frame")
        frames = group["frame"].to_numpy()
        coords = group[["x", "y"]].to_numpy(dtype=float)
        if len(group) < 2:
            continue

        dframe = np.diff(frames)
        dxy = np.diff(coords, axis=0)
        step = np.sqrt(np.sum(dxy * dxy, axis=1))
        valid_step = dframe > 0

        for i, is_valid in enumerate(valid_step):
            if not is_valid:
                continue
            row = {
                "track_id": int(track_id),
                "state_id": int(group.iloc[i]["state_id"]),
                "frame": int(frames[i]),
                "next_frame": int(frames[i + 1]),
                "frame_lag": int(dframe[i]),
                "step_px": float(step[i]),
                "speed_px_per_frame": float(step[i] / dframe[i]),
                "speed_px_per_s": float(step[i] / (dframe[i] * dt)) if dt > 0 else np.nan,
                "nn_dist_px": float(group.iloc[i]["nn_dist_px"])
                if pd.notna(group.iloc[i]["nn_dist_px"])
                else np.nan,
            }
            for label in threshold_labels:
                row[f"near_{label}px"] = bool(group.iloc[i][f"near_{label}px"])
                row[f"near_window_{label}px"] = bool(group.iloc[i][f"near_window_{label}px"])
            rows.append(row)
            row_by_state_id[int(row["state_id"])] = row

        if len(group) < 3:
            continue

        for i in range(1, len(group) - 1):
            if frames[i] - frames[i - 1] <= 0 or frames[i + 1] - frames[i] <= 0:
                continue

            v_prev = coords[i] - coords[i - 1]
            v_next = coords[i + 1] - coords[i]
            norm_prev = float(np.linalg.norm(v_prev))
            norm_next = float(np.linalg.norm(v_next))
            if norm_prev == 0 or norm_next == 0:
                turn_angle = np.nan
                velocity_cosine = np.nan
            else:
                cross = v_prev[0] * v_next[1] - v_prev[1] * v_next[0]
                dot = float(np.dot(v_prev, v_next))
                turn_angle = float(math.atan2(cross, dot))
                velocity_cosine = float(np.clip(dot / (norm_prev * norm_next), -1.0, 1.0))

            state_id = int(group.iloc[i]["state_id"])
            row = row_by_state_id.get(state_id)
            if row is not None:
                row["turn_angle_rad"] = turn_angle
                row["abs_turn_angle_rad"] = abs(turn_angle) if pd.notna(turn_angle) else np.nan
                row["velocity_cosine_lag1"] = velocity_cosine

    motion = pd.DataFrame(rows)
    for col in ["turn_angle_rad", "abs_turn_angle_rad", "velocity_cosine_lag1"]:
        if col not in motion.columns:
            motion[col] = np.nan
    return motion


def threshold_summary(states: pd.DataFrame, thresholds: Iterable[float]) -> pd.DataFrame:
    rows = []
    n_states = len(states)
    for threshold in thresholds:
        label = fmt_threshold(threshold)
        near_col = f"near_{label}px"
        window_col = f"near_window_{label}px"
        rows.append(
            {
                "threshold_px": threshold,
                "n_states": n_states,
                "n_near_states": int(states[near_col].sum()),
                "fraction_near": float(states[near_col].mean()),
                "n_window_states": int(states[window_col].sum()),
                "fraction_window": float(states[window_col].mean()),
            }
        )
    return pd.DataFrame(rows)


def per_track_summary(states: pd.DataFrame, thresholds: Iterable[float]) -> pd.DataFrame:
    summaries = []
    for track_id, group in states.groupby("track_id", sort=True):
        row = {
            "track_id": int(track_id),
            "n_states": int(len(group)),
            "frame_min": int(group["frame"].min()),
            "frame_max": int(group["frame"].max()),
            "median_nn_dist_px": float(group["nn_dist_px"].median()),
        }
        for threshold in thresholds:
            label = fmt_threshold(threshold)
            row[f"fraction_near_{label}px"] = float(group[f"near_{label}px"].mean())
            row[f"fraction_near_window_{label}px"] = float(group[f"near_window_{label}px"].mean())
        summaries.append(row)
    return pd.DataFrame(summaries)


def motion_group_summary(motion: pd.DataFrame, thresholds: Iterable[float]) -> pd.DataFrame:
    rows = []
    metrics = ["step_px", "speed_px_per_frame", "speed_px_per_s", "abs_turn_angle_rad", "velocity_cosine_lag1"]

    for threshold in thresholds:
        label = fmt_threshold(threshold)
        for state_col, class_name in [
            (f"near_{label}px", "instantaneous"),
            (f"near_window_{label}px", "padded_window"),
        ]:
            for is_near, group_name in [(False, "isolated"), (True, "near_neighbor")]:
                subset = motion[motion[state_col] == is_near]
                row = {
                    "threshold_px": threshold,
                    "classification": class_name,
                    "group": group_name,
                    "n_steps": int(len(subset)),
                }
                for metric in metrics:
                    values = subset[metric].dropna()
                    row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
                    row[f"{metric}_median"] = float(values.median()) if len(values) else np.nan
                    row[f"{metric}_p25"] = float(values.quantile(0.25)) if len(values) else np.nan
                    row[f"{metric}_p75"] = float(values.quantile(0.75)) if len(values) else np.nan
                rows.append(row)

    return pd.DataFrame(rows)


def close_approach_windows(states: pd.DataFrame, thresholds: Iterable[float]) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        label = fmt_threshold(threshold)
        for track_id, group in states.groupby("track_id", sort=True):
            group = group.sort_values("frame")
            mask = group[f"near_window_{label}px"].to_numpy(dtype=bool)
            if not mask.any():
                continue
            frames = group["frame"].to_numpy()
            run_start = None
            previous_frame = None
            for frame, in_window in zip(frames, mask):
                if in_window and run_start is None:
                    run_start = frame
                if (not in_window or (previous_frame is not None and frame != previous_frame + 1)) and run_start is not None:
                    end_frame = previous_frame
                    rows.append(
                        {
                            "threshold_px": threshold,
                            "track_id": int(track_id),
                            "window_start_frame": int(run_start),
                            "window_end_frame": int(end_frame),
                            "duration_frames": int(end_frame - run_start + 1),
                        }
                    )
                    run_start = frame if in_window else None
                previous_frame = frame
            if run_start is not None:
                rows.append(
                    {
                        "threshold_px": threshold,
                        "track_id": int(track_id),
                        "window_start_frame": int(run_start),
                        "window_end_frame": int(frames[-1]),
                        "duration_frames": int(frames[-1] - run_start + 1),
                    }
                )
    return pd.DataFrame(rows)


def plot_nearest_neighbor_distribution(states: pd.DataFrame, thresholds: Iterable[float], output_dir: Path, stem: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    values = states["nn_dist_px"].dropna()
    upper = values.quantile(0.995) if len(values) else 1
    bins = np.linspace(0, max(float(upper), max(thresholds) * 1.2), 80)
    ax.hist(values, bins=bins, color="#4c78a8", alpha=0.8)
    for threshold in thresholds:
        ax.axvline(threshold, color="#c44e52", linestyle="--", linewidth=1.2)
        ax.text(threshold, ax.get_ylim()[1] * 0.92, f"{threshold:g}px", rotation=90, va="top", ha="right", fontsize=8)
    ax.set_xlabel("nearest-neighbor distance (px)")
    ax.set_ylabel("particle states")
    ax.set_title("Nearest-neighbor distance distribution")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    path = output_dir / f"{stem}_nearest_neighbor_distance.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_threshold_fractions(summary: pd.DataFrame, output_dir: Path, stem: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(summary))
    width = 0.36
    ax.bar(x - width / 2, summary["fraction_near"], width, label="instantaneous", color="#4c78a8")
    ax.bar(x + width / 2, summary["fraction_window"], width, label="padded window", color="#f58518")
    ax.set_xticks(x, [f"{t:g}" for t in summary["threshold_px"]])
    ax.set_xlabel("threshold (px)")
    ax.set_ylabel("fraction of particle states")
    ax.set_title("Close-neighbor state fraction")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    path = output_dir / f"{stem}_threshold_fractions.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_per_track_fraction(per_track: pd.DataFrame, thresholds: Iterable[float], output_dir: Path, stem: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [fmt_threshold(t) for t in thresholds]
    data = [per_track[f"fraction_near_{label}px"].to_numpy() for label in labels]
    ax.boxplot(data, tick_labels=[f"{t:g}" for t in thresholds], showfliers=False)
    ax.set_xlabel("threshold (px)")
    ax.set_ylabel("per-track close-neighbor fraction")
    ax.set_title("Track-level exposure to close neighbors")
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    path = output_dir / f"{stem}_per_track_close_fraction.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_motion_comparison(motion: pd.DataFrame, compare_threshold: float, output_dir: Path, stem: str) -> Path:
    label = fmt_threshold(compare_threshold)
    near_col = f"near_{label}px"

    isolated = motion.loc[~motion[near_col]]
    near = motion.loc[motion[near_col]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, xlabel in [
        (axes[0], "speed_px_per_frame", "speed (px/frame)"),
        (axes[1], "abs_turn_angle_rad", "absolute turning angle (rad)"),
    ]:
        values = [isolated[metric].dropna(), near[metric].dropna()]
        ax.hist(values, bins=60, density=True, label=["isolated", "near neighbor"], color=["#4c78a8", "#c44e52"], alpha=0.6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.2)
        ax.legend()

    fig.suptitle(f"Motion comparison at {compare_threshold:g}px threshold")
    fig.tight_layout()
    path = output_dir / f"{stem}_speed_turning_comparison_{label}px.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_velocity_correlation(motion_summary: pd.DataFrame, output_dir: Path, stem: str) -> Path:
    data = motion_summary[
        (motion_summary["classification"] == "instantaneous")
        & (motion_summary["group"].isin(["isolated", "near_neighbor"]))
    ].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    for group_name, color in [("isolated", "#4c78a8"), ("near_neighbor", "#c44e52")]:
        group = data[data["group"] == group_name]
        ax.plot(
            group["threshold_px"],
            group["velocity_cosine_lag1_mean"],
            marker="o",
            color=color,
            label=group_name.replace("_", " "),
        )
    ax.set_xlabel("threshold (px)")
    ax.set_ylabel("mean lag-1 velocity direction cosine")
    ax.set_title("Short-lag velocity autocorrelation proxy")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    path = output_dir / f"{stem}_velocity_autocorr_lag1.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_outputs(
    states: pd.DataFrame,
    motion: pd.DataFrame,
    thresholds: list[float],
    output_dir: Path,
    stem: str,
    compare_threshold: float,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold_stats = threshold_summary(states, thresholds)
    per_track = per_track_summary(states, thresholds)
    motion_stats = motion_group_summary(motion, thresholds)
    windows = close_approach_windows(states, thresholds)

    paths = {
        "states_csv": output_dir / f"{stem}_nearest_neighbor_states.csv",
        "threshold_summary_csv": output_dir / f"{stem}_threshold_summary.csv",
        "per_track_csv": output_dir / f"{stem}_per_track_close_fraction.csv",
        "motion_csv": output_dir / f"{stem}_motion_state_metrics.csv",
        "motion_summary_csv": output_dir / f"{stem}_motion_group_summary.csv",
        "windows_csv": output_dir / f"{stem}_close_approach_windows.csv",
    }

    states.to_csv(paths["states_csv"], index=False)
    threshold_stats.to_csv(paths["threshold_summary_csv"], index=False)
    per_track.to_csv(paths["per_track_csv"], index=False)
    motion.to_csv(paths["motion_csv"], index=False)
    motion_stats.to_csv(paths["motion_summary_csv"], index=False)
    windows.to_csv(paths["windows_csv"], index=False)

    paths["nn_plot"] = plot_nearest_neighbor_distribution(states, thresholds, output_dir, stem)
    paths["threshold_plot"] = plot_threshold_fractions(threshold_stats, output_dir, stem)
    paths["per_track_plot"] = plot_per_track_fraction(per_track, thresholds, output_dir, stem)
    paths["motion_plot"] = plot_motion_comparison(motion, compare_threshold, output_dir, stem)
    paths["vacf_plot"] = plot_velocity_correlation(motion_stats, output_dir, stem)
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", required=True, type=Path, help="Track CSV with track_id, frame, x, y columns.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/interactions"),
        help="Directory for CSV and PNG outputs.",
    )
    parser.add_argument(
        "--thresholds",
        type=parse_thresholds,
        default=parse_thresholds("30,50,75"),
        help="Comma-separated close-neighbor thresholds in pixels.",
    )
    parser.add_argument("--window-padding", type=int, default=5, help="Frames before/after close states to mark as a window.")
    parser.add_argument("--dt", type=float, default=1.0, help="Seconds per frame for speed_px_per_s.")
    parser.add_argument(
        "--compare-threshold",
        type=float,
        default=None,
        help="Threshold used for the speed/turning overlay plot. Defaults to the median configured threshold.",
    )
    parser.add_argument(
        "--include-interpolated",
        action="store_true",
        help="Include rows flagged is_interpolated=True. By default they are excluded.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.window_padding < 0:
        raise ValueError("--window-padding must be non-negative.")
    if args.dt <= 0:
        raise ValueError("--dt must be positive.")

    compare_threshold = args.compare_threshold
    if compare_threshold is None:
        compare_threshold = args.thresholds[len(args.thresholds) // 2]
    if compare_threshold not in args.thresholds:
        args.thresholds = sorted(set(args.thresholds + [compare_threshold]))

    tracks = load_tracks(args.tracks, include_interpolated=args.include_interpolated)
    states = compute_nearest_neighbors(tracks, args.thresholds)
    states = add_close_windows(states, args.thresholds, args.window_padding)
    motion = compute_motion_metrics(states, args.thresholds, args.dt)
    paths = write_outputs(states, motion, args.thresholds, args.output_dir, args.tracks.stem, compare_threshold)

    summary = threshold_summary(states, args.thresholds)
    motion_stats = motion_group_summary(motion, args.thresholds)

    print(f"Loaded {len(tracks):,} particle states from {tracks['track_id'].nunique():,} tracks.")
    print(f"Frames: {tracks['frame'].min()}..{tracks['frame'].max()} ({tracks['frame'].nunique():,} unique)")
    print("Close-neighbor fractions:")
    for row in summary.itertuples(index=False):
        print(
            f"  <= {row.threshold_px:g}px: {row.fraction_near:.4f} instantaneous, "
            f"{row.fraction_window:.4f} with window padding"
        )

    print("Motion comparison, instantaneous states:")
    instant = motion_stats[motion_stats["classification"] == "instantaneous"]
    for row in instant.itertuples(index=False):
        print(
            f"  <= {row.threshold_px:g}px {row.group}: n={row.n_steps:,}, "
            f"median speed={row.speed_px_per_frame_median:.3f}px/frame, "
            f"median |turn|={row.abs_turn_angle_rad_median:.3f}rad, "
            f"mean lag1 cos={row.velocity_cosine_lag1_mean:.3f}"
        )

    print("Outputs:")
    for key, path in paths.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
