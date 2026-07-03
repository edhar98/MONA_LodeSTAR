#!/usr/bin/env python3
"""
Compare ABP parameters after filtering tracks by nearest-neighbor distance.

The script joins a tracks CSV to a nearest-neighbor state CSV, keeps real
detections only, and evaluates translational MSD, angular MSD, and ABP fit
parameters for a no-filter baseline plus configurable nn_dist_px thresholds.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mona_mplconfig")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_tracks import abp_msd_model, fit_angular_msd, fit_msd


REQUIRED_TRACK_COLUMNS = {"track_id", "frame", "x", "y", "phi"}
REQUIRED_NN_COLUMNS = {"track_id", "frame", "nn_dist_px"}


def parse_thresholds(raw: str) -> list[float]:
    thresholds = sorted({float(part.strip()) for part in raw.split(",") if part.strip()})
    if any(value <= 0 for value in thresholds):
        raise argparse.ArgumentTypeError("Thresholds must be positive pixel values.")
    return thresholds


def threshold_label(threshold: float | None) -> str:
    if threshold is None:
        return "no_filter"
    if float(threshold).is_integer():
        return f"gt{int(threshold)}px"
    return f"gt{str(threshold).replace('.', 'p')}px"


def threshold_name(threshold: float | None) -> str:
    return "no filter" if threshold is None else f">{threshold:g} px"


def load_inputs(tracks_path: Path, nn_path: Path) -> pd.DataFrame:
    tracks = pd.read_csv(tracks_path)
    nn = pd.read_csv(nn_path, usecols=lambda col: col in REQUIRED_NN_COLUMNS)

    missing_tracks = REQUIRED_TRACK_COLUMNS.difference(tracks.columns)
    missing_nn = REQUIRED_NN_COLUMNS.difference(nn.columns)
    if missing_tracks:
        raise ValueError(f"Missing required columns in {tracks_path}: {sorted(missing_tracks)}")
    if missing_nn:
        raise ValueError(f"Missing required columns in {nn_path}: {sorted(missing_nn)}")

    tracks = tracks.copy()
    if "is_interpolated" not in tracks.columns:
        tracks["is_interpolated"] = False
    tracks["is_interpolated"] = tracks["is_interpolated"].astype(bool)

    for frame_df in (tracks, nn):
        frame_df["track_id"] = frame_df["track_id"].astype(int)
        frame_df["frame"] = frame_df["frame"].astype(int)

    for col in ["x", "y", "phi"]:
        tracks[col] = pd.to_numeric(tracks[col], errors="coerce")
    nn["nn_dist_px"] = pd.to_numeric(nn["nn_dist_px"], errors="coerce")

    tracks = tracks.dropna(subset=["x", "y"]).copy()
    nn = nn.drop_duplicates(subset=["track_id", "frame"], keep="first")

    joined = tracks.merge(nn, on=["track_id", "frame"], how="left", validate="many_to_one")
    joined = joined[~joined["is_interpolated"]].copy()
    joined = joined.sort_values(["track_id", "frame"]).reset_index(drop=True)
    return joined


def frame_lag_msd(tracks: pd.DataFrame, max_lag: int, min_track: int) -> pd.DataFrame:
    """MSD over exact frame lags, preserving gaps introduced by filtering."""
    msd_accum = np.zeros(max_lag, dtype=float)
    counts = np.zeros(max_lag, dtype=int)

    for _, group in tracks.groupby("track_id", sort=False):
        g = group.sort_values("frame")
        if len(g) < min_track:
            continue

        coords = g[["x", "y"]].to_numpy(dtype=float)
        frame_to_index = {int(frame): idx for idx, frame in enumerate(g["frame"].to_numpy())}
        frames = np.array(sorted(frame_to_index), dtype=int)

        for lag in range(1, max_lag + 1):
            target_frames = frames + lag
            pairs = [(frame_to_index[int(frame)], frame_to_index[int(target)]) for frame, target in zip(frames, target_frames) if int(target) in frame_to_index]
            if not pairs:
                continue
            left, right = np.array(pairs, dtype=int).T
            delta = coords[right] - coords[left]
            msd_accum[lag - 1] += np.sum(delta[:, 0] ** 2 + delta[:, 1] ** 2)
            counts[lag - 1] += len(delta)

    msd = np.where(counts > 0, msd_accum / np.maximum(counts, 1), np.nan)
    return pd.DataFrame({"lag": np.arange(1, max_lag + 1), "msd": msd, "n_samples": counts})


def frame_lag_angular_msd(tracks: pd.DataFrame, max_lag: int, min_track: int) -> pd.DataFrame:
    """Angular MSD over exact frame lags, preserving gaps introduced by filtering."""
    amsd_accum = np.zeros(max_lag, dtype=float)
    counts = np.zeros(max_lag, dtype=int)

    for _, group in tracks.groupby("track_id", sort=False):
        g = group[group["phi"].notna()].sort_values("frame")
        if len(g) < min_track:
            continue

        phi = np.unwrap(g["phi"].to_numpy(dtype=float))
        frame_to_index = {int(frame): idx for idx, frame in enumerate(g["frame"].to_numpy())}
        frames = np.array(sorted(frame_to_index), dtype=int)

        for lag in range(1, max_lag + 1):
            target_frames = frames + lag
            pairs = [(frame_to_index[int(frame)], frame_to_index[int(target)]) for frame, target in zip(frames, target_frames) if int(target) in frame_to_index]
            if not pairs:
                continue
            left, right = np.array(pairs, dtype=int).T
            dphi = phi[right] - phi[left]
            amsd_accum[lag - 1] += np.sum(dphi**2)
            counts[lag - 1] += len(dphi)

    amsd = np.where(counts > 0, amsd_accum / np.maximum(counts, 1), np.nan)
    return pd.DataFrame({"lag": np.arange(1, max_lag + 1), "amsd": amsd, "n_samples": counts})


def filtered_dataset(base: pd.DataFrame, threshold: float | None) -> pd.DataFrame:
    if threshold is None:
        return base.copy()
    return base[base["nn_dist_px"] > threshold].copy()


def analyze_subset(
    subset: pd.DataFrame,
    all_real_rows: int,
    threshold: float | None,
    dt: float,
    max_lag: int,
    min_track: int,
    px_size: float,
) -> tuple[dict[str, float | int | str | None], pd.DataFrame, pd.DataFrame]:
    msd = frame_lag_msd(subset, max_lag=max_lag, min_track=min_track)
    amsd = frame_lag_angular_msd(subset, max_lag=max_lag, min_track=min_track)

    fit_params_px = fit_msd(msd, dt)
    d_r_angular = fit_angular_msd(amsd, dt)

    if fit_params_px is None:
        d_t_um2_s = v0_um_s = d_r_msd = np.nan
    else:
        d_t_px2_s, v0_px_s, d_r_msd = fit_params_px
        d_t_um2_s = d_t_px2_s * px_size**2
        v0_um_s = v0_px_s * px_size

    msd = msd.copy()
    msd["threshold_px"] = np.nan if threshold is None else threshold
    msd["filter"] = threshold_name(threshold)
    msd["msd_um2"] = msd["msd"] * px_size**2

    amsd = amsd.copy()
    amsd["threshold_px"] = np.nan if threshold is None else threshold
    amsd["filter"] = threshold_name(threshold)

    row = {
        "filter": threshold_name(threshold),
        "threshold_px": np.nan if threshold is None else threshold,
        "rows_used": int(len(subset)),
        "tracks_used": int(subset["track_id"].nunique()),
        "eligible_tracks_translational": int(subset.groupby("track_id").size().ge(min_track).sum()),
        "eligible_tracks_angular": int(subset[subset["phi"].notna()].groupby("track_id").size().ge(min_track).sum()),
        "fraction_retained": float(len(subset) / all_real_rows) if all_real_rows else np.nan,
        "msd_lag1_px2": float(msd.loc[msd["lag"] == 1, "msd"].iloc[0]),
        "msd_lag1_um2": float(msd.loc[msd["lag"] == 1, "msd_um2"].iloc[0]),
        "msd_lag1_samples": int(msd.loc[msd["lag"] == 1, "n_samples"].iloc[0]),
        "amsd_lag1_rad2": float(amsd.loc[amsd["lag"] == 1, "amsd"].iloc[0]),
        "amsd_lag1_samples": int(amsd.loc[amsd["lag"] == 1, "n_samples"].iloc[0]),
        "D_t_um2_s": float(d_t_um2_s),
        "v0_um_s": float(v0_um_s),
        "D_r_msd_rad2_s": float(d_r_msd),
        "D_r_angular_rad2_s": float(d_r_angular) if d_r_angular is not None else np.nan,
    }
    return row, msd, amsd


def plot_parameters(summary: pd.DataFrame, output_dir: Path, stem: str) -> Path:
    plot_df = summary.copy()
    plot_df["x"] = plot_df["threshold_px"].fillna(0)
    plot_df["x_label"] = plot_df["filter"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    metrics = [
        ("D_t_um2_s", "D_t (um^2/s)"),
        ("v0_um_s", "v0 (um/s)"),
        ("D_r_msd_rad2_s", "D_r from MSD (rad^2/s)"),
        ("D_r_angular_rad2_s", "D_r from angular MSD (rad^2/s)"),
    ]
    for ax, (column, ylabel) in zip(axes.ravel(), metrics):
        ax.plot(plot_df["x"], plot_df[column], marker="o", linewidth=1.6)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    for ax in axes[-1]:
        ax.set_xticks(plot_df["x"], plot_df["x_label"], rotation=20, ha="right")
        ax.set_xlabel("nearest-neighbor filter")
    fig.suptitle("ABP parameters vs nearest-neighbor distance filter")
    fig.tight_layout()
    path = output_dir / f"{stem}_filtered_abp_parameters.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_msd_curves(msd_all: pd.DataFrame, amsd_all: pd.DataFrame, dt: float, output_dir: Path, stem: str) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for label, group in msd_all.groupby("filter", sort=False):
        valid = group["msd_um2"].notna() & (group["msd_um2"] > 0)
        axes[0].loglog(group.loc[valid, "lag"] * dt, group.loc[valid, "msd_um2"], marker="o", ms=2.5, linewidth=1, label=label)
    axes[0].set_xlabel("lag time (s)")
    axes[0].set_ylabel("MSD (um^2)")
    axes[0].set_title("Translational MSD")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    for label, group in amsd_all.groupby("filter", sort=False):
        valid = group["amsd"].notna() & (group["amsd"] > 0)
        axes[1].plot(group.loc[valid, "lag"] * dt, group.loc[valid, "amsd"], marker="o", ms=2.5, linewidth=1, label=label)
    axes[1].set_xlabel("lag time (s)")
    axes[1].set_ylabel("Angular MSD (rad^2)")
    axes[1].set_title("Angular MSD")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    path = output_dir / f"{stem}_filtered_abp_msd_curves.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_filtered_tracks(subset: pd.DataFrame, output_dir: Path, stem: str, threshold: float | None) -> Path:
    path = output_dir / f"{stem}_{threshold_label(threshold)}_tracks.csv"
    subset.to_csv(path, index=False)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, required=True, help="Input track CSV.")
    parser.add_argument("--nearest-neighbor", type=Path, required=True, help="Nearest-neighbor states CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/model_comparison/filtered_abp"),
        help="Directory for filtered ABP outputs.",
    )
    parser.add_argument("--thresholds", type=parse_thresholds, default=parse_thresholds("30,50,75"), help="Comma-separated pixel thresholds.")
    parser.add_argument("--dt", type=float, default=1 / 30, help="Frame interval in seconds.")
    parser.add_argument("--max-lag", type=int, default=100, help="Maximum frame lag for MSD curves.")
    parser.add_argument("--min-track", type=int, default=50, help="Minimum retained real rows per track.")
    parser.add_argument("--px-size", type=float, default=0.078, help="Pixel size in um/px.")
    parser.add_argument("--skip-filtered-track-csv", action="store_true", help="Do not write per-filter track CSVs.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dt <= 0:
        raise ValueError("--dt must be positive.")
    if args.max_lag < 1:
        raise ValueError("--max-lag must be at least 1.")
    if args.min_track < 2:
        raise ValueError("--min-track must be at least 2.")
    if args.px_size <= 0:
        raise ValueError("--px-size must be positive.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.tracks.stem
    real_tracks = load_inputs(args.tracks, args.nearest_neighbor)
    all_real_rows = len(real_tracks)

    summary_rows = []
    msd_tables = []
    amsd_tables = []
    filtered_paths = []

    for threshold in [None] + args.thresholds:
        subset = filtered_dataset(real_tracks, threshold)
        if not args.skip_filtered_track_csv:
            filtered_paths.append(write_filtered_tracks(subset, args.output_dir, stem, threshold))
        row, msd, amsd = analyze_subset(
            subset=subset,
            all_real_rows=all_real_rows,
            threshold=threshold,
            dt=args.dt,
            max_lag=args.max_lag,
            min_track=args.min_track,
            px_size=args.px_size,
        )
        summary_rows.append(row)
        msd_tables.append(msd)
        amsd_tables.append(amsd)

    summary = pd.DataFrame(summary_rows)
    msd_all = pd.concat(msd_tables, ignore_index=True)
    amsd_all = pd.concat(amsd_tables, ignore_index=True)

    summary_path = args.output_dir / f"{stem}_filtered_abp_summary.csv"
    msd_path = args.output_dir / f"{stem}_filtered_abp_msd.csv"
    amsd_path = args.output_dir / f"{stem}_filtered_abp_amsd.csv"
    summary.to_csv(summary_path, index=False)
    msd_all.to_csv(msd_path, index=False)
    amsd_all.to_csv(amsd_path, index=False)

    parameter_plot = plot_parameters(summary, args.output_dir, stem)
    msd_plot = plot_msd_curves(msd_all, amsd_all, args.dt, args.output_dir, stem)

    print(f"Loaded {all_real_rows:,} real rows from {real_tracks['track_id'].nunique():,} tracks.")
    print(f"Outputs written under {args.output_dir}")
    print(f"  summary: {summary_path}")
    print(f"  msd: {msd_path}")
    print(f"  amsd: {amsd_path}")
    print(f"  parameter_plot: {parameter_plot}")
    print(f"  msd_plot: {msd_plot}")
    if filtered_paths:
        print("  filtered_track_csvs:")
        for path in filtered_paths:
            print(f"    {path}")

    print("Threshold-dependent ABP parameters:")
    for row in summary.itertuples(index=False):
        print(
            f"  {row.filter}: rows={row.rows_used:,}, tracks={row.tracks_used:,}, "
            f"retained={row.fraction_retained:.3f}, "
            f"D_t={row.D_t_um2_s:.6g} um^2/s, v0={row.v0_um_s:.6g} um/s, "
            f"D_r(MSD)={row.D_r_msd_rad2_s:.6g} rad^2/s, "
            f"D_r(AMSD)={row.D_r_angular_rad2_s:.6g} rad^2/s"
        )


if __name__ == "__main__":
    main()
