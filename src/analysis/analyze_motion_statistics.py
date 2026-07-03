#!/usr/bin/env python3
"""
Raw motion statistics for comparing particle-track variants.

This deliberately avoids ABP/LSTM fitting. It summarizes observed motion over
lag frames so track-quality variants can be compared before model assumptions
are trusted.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"track_id", "frame", "x", "y", "phi"}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def load_tracks(path: str, include_interpolated: bool) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    if "is_interpolated" not in df.columns:
        df["is_interpolated"] = False

    df["is_interpolated"] = df["is_interpolated"].fillna(False).astype(bool)
    total_rows = int(len(df))
    total_tracks = int(df["track_id"].nunique())
    interpolated_rows = int(df["is_interpolated"].sum())

    if not include_interpolated:
        df = df[~df["is_interpolated"]].copy()

    df = df.dropna(subset=["track_id", "frame", "x", "y"]).copy()
    df["track_id"] = df["track_id"].astype(int)
    df["frame"] = df["frame"].astype(int)
    df = df.sort_values(["track_id", "frame"]).reset_index(drop=True)

    metadata = {
        "input_rows": total_rows,
        "input_tracks": total_tracks,
        "input_interpolated_rows": interpolated_rows,
        "include_interpolated": bool(include_interpolated),
        "rows_used": int(len(df)),
        "tracks_used": int(df["track_id"].nunique()) if len(df) else 0,
        "interpolated_rows_used": int(df["is_interpolated"].sum()) if len(df) else 0,
    }
    return df, metadata


def tracks_with_min_points(df: pd.DataFrame, min_track_length: int) -> pd.DataFrame:
    if df.empty:
        return df
    lengths = df.groupby("track_id").size()
    keep_ids = lengths[lengths >= min_track_length].index
    return df[df["track_id"].isin(keep_ids)].copy()


def contiguous_segments(group: pd.DataFrame) -> list[pd.DataFrame]:
    """Split a track anywhere frame numbers are not consecutive."""
    group = group.sort_values("frame")
    if len(group) == 0:
        return []
    breaks = group["frame"].diff().fillna(1).ne(1).cumsum()
    return [seg for _, seg in group.groupby(breaks) if len(seg) > 0]


def lagged_arrays(df: pd.DataFrame, max_lag: int, need_phi: bool = False):
    for _, group in df.groupby("track_id", sort=False):
        if need_phi:
            group = group.dropna(subset=["phi"])
        for seg in contiguous_segments(group):
            n = len(seg)
            if n < 2:
                continue
            x = seg["x"].to_numpy(dtype=float)
            y = seg["y"].to_numpy(dtype=float)
            phi = np.unwrap(seg["phi"].to_numpy(dtype=float)) if need_phi else None
            max_seg_lag = min(max_lag, n - 1)
            for lag in range(1, max_seg_lag + 1):
                yield lag, x, y, phi


def compute_lag_statistics(df: pd.DataFrame, max_lag: int, px_size: float, frame_rate: float) -> pd.DataFrame:
    rows = []
    accum = {
        lag: {
            "msd_sum": 0.0,
            "amsd_sum": 0.0,
            "disp_auto_sum": 0.0,
            "orient_auto_sum": 0.0,
            "msd_n": 0,
            "amsd_n": 0,
            "disp_auto_n": 0,
            "orient_auto_n": 0,
        }
        for lag in range(1, max_lag + 1)
    }

    for _, group in df.groupby("track_id", sort=False):
        pos_group = group.dropna(subset=["x", "y"])
        for seg in contiguous_segments(pos_group):
            n = len(seg)
            if n < 2:
                continue
            x = seg["x"].to_numpy(dtype=float)
            y = seg["y"].to_numpy(dtype=float)
            dx1 = np.diff(x)
            dy1 = np.diff(y)
            max_seg_lag = min(max_lag, n - 1)
            for lag in range(1, max_seg_lag + 1):
                dx = x[lag:] - x[:-lag]
                dy = y[lag:] - y[:-lag]
                dr2_um2 = (dx * px_size) ** 2 + (dy * px_size) ** 2
                acc = accum[lag]
                acc["msd_sum"] += float(np.sum(dr2_um2))
                acc["msd_n"] += int(len(dr2_um2))

                if len(dx1) > lag:
                    dot = (dx1[lag:] * dx1[:-lag] + dy1[lag:] * dy1[:-lag]) * px_size**2
                    acc["disp_auto_sum"] += float(np.sum(dot))
                    acc["disp_auto_n"] += int(len(dot))

        phi_group = group.dropna(subset=["phi"])
        for seg in contiguous_segments(phi_group):
            n = len(seg)
            if n < 2:
                continue
            phi = np.unwrap(seg["phi"].to_numpy(dtype=float))
            max_seg_lag = min(max_lag, n - 1)
            for lag in range(1, max_seg_lag + 1):
                dphi = phi[lag:] - phi[:-lag]
                acc = accum[lag]
                acc["amsd_sum"] += float(np.sum(dphi**2))
                acc["amsd_n"] += int(len(dphi))
                acc["orient_auto_sum"] += float(np.sum(np.cos(dphi)))
                acc["orient_auto_n"] += int(len(dphi))

    for lag, acc in accum.items():
        dt = lag / frame_rate
        rows.append(
            {
                "lag_frames": lag,
                "lag_seconds": dt,
                "translational_msd_um2": mean_or_nan(acc["msd_sum"], acc["msd_n"]),
                "translational_msd_n": acc["msd_n"],
                "angular_msd_rad2": mean_or_nan(acc["amsd_sum"], acc["amsd_n"]),
                "angular_msd_n": acc["amsd_n"],
                "displacement_autocorr_um2": mean_or_nan(
                    acc["disp_auto_sum"], acc["disp_auto_n"]
                ),
                "displacement_autocorr_n": acc["disp_auto_n"],
                "orientation_autocorr": mean_or_nan(acc["orient_auto_sum"], acc["orient_auto_n"]),
                "orientation_autocorr_n": acc["orient_auto_n"],
            }
        )
    return pd.DataFrame(rows)


def mean_or_nan(total: float, count: int) -> float:
    return total / count if count else np.nan


def compute_step_distributions(df: pd.DataFrame, px_size: float, frame_rate: float) -> pd.DataFrame:
    rows = []
    for tid, group in df.groupby("track_id", sort=False):
        for seg in contiguous_segments(group.dropna(subset=["x", "y"])):
            if len(seg) < 2:
                continue
            frames = seg["frame"].to_numpy(dtype=int)
            x = seg["x"].to_numpy(dtype=float)
            y = seg["y"].to_numpy(dtype=float)
            dx_um = np.diff(x) * px_size
            dy_um = np.diff(y) * px_size
            disp_um = np.hypot(dx_um, dy_um)
            speed = disp_um * frame_rate

            step_heading = np.arctan2(dy_um, dx_um)
            turning = np.full(len(dx_um), np.nan)
            if len(step_heading) >= 2:
                turning[1:] = np.angle(np.exp(1j * np.diff(step_heading)))

            orientation_delta = np.full(len(dx_um), np.nan)
            if seg["phi"].notna().all():
                phi = np.unwrap(seg["phi"].to_numpy(dtype=float))
                orientation_delta = np.angle(np.exp(1j * np.diff(phi)))

            for i in range(len(dx_um)):
                rows.append(
                    {
                        "track_id": int(tid),
                        "frame": int(frames[i]),
                        "next_frame": int(frames[i + 1]),
                        "dx_um": dx_um[i],
                        "dy_um": dy_um[i],
                        "displacement_um": disp_um[i],
                        "speed_um_s": speed[i],
                        "turning_angle_rad": turning[i],
                        "orientation_delta_phi_rad": orientation_delta[i],
                    }
                )
    return pd.DataFrame(rows)


def summarize_distribution(values: pd.Series, name: str) -> dict:
    finite = values[np.isfinite(values)]
    if finite.empty:
        return {
            "statistic": name,
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p05": np.nan,
            "p25": np.nan,
            "median": np.nan,
            "p75": np.nan,
            "p95": np.nan,
            "max": np.nan,
        }
    qs = finite.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "statistic": name,
        "n": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "min": float(finite.min()),
        "p05": float(qs.loc[0.05]),
        "p25": float(qs.loc[0.25]),
        "median": float(qs.loc[0.5]),
        "p75": float(qs.loc[0.75]),
        "p95": float(qs.loc[0.95]),
        "max": float(finite.max()),
    }


def write_metadata(path: Path, metadata: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")


def plot_lag_statistics(lag_df: pd.DataFrame, output_dir: Path, base: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    valid = lag_df["translational_msd_n"] > 0
    ax.loglog(
        lag_df.loc[valid, "lag_seconds"],
        lag_df.loc[valid, "translational_msd_um2"],
        "o-",
        ms=3,
    )
    ax.set_xlabel("lag time (s)")
    ax.set_ylabel("translational MSD (um^2)")
    ax.set_title("Translational MSD")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    valid = lag_df["angular_msd_n"] > 0
    ax.plot(
        lag_df.loc[valid, "lag_seconds"],
        lag_df.loc[valid, "angular_msd_rad2"],
        "o-",
        ms=3,
    )
    ax.set_xlabel("lag time (s)")
    ax.set_ylabel("angular MSD (rad^2)")
    ax.set_title("Angular MSD")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    valid = lag_df["displacement_autocorr_n"] > 0
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.plot(
        lag_df.loc[valid, "lag_seconds"],
        lag_df.loc[valid, "displacement_autocorr_um2"],
        "o-",
        ms=3,
    )
    ax.set_xlabel("lag time (s)")
    ax.set_ylabel("<dr(t+tau) . dr(t)> (um^2)")
    ax.set_title("Displacement Autocorrelation")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    valid = lag_df["orientation_autocorr_n"] > 0
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.plot(
        lag_df.loc[valid, "lag_seconds"],
        lag_df.loc[valid, "orientation_autocorr"],
        "o-",
        ms=3,
    )
    ax.set_xlabel("lag time (s)")
    ax.set_ylabel("<cos(phi(t+tau)-phi(t))>")
    ax.set_title("Orientation Autocorrelation")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / f"{base}_lag_statistics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_distributions(step_df: pd.DataFrame, output_dir: Path, base: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].hist(step_df["displacement_um"].dropna(), bins=80, color="steelblue", alpha=0.85)
    axes[0].set_xlabel("one-frame displacement (um)")
    axes[0].set_ylabel("count")
    axes[0].set_title("Displacement Magnitude")
    axes[0].grid(True, alpha=0.25)

    axes[1].hist(step_df["speed_um_s"].dropna(), bins=80, color="darkorange", alpha=0.85)
    axes[1].set_xlabel("speed (um/s)")
    axes[1].set_ylabel("count")
    axes[1].set_title("Speed")
    axes[1].grid(True, alpha=0.25)

    axes[2].hist(
        step_df["turning_angle_rad"].dropna(),
        bins=np.linspace(-np.pi, np.pi, 73),
        color="seagreen",
        alpha=0.85,
    )
    axes[2].set_xlabel("turning angle (rad/frame)")
    axes[2].set_ylabel("count")
    axes[2].set_title("Turning Angle")
    axes[2].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / f"{base}_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute raw motion statistics from particle tracks."
    )
    parser.add_argument("--tracks", required=True, help="Input tracks CSV")
    parser.add_argument(
        "--output",
        default="analysis_outputs/motion_statistics",
        help="Output directory (default: analysis_outputs/motion_statistics)",
    )
    parser.add_argument("--max-lag", type=positive_int, default=100, help="Maximum lag in frames")
    parser.add_argument(
        "--min-track-length",
        type=positive_int,
        default=2,
        help="Minimum rows per track after filtering (default: 2)",
    )
    parser.add_argument("--px-size", type=float, default=0.078, help="Pixel size in um/px")
    parser.add_argument("--frame-rate", type=float, default=30.0, help="Frame rate in Hz")
    parser.add_argument(
        "--include-interpolated",
        action="store_true",
        help="Include rows marked is_interpolated; default uses real detections only",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional suffix for output filenames, useful when comparing variants",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.px_size <= 0:
        raise ValueError("--px-size must be positive")
    if args.frame_rate <= 0:
        raise ValueError("--frame-rate must be positive")

    tracks_path = Path(args.tracks)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracks, metadata = load_tracks(str(tracks_path), args.include_interpolated)
    tracks = tracks_with_min_points(tracks, args.min_track_length)
    metadata.update(
        {
            "input_path": str(tracks_path),
            "max_lag": int(args.max_lag),
            "min_track_length": int(args.min_track_length),
            "px_size_um_per_px": float(args.px_size),
            "frame_rate_hz": float(args.frame_rate),
            "frame_interval_s": float(1.0 / args.frame_rate),
            "rows_used_after_min_track_length": int(len(tracks)),
            "tracks_used_after_min_track_length": int(tracks["track_id"].nunique()) if len(tracks) else 0,
        }
    )

    label = args.label or ("with_interpolated" if args.include_interpolated else "real_only")
    base = f"{tracks_path.stem}_{label}"

    lag_df = compute_lag_statistics(tracks, args.max_lag, args.px_size, args.frame_rate)
    step_df = compute_step_distributions(tracks, args.px_size, args.frame_rate)
    summary_df = pd.DataFrame(
        [
            summarize_distribution(step_df["displacement_um"], "displacement_um"),
            summarize_distribution(step_df["speed_um_s"], "speed_um_s"),
            summarize_distribution(step_df["turning_angle_rad"], "turning_angle_rad"),
            summarize_distribution(step_df["orientation_delta_phi_rad"], "orientation_delta_phi_rad"),
        ]
    )

    metadata["step_rows"] = int(len(step_df))
    metadata["lag_rows"] = int(len(lag_df))
    if not lag_df.empty:
        first = lag_df.iloc[0]
        metadata["lag1_translational_msd_um2"] = float(first["translational_msd_um2"])
        metadata["lag1_angular_msd_rad2"] = float(first["angular_msd_rad2"])
        metadata["lag1_orientation_autocorr"] = float(first["orientation_autocorr"])
    if not summary_df.empty:
        for _, row in summary_df.iterrows():
            metadata[f"{row['statistic']}_mean"] = float(row["mean"])
            metadata[f"{row['statistic']}_median"] = float(row["median"])

    lag_path = output_dir / f"{base}_lag_statistics.csv"
    step_path = output_dir / f"{base}_step_distributions.csv"
    summary_path = output_dir / f"{base}_distribution_summary.csv"
    metadata_path = output_dir / f"{base}_metadata.json"

    lag_df.to_csv(lag_path, index=False)
    step_df.to_csv(step_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    write_metadata(metadata_path, metadata)
    plot_lag_statistics(lag_df, output_dir, base)
    plot_distributions(step_df, output_dir, base)

    print(f"Loaded {metadata['input_rows']} rows across {metadata['input_tracks']} tracks")
    print(
        f"Used {metadata['rows_used_after_min_track_length']} rows across "
        f"{metadata['tracks_used_after_min_track_length']} tracks"
    )
    print(f"include_interpolated={args.include_interpolated}")
    print(f"px_size={args.px_size} um/px, frame_rate={args.frame_rate} Hz, max_lag={args.max_lag}")
    print(f"Saved lag statistics: {lag_path}")
    print(f"Saved step distributions: {step_path}")
    print(f"Saved distribution summary: {summary_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved plots: {output_dir / f'{base}_lag_statistics.png'}")
    print(f"Saved plots: {output_dir / f'{base}_distributions.png'}")


if __name__ == "__main__":
    main()
