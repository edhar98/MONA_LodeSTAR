#!/usr/bin/env python3
"""
Velocity-persistence analysis for particle trajectories.

This estimates AOUP / persistent-random-walk style parameters directly from
the velocity autocorrelation function, avoiding orientation-based assumptions.
"""

from __future__ import annotations

import argparse
import json
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
    from scipy.optimize import curve_fit
except ImportError:  # pragma: no cover - scipy is available in the project env.
    curve_fit = None


DEFAULT_TRACKS = Path(
    "detection_results/JP_FE/wf_2_40/JP_Fe_wf_2_40_5m4rtzfx/04/tracks/"
    "JP_Fe_wf_2_40_slm075_tracks.csv"
)
DEFAULT_OUTPUT_DIR = Path("analysis_outputs/model_comparison/velocity_persistence")
DEFAULT_NN_CSV = Path(
    "analysis_outputs/interactions/JP_Fe_wf_2_40_slm075_tracks_nearest_neighbor_states.csv"
)
REQUIRED_TRACK_COLUMNS = {"track_id", "frame", "x", "y"}


def positive_float(raw: str) -> float:
    value = float(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def parse_thresholds(raw: str) -> list[float]:
    if not raw.strip():
        return []
    values = sorted({float(part.strip()) for part in raw.split(",") if part.strip()})
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("thresholds must be positive")
    return values


def fmt_threshold(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def load_tracks(path: Path, include_interpolated: bool) -> tuple[pd.DataFrame, dict[str, int | bool]]:
    tracks = pd.read_csv(path)
    missing = REQUIRED_TRACK_COLUMNS.difference(tracks.columns)
    if missing:
        raise ValueError(f"missing required columns in {path}: {sorted(missing)}")

    if "is_interpolated" not in tracks.columns:
        tracks["is_interpolated"] = False

    tracks = tracks.copy()
    tracks["is_interpolated"] = tracks["is_interpolated"].fillna(False).astype(bool)
    input_rows = int(len(tracks))
    input_tracks = int(tracks["track_id"].nunique())
    input_interpolated_rows = int(tracks["is_interpolated"].sum())

    if not include_interpolated:
        tracks = tracks[~tracks["is_interpolated"]].copy()

    tracks["track_id"] = pd.to_numeric(tracks["track_id"], errors="coerce")
    tracks["frame"] = pd.to_numeric(tracks["frame"], errors="coerce")
    tracks["x"] = pd.to_numeric(tracks["x"], errors="coerce")
    tracks["y"] = pd.to_numeric(tracks["y"], errors="coerce")
    tracks = tracks.dropna(subset=["track_id", "frame", "x", "y"]).copy()
    tracks["track_id"] = tracks["track_id"].astype(int)
    tracks["frame"] = tracks["frame"].astype(int)
    tracks = tracks.sort_values(["track_id", "frame"]).reset_index(drop=True)

    metadata = {
        "input_rows": input_rows,
        "input_tracks": input_tracks,
        "input_interpolated_rows": input_interpolated_rows,
        "include_interpolated": bool(include_interpolated),
        "rows_used": int(len(tracks)),
        "tracks_used": int(tracks["track_id"].nunique()),
        "interpolated_rows_used": int(tracks["is_interpolated"].sum()),
    }
    return tracks, metadata


def compute_consecutive_velocities(tracks: pd.DataFrame, frame_rate: float, px_size: float) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    dt = 1.0 / frame_rate
    for track_id, group in tracks.groupby("track_id", sort=False):
        group = group.sort_values("frame")
        next_group = group.shift(-1)
        consecutive = (next_group["frame"] - group["frame"]) == 1
        if not consecutive.any():
            continue

        starts = group.loc[consecutive, ["frame", "x", "y"]].reset_index(drop=True)
        ends = next_group.loc[consecutive, ["frame", "x", "y"]].reset_index(drop=True)
        dx = (ends["x"].to_numpy(dtype=float) - starts["x"].to_numpy(dtype=float)) * px_size
        dy = (ends["y"].to_numpy(dtype=float) - starts["y"].to_numpy(dtype=float)) * px_size
        vx = dx / dt
        vy = dy / dt
        part = pd.DataFrame(
            {
                "track_id": int(track_id),
                "start_frame": starts["frame"].to_numpy(dtype=int),
                "end_frame": ends["frame"].to_numpy(dtype=int),
                "x_start_px": starts["x"].to_numpy(dtype=float),
                "y_start_px": starts["y"].to_numpy(dtype=float),
                "dx": dx,
                "dy": dy,
                "vx": vx,
                "vy": vy,
                "speed": np.sqrt(vx * vx + vy * vy),
            }
        )
        rows.append(part)

    if not rows:
        return pd.DataFrame(
            columns=[
                "track_id",
                "start_frame",
                "end_frame",
                "x_start_px",
                "y_start_px",
                "dx",
                "dy",
                "vx",
                "vy",
                "speed",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def load_nn_states(path: Path) -> pd.DataFrame:
    nn = pd.read_csv(path)
    required = {"track_id", "frame"}
    if "nn_dist_px" in nn.columns:
        dist_col = "nn_dist_px"
    elif "nn_distance" in nn.columns:
        dist_col = "nn_distance"
    else:
        raise ValueError(f"{path} must contain nn_dist_px or nn_distance")
    missing = required.difference(nn.columns)
    if missing:
        raise ValueError(f"missing required NN columns in {path}: {sorted(missing)}")

    nn = nn[["track_id", "frame", dist_col]].copy()
    nn = nn.rename(columns={dist_col: "nn_dist_px"})
    nn["track_id"] = pd.to_numeric(nn["track_id"], errors="coerce")
    nn["frame"] = pd.to_numeric(nn["frame"], errors="coerce")
    nn["nn_dist_px"] = pd.to_numeric(nn["nn_dist_px"], errors="coerce")
    nn = nn.dropna(subset=["track_id", "frame"]).copy()
    nn["track_id"] = nn["track_id"].astype(int)
    nn["frame"] = nn["frame"].astype(int)
    return nn.drop_duplicates(subset=["track_id", "frame"])


def attach_step_nn_distances(velocities: pd.DataFrame, nn_states: pd.DataFrame) -> pd.DataFrame:
    start_nn = nn_states.rename(
        columns={"frame": "start_frame", "nn_dist_px": "nn_dist_start_px"}
    )
    end_nn = nn_states.rename(columns={"frame": "end_frame", "nn_dist_px": "nn_dist_end_px"})
    merged = velocities.merge(start_nn, on=["track_id", "start_frame"], how="left")
    merged = merged.merge(end_nn, on=["track_id", "end_frame"], how="left")
    merged["step_nn_min_px"] = merged[["nn_dist_start_px", "nn_dist_end_px"]].min(axis=1)
    return merged


def filter_velocities(velocities: pd.DataFrame, threshold: float | None) -> pd.DataFrame:
    if threshold is None:
        return velocities.copy()
    if "step_nn_min_px" not in velocities.columns:
        raise ValueError("NN-filtered analysis requires attached step_nn_min_px")
    return velocities[velocities["step_nn_min_px"] > threshold].copy()


def compute_vacf(velocities: pd.DataFrame, max_lag: int, frame_rate: float) -> pd.DataFrame:
    accum = {
        lag: {"dot_sum": 0.0, "count": 0}
        for lag in range(max_lag + 1)
    }

    for _, group in velocities.groupby("track_id", sort=False):
        group = group.sort_values("start_frame")
        if group.empty:
            continue
        frames = group["start_frame"].to_numpy(dtype=int)
        v = group[["vx", "vy"]].to_numpy(dtype=float)
        by_frame = {int(frame): idx for idx, frame in enumerate(frames)}

        accum[0]["dot_sum"] += float(np.sum(v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1]))
        accum[0]["count"] += int(len(v))

        for lag in range(1, max_lag + 1):
            left_indices = []
            right_indices = []
            for idx, frame in enumerate(frames):
                right_idx = by_frame.get(int(frame + lag))
                if right_idx is not None:
                    left_indices.append(idx)
                    right_indices.append(right_idx)
            if not left_indices:
                continue
            left = v[np.asarray(left_indices)]
            right = v[np.asarray(right_indices)]
            accum[lag]["dot_sum"] += float(np.sum(left[:, 0] * right[:, 0] + left[:, 1] * right[:, 1]))
            accum[lag]["count"] += int(len(left_indices))

    c0 = np.nan
    rows = []
    if accum[0]["count"] > 0:
        c0 = accum[0]["dot_sum"] / accum[0]["count"]

    for lag, values in accum.items():
        count = int(values["count"])
        cv = values["dot_sum"] / count if count else np.nan
        rows.append(
            {
                "lag_frames": lag,
                "tau": lag / frame_rate,
                "vacf": cv,
                "vacf_normalized": cv / c0 if count and np.isfinite(c0) and c0 != 0 else np.nan,
                "n_pairs": count,
            }
        )
    return pd.DataFrame(rows)


def _exp_no_offset(tau: np.ndarray, tau_p: float) -> np.ndarray:
    return np.exp(-tau / tau_p)


def _exp_with_offset(tau: np.ndarray, tau_p: float, offset: float) -> np.ndarray:
    return (1.0 - offset) * np.exp(-tau / tau_p) + offset


def fit_normalized_vacf(vacf: pd.DataFrame, min_pairs: int, fit_offset: bool) -> dict[str, float | str]:
    fit_data = vacf[
        (vacf["lag_frames"] > 0)
        & np.isfinite(vacf["vacf_normalized"])
        & (vacf["n_pairs"] >= min_pairs)
    ].copy()
    fit_data = fit_data[fit_data["vacf_normalized"] > 0]
    if len(fit_data) < (3 if fit_offset else 2):
        return {"fit_status": "insufficient_positive_points", "tau_p": np.nan, "offset": np.nan}

    tau = fit_data["tau"].to_numpy(dtype=float)
    y = fit_data["vacf_normalized"].to_numpy(dtype=float)
    weights = np.sqrt(fit_data["n_pairs"].to_numpy(dtype=float))

    if curve_fit is not None:
        try:
            if fit_offset:
                params, covariance = curve_fit(
                    _exp_with_offset,
                    tau,
                    y,
                    p0=(max(float(np.median(tau)), 1e-9), 0.0),
                    sigma=1.0 / np.maximum(weights, 1.0),
                    bounds=([1e-9, -1.0], [np.inf, 1.0]),
                    maxfev=20000,
                )
                tau_p, offset = float(params[0]), float(params[1])
                tau_p_se = float(np.sqrt(covariance[0, 0])) if covariance.size else np.nan
            else:
                params, covariance = curve_fit(
                    _exp_no_offset,
                    tau,
                    y,
                    p0=(max(float(np.median(tau)), 1e-9),),
                    sigma=1.0 / np.maximum(weights, 1.0),
                    bounds=([1e-9], [np.inf]),
                    maxfev=20000,
                )
                tau_p, offset = float(params[0]), 0.0
                tau_p_se = float(np.sqrt(covariance[0, 0])) if covariance.size else np.nan
            return {
                "fit_status": "ok",
                "tau_p": tau_p,
                "tau_p_se": tau_p_se,
                "offset": offset,
                "fit_points": int(len(fit_data)),
                "fit_model": "normalized_exp_offset" if fit_offset else "normalized_exp",
            }
        except Exception as exc:  # pragma: no cover - depends on data quality.
            status = f"curve_fit_failed:{type(exc).__name__}"
    else:
        status = "scipy_unavailable_log_linear"

    log_y = np.log(y)
    slope, intercept = np.polyfit(tau, log_y, 1, w=weights)
    if slope >= 0:
        return {"fit_status": status + ":nonnegative_log_slope", "tau_p": np.nan, "offset": 0.0}
    return {
        "fit_status": status,
        "tau_p": float(-1.0 / slope),
        "tau_p_se": np.nan,
        "offset": 0.0,
        "fit_points": int(len(fit_data)),
        "fit_model": "log_linear_normalized_exp",
        "log_intercept": float(intercept),
    }


def diffusion_proxy(vacf: pd.DataFrame, dimension: int) -> float:
    valid = vacf[(vacf["n_pairs"] > 0) & np.isfinite(vacf["vacf"])].copy()
    if len(valid) < 2:
        return np.nan
    tau = valid["tau"].to_numpy(dtype=float)
    cv = valid["vacf"].to_numpy(dtype=float)
    positive_prefix = np.ones(len(cv), dtype=bool)
    if np.any(cv[1:] <= 0):
        first_nonpositive = int(np.where(cv[1:] <= 0)[0][0] + 1)
        positive_prefix[first_nonpositive + 1 :] = False
    tau = tau[positive_prefix]
    cv = cv[positive_prefix]
    if len(cv) < 2:
        return np.nan
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(cv, tau) / dimension)


def summarize_speed(velocities: pd.DataFrame) -> dict[str, float | int]:
    if velocities.empty:
        return {
            "n_velocity_steps": 0,
            "n_tracks_with_steps": 0,
            "speed_mean": np.nan,
            "speed_median": np.nan,
            "speed_std": np.nan,
            "speed_p10": np.nan,
            "speed_p90": np.nan,
            "c0_mean_speed_squared": np.nan,
        }
    speed = velocities["speed"].to_numpy(dtype=float)
    v = velocities[["vx", "vy"]].to_numpy(dtype=float)
    return {
        "n_velocity_steps": int(len(velocities)),
        "n_tracks_with_steps": int(velocities["track_id"].nunique()),
        "speed_mean": float(np.mean(speed)),
        "speed_median": float(np.median(speed)),
        "speed_std": float(np.std(speed, ddof=1)) if len(speed) > 1 else 0.0,
        "speed_p10": float(np.percentile(speed, 10)),
        "speed_p90": float(np.percentile(speed, 90)),
        "c0_mean_speed_squared": float(np.mean(v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1])),
    }


def fit_curve_for_plot(tau: np.ndarray, fit: dict[str, float | str], fit_offset: bool) -> np.ndarray:
    tau_p = float(fit.get("tau_p", np.nan))
    if not np.isfinite(tau_p) or tau_p <= 0:
        return np.full_like(tau, np.nan, dtype=float)
    if fit_offset:
        offset = float(fit.get("offset", 0.0))
        return _exp_with_offset(tau, tau_p, offset)
    return _exp_no_offset(tau, tau_p)


def plot_vacf(vacf: pd.DataFrame, fit: dict[str, float | str], out_path: Path, title: str, fit_offset: bool) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    valid = vacf[vacf["n_pairs"] > 0]

    axes[0].plot(valid["tau"], valid["vacf"], marker="o", ms=3, lw=1)
    axes[0].axhline(0, color="0.4", lw=0.8)
    axes[0].set_xlabel("Lag time")
    axes[0].set_ylabel("C_v(tau)")
    axes[0].set_title("Velocity autocorrelation")

    axes[1].plot(valid["tau"], valid["vacf_normalized"], marker="o", ms=3, lw=1, label="data")
    fit_tau = valid.loc[valid["lag_frames"] > 0, "tau"].to_numpy(dtype=float)
    if len(fit_tau):
        axes[1].plot(fit_tau, fit_curve_for_plot(fit_tau, fit, fit_offset), lw=1.5, label="fit")
    axes[1].axhline(0, color="0.4", lw=0.8)
    axes[1].set_xlabel("Lag time")
    axes[1].set_ylabel("C_v(tau) / C_v(0)")
    axes[1].set_title("Normalized VACF")
    axes[1].legend(frameon=False)

    fig.suptitle(title)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_summary(summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    labels = summary["analysis"].astype(str).to_numpy()
    x = np.arange(len(summary))

    axes[0].bar(x, summary["tau_p"].to_numpy(dtype=float))
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
    axes[0].set_ylabel("Persistence time tau_p")
    axes[0].set_title("Fitted persistence")

    axes[1].bar(x, summary["diffusion_proxy_vacf"].to_numpy(dtype=float))
    axes[1].set_xticks(x, labels, rotation=25, ha="right")
    axes[1].set_ylabel("Integrated VACF / dimension")
    axes[1].set_title("Long-time diffusion proxy")

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def analyze_one(
    velocities: pd.DataFrame,
    analysis_name: str,
    threshold: float | None,
    max_lag: int,
    frame_rate: float,
    dimension: int,
    min_pairs: int,
    fit_offset: bool,
    output_dir: Path,
) -> dict[str, float | int | str | None]:
    filtered = filter_velocities(velocities, threshold)
    vacf = compute_vacf(filtered, max_lag=max_lag, frame_rate=frame_rate)
    fit = fit_normalized_vacf(vacf, min_pairs=min_pairs, fit_offset=fit_offset)
    speed_stats = summarize_speed(filtered)

    row: dict[str, float | int | str | None] = {
        "analysis": analysis_name,
        "nn_threshold_px": threshold,
        "max_lag_frames": max_lag,
        "frame_rate": frame_rate,
        "fit_min_pairs": min_pairs,
        "diffusion_proxy_vacf": diffusion_proxy(vacf, dimension=dimension),
    }
    row.update(speed_stats)
    row.update(fit)

    vacf_path = output_dir / f"{analysis_name}_vacf.csv"
    steps_path = output_dir / f"{analysis_name}_velocity_steps.csv"
    plot_path = output_dir / f"{analysis_name}_vacf.png"
    filtered.to_csv(steps_path, index=False)
    vacf.to_csv(vacf_path, index=False)
    plot_vacf(vacf, fit, plot_path, title=analysis_name, fit_offset=fit_offset)

    row["velocity_steps_csv"] = str(steps_path)
    row["vacf_csv"] = str(vacf_path)
    row["vacf_png"] = str(plot_path)
    return row


def resolve_nn_csv(path_arg: str | None, tracks_path: Path) -> Path | None:
    if path_arg:
        path = Path(path_arg)
        return path if path.exists() else None
    if DEFAULT_NN_CSV.exists():
        return DEFAULT_NN_CSV
    candidate = Path("analysis_outputs/interactions") / f"{tracks_path.stem}_nearest_neighbor_states.csv"
    if candidate.exists():
        return candidate
    return None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, default=DEFAULT_TRACKS, help="Track CSV path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--include-interpolated", action="store_true")
    parser.add_argument("--frame-rate", type=positive_float, default=1.0, help="Frames per time unit.")
    parser.add_argument("--px-size", type=positive_float, default=1.0, help="Distance units per pixel.")
    parser.add_argument("--max-lag", type=positive_int, default=50)
    parser.add_argument("--fit-min-pairs", type=positive_int, default=25)
    parser.add_argument("--fit-offset", action="store_true", help="Fit normalized exp + offset.")
    parser.add_argument("--dimension", type=positive_int, default=2)
    parser.add_argument("--nn-csv", default=None, help="Nearest-neighbor state CSV. Auto-detected if omitted.")
    parser.add_argument(
        "--nn-thresholds",
        type=parse_thresholds,
        default=parse_thresholds("30,50,75"),
        help="Comma-separated NN thresholds in pixels.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tracks, metadata = load_tracks(args.tracks, include_interpolated=args.include_interpolated)
    velocities = compute_consecutive_velocities(tracks, frame_rate=args.frame_rate, px_size=args.px_size)
    nn_csv = resolve_nn_csv(args.nn_csv, args.tracks)
    analyses: list[tuple[str, float | None]] = [("unfiltered", None)]

    nn_metadata: dict[str, str | int | bool] = {"nn_csv": "", "nn_filtering_available": False}
    if nn_csv is not None:
        nn_states = load_nn_states(nn_csv)
        velocities = attach_step_nn_distances(velocities, nn_states)
        analyses.extend((f"nn_gt_{fmt_threshold(threshold)}px", threshold) for threshold in args.nn_thresholds)
        nn_metadata = {
            "nn_csv": str(nn_csv),
            "nn_filtering_available": True,
            "nn_state_rows": int(len(nn_states)),
        }
    elif args.nn_csv:
        print(f"WARNING: NN CSV not found: {args.nn_csv}; writing unfiltered analysis only.")
    else:
        print("WARNING: no NN CSV auto-detected; writing unfiltered analysis only.")

    summaries = [
        analyze_one(
            velocities=velocities,
            analysis_name=name,
            threshold=threshold,
            max_lag=args.max_lag,
            frame_rate=args.frame_rate,
            dimension=args.dimension,
            min_pairs=args.fit_min_pairs,
            fit_offset=args.fit_offset,
            output_dir=args.output_dir,
        )
        for name, threshold in analyses
    ]
    summary = pd.DataFrame(summaries)
    summary_path = args.output_dir / "velocity_persistence_summary.csv"
    summary.to_csv(summary_path, index=False)
    plot_summary(summary, args.output_dir / "velocity_persistence_summary.png")

    metadata_path = args.output_dir / "velocity_persistence_metadata.json"
    metadata_out = {
        **metadata,
        **nn_metadata,
        "tracks_csv": str(args.tracks),
        "output_dir": str(args.output_dir),
        "px_size": args.px_size,
        "frame_rate": args.frame_rate,
        "max_lag": args.max_lag,
        "fit_offset": bool(args.fit_offset),
        "fit_min_pairs": args.fit_min_pairs,
    }
    metadata_path.write_text(json.dumps(metadata_out, indent=2) + "\n")

    print(f"Wrote {summary_path}")
    print(summary[["analysis", "n_velocity_steps", "tau_p", "offset", "diffusion_proxy_vacf", "fit_status"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
