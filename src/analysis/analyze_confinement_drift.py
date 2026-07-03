#!/usr/bin/env python3
"""
Confinement and drift-field analysis for tracked active particles.

The script tests for signatures that are not captured by a plain isolated ABP:
radial occupancy, central radial drift, speed changes with radius, and spatially
resolved drift fields.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib_cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"track_id", "frame", "x", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze radial confinement and spatial drift in particle tracks."
    )
    parser.add_argument("--tracks", required=True, help="Input track CSV.")
    parser.add_argument(
        "--output-dir",
        default="analysis_outputs/confinement",
        help="Directory for CSV and PNG outputs.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="Seconds per frame. Velocities are reported in position units per second.",
    )
    parser.add_argument(
        "--px-um",
        type=float,
        default=1.0,
        help="Position scale in microns per pixel. Default keeps native pixel units.",
    )
    parser.add_argument("--center-x", type=float, default=None, help="Override center x.")
    parser.add_argument("--center-y", type=float, default=None, help="Override center y.")
    parser.add_argument(
        "--center-method",
        choices=("quantile-midpoint", "median", "mean"),
        default="quantile-midpoint",
        help="Automatic center estimator when center-x/y are not both provided.",
    )
    parser.add_argument(
        "--center-quantile",
        type=float,
        default=0.01,
        help="Tail quantile for quantile-midpoint center estimate.",
    )
    parser.add_argument(
        "--include-interpolated",
        action="store_true",
        help="Include interpolated rows. Default uses only real detections.",
    )
    parser.add_argument(
        "--radial-bins",
        type=int,
        default=40,
        help="Number of radial bins for occupancy and radial profiles.",
    )
    parser.add_argument(
        "--spatial-bins",
        type=int,
        default=24,
        help="Number of bins per axis for spatial drift field.",
    )
    parser.add_argument(
        "--min-bin-count",
        type=int,
        default=20,
        help="Minimum samples for plotted/fit binned means.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit for smoke tests.",
    )
    return parser.parse_args()


def load_tracks(path: Path, include_interpolated: bool, max_rows: int | None) -> pd.DataFrame:
    tracks = pd.read_csv(path, nrows=max_rows)
    missing = REQUIRED_COLUMNS - set(tracks.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    tracks = tracks.copy()
    if "is_interpolated" not in tracks.columns:
        tracks["is_interpolated"] = False
    else:
        tracks["is_interpolated"] = tracks["is_interpolated"].astype(bool)

    if not include_interpolated:
        tracks = tracks[~tracks["is_interpolated"]].copy()

    tracks = tracks.sort_values(["track_id", "frame"]).reset_index(drop=True)
    return tracks


def estimate_center(
    tracks: pd.DataFrame,
    method: str,
    quantile: float,
    center_x: float | None,
    center_y: float | None,
) -> tuple[float, float]:
    if center_x is not None and center_y is not None:
        return float(center_x), float(center_y)

    q = min(max(quantile, 0.0), 0.49)
    if method == "median":
        auto_x = tracks["x"].median()
        auto_y = tracks["y"].median()
    elif method == "mean":
        auto_x = tracks["x"].mean()
        auto_y = tracks["y"].mean()
    else:
        x_lo, x_hi = tracks["x"].quantile([q, 1.0 - q])
        y_lo, y_hi = tracks["y"].quantile([q, 1.0 - q])
        auto_x = 0.5 * (x_lo + x_hi)
        auto_y = 0.5 * (y_lo + y_hi)

    return (
        float(auto_x if center_x is None else center_x),
        float(auto_y if center_y is None else center_y),
    )


def compute_steps(tracks: pd.DataFrame, dt: float, center_x: float, center_y: float) -> pd.DataFrame:
    current = tracks[["track_id", "frame", "x", "y", "is_interpolated"]].copy()
    nxt = current.groupby("track_id", sort=False)[["frame", "x", "y", "is_interpolated"]].shift(-1)

    steps = pd.DataFrame(
        {
            "track_id": current["track_id"],
            "frame": current["frame"],
            "frame_next": nxt["frame"],
            "x": current["x"],
            "y": current["y"],
            "x_next": nxt["x"],
            "y_next": nxt["y"],
            "is_interpolated": current["is_interpolated"],
            "is_interpolated_next": nxt["is_interpolated"],
        }
    )
    steps = steps.dropna(subset=["frame_next", "x_next", "y_next"]).copy()
    steps["frame_delta"] = steps["frame_next"] - steps["frame"]
    steps = steps[steps["frame_delta"] > 0].copy()

    steps["dt_step"] = steps["frame_delta"] * dt
    steps["dx"] = steps["x_next"] - steps["x"]
    steps["dy"] = steps["y_next"] - steps["y"]
    steps["vx"] = steps["dx"] / steps["dt_step"]
    steps["vy"] = steps["dy"] / steps["dt_step"]
    steps["speed"] = np.hypot(steps["vx"], steps["vy"])
    steps["x_mid"] = 0.5 * (steps["x"] + steps["x_next"])
    steps["y_mid"] = 0.5 * (steps["y"] + steps["y_next"])

    rx = steps["x_mid"] - center_x
    ry = steps["y_mid"] - center_y
    r = np.hypot(rx, ry)
    inv_r = np.divide(1.0, r, out=np.zeros_like(r, dtype=float), where=r > 0)
    er_x = rx * inv_r
    er_y = ry * inv_r
    et_x = -er_y
    et_y = er_x

    steps["r"] = r
    steps["theta"] = np.arctan2(ry, rx)
    steps["v_radial"] = steps["vx"] * er_x + steps["vy"] * er_y
    steps["v_tangential"] = steps["vx"] * et_x + steps["vy"] * et_y
    return steps.reset_index(drop=True)


def add_scaled_columns(tracks: pd.DataFrame, steps: pd.DataFrame, px_um: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    tracks = tracks.copy()
    steps = steps.copy()
    for col in ("x", "y"):
        tracks[f"{col}_um"] = tracks[col] * px_um
    for col in ("x", "y", "x_next", "y_next", "x_mid", "y_mid", "dx", "dy", "r"):
        steps[f"{col}_um"] = steps[col] * px_um
    for col in ("vx", "vy", "speed", "v_radial", "v_tangential"):
        steps[f"{col}_um_s"] = steps[col] * px_um
    return tracks, steps


def radial_profiles(steps: pd.DataFrame, radial_bins: int) -> pd.DataFrame:
    r_max = float(steps["r"].quantile(0.995))
    edges = np.linspace(0.0, r_max, radial_bins + 1)
    binned = steps.copy()
    binned["radius_bin"] = pd.cut(binned["r"], edges, include_lowest=True)
    profile = (
        binned.groupby("radius_bin", observed=True)
        .agg(
            r_mean=("r", "mean"),
            r_min=("r", "min"),
            r_max=("r", "max"),
            count=("r", "size"),
            mean_v_radial=("v_radial", "mean"),
            sem_v_radial=("v_radial", lambda s: s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else np.nan),
            mean_v_tangential=("v_tangential", "mean"),
            mean_speed=("speed", "mean"),
            median_speed=("speed", "median"),
        )
        .reset_index(drop=True)
    )
    return profile


def radial_occupancy(tracks: pd.DataFrame, center_x: float, center_y: float, radial_bins: int) -> pd.DataFrame:
    r = np.hypot(tracks["x"] - center_x, tracks["y"] - center_y)
    r_max = float(r.quantile(0.995))
    edges = np.linspace(0.0, r_max, radial_bins + 1)
    counts, edges = np.histogram(r, bins=edges)
    area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    density = counts / np.maximum(area, 1e-12)
    probability = counts / max(counts.sum(), 1)
    return pd.DataFrame(
        {
            "r_min": edges[:-1],
            "r_max": edges[1:],
            "r_center": 0.5 * (edges[:-1] + edges[1:]),
            "count": counts,
            "probability": probability,
            "area_normalized_density": density,
        }
    )


def spatial_drift_field(tracks: pd.DataFrame, steps: pd.DataFrame, spatial_bins: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    x_edges = np.linspace(tracks["x"].quantile(0.005), tracks["x"].quantile(0.995), spatial_bins + 1)
    y_edges = np.linspace(tracks["y"].quantile(0.005), tracks["y"].quantile(0.995), spatial_bins + 1)
    occupancy, _, _ = np.histogram2d(tracks["x"], tracks["y"], bins=[x_edges, y_edges])

    binned = steps.copy()
    binned["x_bin"] = pd.cut(binned["x_mid"], x_edges, labels=False, include_lowest=True)
    binned["y_bin"] = pd.cut(binned["y_mid"], y_edges, labels=False, include_lowest=True)
    binned = binned.dropna(subset=["x_bin", "y_bin"]).copy()
    binned["x_bin"] = binned["x_bin"].astype(int)
    binned["y_bin"] = binned["y_bin"].astype(int)

    field = (
        binned.groupby(["x_bin", "y_bin"], observed=True)
        .agg(
            count=("vx", "size"),
            x_mid_mean=("x_mid", "mean"),
            y_mid_mean=("y_mid", "mean"),
            mean_vx=("vx", "mean"),
            mean_vy=("vy", "mean"),
            mean_speed=("speed", "mean"),
            mean_v_radial=("v_radial", "mean"),
            mean_v_tangential=("v_tangential", "mean"),
        )
        .reset_index()
    )
    field["drift_magnitude"] = np.hypot(field["mean_vx"], field["mean_vy"])
    field["x_center"] = 0.5 * (x_edges[field["x_bin"].to_numpy()] + x_edges[field["x_bin"].to_numpy() + 1])
    field["y_center"] = 0.5 * (y_edges[field["y_bin"].to_numpy()] + y_edges[field["y_bin"].to_numpy() + 1])
    return field, x_edges, y_edges, occupancy


def summarize(
    tracks: pd.DataFrame,
    steps: pd.DataFrame,
    radial_profile: pd.DataFrame,
    field: pd.DataFrame,
    center_x: float,
    center_y: float,
    px_um: float,
    min_bin_count: int,
    include_interpolated: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_profile = radial_profile[radial_profile["count"] >= min_bin_count].copy()
    if len(valid_profile) >= 2:
        weights = np.sqrt(valid_profile["count"].to_numpy())
        slope, intercept = np.polyfit(
            valid_profile["r_mean"].to_numpy(),
            valid_profile["mean_v_radial"].to_numpy(),
            1,
            w=weights,
        )
        valid_profile["radial_drift_fit"] = intercept + slope * valid_profile["r_mean"]
        valid_profile["radial_drift_residual"] = valid_profile["mean_v_radial"] - valid_profile["radial_drift_fit"]
    else:
        slope = np.nan
        intercept = np.nan
        valid_profile["radial_drift_fit"] = np.nan
        valid_profile["radial_drift_residual"] = np.nan

    r = steps["r"].to_numpy()
    vr = steps["v_radial"].to_numpy()
    speed = steps["speed"].to_numpy()
    corr_r_vr = np.corrcoef(r, vr)[0, 1] if len(steps) > 1 and np.std(r) > 0 and np.std(vr) > 0 else np.nan
    corr_r_speed = (
        np.corrcoef(r, speed)[0, 1] if len(steps) > 1 and np.std(r) > 0 and np.std(speed) > 0 else np.nan
    )

    field_valid = field[field["count"] >= min_bin_count]
    summary = pd.DataFrame(
        [
            {
                "n_positions": len(tracks),
                "n_steps": len(steps),
                "n_tracks": tracks["track_id"].nunique(),
                "include_interpolated": include_interpolated,
                "center_x": center_x,
                "center_y": center_y,
                "center_x_um": center_x * px_um,
                "center_y_um": center_y * px_um,
                "arena_radius_p995": steps["r"].quantile(0.995),
                "arena_radius_p995_um": steps["r"].quantile(0.995) * px_um,
                "mean_speed": steps["speed"].mean(),
                "mean_speed_um_s": steps["speed"].mean() * px_um,
                "mean_v_radial": steps["v_radial"].mean(),
                "mean_v_radial_um_s": steps["v_radial"].mean() * px_um,
                "mean_v_tangential": steps["v_tangential"].mean(),
                "mean_v_tangential_um_s": steps["v_tangential"].mean() * px_um,
                "corr_radius_v_radial": corr_r_vr,
                "corr_radius_speed": corr_r_speed,
                "radial_drift_slope": slope,
                "radial_drift_slope_per_s": slope,
                "radial_drift_intercept": intercept,
                "drift_field_bins": len(field),
                "drift_field_bins_ge_min_count": len(field_valid),
                "mean_drift_field_magnitude": field_valid["drift_magnitude"].mean(),
                "max_drift_field_magnitude": field_valid["drift_magnitude"].max(),
            }
        ]
    )
    return summary, valid_profile


def save_plots(
    output_dir: Path,
    base: str,
    tracks: pd.DataFrame,
    radial_occ: pd.DataFrame,
    radial_profile: pd.DataFrame,
    field: pd.DataFrame,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    occupancy: np.ndarray,
    center_x: float,
    center_y: float,
    min_bin_count: int,
) -> None:
    valid_profile = radial_profile[radial_profile["count"] >= min_bin_count]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        radial_occ["r_center"],
        radial_occ["probability"],
        width=radial_occ["r_max"] - radial_occ["r_min"],
        align="center",
        color="steelblue",
        alpha=0.8,
    )
    ax.set_xlabel("radius (px)")
    ax.set_ylabel("position probability")
    ax.set_title("Radial occupancy")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"{base}_radial_occupancy.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(0.0, color="0.25", lw=1)
    ax.errorbar(
        valid_profile["r_mean"],
        valid_profile["mean_v_radial"],
        yerr=valid_profile["sem_v_radial"],
        fmt="o-",
        ms=4,
        lw=1.5,
        color="firebrick",
    )
    ax.set_xlabel("radius (px)")
    ax.set_ylabel("mean radial velocity (px/s)")
    ax.set_title("Mean radial drift vs radius")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"{base}_mean_radial_velocity_vs_radius.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(valid_profile["r_mean"], valid_profile["mean_speed"], "o-", ms=4, color="darkgreen")
    ax.set_xlabel("radius (px)")
    ax.set_ylabel("mean speed (px/s)")
    ax.set_title("Speed vs radius")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"{base}_speed_vs_radius.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    image = ax.imshow(
        occupancy.T,
        origin="lower",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        cmap="Greys",
        alpha=0.75,
        aspect="equal",
    )
    fig.colorbar(image, ax=ax, label="position count")
    field_plot = field[field["count"] >= min_bin_count]
    ax.quiver(
        field_plot["x_center"],
        field_plot["y_center"],
        field_plot["mean_vx"],
        field_plot["mean_vy"],
        field_plot["drift_magnitude"],
        cmap="viridis",
        angles="xy",
        scale_units="xy",
        scale=None,
        width=0.004,
    )
    ax.plot(center_x, center_y, "+", color="crimson", ms=12, mew=2, label="estimated center")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title("Spatial drift field over occupancy")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / f"{base}_quiver_drift_field.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    sample = tracks.sample(min(len(tracks), 60000), random_state=7) if len(tracks) > 60000 else tracks
    ax.hist2d(sample["x"], sample["y"], bins=140, cmap="magma")
    ax.plot(center_x, center_y, "+", color="cyan", ms=12, mew=2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title("Position density")
    fig.tight_layout()
    fig.savefig(output_dir / f"{base}_position_density.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    tracks_path = Path(args.tracks)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = tracks_path.stem

    tracks = load_tracks(tracks_path, args.include_interpolated, args.max_rows)
    if tracks.empty:
        raise ValueError("No rows available after filtering.")

    center_x, center_y = estimate_center(
        tracks,
        args.center_method,
        args.center_quantile,
        args.center_x,
        args.center_y,
    )
    steps = compute_steps(tracks, args.dt, center_x, center_y)
    if steps.empty:
        raise ValueError("No valid per-track steps available after filtering.")

    tracks_scaled, steps_scaled = add_scaled_columns(tracks, steps, args.px_um)
    radial_occ = radial_occupancy(tracks, center_x, center_y, args.radial_bins)
    radial_profile = radial_profiles(steps, args.radial_bins)
    field, x_edges, y_edges, occupancy = spatial_drift_field(tracks, steps, args.spatial_bins)
    summary, residuals = summarize(
        tracks,
        steps,
        radial_profile,
        field,
        center_x,
        center_y,
        args.px_um,
        args.min_bin_count,
        args.include_interpolated,
    )

    tracks_summary = (
        tracks.groupby("track_id", observed=True)
        .agg(
            n_positions=("frame", "size"),
            first_frame=("frame", "min"),
            last_frame=("frame", "max"),
            x_mean=("x", "mean"),
            y_mean=("y", "mean"),
            interpolated_positions=("is_interpolated", "sum"),
        )
        .reset_index()
    )

    steps_scaled.to_csv(output_dir / f"{base}_per_step_kinematics.csv", index=False)
    tracks_summary.to_csv(output_dir / f"{base}_track_summary.csv", index=False)
    radial_occ.to_csv(output_dir / f"{base}_radial_occupancy.csv", index=False)
    radial_profile.to_csv(output_dir / f"{base}_radial_profiles.csv", index=False)
    field.to_csv(output_dir / f"{base}_spatial_drift_field.csv", index=False)
    residuals.to_csv(output_dir / f"{base}_radial_drift_residuals.csv", index=False)
    summary.to_csv(output_dir / f"{base}_confinement_summary.csv", index=False)

    save_plots(
        output_dir,
        base,
        tracks_scaled,
        radial_occ,
        radial_profile,
        field,
        x_edges,
        y_edges,
        occupancy,
        center_x,
        center_y,
        args.min_bin_count,
    )

    row = summary.iloc[0]
    print(f"Loaded {row.n_positions:.0f} positions from {row.n_tracks:.0f} tracks")
    print(f"Computed {row.n_steps:.0f} steps; include_interpolated={bool(row.include_interpolated)}")
    print(f"Estimated center: x={row.center_x:.3f}, y={row.center_y:.3f}")
    print(f"Arena radius p99.5: {row.arena_radius_p995:.3f} px")
    print(f"Mean speed: {row.mean_speed:.6g} px/s")
    print(f"Mean radial velocity: {row.mean_v_radial:.6g} px/s")
    print(f"Radial drift slope: {row.radial_drift_slope:.6g} 1/s")
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
