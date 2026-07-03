#!/usr/bin/env python3
"""
Physics analysis of particle tracks (Active Brownian Particle model).

Computes:
  - Translational MSD  → D_t (diffusion) and v0 (self-propulsion)
  - Angular MSD        → D_r (rotational diffusion)
  - Per-track summary statistics

Usage:
    python src/analyze_tracks.py \
        --tracks detection_results/.../tracks/..._tracks.csv \
        --dt 1/30 \
        --output detection_results/.../tracks/
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# MSD computation
# ---------------------------------------------------------------------------

def compute_msd(
    tracks: pd.DataFrame,
    max_lag: int,
    min_track: int = 50,
    include_interpolated: bool = False,
) -> pd.DataFrame:
    """
    Ensemble-averaged translational MSD over lag times 1..max_lag.
    Uses real detections by default; optionally includes interpolated rows.
    Returns DataFrame with columns: lag, msd, n_samples.
    """
    long_tracks = tracks.groupby("track_id").filter(
        lambda g: (len(g) if include_interpolated else (~g["is_interpolated"]).sum()) >= min_track
    )

    msd_accum = np.zeros(max_lag)
    counts = np.zeros(max_lag, dtype=int)

    for tid, group in long_tracks.groupby("track_id"):
        if include_interpolated:
            g = group.sort_values("frame")
        else:
            g = group[~group["is_interpolated"]].sort_values("frame")
        x = g["x"].values
        y = g["y"].values
        n = len(x)
        for lag in range(1, min(max_lag + 1, n)):
            dx = x[lag:] - x[:-lag]
            dy = y[lag:] - y[:-lag]
            msd_accum[lag - 1] += (dx**2 + dy**2).sum()
            counts[lag - 1] += len(dx)

    valid = counts > 0
    msd = np.where(valid, msd_accum / np.maximum(counts, 1), np.nan)
    lags = np.arange(1, max_lag + 1)
    return pd.DataFrame({"lag": lags, "msd": msd, "n_samples": counts})


def compute_angular_msd(
    tracks: pd.DataFrame,
    max_lag: int,
    min_track: int = 50,
    include_interpolated: bool = False,
) -> pd.DataFrame:
    """
    Ensemble-averaged angular MSD <(Δφ)²> vs lag time.
    Uses circular difference to handle angle wrapping.
    """
    def has_enough_rows(group):
        if include_interpolated:
            return group["phi"].notna().sum() >= min_track
        return (~group["is_interpolated"] & group["phi"].notna()).sum() >= min_track

    long_tracks = tracks.groupby("track_id").filter(has_enough_rows)

    amsd_accum = np.zeros(max_lag)
    counts = np.zeros(max_lag, dtype=int)

    for tid, group in long_tracks.groupby("track_id"):
        if include_interpolated:
            g = group[group["phi"].notna()].sort_values("frame")
        else:
            g = group[~group["is_interpolated"] & group["phi"].notna()].sort_values("frame")
        phi = np.unwrap(g["phi"].values)
        n = len(phi)
        for lag in range(1, min(max_lag + 1, n)):
            dphi = phi[lag:] - phi[:-lag]
            amsd_accum[lag - 1] += (dphi**2).sum()
            counts[lag - 1] += len(dphi)

    valid = counts > 0
    amsd = np.where(valid, amsd_accum / np.maximum(counts, 1), np.nan)
    lags = np.arange(1, max_lag + 1)
    return pd.DataFrame({"lag": lags, "amsd": amsd, "n_samples": counts})


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def abp_msd_model(t, D_t, v0, D_r):
    """
    ABP translational MSD (2D):
      MSD(t) = 4*D_t*t + 2*v0²/D_r * [t - (1 - exp(-D_r*t))/D_r]
    """
    return 4 * D_t * t + 2 * v0**2 / D_r * (t - (1 - np.exp(-D_r * t)) / D_r)


def fit_msd(msd_df: pd.DataFrame, dt: float):
    """Fit ABP MSD model; returns (D_t, v0, D_r) in physical units."""
    t = msd_df["lag"].values * dt
    msd = msd_df["msd"].values
    valid = np.isfinite(msd) & (msd > 0)
    t, msd = t[valid], msd[valid]

    # Initial guess: simple diffusion fit on short lags
    short = t < t.max() * 0.1
    if short.sum() >= 2:
        D_t0 = np.polyfit(t[short], msd[short], 1)[0] / 4
    else:
        D_t0 = msd[0] / (4 * t[0])

    try:
        popt, _ = curve_fit(
            abp_msd_model, t, msd,
            p0=[D_t0, 1.0, 0.1],
            bounds=([0, 0, 1e-6], [np.inf, np.inf, np.inf]),
            maxfev=10000,
        )
        return popt  # D_t, v0, D_r
    except Exception:
        return None


def fit_angular_msd(amsd_df: pd.DataFrame, dt: float):
    """Linear fit <Δφ²> = 2*D_r*t → returns D_r."""
    t = amsd_df["lag"].values * dt
    amsd = amsd_df["amsd"].values
    valid = np.isfinite(amsd) & (amsd > 0)
    # Use short lags for linear regime
    cutoff = min(int(len(t) * 0.3), 30)
    t_fit, amsd_fit = t[valid][:cutoff], amsd[valid][:cutoff]
    if len(t_fit) < 2:
        return None
    slope, _ = np.polyfit(t_fit, amsd_fit, 1)
    return slope / 2  # D_r


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_msd(msd_df, amsd_df, dt, fit_params, D_r_angular, output_dir, base, px_um=1.0):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    spatial_unit = "µm" if px_um != 1.0 else "px"

    # Translational MSD
    ax = axes[0]
    t = msd_df["lag"].values * dt
    # Use pre-converted column if available, otherwise scale on the fly
    msd_phys = msd_df["msd_um2"].values if "msd_um2" in msd_df.columns else msd_df["msd"].values * px_um**2
    ax.loglog(t, msd_phys, "o", ms=3, alpha=0.7, label="data")
    if fit_params is not None:
        D_t, v0, D_r = fit_params
        t_fit = np.logspace(np.log10(t[0]), np.log10(t[-1]), 200)
        ax.loglog(t_fit, abp_msd_model(t_fit, D_t, v0, D_r), "-",
                  label=f"ABP fit\nD_t={D_t:.4f} {spatial_unit}²/s\n"
                        f"v₀={v0:.3f} {spatial_unit}/s\nD_r={D_r:.4f} rad²/s")
        ax.loglog(t_fit, 4 * D_t * t_fit, "--", alpha=0.5, label="4D_t·t")
    ax.set_xlabel("lag time (s)")
    ax.set_ylabel(f"MSD ({spatial_unit}²)")
    ax.set_title("Translational MSD")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Angular MSD
    ax = axes[1]
    t_a = amsd_df["lag"].values * dt
    ax.plot(t_a, amsd_df["amsd"].values, "o", ms=3, alpha=0.7, label="data")
    if D_r_angular is not None:
        t_fit = np.linspace(0, t_a[-1], 200)
        ax.plot(t_fit, 2 * D_r_angular * t_fit, "-",
                label=f"linear fit\nD_r={D_r_angular:.4f} rad²/s")
    ax.set_xlabel("lag time (s)")
    ax.set_ylabel("Angular MSD (rad²)")
    ax.set_title("Rotational MSD")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f"{base}_msd.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved MSD plot → {path}")


def plot_sample_tracks(tracks, output_dir, base, n=12, px_um=1.0):
    """Plot n longest tracks as x(t), y(t), phi(t) time series."""
    lengths = tracks.groupby("track_id").size()
    top_ids = lengths.nlargest(n).index

    fig, axes = plt.subplots(n, 3, figsize=(14, n * 1.8), sharex=False)
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, tid in enumerate(top_ids):
        g = tracks[tracks.track_id == tid].sort_values("frame")
        t = g["frame"].values
        real = ~g["is_interpolated"].values

        for col, (col_name, ylabel) in enumerate([("x", "x (µm)"), ("y", "y (µm)"), ("phi", "φ (rad)")]):
            ax = axes[row, col]
            scale = px_um if col_name in ("x", "y") else 1.0
            ax.plot(t[real], g[col_name].values[real] * scale, ".", ms=2, color="steelblue")
            ax.plot(t[~real], g[col_name].values[~real] * scale, "x", ms=3, color="salmon", alpha=0.7)
            if col == 0:
                ax.set_ylabel(f"tr {tid}\n{ylabel}", fontsize=7)
            else:
                ax.set_ylabel(ylabel, fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.2)

        axes[row, 0].set_xlabel("frame", fontsize=6)

    axes[0, 0].set_title("x(t)", fontsize=9)
    axes[0, 1].set_title("y(t)", fontsize=9)
    axes[0, 2].set_title("φ(t)  [blue=real, red×=interp]", fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, f"{base}_sample_tracks.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved sample tracks → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Physics analysis of particle tracks")
    parser.add_argument("--tracks",    required=True, help="Tracks CSV path")
    parser.add_argument("--dt",        type=float, default=1/30,
                        help="Frame interval in seconds (default: 1/30)")
    parser.add_argument("--max-lag",   type=int, default=100,
                        help="Max lag in frames for MSD (default: 100)")
    parser.add_argument("--min-track", type=int, default=50,
                        help="Min real frames per track for MSD (default: 50)")
    parser.add_argument("--px-size",   type=float, default=0.078,
                        help="Pixel size in µm/px (default: 0.078)")
    parser.add_argument("--output",    required=True, help="Output directory")
    parser.add_argument("--include-interpolated", action="store_true",
                        help="Include interpolated/refined rows in MSD/ABP analysis")
    args = parser.parse_args()

    px = args.px_size  # µm/px
    tracks = pd.read_csv(args.tracks)
    os.makedirs(args.output, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.tracks))[0]

    n_tracks = tracks["track_id"].nunique()
    n_real = (~tracks["is_interpolated"]).sum()
    n_interp = tracks["is_interpolated"].sum()
    mode = "all rows including interpolated" if args.include_interpolated else "real detections only"
    print(f"Loaded {n_tracks} tracks, {n_real} real detections, {n_interp} interpolated rows, "
          f"dt={args.dt:.4f}s, px={px} µm/px, mode={mode}")

    # MSD (computed in pixels, converted to µm² after fitting)
    print("Computing translational MSD...")
    msd_df = compute_msd(
        tracks,
        args.max_lag,
        args.min_track,
        include_interpolated=args.include_interpolated,
    )

    print("Computing angular MSD...")
    amsd_df = compute_angular_msd(
        tracks,
        args.max_lag,
        args.min_track,
        include_interpolated=args.include_interpolated,
    )

    # Fit in pixel space, then convert
    fit_params_px = fit_msd(msd_df, args.dt)
    D_r_angular = fit_angular_msd(amsd_df, args.dt)

    # Convert MSD table to µm²
    msd_df["msd_um2"] = msd_df["msd"] * px**2

    print()
    print("=== Physics parameters ===")
    if fit_params_px is not None:
        D_t_px, v0_px, D_r_msd = fit_params_px
        D_t = D_t_px * px**2   # µm²/s
        v0  = v0_px  * px      # µm/s
        fit_params = (D_t, v0, D_r_msd)
        print(f"  D_t  = {D_t:.4f}  µm²/s   (translational diffusion)")
        print(f"  v₀   = {v0:.4f}  µm/s    (self-propulsion speed)")
        print(f"  D_r  = {D_r_msd:.5f} rad²/s  (rotational diffusion, from MSD fit)")
    else:
        fit_params = None
        print("  ABP MSD fit failed — check data quality")
    if D_r_angular is not None:
        print(f"  D_r  = {D_r_angular:.5f} rad²/s  (rotational diffusion, from angular MSD)")

    # Save MSD tables
    msd_path  = os.path.join(args.output, f"{base}_msd.csv")
    amsd_path = os.path.join(args.output, f"{base}_amsd.csv")
    msd_df.to_csv(msd_path, index=False)
    amsd_df.to_csv(amsd_path, index=False)
    print(f"\nSaved MSD data  → {msd_path}")
    print(f"Saved AMSD data → {amsd_path}")

    # Plots
    plot_msd(msd_df, amsd_df, args.dt, fit_params, D_r_angular,
             args.output, base, px_um=px)
    plot_sample_tracks(tracks, args.output, base, n=12, px_um=px)


if __name__ == "__main__":
    main()
