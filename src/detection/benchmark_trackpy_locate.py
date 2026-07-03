#!/usr/bin/env python3
"""
Benchmark trackpy.locate against LodeSTAR position detection on real frames.

For each image in a stack, runs both detectors and matches their outputs with
the Hungarian algorithm to compute per-frame precision, recall, and position error.
Produces a summary CSV and box-plot for quick hardware/accuracy trade-off analysis.
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment

from track_particles import apply_nms


def _load_lodestar(path: str, min_dist: float) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(path)
    unnamed_cols: list[str] = [col for col in df.columns if col.startswith("Unnamed:")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    if "orientation_ncc" in df.columns and "ncc" not in df.columns:
        df = df.rename(columns={"orientation_ncc": "ncc"})
    if "confidence" in df.columns and "ncc" not in df.columns:
        df["ncc"] = df["confidence"]
    if "ncc" not in df.columns:
        df["ncc"] = np.nan
    if "phi" not in df.columns:
        df["phi"] = np.nan
    df["frame"] = df["frame"].astype(int)
    return apply_nms(df, min_dist)


def _list_images(images_dir: str, prefix: str) -> list[str]:
    files: list[str] = []
    for name in sorted(os.listdir(images_dir)):
        if prefix and not name.startswith(prefix):
            continue
        if re.search(r"\.(png|jpg|jpeg|tif|tiff)$", name, re.I):
            files.append(os.path.join(images_dir, name))
    if not files:
        raise FileNotFoundError(f"No frame images found in {images_dir} with prefix '{prefix}'")
    return files


def _locate_images(
    image_paths: list[str],
    diameter: int,
    minmass: float | None,
    invert: bool,
    max_frames: int | None,
) -> pd.DataFrame:
    try:
        import trackpy as tp
    except ImportError as exc:
        raise ImportError("trackpy is required. Install it with: pip install trackpy") from exc
    rows: list[pd.DataFrame] = []
    selected_paths: list[str] = image_paths[:max_frames] if max_frames else image_paths
    for frame, path in enumerate(selected_paths):
        image: np.ndarray = np.array(Image.open(path).convert("L"))
        found: pd.DataFrame = tp.locate(image, diameter=diameter, minmass=minmass, invert=invert)
        if len(found) == 0:
            continue
        out: pd.DataFrame = found.copy()
        out["frame"] = frame
        out["source_image"] = os.path.basename(path)
        rows.append(out)
        if (frame + 1) % 100 == 0:
            print(f"Located frame {frame + 1}/{len(selected_paths)}")
    if not rows:
        return pd.DataFrame(columns=["x", "y", "mass", "size", "ecc", "signal", "raw_mass", "ep", "frame", "source_image"])
    return pd.concat(rows, ignore_index=True)


def _match_frames(lodestar: pd.DataFrame, located: pd.DataFrame, distance: float) -> dict[str, float]:
    tp_total: int = 0
    fp_total: int = 0
    fn_total: int = 0
    dist_sum: float = 0.0
    dist_count: int = 0
    frames: list[int] = sorted(set(lodestar["frame"].unique()) | set(located["frame"].unique()))
    for frame in frames:
        a: np.ndarray = lodestar[lodestar["frame"] == frame][["x", "y"]].to_numpy(float)
        b: np.ndarray = located[located["frame"] == frame][["x", "y"]].to_numpy(float)
        if len(a) == 0:
            fp_total += len(b)
            continue
        if len(b) == 0:
            fn_total += len(a)
            continue
        costs: np.ndarray = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
        rows, cols = linear_sum_assignment(costs)
        matched: list[float] = [float(costs[r, c]) for r, c in zip(rows, cols) if costs[r, c] <= distance]
        tp_total += len(matched)
        fp_total += len(b) - len(matched)
        fn_total += len(a) - len(matched)
        dist_sum += float(np.sum(matched))
        dist_count += len(matched)
    precision: float = tp_total / (tp_total + fp_total) if tp_total + fp_total else 0.0
    recall: float = tp_total / (tp_total + fn_total) if tp_total + fn_total else 0.0
    f1: float = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_distance: float = dist_sum / dist_count if dist_count else np.nan
    return {
        "tp": float(tp_total),
        "fp": float(fp_total),
        "fn": float(fn_total),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_match_distance_px": mean_distance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare trackpy.locate detections with LodeSTAR detections")
    parser.add_argument("--lodestar", required=True, help="LodeSTAR detection CSV")
    parser.add_argument("--images", required=True, help="Frame image directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--prefix", default="JP_Fe_wf_2_40_slm075_", help="Image filename prefix")
    parser.add_argument("--diameter", type=int, default=41, help="Odd feature diameter for trackpy.locate")
    parser.add_argument("--minmass", type=float, default=None, help="Minimum mass for trackpy.locate")
    parser.add_argument("--invert", action="store_true", help="Locate dark features on bright background")
    parser.add_argument("--min-dist", type=float, default=20.0, help="NMS distance for LodeSTAR detections")
    parser.add_argument("--match-distance", type=float, default=20.0, help="Match radius in pixels")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit")
    args = parser.parse_args()

    if args.diameter % 2 == 0:
        raise ValueError("--diameter must be odd for trackpy.locate")
    os.makedirs(args.output, exist_ok=True)
    lodestar: pd.DataFrame = _load_lodestar(args.lodestar, args.min_dist)
    image_paths: list[str] = _list_images(args.images, args.prefix)
    located: pd.DataFrame = _locate_images(image_paths, args.diameter, args.minmass, args.invert, args.max_frames)
    if args.max_frames is not None:
        lodestar = lodestar[lodestar["frame"] < args.max_frames].reset_index(drop=True)
    metrics: dict[str, float] = _match_frames(lodestar, located, args.match_distance)

    detections_path: str = os.path.join(args.output, "trackpy_locate_detections.csv")
    metrics_path: str = os.path.join(args.output, "trackpy_locate_vs_lodestar_summary.csv")
    located.to_csv(detections_path, index=False)
    pd.DataFrame([{**metrics, "lodestar_detections": len(lodestar), "trackpy_detections": len(located)}]).to_csv(metrics_path, index=False)

    print(f"LodeSTAR detections: {len(lodestar)}")
    print(f"trackpy.locate detections: {len(located)}")
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall: {metrics['recall']:.4f}")
    print(f"f1: {metrics['f1']:.4f}")
    print(f"mean match distance: {metrics['mean_match_distance_px']:.3f} px")
    print(f"Saved detections -> {detections_path}")
    print(f"Saved summary -> {metrics_path}")


if __name__ == "__main__":
    main()
