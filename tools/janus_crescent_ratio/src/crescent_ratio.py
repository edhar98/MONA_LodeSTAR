#!/usr/bin/env python3
"""Measure Janus crescent area ratio from frame 0 of TDMS/image files.

The primary output is:

    crescent_area_ratio = segmented_crescent_area_px / rim_excluded_disk_area_px
    theta_deg = degrees(arccos(2 * crescent_area_ratio - 1))
    out_of_plane_angle_deg = degrees(arcsin(1 - 2 * crescent_area_ratio))

This module is intentionally standalone from the LodeSTAR training/tracking
pipeline. It can be imported from notebooks or run as a CLI.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage import filters, measure, morphology


DEFAULT_DATA_ROOT = Path("/mnt/75/Data/Akshay/5CB paper/Measurements/Janus/30.07.26/Janus")
DEFAULT_OUTPUT_DIR = Path("tools/janus_crescent_ratio/outputs")


@dataclass(frozen=True)
class ParticleDetection:
    center_x: float
    center_y: float
    radius_px: float
    method: str
    score: float


@dataclass(frozen=True)
class CropRegion:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def extract(self, image: np.ndarray) -> np.ndarray:
        return image[self.y0 : self.y1, self.x0 : self.x1]


@dataclass(frozen=True)
class CrescentMeasurement:
    path: str
    folder: str
    file: str
    frame: int
    crop_x0: int
    crop_y0: int
    crop_x1: int
    crop_y1: int
    center_x: float
    center_y: float
    radius_px: float
    disk_area_px: int
    interior_area_px: int
    excluded_annulus_area_px: int
    rim_exclusion_px: float
    crescent_area_px: int
    crescent_area_ratio: float
    theta_deg: float
    theta_mapping: str
    out_of_plane_angle_deg: float
    out_of_plane_angle_mapping: str
    threshold_value: float
    threshold_percentile: float | None
    background_value: float
    mean_disk_intensity: float
    mean_crescent_intensity: float
    polarity: str
    detection_method: str
    qc_status: str
    qc_reason: str


def discover_tdms_files(root: Path, recursive: bool = True) -> list[Path]:
    """Find TDMS files under root, excluding index sidecar files."""
    pattern = "**/*.tdms" if recursive else "*.tdms"
    return sorted(path for path in root.glob(pattern) if path.is_file() and not path.name.endswith(".tdms_index"))


def crescent_ratio_to_out_of_plane_angle_deg(crescent_area_ratio: float) -> float:
    ratio = float(crescent_area_ratio)
    if not np.isfinite(ratio):
        return float("nan")
    argument = np.clip(1.0 - 2.0 * ratio, 0.0, 1.0)
    return float(np.degrees(np.arcsin(argument)))


def crescent_ratio_to_theta_deg(crescent_area_ratio: float) -> float:
    ratio = float(crescent_area_ratio)
    if not np.isfinite(ratio):
        return float("nan")
    argument = np.clip(2.0 * ratio - 1.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(argument)))


def load_frame0(path: Path | str) -> np.ndarray:
    """Load frame 0 from a TDMS or common image file."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".tdms":
        from tdms_explorer import TDMSFileExplorer

        images = TDMSFileExplorer(str(path.resolve())).extract_images()
        if images is None or len(images) == 0:
            raise ValueError(f"No image frames found in TDMS: {path}")
        return np.asarray(images[0])

    with Image.open(path) as image:
        return np.asarray(image)


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Convert an arbitrary frame array to a 2D float64 image."""
    arr = np.asarray(frame)
    if arr.ndim == 2:
        return arr.astype(np.float64)
    if arr.ndim == 3:
        rgb = arr[..., :3].astype(np.float64)
        return rgb @ np.array([0.299, 0.587, 0.114])
    raise ValueError(f"Unsupported frame shape: {arr.shape}")


def normalize_uint8(image: np.ndarray, lower_percentile: float = 0.5, upper_percentile: float = 99.5) -> np.ndarray:
    """Robustly scale an image to uint8 for visualization and OpenCV."""
    img = np.asarray(image, dtype=np.float64)
    lo, hi = np.percentile(img, [lower_percentile, upper_percentile])
    if hi <= lo:
        lo, hi = float(np.min(img)), float(np.max(img))
    if hi <= lo:
        return np.zeros(img.shape, dtype=np.uint8)
    scaled = (img - lo) / (hi - lo)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def select_analysis_crop(
    image: np.ndarray,
    crop_size: int = 180,
    center_x: float | None = None,
    center_y: float | None = None,
) -> tuple[np.ndarray, CropRegion]:
    """Extract a square ROI while retaining its full-frame coordinates."""
    if image.ndim != 2:
        raise ValueError("select_analysis_crop expects a 2D grayscale image")
    if crop_size <= 0:
        raise ValueError("crop_size must be positive")

    height, width = image.shape
    size = min(int(crop_size), height, width)
    cx = width / 2.0 if center_x is None else float(center_x)
    cy = height / 2.0 if center_y is None else float(center_y)
    x0 = int(round(cx - size / 2.0))
    y0 = int(round(cy - size / 2.0))
    x0 = min(max(x0, 0), width - size)
    y0 = min(max(y0, 0), height - size)
    region = CropRegion(x0=x0, y0=y0, x1=x0 + size, y1=y0 + size)
    return region.extract(image), region


def circular_mask(shape: tuple[int, int], center_x: float, center_y: float, radius: float) -> np.ndarray:
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius ** 2


def annulus_mask(
    shape: tuple[int, int],
    center_x: float,
    center_y: float,
    inner_radius: float,
    outer_radius: float,
) -> np.ndarray:
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    dist2 = (xx - center_x) ** 2 + (yy - center_y) ** 2
    return (dist2 >= inner_radius ** 2) & (dist2 <= outer_radius ** 2)


def _hough_candidates(
    image_u8: np.ndarray,
    center_window: int,
    min_radius: int,
    max_radius: int,
    hough_param2: float,
) -> list[ParticleDetection]:
    h, w = image_u8.shape
    cx0, cy0 = w / 2.0, h / 2.0
    half = int(center_window // 2)
    x0 = max(0, int(round(cx0 - half)))
    x1 = min(w, int(round(cx0 + half)))
    y0 = max(0, int(round(cy0 - half)))
    y1 = min(h, int(round(cy0 + half)))
    crop = image_u8[y0:y1, x0:x1]
    if crop.size == 0:
        return []

    crop = cv2.GaussianBlur(crop, (5, 5), 0)
    local_background = cv2.GaussianBlur(crop, (0, 0), sigmaX=12.0)
    saliency = cv2.absdiff(crop, local_background).astype(np.float64) / 255.0
    circles = cv2.HoughCircles(
        crop,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(8, min_radius),
        param1=60,
        param2=hough_param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return []

    candidates: list[ParticleDetection] = []
    for row in np.squeeze(circles, axis=0):
        local_x, local_y, r = float(row[0]), float(row[1]), float(row[2])
        x, y = local_x + x0, local_y + y0
        center_dist = float(np.hypot(x - cx0, y - cy0))
        edge_strength = _edge_ring_score(image_u8, x, y, r)
        candidate_disk = circular_mask(crop.shape, local_x, local_y, max(1.0, r - 2.0))
        signal_strength = float(np.mean(saliency[candidate_disk])) if np.any(candidate_disk) else 0.0
        score = edge_strength + 1.5 * signal_strength - 0.002 * center_dist
        candidates.append(ParticleDetection(x, y, r, "hough", score))
    return candidates


def _edge_ring_score(image_u8: np.ndarray, center_x: float, center_y: float, radius: float) -> float:
    edges = cv2.Canny(image_u8, 40, 120).astype(bool)
    ring = annulus_mask(edges.shape, center_x, center_y, max(1, radius - 2), radius + 2)
    if not np.any(ring):
        return 0.0
    return float(edges[ring].mean())


def _radial_fallback(
    image_u8: np.ndarray,
    min_radius: int,
    max_radius: int,
) -> ParticleDetection:
    """Fallback for centered particles when Hough fails."""
    h, w = image_u8.shape
    cx, cy = w / 2.0, h / 2.0
    img = image_u8.astype(np.float64)
    scores = []
    for radius in range(min_radius, max_radius + 1):
        inner = annulus_mask(img.shape, cx, cy, max(1, radius - 2), radius)
        outer = annulus_mask(img.shape, cx, cy, radius + 1, radius + 4)
        if not np.any(inner) or not np.any(outer):
            continue
        scores.append((abs(float(img[inner].mean() - img[outer].mean())), radius))
    if not scores:
        return ParticleDetection(cx, cy, float((min_radius + max_radius) / 2), "center_fallback", 0.0)
    score, radius = max(scores)
    return ParticleDetection(cx, cy, float(radius), "radial_center_fallback", float(score))


def detect_particle(
    image: np.ndarray,
    center_window: int = 180,
    min_radius: int = 18,
    max_radius: int = 35,
    hough_param2: float = 22.0,
    seed: ParticleDetection | None = None,
) -> ParticleDetection:
    """Detect the inner Janus particle, not a larger enclosing ring."""
    if seed is not None:
        return seed
    image_u8 = normalize_uint8(image)
    candidates = _hough_candidates(image_u8, center_window, min_radius, max_radius, hough_param2)
    if candidates:
        return max(candidates, key=lambda item: item.score)
    return _radial_fallback(image_u8, min_radius, max_radius)


def segment_crescent(
    image: np.ndarray,
    detection: ParticleDetection,
    polarity: str = "bright",
    rim_exclusion_px: float = 5.0,
    threshold_percentile: float | None = None,
    min_crescent_area: int = 20,
    max_crescent_fraction: float = 0.65,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, str, str]:
    """Segment the crescent inside the disk while excluding its bright rim."""
    if polarity not in {"bright", "dark"}:
        raise ValueError("polarity must be 'bright' or 'dark'")
    if rim_exclusion_px < 0:
        raise ValueError("rim_exclusion_px must be non-negative")
    if threshold_percentile is not None and not 0 <= threshold_percentile <= 100:
        raise ValueError("threshold_percentile must be between 0 and 100")

    disk = circular_mask(image.shape, detection.center_x, detection.center_y, detection.radius_px)
    interior_radius = detection.radius_px - rim_exclusion_px
    if interior_radius <= 0:
        empty = np.zeros(image.shape, dtype=bool)
        return disk, empty, empty, empty, float("nan"), float("nan"), "fail", "rim exclusion removes entire disk"
    interior = circular_mask(image.shape, detection.center_x, detection.center_y, interior_radius)
    bg = annulus_mask(image.shape, detection.center_x, detection.center_y, detection.radius_px + 5, detection.radius_px + 15)
    if not np.any(bg):
        bg = ~disk

    background_value = float(np.median(image[bg])) if np.any(bg) else float(np.median(image))
    corrected = image.astype(np.float64) - background_value
    signal = corrected if polarity == "bright" else -corrected
    disk_values = signal[interior]
    if disk_values.size == 0:
        empty = np.zeros(image.shape, dtype=bool)
        return disk, interior, bg, empty, float("nan"), background_value, "fail", "empty interior mask"

    if threshold_percentile is None:
        threshold_value = _adaptive_threshold(disk_values)
    else:
        threshold_value = float(np.percentile(disk_values, threshold_percentile))
    crescent = np.zeros(image.shape, dtype=bool)
    crescent[interior] = signal[interior] > threshold_value
    crescent = morphology.remove_small_objects(crescent, min_size=max(1, int(min_crescent_area)))
    crescent = morphology.binary_closing(crescent, morphology.disk(2))
    crescent &= interior

    disk_area = int(disk.sum())
    bright_area = int(crescent.sum())
    frac = bright_area / max(int(interior.sum()), 1)
    qc_status = "ok"
    qc_reason = ""
    if bright_area < min_crescent_area:
        qc_status = "warn"
        qc_reason = "crescent area below minimum"
    elif frac > max_crescent_fraction:
        qc_status = "warn"
        qc_reason = "crescent area fraction unusually high"
    return disk, interior, bg, crescent, float(threshold_value), background_value, qc_status, qc_reason


def _adaptive_threshold(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    finite = finite.astype(np.float64)
    try:
        threshold = float(filters.threshold_otsu(finite))
    except ValueError:
        threshold = float(np.percentile(finite, 90))
    frac = float(np.mean(finite > threshold))
    if not np.isfinite(threshold) or frac < 0.005 or frac > 0.75:
        threshold = float(np.percentile(finite, 90))
    return threshold


def measure_frame(
    frame: np.ndarray,
    path: Path,
    root: Path | None = None,
    polarity: str = "bright",
    seed: ParticleDetection | None = None,
    crop_size: int = 180,
    crop_center_x: float | None = None,
    crop_center_y: float | None = None,
    min_radius: int = 18,
    max_radius: int = 35,
    rim_exclusion_px: float = 5.0,
    hough_param2: float = 22.0,
    threshold_percentile: float | None = None,
    selected_crop: CropRegion | None = None,
) -> tuple[CrescentMeasurement, dict[str, np.ndarray | ParticleDetection | CropRegion]]:
    gray = to_grayscale(frame)
    requested_center_x = crop_center_x if crop_center_x is not None else (seed.center_x if seed is not None else None)
    requested_center_y = crop_center_y if crop_center_y is not None else (seed.center_y if seed is not None else None)
    if selected_crop is None:
        crop, crop_region = select_analysis_crop(gray, crop_size, requested_center_x, requested_center_y)
    else:
        height, width = gray.shape
        if not (0 <= selected_crop.x0 < selected_crop.x1 <= width and 0 <= selected_crop.y0 < selected_crop.y1 <= height):
            raise ValueError("selected_crop must lie inside the frame")
        crop_region = selected_crop
        crop = crop_region.extract(gray)
    if seed is None:
        local_detection = detect_particle(
            crop,
            center_window=crop_size,
            min_radius=min_radius,
            max_radius=max_radius,
            hough_param2=hough_param2,
        )
        detection = ParticleDetection(
            center_x=local_detection.center_x + crop_region.x0,
            center_y=local_detection.center_y + crop_region.y0,
            radius_px=local_detection.radius_px,
            method=f"cropped_{local_detection.method}",
            score=local_detection.score,
        )
    else:
        detection = seed
    disk, interior, bg, crescent, threshold, background, qc_status, qc_reason = segment_crescent(
        gray,
        detection,
        polarity=polarity,
        rim_exclusion_px=rim_exclusion_px,
        threshold_percentile=threshold_percentile,
    )
    disk_values = gray[disk]
    crescent_values = gray[crescent]
    disk_area = int(disk.sum())
    interior_area = int(interior.sum())
    crescent_area = int(crescent.sum())
    crescent_area_ratio = float(crescent_area / max(interior_area, 1))

    folder = path.parent.name
    if root is not None:
        try:
            rel_parent = path.parent.relative_to(root)
            folder = str(rel_parent) if str(rel_parent) != "." else "root"
        except ValueError:
            pass

    measurement = CrescentMeasurement(
        path=str(path),
        folder=folder,
        file=path.name,
        frame=0,
        crop_x0=crop_region.x0,
        crop_y0=crop_region.y0,
        crop_x1=crop_region.x1,
        crop_y1=crop_region.y1,
        center_x=float(detection.center_x),
        center_y=float(detection.center_y),
        radius_px=float(detection.radius_px),
        disk_area_px=disk_area,
        interior_area_px=interior_area,
        excluded_annulus_area_px=disk_area - interior_area,
        rim_exclusion_px=float(rim_exclusion_px),
        crescent_area_px=crescent_area,
        crescent_area_ratio=crescent_area_ratio,
        theta_deg=crescent_ratio_to_theta_deg(crescent_area_ratio),
        theta_mapping="polar cosine: theta_deg = degrees(arccos(clip(2 * crescent_area_ratio - 1, -1, 1)))",
        out_of_plane_angle_deg=crescent_ratio_to_out_of_plane_angle_deg(crescent_area_ratio),
        out_of_plane_angle_mapping="projected hemisphere: out_of_plane_angle_deg = degrees(arcsin(clip(1 - 2 * crescent_area_ratio, 0, 1)))",
        threshold_value=float(threshold),
        threshold_percentile=None if threshold_percentile is None else float(threshold_percentile),
        background_value=float(background),
        mean_disk_intensity=float(np.mean(disk_values)) if disk_values.size else float("nan"),
        mean_crescent_intensity=float(np.mean(crescent_values)) if crescent_values.size else float("nan"),
        polarity=polarity,
        detection_method=detection.method,
        qc_status=qc_status,
        qc_reason=qc_reason,
    )
    debug = {
        "gray": gray,
        "crop": crop,
        "crop_region": crop_region,
        "detection": detection,
        "disk": disk,
        "interior": interior,
        "excluded_annulus": disk & ~interior,
        "background": bg,
        "crescent": crescent,
    }
    return measurement, debug


def load_seed_csv(path: Path | str | None) -> dict[str, ParticleDetection]:
    if path is None:
        return {}
    path = Path(path)
    seeds: dict[str, ParticleDetection] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = row.get("path") or row.get("file")
            if not key:
                continue
            seeds[str(key)] = ParticleDetection(
                center_x=float(row["center_x"]),
                center_y=float(row["center_y"]),
                radius_px=float(row["radius_px"]),
                method="manual_seed",
                score=1.0,
            )
    return seeds


def seed_for_path(seeds: dict[str, ParticleDetection], path: Path) -> ParticleDetection | None:
    return seeds.get(str(path)) or seeds.get(path.name)


def save_overlay(
    output_path: Path | BytesIO,
    gray: np.ndarray,
    disk: np.ndarray,
    interior: np.ndarray,
    background: np.ndarray,
    crescent: np.ndarray,
    detection: ParticleDetection,
    crop_region: CropRegion,
    title: str | None = None,
) -> None:
    if isinstance(output_path, Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 7, figsize=(24, 4), constrained_layout=True)
    image_u8 = normalize_uint8(gray)
    crop_u8 = crop_region.extract(image_u8)
    excluded_annulus = disk & ~interior
    overlays = [
        ("cropped frame", None),
        ("particle disk", crop_region.extract(disk)),
        ("measurement interior", crop_region.extract(interior)),
        ("excluded bright rim", crop_region.extract(excluded_annulus)),
        ("background annulus", crop_region.extract(background)),
        ("crescent mask", crop_region.extract(crescent)),
    ]

    axes[0].imshow(image_u8, cmap="gray")
    rectangle = plt.Rectangle(
        (crop_region.x0, crop_region.y0),
        crop_region.width,
        crop_region.height,
        fill=False,
        color="magenta",
        lw=1.5,
    )
    axes[0].add_patch(rectangle)
    axes[0].set_title("full frame + crop")
    axes[0].set_axis_off()

    local_center_x = detection.center_x - crop_region.x0
    local_center_y = detection.center_y - crop_region.y0
    for ax, (label, mask) in zip(axes[1:], overlays):
        ax.imshow(crop_u8, cmap="gray")
        if mask is not None:
            rgba = np.zeros((*mask.shape, 4), dtype=float)
            color = {
                "particle disk": (0.1, 0.8, 1.0, 0.35),
                "measurement interior": (0.2, 0.9, 0.4, 0.35),
                "excluded bright rim": (1.0, 0.35, 0.1, 0.55),
                "background annulus": (1.0, 0.8, 0.1, 0.35),
                "crescent mask": (1.0, 0.1, 0.1, 0.55),
            }[label]
            rgba[mask] = color
            ax.imshow(rgba)
        circle = plt.Circle((local_center_x, local_center_y), detection.radius_px, fill=False, color="lime", lw=1.5)
        ax.add_patch(circle)
        ax.set_title(label)
        ax.set_axis_off()
    if title:
        fig.suptitle(title)
    fig.savefig(output_path, dpi=150, format="png")
    plt.close(fig)


def analyze_files(
    files: Sequence[Path],
    root: Path | None,
    output_dir: Path,
    polarity: str,
    seed_csv: Path | None = None,
    overlay_limit: int = 20,
    crop_size: int = 180,
    crop_center_x: float | None = None,
    crop_center_y: float | None = None,
    min_radius: int = 18,
    max_radius: int = 35,
    rim_exclusion_px: float = 5.0,
    hough_param2: float = 22.0,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = load_seed_csv(seed_csv)
    rows: list[dict] = []
    overlay_dir = output_dir / "overlays"
    for index, path in enumerate(files):
        try:
            frame = load_frame0(path)
            measurement, debug = measure_frame(
                frame,
                path,
                root=root,
                polarity=polarity,
                seed=seed_for_path(seeds, path),
                crop_size=crop_size,
                crop_center_x=crop_center_x,
                crop_center_y=crop_center_y,
                min_radius=min_radius,
                max_radius=max_radius,
                rim_exclusion_px=rim_exclusion_px,
                hough_param2=hough_param2,
            )
        except Exception as exc:
            folder = path.parent.name
            rows.append(
                {
                    "path": str(path),
                    "folder": folder,
                    "file": path.name,
                    "frame": 0,
                    "qc_status": "fail",
                    "qc_reason": str(exc),
                }
            )
            continue

        rows.append(asdict(measurement))
        if index < overlay_limit:
            stem = path.with_suffix("").name
            safe_folder = measurement.folder.replace("/", "_")
            save_overlay(
                overlay_dir / f"{safe_folder}_{stem}_frame0_overlay.png",
                debug["gray"],  # type: ignore[arg-type]
                debug["disk"],  # type: ignore[arg-type]
                debug["interior"],  # type: ignore[arg-type]
                debug["background"],  # type: ignore[arg-type]
                debug["crescent"],  # type: ignore[arg-type]
                debug["detection"],  # type: ignore[arg-type]
                debug["crop_region"],  # type: ignore[arg-type]
                title=f"{measurement.folder}/{measurement.file} ratio={measurement.crescent_area_ratio:.3f}",
            )
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "janus_crescent_ratio_frame0_measurements.csv", index=False)
    write_summaries(df, output_dir)
    return df


def write_summaries(df: pd.DataFrame, output_dir: Path) -> None:
    ok = df[df["qc_status"].isin(["ok", "warn"])].copy() if "qc_status" in df else df.copy()
    if ok.empty or "crescent_area_ratio" not in ok:
        return
    file_summary = ok.groupby(["folder", "file"], dropna=False).agg(
        n=("crescent_area_ratio", "count"),
        ratio_mean=("crescent_area_ratio", "mean"),
        ratio_median=("crescent_area_ratio", "median"),
        ratio_std=("crescent_area_ratio", "std"),
        qc_status=("qc_status", lambda x: ",".join(sorted(set(map(str, x))))),
    )
    file_summary.reset_index().to_csv(output_dir / "janus_crescent_ratio_file_summary.csv", index=False)

    folder_summary = ok.groupby(["folder"], dropna=False).agg(
        n_files=("file", "count"),
        ratio_mean=("crescent_area_ratio", "mean"),
        ratio_median=("crescent_area_ratio", "median"),
        ratio_std=("crescent_area_ratio", "std"),
        ratio_min=("crescent_area_ratio", "min"),
        ratio_max=("crescent_area_ratio", "max"),
    )
    folder_summary.reset_index().to_csv(output_dir / "janus_crescent_ratio_folder_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.hist(ok["crescent_area_ratio"].dropna(), bins=30, color="#4c78a8", edgecolor="white")
    ax.set_xlabel("crescent area ratio")
    ax.set_ylabel("count")
    ax.set_title("Frame-0 Janus crescent area ratios")
    fig.savefig(output_dir / "janus_crescent_ratio_histogram.png", dpi=150)
    plt.close(fig)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_DATA_ROOT, help="Root directory containing TDMS files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True, help="Recursively discover TDMS files.")
    parser.add_argument("--polarity", choices=["bright", "dark"], default="bright", help="Whether the crescent is brighter or darker than background.")
    parser.add_argument("--seed-csv", type=Path, default=None, help="Optional CSV with file/path, center_x, center_y, radius_px.")
    parser.add_argument("--overlay-limit", type=int, default=20, help="Maximum number of QC overlay PNGs to write.")
    parser.add_argument(
        "--crop-size",
        "--center-window",
        dest="crop_size",
        type=int,
        default=180,
        help="Square full-frame crop size in pixels (legacy alias: --center-window).",
    )
    parser.add_argument("--crop-center-x", type=float, default=None, help="Optional full-frame crop center x; default is frame center.")
    parser.add_argument("--crop-center-y", type=float, default=None, help="Optional full-frame crop center y; default is frame center.")
    parser.add_argument("--min-radius", type=int, default=18, help="Minimum inner-particle radius in pixels.")
    parser.add_argument("--max-radius", type=int, default=35, help="Maximum inner-particle radius; keep below the enclosing-ring radius.")
    parser.add_argument(
        "--rim-exclusion-px",
        type=float,
        default=5.0,
        help="Width removed from inside the particle boundary before crescent segmentation.",
    )
    parser.add_argument("--hough-param2", type=float, default=22.0, help="OpenCV Hough circle accumulator threshold.")
    parser.add_argument("--limit", type=int, default=None, help="Optional file limit for smoke tests.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    files = discover_tdms_files(args.input_root, recursive=args.recursive)
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"No TDMS files found under {args.input_root}")
    config = {
        "input_root": str(args.input_root),
        "n_files": len(files),
        "polarity": args.polarity,
        "frame": 0,
        "crop_size": args.crop_size,
        "crop_center_x": args.crop_center_x,
        "crop_center_y": args.crop_center_y,
        "min_radius": args.min_radius,
        "max_radius": args.max_radius,
        "rim_exclusion_px": args.rim_exclusion_px,
        "hough_param2": args.hough_param2,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2))
    df = analyze_files(
        files,
        root=args.input_root,
        output_dir=args.output_dir,
        polarity=args.polarity,
        seed_csv=args.seed_csv,
        overlay_limit=args.overlay_limit,
        crop_size=args.crop_size,
        crop_center_x=args.crop_center_x,
        crop_center_y=args.crop_center_y,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        rim_exclusion_px=args.rim_exclusion_px,
        hough_param2=args.hough_param2,
    )
    n_ok = int(df["qc_status"].isin(["ok", "warn"]).sum()) if "qc_status" in df else 0
    print(f"Processed {len(df)} files ({n_ok} measured). Outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
