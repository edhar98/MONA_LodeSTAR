#!/usr/bin/env python3
"""
Visualize particle tracks:
  1. tracks_overview.png  — all trajectories overlaid on first frame
  2. tracks_video.mp4     — per-frame animation with accumulated trails

Usage:
    python src/visualize_tracks.py \
        --tracks  detection_results/.../tracks/..._tracks.csv \
        --images  data/JP_FE/wf_2_40/04/images/ \
        --output  detection_results/.../tracks/
"""

import argparse
import os
import re

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def load_frame(images_dir: str, frame_idx: int) -> np.ndarray:
    """Load a frame by index (1-based filenames)."""
    candidates = sorted(os.listdir(images_dir))
    # Filter to PNG/JPG only
    candidates = [f for f in candidates
                  if re.search(r'\.(png|jpg|jpeg|tif|tiff)$', f, re.I)]
    # Frame index is 0-based in tracks CSV; filenames are 1-based
    filename_idx = frame_idx + 1
    # Try exact match on trailing number
    for fname in candidates:
        m = re.search(r'_(\d+)\.\w+$', fname)
        if m and int(m.group(1)) == filename_idx:
            path = os.path.join(images_dir, fname)
            img = np.array(Image.open(path).convert("L"))
            return img
    # Fallback: positional
    if 0 <= frame_idx < len(candidates):
        path = os.path.join(images_dir, candidates[frame_idx])
        return np.array(Image.open(path).convert("L"))
    return None


def track_colormap(n_tracks: int):
    """Assign a distinct BGR colour to each track_id."""
    cmap = cm.get_cmap("hsv", n_tracks)
    colors = {}
    ids = sorted(range(n_tracks))
    for i, tid in enumerate(ids):
        r, g, b, _ = cmap(i / max(n_tracks - 1, 1))
        colors[tid] = (int(b * 255), int(g * 255), int(r * 255))  # BGR
    return colors


# ---------------------------------------------------------------------------
# 1. Static overview PNG
# ---------------------------------------------------------------------------

def make_overview(df: pd.DataFrame, images_dir: str, output_path: str):
    # Use middle frame as background for representative content
    mid_frame = int(df["frame"].median())
    bg = load_frame(images_dir, mid_frame)
    if bg is None:
        bg = np.zeros((1024, 1024), dtype=np.uint8)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(bg, cmap="gray", vmin=0, vmax=255)

    track_ids = df["track_id"].unique()
    cmap = cm.get_cmap("hsv", len(track_ids))
    id_to_color = {tid: cmap(i / max(len(track_ids) - 1, 1))
                   for i, tid in enumerate(sorted(track_ids))}

    for tid, group in df.groupby("track_id"):
        group = group.sort_values("frame")
        real = group[~group["is_interpolated"]]
        interp = group[group["is_interpolated"]]

        color = id_to_color[tid]

        # Solid line for real detections
        ax.plot(group["x"], group["y"], color=color, linewidth=0.6, alpha=0.7)
        # Dots for real detections
        if len(real):
            ax.scatter(real["x"], real["y"], color=color, s=3, zorder=3, alpha=0.8)
        # Dashed segments for interpolated frames
        if len(interp):
            ax.scatter(interp["x"], interp["y"], color=color, s=3,
                       marker="x", zorder=3, alpha=0.5)

    ax.set_title(
        f"Particle tracks — {len(track_ids)} tracks, {len(df)} rows "
        f"({int(df['is_interpolated'].sum())} interpolated)",
        fontsize=11,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved overview → {output_path}")


# ---------------------------------------------------------------------------
# 2. Per-frame video MP4
# ---------------------------------------------------------------------------

def make_video(
    df: pd.DataFrame,
    images_dir: str,
    output_path: str,
    fps: int = 10,
    trail_frames: int = 20,
):
    frames = sorted(df["frame"].unique())
    track_ids = sorted(df["track_id"].unique())
    colors = track_colormap(len(track_ids))
    id_to_color = {tid: colors[i] for i, tid in enumerate(track_ids)}

    # Determine frame size from first image
    first_bg = load_frame(images_dir, frames[0])
    if first_bg is not None:
        h, w = first_bg.shape[:2]
    else:
        h, w = 1024, 1024

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame_idx in frames:
        bg = load_frame(images_dir, frame_idx)
        if bg is None:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            canvas = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)

        # Draw trail: last `trail_frames` frames up to current
        trail_start = max(frames[0], frame_idx - trail_frames)
        trail_df = df[(df["frame"] >= trail_start) & (df["frame"] <= frame_idx)]

        for tid, group in trail_df.groupby("track_id"):
            group = group.sort_values("frame")
            color = id_to_color.get(tid, (255, 255, 255))
            pts = group[["x", "y"]].values.astype(np.int32)

            # Draw polyline trail
            if len(pts) >= 2:
                cv2.polylines(canvas, [pts.reshape(-1, 1, 2)],
                              isClosed=False, color=color, thickness=1,
                              lineType=cv2.LINE_AA)

            # Current-frame dot
            cur = group[group["frame"] == frame_idx]
            if len(cur):
                cx, cy = int(cur.iloc[0].x), int(cur.iloc[0].y)
                is_interp = bool(cur.iloc[0].is_interpolated)
                marker = cv2.MARKER_CROSS if is_interp else cv2.MARKER_TILTED_CROSS
                cv2.drawMarker(canvas, (cx, cy), color,
                               markerType=marker, markerSize=6, thickness=1)

        # Frame counter
        cv2.putText(canvas, f"frame {frame_idx:03d}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1,
                    cv2.LINE_AA)

        writer.write(canvas)

    writer.release()
    print(f"Saved video   → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize particle tracks")
    parser.add_argument("--tracks",  required=True, help="Tracks CSV path")
    parser.add_argument("--images",  required=True, help="Directory of per-frame images")
    parser.add_argument("--output",  required=True, help="Output directory")
    parser.add_argument("--fps",     type=int, default=30, help="Video FPS (default: 30)")
    parser.add_argument("--trail",   type=int, default=20,
                        help="Trail length in frames for video (default: 20)")
    args = parser.parse_args()

    df = pd.read_csv(args.tracks)
    os.makedirs(args.output, exist_ok=True)

    # Derive a base name from the tracks file
    base = os.path.splitext(os.path.basename(args.tracks))[0]  # e.g. ..._tracks

    make_overview(
        df,
        args.images,
        os.path.join(args.output, f"{base}_overview.png"),
    )

    make_video(
        df,
        args.images,
        os.path.join(args.output, f"{base}_video.mp4"),
        fps=args.fps,
        trail_frames=args.trail,
    )


if __name__ == "__main__":
    main()
