#!/usr/bin/env python3
"""
Visualize particle tracks:
  1. tracks_overview.png  — all trajectories overlaid on a background frame
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
from concurrent.futures import ThreadPoolExecutor

import cv2
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from PIL import Image


def list_frame_images(images_dir: str) -> list[str]:
    candidates = sorted(os.listdir(images_dir))
    return [
        os.path.join(images_dir, f)
        for f in candidates
        if re.search(r'\.(png|jpg|jpeg|tif|tiff)$', f, re.I)
    ]


def load_frame(images_dir: str, frame_idx: int) -> np.ndarray:
    candidates = list_frame_images(images_dir)
    return load_frame_from_paths(candidates, frame_idx)


def load_frame_from_paths(candidates: list[str], frame_idx: int) -> np.ndarray:
    if 0 <= frame_idx < len(candidates):
        return np.array(Image.open(candidates[frame_idx]).convert("L"))

    filename_idx = frame_idx + 1
    for path in candidates:
        fname = os.path.basename(path)
        m = re.search(r'_(\d+)\.\w+$', fname)
        if m and int(m.group(1)) == filename_idx:
            return np.array(Image.open(path).convert("L"))
    return None


def track_colormap(n_tracks: int):
    cmap = cm.get_cmap("hsv", max(n_tracks, 1))
    colors = {}
    for i in range(n_tracks):
        r, g, b, _ = cmap(i / max(n_tracks - 1, 1))
        colors[i] = (int(b * 255), int(g * 255), int(r * 255))
    return colors


def _contrast_bgr(bg: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(bg, [1, 99.5])
    if hi <= lo:
        lo, hi = 0, 255
    stretched = np.clip((bg.astype(np.float32) - lo) / (hi - lo + 1e-8) * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)


def _track_series(df: pd.DataFrame):
    track_ids = sorted(df["track_id"].unique())
    colors = track_colormap(len(track_ids))
    id_to_color = {tid: colors[i] for i, tid in enumerate(track_ids)}
    series = {}
    for tid, group in df.groupby("track_id", sort=False):
        g = group.sort_values("frame")
        series[tid] = (
            g["frame"].to_numpy(dtype=np.int32),
            g[["x", "y"]].to_numpy(dtype=np.float32),
            g["is_interpolated"].to_numpy(dtype=bool),
        )
    return track_ids, id_to_color, series


def make_overview(df: pd.DataFrame, images_dir: str, output_path: str, get_frame=None):
    mid_frame = int(df["frame"].median())
    if get_frame is not None:
        bg = get_frame(mid_frame)
    else:
        bg = load_frame(images_dir, mid_frame)
    if bg is None:
        bg = np.zeros((1024, 1024), dtype=np.uint8)

    canvas = _contrast_bgr(bg)
    _, id_to_color, series = _track_series(df)

    for tid, (frames, xy, interp) in series.items():
        color = id_to_color[tid]
        pts = np.round(xy).astype(np.int32)
        if len(pts) >= 2:
            cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False, color, 1, cv2.LINE_AA)
        real = pts[~interp]
        for p in real:
            cv2.circle(canvas, (int(p[0]), int(p[1])), 2, color, -1, cv2.LINE_AA)
        for p in pts[interp]:
            cv2.drawMarker(canvas, (int(p[0]), int(p[1])), color,
                           markerType=cv2.MARKER_CROSS, markerSize=5, thickness=1)

    n_interp = int(df["is_interpolated"].sum())
    label = f"tracks={len(series)} rows={len(df)} interp={n_interp} bg={mid_frame}"
    cv2.putText(canvas, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(output_path, canvas)
    print(f"Saved overview → {output_path}")


def make_video(
    df: pd.DataFrame,
    images_dir: str,
    output_path: str,
    fps: int = 10,
    trail_frames: int = 20,
    get_frame=None,
    workers: int = 0,
):
    image_paths = list_frame_images(images_dir) if get_frame is None else []
    df = df.sort_values(["frame", "track_id"]).reset_index(drop=True)
    frames = sorted(int(f) for f in df["frame"].unique())
    _, id_to_color, series = _track_series(df)

    if get_frame is not None:
        first_bg = get_frame(frames[0])
    else:
        first_bg = load_frame_from_paths(image_paths, frames[0])
    if first_bg is not None:
        h, w = first_bg.shape[:2]
    else:
        h, w = 1024, 1024

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    n_workers = workers if workers > 0 else min(8, max(1, (os.cpu_count() or 4) - 1))
    prev_cv_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)

    def render(frame_idx: int) -> np.ndarray:
        if get_frame is not None:
            bg = get_frame(frame_idx)
        else:
            bg = load_frame_from_paths(image_paths, frame_idx)
        if bg is None:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            canvas = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)

        trail_start = frame_idx - trail_frames
        for tid, (tframes, xy, interp) in series.items():
            mask = (tframes >= trail_start) & (tframes <= frame_idx)
            if not np.any(mask):
                continue
            color = id_to_color.get(tid, (255, 255, 255))
            pts = np.round(xy[mask]).astype(np.int32)
            if len(pts) >= 2:
                cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False, color, 1, cv2.LINE_AA)
            cur = np.where(tframes[mask] == frame_idx)[0]
            if len(cur):
                i = int(cur[-1])
                cx, cy = int(pts[i, 0]), int(pts[i, 1])
                marker = cv2.MARKER_CROSS if bool(interp[mask][i]) else cv2.MARKER_TILTED_CROSS
                cv2.drawMarker(canvas, (cx, cy), color, markerType=marker, markerSize=6, thickness=1)

        cv2.putText(canvas, f"frame {frame_idx:03d}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    total_frames = len(frames)
    batch = max(1, n_workers * 2)
    try:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for start in range(0, total_frames, batch):
                chunk = frames[start:start + batch]
                for i, canvas in enumerate(pool.map(render, chunk), start=start + 1):
                    writer.write(canvas)
                    if i == 1 or i % 100 == 0 or i == total_frames:
                        print(f"Rendered video frame {i}/{total_frames}", flush=True)
    finally:
        cv2.setNumThreads(prev_cv_threads)
        writer.release()
    print(f"Saved video   → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize particle tracks")
    parser.add_argument("--tracks", required=True, help="Tracks CSV path")
    parser.add_argument("--images", required=True, help="Directory of per-frame images")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS (default: 30)")
    parser.add_argument("--trail", type=int, default=20,
                        help="Trail length in frames for video (default: 20)")
    parser.add_argument("--overview-only", action="store_true",
                        help="Only write the static overview PNG; skip MP4 rendering")
    parser.add_argument("--workers", type=int, default=0,
                        help="CPU workers for video frames (0 = auto)")
    args = parser.parse_args()

    df = pd.read_csv(args.tracks)
    os.makedirs(args.output, exist_ok=True)

    base = os.path.splitext(os.path.basename(args.tracks))[0]
    overview_path = os.path.join(args.output, f"{base}_overview.png")
    make_overview(df, args.images, overview_path)

    if not args.overview_only:
        video_path = os.path.join(args.output, f"{base}_video.mp4")
        make_video(df, args.images, video_path, fps=args.fps, trail_frames=args.trail,
                   workers=args.workers)


if __name__ == "__main__":
    main()
