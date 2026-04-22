import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def crop_detections(
    data_dir: str,
    output_dir: str,
    crop_size: int = 64,
) -> int:
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    csv_dir = data_path / "csv"
    img_dir = data_path / "images"

    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    csv_files = sorted(csv_dir.glob("*_video.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")

    half = crop_size // 2
    total_saved = 0

    for csv_file in csv_files:
        video_id = csv_file.stem.replace("_video", "").split("_")[-1]
        prefix = csv_file.stem.replace(f"_{video_id}_video", "")

        df = pd.read_csv(csv_file, index_col=0)
        df["phi"] = pd.to_numeric(df["phi"], errors="coerce")
        df.dropna(subset=["x", "y", "phi", "frame"], inplace=True)

        video_out = out_path / video_id
        video_out.mkdir(parents=True, exist_ok=True)

        sorted_frames = sorted(df["frame"].unique())
        frame_to_local = {f: i for i, f in enumerate(sorted_frames)}

        frame_groups = df.groupby("frame")
        img_cache: dict[int, np.ndarray | None] = {}

        for frame_idx, detections in frame_groups:
            frame_num = int(frame_idx)

            if frame_num not in img_cache:
                local_idx = frame_to_local[frame_num]
                img_name = f"{prefix}_{video_id}_{local_idx + 1:03d}.png"
                img_path = img_dir / img_name
                if img_path.exists():
                    img_cache[frame_num] = np.array(Image.open(img_path))
                else:
                    img_cache[frame_num] = None
                    print(f"WARNING: {img_path} not found, skipping frame {frame_num}")

            img = img_cache[frame_num]
            if img is None:
                continue

            h, w = img.shape[:2]

            for det_idx, (_, row) in enumerate(detections.iterrows()):
                cx, cy = int(round(row["x"])), int(round(row["y"]))
                phi = float(row["phi"])

                y1 = max(cy - half, 0)
                y2 = min(cy + half, h)
                x1 = max(cx - half, 0)
                x2 = min(cx + half, w)

                if (y2 - y1) < half or (x2 - x1) < half:
                    continue

                crop = img[y1:y2, x1:x2]
                phi_deg = np.degrees(phi) % 360

                fname = f"f{frame_num:03d}_d{det_idx:03d}_phi{phi_deg:06.1f}.png"
                Image.fromarray(crop).save(video_out / fname)
                total_saved += 1

        img_cache.clear()
        print(f"[{video_id}] {total_saved} crops saved so far")

    return total_saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop detections from images using CSV coordinates")
    parser.add_argument("data_dir", type=str, help="Path containing images/ and csv/ subdirectories")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output directory (default: data_dir/crops)")
    parser.add_argument("-s", "--crop-size", type=int, default=64, help="Crop size in pixels (default: 64)")
    args = parser.parse_args()

    output = args.output or str(Path(args.data_dir) / "crops")
    n = crop_detections(args.data_dir, output, args.crop_size)
    print(f"Done. {n} crops saved to {output}")
