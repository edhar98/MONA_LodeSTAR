#!/usr/bin/env python3
"""Run LodeSTAR detection on one real JP stack for a controlled smoke test."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
_REPO_ROOT = Path(__file__).resolve().parents[2]
_IMPORT_CWD = Path.cwd()
os.chdir("/tmp")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/ for utils

import deeptrack as dt
import numpy as np

import utils
from test_single_particle import (
    detect_particles,
    load_trained_model,
    save_detections_to_csv,
    visualize_detection_results,
)

os.chdir(_IMPORT_CWD)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/JP_FE/wf_2_40")
    parser.add_argument("--run", default="01")
    parser.add_argument("--stack", required=True, help="Stack id, for example 420")
    parser.add_argument("--particle", default="JP_Fe_wf_2_40")
    parser.add_argument("--model", default="models/5m4rtzfx/JP_Fe_wf_2_40_weights.pth")
    parser.add_argument("--config", default="src/config.yaml")
    parser.add_argument(
        "--output-run-dir",
        default=None,
        help="Defaults to detection_results/JP_FE/wf_2_40/<particle>_5m4rtzfx/<run>",
    )
    parser.add_argument(
        "--detection-mode",
        default="template",
        choices=["standard", "area", "watershed", "template"],
    )
    parser.add_argument(
        "--orientation-template",
        default="data/Samples/JP_Fe_wf_2_40/Samples/f000_d003_phi0234.0.png",
    )
    parser.add_argument("--template-angle-step", type=int, default=2)
    parser.add_argument("--template-phi-deg", type=float, default=None)
    parser.add_argument("--template-refine-radius", type=int, default=25)
    parser.add_argument("--template-search-radius", type=int, default=5)
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write run-level detections/ and weightmaps/ PNGs.",
    )
    return parser.parse_args()


def _repo_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else _REPO_ROOT / path


def main() -> None:
    args = parse_args()
    data_root = _repo_path(args.data_root)
    model_path = _repo_path(args.model)
    config_path = _repo_path(args.config)
    orientation_template = _repo_path(args.orientation_template)

    config = utils.load_yaml(str(config_path))
    model = load_trained_model(str(model_path), config)
    if model is None:
        raise FileNotFoundError(model_path)

    if args.output_run_dir is None:
        model_id = model_path.parent.name
        args.output_run_dir = (
            f"detection_results/JP_FE/wf_2_40/{args.particle}_{model_id}/{args.run}"
        )
    output_run_dir = _repo_path(args.output_run_dir)

    images_dir = data_root / args.run / "images"
    pattern = f"JP_Fe_wf_2_40_slm075_{args.stack}_*.png"
    image_paths = sorted(images_dir.glob(pattern))
    if not image_paths:
        raise FileNotFoundError(f"No images found for {images_dir / pattern}")

    template_bank = None
    if args.detection_mode == "template":
        template_bank = utils.build_template_bank(
            sample_path=str(orientation_template),
            angle_step=args.template_angle_step,
            template_phi_deg=args.template_phi_deg,
        )

    results = []
    for image_path in image_paths:
        image = np.array(dt.LoadImage(str(image_path)).resolve()).astype(np.float32)
        detections, prediction, detection_labels, model_output = detect_particles(
            model,
            image,
            config,
            particle_type=args.particle,
            detection_mode=args.detection_mode,
            template_bank=template_bank,
            template_refine_radius=args.template_refine_radius,
            template_search_radius=args.template_search_radius,
        )
        result = (
            {
                "image_file": image_path.name,
                "detections": detections,
                "detection_labels": detection_labels,
                "prediction": prediction,
                "model_output": model_output,
                "orientations": prediction.get("orientations")
                if isinstance(prediction, dict)
                else None,
                "orientation_ncc": prediction.get("orientation_ncc")
                if isinstance(prediction, dict)
                else None,
            }
        )
        results.append(result)

        if args.visualize:
            video_prefix = f"{args.particle}_{args.run}_{image_path.stem}"
            visualize_detection_results(
                image,
                np.empty((0, 2)),
                detections,
                prediction,
                title=video_prefix,
                save_dir=str(output_run_dir),
                gt_labels=[],
                detection_labels=detection_labels,
                model_output=model_output,
                model=model,
            )

    os.makedirs(output_run_dir, exist_ok=True)
    save_detections_to_csv(results, str(output_run_dir))

    total_detections = int(sum(len(result["detections"]) for result in results))
    print(f"stack={args.stack}")
    print(f"frames={len(results)}")
    print(f"detections={total_detections}")
    print(f"output_dir={output_run_dir}")


if __name__ == "__main__":
    main()
