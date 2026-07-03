#!/usr/bin/env python3
"""Merge per-stack LodeSTAR detection CSVs into one global-frame CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import utils  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing per-stack *_detections.csv files.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Default: infer <common_prefix>_detections.csv in input dir.",
    )
    parser.add_argument(
        "--pattern",
        default="*_detections.csv",
        help="Glob pattern for input CSVs. Files without _<stack>_detections.csv are ignored.",
    )
    parser.add_argument(
        "--frames-per-stack",
        type=int,
        default=100,
        help="Number of frames per stack used for global-frame offsets.",
    )
    parser.add_argument(
        "--no-stack-columns",
        action="store_true",
        help="Do not add stack and frame_local columns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, summary = utils.merge_detection_csvs(
        input_dir=args.input_dir,
        output_path=args.output,
        pattern=args.pattern,
        frames_per_stack=args.frames_per_stack,
        add_stack_columns=not args.no_stack_columns,
    )

    print(f"output={summary['output_path']}")
    print(f"files={summary['n_files']}")
    print(f"rows={summary['n_rows']}")
    print(f"frames={summary['frame_min']}..{summary['frame_max']}")
    print(f"stacks={summary['stacks'][0]}..{summary['stacks'][-1]}")


if __name__ == "__main__":
    main()
