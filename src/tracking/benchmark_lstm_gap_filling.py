#!/usr/bin/env python3
"""
Masked-gap benchmark for LSTM trajectory filling.

This script removes known real detections from consecutive track segments and
asks several methods to reconstruct them:
  - linear interpolation between the pre-gap and post-gap detections
  - persistence from the last pre-gap detection
  - constant velocity from the last two pre-gap detections
  - iterative LSTM rollout

The output CSV is long-form: one row per masked frame and method.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from lstm_track_predictor import (
    FEATURE_COLUMNS,
    LSTMTrackPredictor,
    Normalizer,
    STATE_COLUMNS,
    _add_motion_features,
    load_tracks,
    reconstruct_absolute,
)


@dataclass(frozen=True)
class GapSample:
    track_id: int
    start: int
    gap_len: int


def _angle_error_deg(pred_phi: float, true_phi: float) -> float:
    if not np.isfinite(pred_phi) or not np.isfinite(true_phi):
        return np.nan
    dphi = ((pred_phi - true_phi + np.pi) % (2 * np.pi)) - np.pi
    return float(abs(dphi) * 180.0 / np.pi)


def _state_row(frame: int, x: float, y: float, phi: float) -> dict:
    return {
        "frame": int(frame),
        "x": float(x),
        "y": float(y),
        "phi": float(phi) if np.isfinite(phi) else np.nan,
        "sin_phi": float(np.sin(phi)) if np.isfinite(phi) else 0.0,
        "cos_phi": float(np.cos(phi)) if np.isfinite(phi) else 0.0,
    }


def _consecutive_segments(group: pd.DataFrame) -> list[pd.DataFrame]:
    group = group.sort_values("frame").reset_index(drop=True)
    if group.empty:
        return []
    breaks = np.flatnonzero(group["frame"].diff().fillna(1).to_numpy() != 1)
    starts = [0] + breaks.tolist()
    stops = breaks.tolist() + [len(group)]
    return [group.iloc[a:b].reset_index(drop=True) for a, b in zip(starts, stops) if b > a]


def collect_samples(
    tracks: pd.DataFrame,
    gap_lengths: list[int],
    seq_len: int,
    samples_per_gap: int,
    seed: int,
) -> list[tuple[GapSample, pd.DataFrame]]:
    rng = np.random.default_rng(seed)
    real_tracks = tracks[~tracks["is_interpolated"]].copy()
    candidates: dict[int, list[tuple[GapSample, pd.DataFrame]]] = {gap: [] for gap in gap_lengths}

    for tid, group in real_tracks.groupby("track_id"):
        for segment in _consecutive_segments(group):
            for gap_len in gap_lengths:
                needed = seq_len + gap_len + 1
                if len(segment) < needed:
                    continue
                max_start = len(segment) - needed
                for start in range(max_start + 1):
                    sample = GapSample(track_id=int(tid), start=start, gap_len=gap_len)
                    candidates[gap_len].append((sample, segment))

    selected: list[tuple[GapSample, pd.DataFrame]] = []
    for gap_len in gap_lengths:
        items = candidates[gap_len]
        if not items:
            continue
        n = min(samples_per_gap, len(items))
        idx = rng.choice(len(items), size=n, replace=False)
        selected.extend(items[int(i)] for i in idx)
    return selected


def load_lstm_model(model_path: str, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device)
    input_columns = checkpoint.get("input_columns", checkpoint.get("feature_columns", FEATURE_COLUMNS))
    state_columns = checkpoint.get("state_columns", STATE_COLUMNS)
    target_mode = checkpoint.get("target_mode", "absolute")
    x_normalizer = Normalizer(**checkpoint.get("x_normalizer", checkpoint["normalizer"]))
    y_normalizer = Normalizer(**checkpoint.get("y_normalizer", checkpoint["normalizer"]))
    model = LSTMTrackPredictor(
        input_size=len(input_columns),
        hidden_size=int(checkpoint["hidden_size"]),
        num_layers=int(checkpoint["layers"]),
        dropout=float(checkpoint["dropout"]),
        output_size=len(state_columns),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, input_columns, target_mode, x_normalizer, y_normalizer


def lstm_rollout(
    model: LSTMTrackPredictor,
    input_columns: list[str],
    target_mode: str,
    x_normalizer: Normalizer,
    y_normalizer: Normalizer,
    history: pd.DataFrame,
    target_frames: list[int],
    device: torch.device,
) -> list[dict]:
    rows = [
        _state_row(int(row.frame), float(row.x), float(row.y), float(row.phi))
        for row in history.itertuples(index=False)
    ]
    predictions: list[dict] = []

    for frame in target_frames:
        window_df = pd.DataFrame(rows).sort_values("frame").tail(len(history)).reset_index(drop=True)
        window_df = _add_motion_features(window_df)
        raw_seq = window_df[input_columns].to_numpy(dtype=np.float32)
        state_seq = window_df[STATE_COLUMNS].to_numpy(dtype=np.float32)
        seq = x_normalizer.transform(raw_seq)[None, ...]
        with torch.no_grad():
            pred_norm = model(torch.from_numpy(seq).to(device)).cpu().numpy()
        pred_target = y_normalizer.inverse(pred_norm)
        pred_state = reconstruct_absolute(state_seq[None, ...], pred_target, target_mode)[0]
        pred_phi = float(np.arctan2(pred_state[2], pred_state[3]))
        pred_row = _state_row(frame, pred_state[0], pred_state[1], pred_phi)
        rows.append(pred_row)
        predictions.append(pred_row)

    return predictions


def _linear_prediction(before, after, target) -> tuple[float, float, float]:
    denom = max(float(after.frame - before.frame), 1.0)
    t = float(target.frame - before.frame) / denom
    x = float(before.x) + t * (float(after.x) - float(before.x))
    y = float(before.y) + t * (float(after.y) - float(before.y))
    phi_a = float(before.phi) if pd.notna(before.phi) else np.nan
    phi_b = float(after.phi) if pd.notna(after.phi) else np.nan
    if np.isfinite(phi_a) and np.isfinite(phi_b):
        dphi = ((phi_b - phi_a + np.pi) % (2 * np.pi)) - np.pi
        phi = phi_a + t * dphi
    else:
        phi = np.nan
    return x, y, phi


def _constant_velocity_prediction(history: pd.DataFrame, target) -> tuple[float, float, float]:
    last = history.iloc[-1]
    prev = history.iloc[-2]
    dt_prev = max(float(last.frame - prev.frame), 1.0)
    dt_target = max(float(target.frame - last.frame), 1.0)
    vx = (float(last.x) - float(prev.x)) / dt_prev
    vy = (float(last.y) - float(prev.y)) / dt_prev
    return float(last.x) + vx * dt_target, float(last.y) + vy * dt_target, float(last.phi)


def add_result(rows: list[dict], method: str, sample: GapSample, target, pred_x: float, pred_y: float, pred_phi: float):
    err = float(np.hypot(pred_x - float(target.x), pred_y - float(target.y)))
    rows.append({
        "method": method,
        "gap_len": sample.gap_len,
        "track_id": sample.track_id,
        "frame": int(target.frame),
        "x": float(target.x),
        "y": float(target.y),
        "phi": float(target.phi) if pd.notna(target.phi) else np.nan,
        "pred_x": float(pred_x),
        "pred_y": float(pred_y),
        "pred_phi": float(pred_phi) if np.isfinite(pred_phi) else np.nan,
        "position_error_px": err,
        "angular_error_deg": _angle_error_deg(pred_phi, float(target.phi) if pd.notna(target.phi) else np.nan),
    })


def run_benchmark(args: argparse.Namespace) -> pd.DataFrame:
    tracks = load_tracks(args.tracks)
    gap_lengths = [int(x) for x in args.gap_lengths.split(",") if x.strip()]
    samples = collect_samples(tracks, gap_lengths, args.seq_len, args.samples_per_gap, args.seed)
    if not samples:
        raise ValueError("No benchmark samples found. Lower --seq-len or --gap-lengths.")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, input_columns, target_mode, x_norm, y_norm = load_lstm_model(args.model, device)

    rows: list[dict] = []
    for sample, segment in samples:
        start = sample.start
        context = segment.iloc[start:start + args.seq_len].reset_index(drop=True)
        targets = segment.iloc[start + args.seq_len:start + args.seq_len + sample.gap_len].reset_index(drop=True)
        after = segment.iloc[start + args.seq_len + sample.gap_len]
        before = context.iloc[-1]
        lstm_preds = lstm_rollout(
            model,
            input_columns,
            target_mode,
            x_norm,
            y_norm,
            context,
            targets["frame"].astype(int).tolist(),
            device,
        )

        for i, target in targets.iterrows():
            x, y, phi = _linear_prediction(before, after, target)
            add_result(rows, "linear", sample, target, x, y, phi)
            add_result(rows, "persistence", sample, target, float(before.x), float(before.y), float(before.phi))
            x, y, phi = _constant_velocity_prediction(context, target)
            add_result(rows, "constant_velocity", sample, target, x, y, phi)
            pred = lstm_preds[i]
            add_result(rows, "lstm", sample, target, pred["x"], pred["y"], pred["phi"])

    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results
        .groupby(["gap_len", "method"], observed=True)["position_error_px"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            p90=lambda s: s.quantile(0.90),
            p95=lambda s: s.quantile(0.95),
        )
        .reset_index()
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark LSTM masked-gap filling against simple baselines")
    parser.add_argument("--tracks", required=True, help="Input tracks CSV")
    parser.add_argument("--model", required=True, help="LSTM checkpoint")
    parser.add_argument("--output", required=True, help="Output long-form benchmark CSV")
    parser.add_argument("--summary-output", default=None, help="Optional summary CSV")
    parser.add_argument("--gap-lengths", default="1,2,3,5,10", help="Comma-separated gap lengths")
    parser.add_argument("--samples-per-gap", type=int, default=200, help="Max random samples per gap length")
    parser.add_argument("--seq-len", type=int, default=10, help="Context length before the masked gap")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="Override device, e.g. cpu or cuda")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    results = run_benchmark(args)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    results.to_csv(args.output, index=False)
    summary = summarize(results)
    if args.summary_output:
        os.makedirs(os.path.dirname(args.summary_output) or ".", exist_ok=True)
        summary.to_csv(args.summary_output, index=False)
    print(f"Saved benchmark -> {args.output} ({len(results)} rows)")
    print(json.dumps(summary.to_dict(orient="records"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
