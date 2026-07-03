#!/usr/bin/env python3
"""
Bidirectional LSTM gap filler for tracked particles.

This model is trained on artificial masked gaps from real consecutive track
segments. Unlike the one-step predictor, it sees both sides of the gap:

    past context + future context + query position inside gap

It predicts a correction to linear interpolation for x/y plus the target
orientation as sin/cos.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "tools")))
try:
    from wandb_logging import WANDB_AVAILABLE, finish_run, set_summary, wandb
except Exception:
    WANDB_AVAILABLE = False
    wandb = None

from lstm_track_predictor import Normalizer, _add_motion_features, load_tracks
from benchmark_lstm_gap_filling import (
    load_lstm_model as load_causal_lstm_model,
    lstm_rollout as causal_lstm_rollout,
)


STATE_COLUMNS = ["x", "y", "sin_phi", "cos_phi"]
INPUT_COLUMNS = ["x", "y", "sin_phi", "cos_phi", "dx_prev", "dy_prev", "dt"]
QUERY_COLUMNS = ["alpha", "gap_len", "offset"]
TARGET_COLUMNS = ["corr_x", "corr_y", "sin_phi", "cos_phi"]


class BiLSTMGapFiller(nn.Module):
    def __init__(
        self,
        input_size: int = len(INPUT_COLUMNS),
        query_size: int = len(QUERY_COLUMNS),
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_size: int = len(TARGET_COLUMNS),
    ):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.past_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.future_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2 + query_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, past: torch.Tensor, future: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        past_out, _ = self.past_lstm(past)
        future_out, _ = self.future_lstm(future)
        features = torch.cat([past_out[:, -1], future_out[:, -1], query], dim=1)
        return self.head(features)


@dataclass
class GapArrays:
    past: np.ndarray
    future: np.ndarray
    query: np.ndarray
    target: np.ndarray
    linear_xy: np.ndarray
    true_state: np.ndarray
    meta: pd.DataFrame


def _consecutive_segments(group: pd.DataFrame) -> list[pd.DataFrame]:
    group = group.sort_values("frame").reset_index(drop=True)
    if group.empty:
        return []
    breaks = np.flatnonzero(group["frame"].diff().fillna(1).to_numpy() != 1)
    starts = [0] + breaks.tolist()
    stops = breaks.tolist() + [len(group)]
    return [group.iloc[a:b].reset_index(drop=True) for a, b in zip(starts, stops) if b > a]


def _linear_state(before, after, target) -> tuple[float, float, float]:
    denom = max(float(after.frame - before.frame), 1.0)
    alpha = float(target.frame - before.frame) / denom
    x = float(before.x) + alpha * (float(after.x) - float(before.x))
    y = float(before.y) + alpha * (float(after.y) - float(before.y))
    phi_a = float(before.phi) if pd.notna(before.phi) else np.nan
    phi_b = float(after.phi) if pd.notna(after.phi) else np.nan
    if np.isfinite(phi_a) and np.isfinite(phi_b):
        dphi = ((phi_b - phi_a + np.pi) % (2 * np.pi)) - np.pi
        phi = phi_a + alpha * dphi
    else:
        phi = np.nan
    return x, y, phi


def _constant_velocity_state(past: pd.DataFrame, target) -> tuple[float, float, float]:
    last = past.iloc[-1]
    prev = past.iloc[-2]
    dt_prev = max(float(last.frame - prev.frame), 1.0)
    dt_target = max(float(target.frame - last.frame), 1.0)
    vx = (float(last.x) - float(prev.x)) / dt_prev
    vy = (float(last.y) - float(prev.y)) / dt_prev
    return float(last.x) + vx * dt_target, float(last.y) + vy * dt_target, float(last.phi)


def _estimate_kalman_noise(observed: pd.DataFrame) -> tuple[float, float]:
    xy = observed[["x", "y"]].to_numpy(dtype=float)
    if len(xy) < 3:
        return 1.0, 0.25
    steps = np.diff(observed["frame"].to_numpy(dtype=float))
    steps[steps <= 0.0] = 1.0
    velocity = np.diff(xy, axis=0) / steps[:, None]
    if len(velocity) < 2:
        return 1.0, 0.25
    acceleration = np.diff(velocity, axis=0)
    accel_var = float(np.nanmedian(np.sum(acceleration * acceleration, axis=1)))
    process_var = max(accel_var * 0.01, 1e-4)
    meas_var = 0.25
    return process_var, meas_var


def _kalman_smoother_xy(past: pd.DataFrame, gap: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
    """Constant-velocity Kalman filter plus RTS smoother over x/y with the gap masked."""
    sequence = pd.concat([past, gap, future], ignore_index=True)
    frames = sequence["frame"].to_numpy(dtype=float)
    observed_mask = np.ones(len(sequence), dtype=bool)
    observed_mask[len(past):len(past) + len(gap)] = False

    observed = pd.concat([past, future], ignore_index=True)
    process_var, meas_var = _estimate_kalman_noise(observed)

    state = np.zeros(4, dtype=float)
    first = past.iloc[0]
    state[:2] = [float(first.x), float(first.y)]
    if len(past) >= 2:
        second = past.iloc[1]
        dt = max(float(second.frame - first.frame), 1.0)
        state[2:] = [(float(second.x) - float(first.x)) / dt, (float(second.y) - float(first.y)) / dt]

    cov = np.diag([meas_var, meas_var, 10.0, 10.0]).astype(float)
    transition_mats: list[np.ndarray] = []
    pred_states = np.zeros((len(sequence), 4), dtype=float)
    pred_covs = np.zeros((len(sequence), 4, 4), dtype=float)
    filt_states = np.zeros((len(sequence), 4), dtype=float)
    filt_covs = np.zeros((len(sequence), 4, 4), dtype=float)
    h_mat = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)
    r_mat = np.eye(2, dtype=float) * meas_var

    for i in range(len(sequence)):
        if i == 0:
            transition = np.eye(4, dtype=float)
        else:
            dt = max(float(frames[i] - frames[i - 1]), 1.0)
            transition = np.array(
                [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                dtype=float,
            )
            q_block = np.array(
                [
                    [dt ** 4 / 4.0, 0.0, dt ** 3 / 2.0, 0.0],
                    [0.0, dt ** 4 / 4.0, 0.0, dt ** 3 / 2.0],
                    [dt ** 3 / 2.0, 0.0, dt ** 2, 0.0],
                    [0.0, dt ** 3 / 2.0, 0.0, dt ** 2],
                ],
                dtype=float,
            )
            state = transition @ state
            cov = transition @ cov @ transition.T + process_var * q_block
        transition_mats.append(transition)
        pred_states[i] = state
        pred_covs[i] = cov

        if observed_mask[i]:
            measurement = sequence.loc[i, ["x", "y"]].to_numpy(dtype=float)
            innovation = measurement - h_mat @ state
            innovation_cov = h_mat @ cov @ h_mat.T + r_mat
            gain = cov @ h_mat.T @ np.linalg.inv(innovation_cov)
            state = state + gain @ innovation
            cov = (np.eye(4, dtype=float) - gain @ h_mat) @ cov
        filt_states[i] = state
        filt_covs[i] = cov

    smooth_states = filt_states.copy()
    smooth_covs = filt_covs.copy()
    for i in range(len(sequence) - 2, -1, -1):
        pred_cov_next = pred_covs[i + 1] + np.eye(4, dtype=float) * 1e-9
        gain = filt_covs[i] @ transition_mats[i + 1].T @ np.linalg.inv(pred_cov_next)
        smooth_states[i] = filt_states[i] + gain @ (smooth_states[i + 1] - pred_states[i + 1])
        smooth_covs[i] = filt_covs[i] + gain @ (smooth_covs[i + 1] - pred_cov_next) @ gain.T

    return smooth_states[len(past):len(past) + len(gap), :2]


def build_gap_dataset(
    tracks: pd.DataFrame,
    context_len: int,
    gap_lengths: list[int],
    max_samples: int | None,
    seed: int,
) -> GapArrays:
    real_tracks = tracks[~tracks["is_interpolated"]].copy()
    segments: list[pd.DataFrame] = []
    candidates: list[tuple[int, int, int, int, int]] = []

    for tid, group in real_tracks.groupby("track_id"):
        for segment in _consecutive_segments(group):
            segment = _add_motion_features(segment)
            segment_id = len(segments)
            segments.append(segment)
            for gap_len in gap_lengths:
                needed = context_len + gap_len + context_len
                if len(segment) < needed:
                    continue
                for start in range(0, len(segment) - needed + 1):
                    for offset in range(1, gap_len + 1):
                        candidates.append((int(tid), segment_id, start, gap_len, offset))

    if not candidates:
        raise ValueError("No gap samples created. Lower --context-len or --gap-lengths.")

    rng = np.random.default_rng(seed)
    if max_samples is not None and len(candidates) > max_samples:
        idx = rng.choice(len(candidates), size=max_samples, replace=False)
        candidates = [candidates[int(i)] for i in idx]

    records = []
    for tid, segment_id, start, gap_len, offset in candidates:
        segment = segments[segment_id]
        needed = context_len + gap_len + context_len
        past = segment.iloc[start:start + context_len]
        gap = segment.iloc[start + context_len:start + context_len + gap_len]
        future = segment.iloc[start + context_len + gap_len:start + needed]
        target = gap.iloc[offset - 1]
        before = past.iloc[-1]
        after = future.iloc[0]
        lin_x, lin_y, _ = _linear_state(before, after, target)
        records.append({
            "track_id": int(tid),
            "gap_len": int(gap_len),
            "offset": int(offset),
            "frame": int(target.frame),
            "past": past[INPUT_COLUMNS].to_numpy(dtype=np.float32),
            "future": future.iloc[::-1][INPUT_COLUMNS].to_numpy(dtype=np.float32),
            "query": np.array([offset / (gap_len + 1), gap_len, offset], dtype=np.float32),
            "target": np.array([
                float(target.x) - lin_x,
                float(target.y) - lin_y,
                np.sin(float(target.phi)) if pd.notna(target.phi) else 0.0,
                np.cos(float(target.phi)) if pd.notna(target.phi) else 0.0,
            ], dtype=np.float32),
            "linear_xy": np.array([lin_x, lin_y], dtype=np.float32),
            "true_state": np.array([
                float(target.x),
                float(target.y),
                np.sin(float(target.phi)) if pd.notna(target.phi) else 0.0,
                np.cos(float(target.phi)) if pd.notna(target.phi) else 0.0,
            ], dtype=np.float32),
        })

    past = np.stack([r["past"] for r in records])
    future = np.stack([r["future"] for r in records])
    query = np.stack([r["query"] for r in records])
    target = np.stack([r["target"] for r in records])
    linear_xy = np.stack([r["linear_xy"] for r in records])
    true_state = np.stack([r["true_state"] for r in records])
    meta = pd.DataFrame([{k: r[k] for k in ["track_id", "gap_len", "offset", "frame"]} for r in records])
    return GapArrays(past=past, future=future, query=query, target=target, linear_xy=linear_xy, true_state=true_state, meta=meta)


def fit_normalizer(values: list[np.ndarray]) -> Normalizer:
    feature_count = values[0].shape[-1]
    joined = np.concatenate([v.reshape(-1, feature_count) for v in values], axis=0)
    mean = joined.mean(axis=0).astype(np.float32)
    std = joined.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return Normalizer(mean=mean.tolist(), std=std.tolist())


def split_indices(n: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    if len(train_idx) == 0:
        train_idx = val_idx
    return train_idx, val_idx


def compute_metrics(pred_target: np.ndarray, arrays: GapArrays, indices: np.ndarray) -> dict[str, float]:
    linear_xy = arrays.linear_xy[indices]
    true = arrays.true_state[indices]
    pred_xy = linear_xy + pred_target[:, :2]
    pos_err = np.linalg.norm(pred_xy - true[:, :2], axis=1)
    pred_phi = np.arctan2(pred_target[:, 2], pred_target[:, 3])
    true_phi = np.arctan2(true[:, 2], true[:, 3])
    dphi = ((pred_phi - true_phi + np.pi) % (2 * np.pi)) - np.pi
    return {
        "mean_position_error_px": float(pos_err.mean()),
        "median_position_error_px": float(np.median(pos_err)),
        "p90_position_error_px": float(np.quantile(pos_err, 0.90)),
        "p95_position_error_px": float(np.quantile(pos_err, 0.95)),
        "mean_angular_error_deg": float(np.mean(np.abs(dphi)) * 180.0 / np.pi),
    }


def parse_int_list(value: str) -> list[int]:
    return [int(v) for v in value.split(",") if v.strip()]


def parse_tags(tags: str | None) -> list[str]:
    if not tags:
        return []
    return [tag.strip() for tag in tags.split(",") if tag.strip()]


def init_wandb(args, n_samples: int, device: torch.device, model: nn.Module):
    if not args.wandb or not WANDB_AVAILABLE or wandb is None:
        return None
    try:
        os.makedirs(args.wandb_dir, exist_ok=True)
        return wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            tags=parse_tags(args.wandb_tags),
            notes=args.wandb_notes,
            dir=args.wandb_dir,
            mode=args.wandb_mode,
            config={
                "tracks": args.tracks,
                "model_out": args.model_out,
                "samples": n_samples,
                "context_len": args.context_len,
                "gap_lengths": args.gap_lengths,
                "hidden_size": args.hidden_size,
                "layers": args.layers,
                "dropout": args.dropout,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "device": str(device),
                "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            },
        )
    except Exception as exc:
        print(f"WandB initialization failed ({exc}); continuing without WandB logging.")
        return None


def train(args: argparse.Namespace) -> int:
    tracks = load_tracks(args.tracks)
    arrays = build_gap_dataset(
        tracks,
        context_len=args.context_len,
        gap_lengths=parse_int_list(args.gap_lengths),
        max_samples=args.max_samples,
        seed=args.seed,
    )
    train_idx, val_idx = split_indices(len(arrays.target), args.val_fraction, args.seed)
    input_norm = fit_normalizer([arrays.past, arrays.future])
    query_norm = fit_normalizer([arrays.query])
    target_norm = fit_normalizer([arrays.target])

    past = input_norm.transform(arrays.past)
    future = input_norm.transform(arrays.future)
    query = query_norm.transform(arrays.query)
    target = target_norm.transform(arrays.target)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = BiLSTMGapFiller(
        input_size=len(INPUT_COLUMNS),
        query_size=len(QUERY_COLUMNS),
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss()
    run = init_wandb(args, len(arrays.target), device, model)

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(past[train_idx]),
            torch.from_numpy(future[train_idx]),
            torch.from_numpy(query[train_idx]),
            torch.from_numpy(target[train_idx]),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_past = torch.from_numpy(past[val_idx]).to(device)
    val_future = torch.from_numpy(future[val_idx]).to(device)
    val_query = torch.from_numpy(query[val_idx]).to(device)
    val_target = torch.from_numpy(target[val_idx]).to(device)

    best_val = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch_past, batch_future, batch_query, batch_target in loader:
            batch_past = batch_past.to(device)
            batch_future = batch_future.to(device)
            batch_query = batch_query.to(device)
            batch_target = batch_target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_past, batch_future, batch_query), batch_target)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(batch_target)
        train_loss = total / max(len(train_idx), 1)
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_past, val_future, val_query), val_target).item())
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if WANDB_AVAILABLE and wandb is not None and wandb.run is not None:
            wandb.log({"epoch": epoch, "train/loss": train_loss, "val/loss": val_loss, "val/best_loss": best_val}, step=epoch)
        print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_norm = model(val_past, val_future, val_query).cpu().numpy()
    pred_target = target_norm.inverse(pred_norm)
    metrics = compute_metrics(pred_target, arrays, val_idx)
    if WANDB_AVAILABLE and wandb is not None and wandb.run is not None:
        wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=args.epochs)
        for key, value in metrics.items():
            set_summary(f"final/{key}", value)

    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "input_normalizer": asdict(input_norm),
        "query_normalizer": asdict(query_norm),
        "target_normalizer": asdict(target_norm),
        "input_columns": INPUT_COLUMNS,
        "query_columns": QUERY_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "context_len": args.context_len,
        "hidden_size": args.hidden_size,
        "layers": args.layers,
        "dropout": args.dropout,
        "metrics": metrics,
    }
    torch.save(checkpoint, args.model_out)
    if run is not None:
        set_summary("model_out", args.model_out)
        finish_run(run)
    print(json.dumps({"model_out": args.model_out, "samples": len(arrays.target), "metrics": metrics}, indent=2))
    return 0


def load_model(model_path: str, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device)
    model = BiLSTMGapFiller(
        input_size=len(checkpoint["input_columns"]),
        query_size=len(checkpoint["query_columns"]),
        hidden_size=int(checkpoint["hidden_size"]),
        num_layers=int(checkpoint["layers"]),
        dropout=float(checkpoint["dropout"]),
        output_size=len(checkpoint["target_columns"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return (
        model,
        checkpoint,
        Normalizer(**checkpoint["input_normalizer"]),
        Normalizer(**checkpoint["query_normalizer"]),
        Normalizer(**checkpoint["target_normalizer"]),
    )


def predict_gap(model, checkpoint, input_norm, query_norm, target_norm, past, future, gap, before, after, device):
    future_rev = future.iloc[::-1].reset_index(drop=True)
    past_x = input_norm.transform(past[checkpoint["input_columns"]].to_numpy(dtype=np.float32))[None, ...]
    future_x = input_norm.transform(future_rev[checkpoint["input_columns"]].to_numpy(dtype=np.float32))[None, ...]
    predictions = []
    for offset, target in enumerate(gap.itertuples(index=False), start=1):
        lin_x, lin_y, _ = _linear_state(before, after, target)
        query = np.array([[offset / (len(gap) + 1), len(gap), offset]], dtype=np.float32)
        query_x = query_norm.transform(query)
        with torch.no_grad():
            pred_norm = model(
                torch.from_numpy(past_x).to(device),
                torch.from_numpy(future_x).to(device),
                torch.from_numpy(query_x).to(device),
            ).cpu().numpy()
        pred = target_norm.inverse(pred_norm)[0]
        pred_phi = float(np.arctan2(pred[2], pred[3]))
        predictions.append((lin_x + float(pred[0]), lin_y + float(pred[1]), pred_phi))
    return predictions


def angle_error(pred_phi: float, true_phi: float) -> float:
    if not np.isfinite(pred_phi) or not np.isfinite(true_phi):
        return np.nan
    dphi = ((pred_phi - true_phi + np.pi) % (2 * np.pi)) - np.pi
    return float(abs(dphi) * 180.0 / np.pi)


def add_result(rows, method, gap_len, track_id, target, pred_x, pred_y, pred_phi):
    rows.append({
        "method": method,
        "gap_len": int(gap_len),
        "track_id": int(track_id),
        "frame": int(target.frame),
        "x": float(target.x),
        "y": float(target.y),
        "phi": float(target.phi) if pd.notna(target.phi) else np.nan,
        "pred_x": float(pred_x),
        "pred_y": float(pred_y),
        "pred_phi": float(pred_phi) if np.isfinite(pred_phi) else np.nan,
        "position_error_px": float(np.hypot(pred_x - float(target.x), pred_y - float(target.y))),
        "angular_error_deg": angle_error(pred_phi, float(target.phi) if pd.notna(target.phi) else np.nan),
    })


def benchmark(args: argparse.Namespace) -> int:
    tracks = load_tracks(args.tracks)
    gap_lengths = parse_int_list(args.gap_lengths)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, checkpoint, input_norm, query_norm, target_norm = load_model(args.model, device)
    causal_model = None
    causal_inputs = None
    causal_target_mode = None
    causal_x_norm = None
    causal_y_norm = None
    if args.causal_model:
        causal_model, causal_inputs, causal_target_mode, causal_x_norm, causal_y_norm = load_causal_lstm_model(
            args.causal_model,
            device,
        )
    context_len = int(checkpoint["context_len"])
    rng = np.random.default_rng(args.seed)

    candidates = []
    real_tracks = tracks[~tracks["is_interpolated"]].copy()
    for tid, group in real_tracks.groupby("track_id"):
        for segment in _consecutive_segments(group):
            segment = _add_motion_features(segment)
            for gap_len in gap_lengths:
                needed = context_len + gap_len + context_len
                if len(segment) < needed:
                    continue
                for start in range(0, len(segment) - needed + 1):
                    candidates.append((int(tid), gap_len, start, segment))

    if not candidates:
        raise ValueError("No benchmark candidates found.")
    if len(candidates) > args.samples:
        idx = rng.choice(len(candidates), size=args.samples, replace=False)
        candidates = [candidates[int(i)] for i in idx]

    rows = []
    for tid, gap_len, start, segment in candidates:
        past = segment.iloc[start:start + context_len].reset_index(drop=True)
        gap = segment.iloc[start + context_len:start + context_len + gap_len].reset_index(drop=True)
        future = segment.iloc[start + context_len + gap_len:start + context_len + gap_len + context_len].reset_index(drop=True)
        before = past.iloc[-1]
        after = future.iloc[0]
        preds = predict_gap(model, checkpoint, input_norm, query_norm, target_norm, past, future, gap, before, after, device)
        kalman_xy = _kalman_smoother_xy(past, gap, future)
        causal_preds = None
        if causal_model is not None:
            causal_preds = causal_lstm_rollout(
                causal_model,
                causal_inputs,
                causal_target_mode,
                causal_x_norm,
                causal_y_norm,
                past,
                gap["frame"].astype(int).tolist(),
                device,
            )
        for i, target in gap.iterrows():
            lin_x, lin_y, lin_phi = _linear_state(before, after, target)
            add_result(rows, "linear", gap_len, tid, target, lin_x, lin_y, lin_phi)
            add_result(rows, "persistence", gap_len, tid, target, float(before.x), float(before.y), float(before.phi))
            vel_x, vel_y, vel_phi = _constant_velocity_state(past, target)
            add_result(rows, "constant_velocity", gap_len, tid, target, vel_x, vel_y, vel_phi)
            _, _, kalman_phi = _linear_state(before, after, target)
            add_result(rows, "kalman_smoother", gap_len, tid, target, kalman_xy[i, 0], kalman_xy[i, 1], kalman_phi)
            if causal_preds is not None:
                pred = causal_preds[i]
                add_result(rows, "causal_lstm", gap_len, tid, target, pred["x"], pred["y"], pred["phi"])
            pred_x, pred_y, pred_phi = preds[i]
            add_result(rows, "bilstm_gap", gap_len, tid, target, pred_x, pred_y, pred_phi)

    results = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    results.to_csv(args.output, index=False)
    summary = (
        results.groupby(["gap_len", "method"], observed=True)["position_error_px"]
        .agg(count="count", mean="mean", median="median", p90=lambda s: s.quantile(0.9), p95=lambda s: s.quantile(0.95))
        .reset_index()
    )
    if args.summary_output:
        os.makedirs(os.path.dirname(args.summary_output) or ".", exist_ok=True)
        summary.to_csv(args.summary_output, index=False)
    print(f"Saved benchmark -> {args.output} ({len(results)} rows)")
    print(json.dumps(summary.to_dict(orient="records"), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bidirectional LSTM gap filler")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train bidirectional gap filler")
    p_train.add_argument("--tracks", required=True)
    p_train.add_argument("--model-out", required=True)
    p_train.add_argument("--context-len", type=int, default=10)
    p_train.add_argument("--gap-lengths", default="1,2,3,5,10")
    p_train.add_argument("--max-samples", type=int, default=120000)
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--batch-size", type=int, default=512)
    p_train.add_argument("--hidden-size", type=int, default=64)
    p_train.add_argument("--layers", type=int, default=2)
    p_train.add_argument("--dropout", type=float, default=0.1)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--weight-decay", type=float, default=1e-4)
    p_train.add_argument("--val-fraction", type=float, default=0.2)
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--device", default=None)
    p_train.add_argument("--wandb", action="store_true")
    p_train.add_argument("--wandb-project", default="MONA_LodeSTAR_LSTM")
    p_train.add_argument("--wandb-entity", default=None)
    p_train.add_argument("--wandb-run-name", default=None)
    p_train.add_argument("--wandb-tags", default="LSTM,gap-filling,bidirectional")
    p_train.add_argument("--wandb-notes", default="")
    p_train.add_argument("--wandb-dir", default="wandb_logs")
    p_train.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    p_train.set_defaults(func=train)

    p_bench = sub.add_parser("benchmark", help="Benchmark trained gap filler against linear interpolation")
    p_bench.add_argument("--tracks", required=True)
    p_bench.add_argument("--model", required=True)
    p_bench.add_argument("--causal-model", default=None, help="Optional causal one-step LSTM checkpoint to include")
    p_bench.add_argument("--output", required=True)
    p_bench.add_argument("--summary-output", default=None)
    p_bench.add_argument("--gap-lengths", default="1,2,3,5,10")
    p_bench.add_argument("--samples", type=int, default=1000)
    p_bench.add_argument("--seed", type=int, default=1)
    p_bench.add_argument("--device", default=None)
    p_bench.set_defaults(func=benchmark)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
