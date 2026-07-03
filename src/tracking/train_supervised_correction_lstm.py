#!/usr/bin/env python3
"""
Train a supervised LSTM residual corrector for LodeSTAR trajectories.

The model consumes LodeSTAR-side sequence features and predicts the residual
from LodeSTAR position to reference/YOLO position at the last timestep:

    target = [x_reference - x_lodestar, y_reference - y_lodestar]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_DATASET = (
    "supervised_correction_outputs/JP_Fe_wf_2_40_04/"
    "JP_Fe_wf_2_40_slm075_tracks_correction_windows_seq10.npz"
)
DEFAULT_MODEL_OUT = (
    "supervised_correction_outputs/JP_Fe_wf_2_40_04/"
    "supervised_lodestar_to_reference_lstm.pt"
)


@dataclass
class Normalizer:
    mean: list[float]
    std: list[float]

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - np.asarray(self.mean, dtype=np.float32)) / np.asarray(self.std, dtype=np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * np.asarray(self.std, dtype=np.float32) + np.asarray(self.mean, dtype=np.float32)


class ResidualCorrectionLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def fit_normalizer(values: np.ndarray) -> Normalizer:
    flat = values.reshape(-1, values.shape[-1])
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return Normalizer(mean=mean.tolist(), std=std.tolist())


def split_by_track(meta: np.ndarray, val_fraction: float, test_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    track_ids = np.unique(meta[:, 0])
    rng.shuffle(track_ids)
    n_test = int(round(len(track_ids) * test_fraction))
    n_val = int(round(len(track_ids) * val_fraction))
    test_tracks = set(track_ids[:n_test].tolist())
    val_tracks = set(track_ids[n_test:n_test + n_val].tolist())

    train_idx = []
    val_idx = []
    test_idx = []
    for idx, track_id in enumerate(meta[:, 0]):
        if int(track_id) in test_tracks:
            test_idx.append(idx)
        elif int(track_id) in val_tracks:
            val_idx.append(idx)
        else:
            train_idx.append(idx)

    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


def parse_frame_ranges(raw: str) -> list[tuple[int, int]]:
    ranges = []
    if not raw.strip():
        return ranges
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            frame = int(part)
            ranges.append((frame, frame))
            continue
        start, stop = part.split(":", 1)
        ranges.append((int(start), int(stop)))
    return ranges


def mask_frame_ranges(frames: np.ndarray, ranges: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(len(frames), dtype=bool)
    for start, stop in ranges:
        mask |= (frames >= start) & (frames <= stop)
    return mask


def split_by_frames(meta: np.ndarray, val_ranges: str, test_ranges: str):
    frames = meta[:, 1]
    val_mask = mask_frame_ranges(frames, parse_frame_ranges(val_ranges))
    test_mask = mask_frame_ranges(frames, parse_frame_ranges(test_ranges))
    if np.any(val_mask & test_mask):
        raise ValueError("Validation and test frame ranges overlap")
    train_mask = ~(val_mask | test_mask)
    return np.flatnonzero(train_mask), np.flatnonzero(val_mask), np.flatnonzero(test_mask)


def metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    err = pred - target
    dist = np.hypot(err[:, 0], err[:, 1])
    baseline = np.hypot(target[:, 0], target[:, 1])
    return {
        "mean_error_px": float(dist.mean()),
        "median_error_px": float(np.median(dist)),
        "p90_error_px": float(np.quantile(dist, 0.90)),
        "p95_error_px": float(np.quantile(dist, 0.95)),
        "baseline_mean_error_px": float(baseline.mean()),
        "baseline_median_error_px": float(np.median(baseline)),
        "improvement_mean_px": float(baseline.mean() - dist.mean()),
    }


def evaluate(model, x_norm, y_true, target_norm: Normalizer, indices, device) -> dict[str, float]:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(indices), 4096):
            batch_idx = indices[start:start + 4096]
            xb = torch.from_numpy(x_norm[batch_idx]).to(device)
            pred_norm = model(xb).cpu().numpy()
            preds.append(target_norm.inverse(pred_norm))
    pred = np.concatenate(preds, axis=0) if preds else np.empty((0, 2), dtype=np.float32)
    return metrics(pred, y_true[indices])


def main() -> int:
    parser = argparse.ArgumentParser(description="Train supervised LodeSTAR trajectory residual corrector")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--model-out", default=DEFAULT_MODEL_OUT)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--split-mode", choices=["track", "frame"], default="track")
    parser.add_argument("--val-frame-ranges", default="900:999",
                        help="Inclusive frame ranges for frame split, e.g. '900:999'")
    parser.add_argument("--test-frame-ranges", default="1000:1099",
                        help="Inclusive frame ranges for frame split, e.g. '1000:1099'")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    data = np.load(args.dataset, allow_pickle=True)
    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.float32)
    meta = data["meta"].astype(np.int64)
    feature_cols = [str(v) for v in data["feature_cols"]]
    target_cols = [str(v) for v in data["target_cols"]]
    if len(x) == 0:
        raise ValueError("Dataset contains no windows")

    if args.split_mode == "frame":
        train_idx, val_idx, test_idx = split_by_frames(meta, args.val_frame_ranges, args.test_frame_ranges)
    else:
        train_idx, val_idx, test_idx = split_by_track(meta, args.val_fraction, args.test_fraction, args.seed)
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            f"Empty split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}. "
            "Check split settings."
        )
    input_norm = fit_normalizer(x[train_idx])
    target_norm = fit_normalizer(y[train_idx])
    x_norm = input_norm.transform(x).astype(np.float32)
    y_norm = target_norm.transform(y).astype(np.float32)

    device = torch.device(args.device)
    model = ResidualCorrectionLSTM(
        input_size=x.shape[-1],
        hidden_size=args.hidden_size,
        layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss()

    train_ds = TensorDataset(torch.from_numpy(x_norm[train_idx]), torch.from_numpy(y_norm[train_idx]))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    best_val = float("inf")
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate(model, x_norm, y, target_norm, val_idx, device)
        train_loss = float(np.mean(losses))
        val_loss = val_metrics["mean_error_px"]
        history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"val_mean_error_px={val_metrics['mean_error_px']:.4f} "
            f"baseline={val_metrics['baseline_mean_error_px']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics = {
        "train": evaluate(model, x_norm, y, target_norm, train_idx, device),
        "val": evaluate(model, x_norm, y, target_norm, val_idx, device),
        "test": evaluate(model, x_norm, y, target_norm, test_idx, device),
    }

    output_path = Path(args.model_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "input_normalizer": asdict(input_norm),
        "target_normalizer": asdict(target_norm),
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "input_size": int(x.shape[-1]),
        "seq_len": int(x.shape[1]),
        "hidden_size": args.hidden_size,
        "layers": args.layers,
        "dropout": args.dropout,
        "dataset": args.dataset,
        "split_mode": args.split_mode,
        "val_frame_ranges": args.val_frame_ranges if args.split_mode == "frame" else None,
        "test_frame_ranges": args.test_frame_ranges if args.split_mode == "frame" else None,
        "split_counts": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "metrics": final_metrics,
        "history": history,
    }
    torch.save(checkpoint, output_path)
    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(_jsonable(checkpoint["metrics"]), indent=2))

    print(json.dumps(_jsonable({"model_out": output_path, "metrics": final_metrics}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
