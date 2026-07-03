#!/usr/bin/env python3
"""
LSTM baseline for particle trajectory prediction.

This is the first step toward learned gap filling and denoising. It trains a
sequence model on existing track CSVs and predicts the next state from a short
history window.

Expected input schema:
    track_id, frame, x, y, phi, ncc, is_interpolated

Usage:
    python src/lstm_track_predictor.py train \
        --tracks detection_results/.../tracks/<name>_tracks.csv \
        --model-out models/lstm_track_predictor.pt

    python src/lstm_track_predictor.py predict \
        --tracks detection_results/.../tracks/<name>_tracks.csv \
        --model models/lstm_track_predictor.pt \
        --output detection_results/.../tracks/<name>_lstm_predictions.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Iterable

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


STATE_COLUMNS = ["x", "y", "sin_phi", "cos_phi"]
FEATURE_SETS = {
    "basic": STATE_COLUMNS,
    "motion": STATE_COLUMNS + ["dx_prev", "dy_prev", "dt"],
}
FEATURE_COLUMNS = FEATURE_SETS["basic"]
TARGET_MODES = ("absolute", "residual")


@dataclass
class Normalizer:
    mean: list[float]
    std: list[float]

    def transform(self, values: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        return (values - mean) / std

    def inverse(self, values: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        return values * std + mean


class LSTMTrackPredictor(nn.Module):
    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_size: int = 4,
    ):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


def _normalise_track_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = {"track_id", "frame", "x", "y"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Track CSV missing required columns: {missing}")
    if "phi" not in df.columns:
        df["phi"] = np.nan
    if "is_interpolated" not in df.columns:
        df["is_interpolated"] = False
    if df["is_interpolated"].dtype == object:
        df["is_interpolated"] = df["is_interpolated"].astype(str).str.lower().isin(["true", "1", "yes"])
    df["frame"] = df["frame"].astype(int)
    df["track_id"] = df["track_id"].astype(int)
    return df.sort_values(["track_id", "frame"]).reset_index(drop=True)


def _add_angle_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    phi = df["phi"].to_numpy(dtype=np.float32)
    valid = np.isfinite(phi)
    df["sin_phi"] = np.where(valid, np.sin(phi), 0.0)
    df["cos_phi"] = np.where(valid, np.cos(phi), 0.0)
    return df


def load_tracks(path: str) -> pd.DataFrame:
    return _add_angle_features(_normalise_track_schema(pd.read_csv(path)))


def _feature_columns(feature_set: str) -> list[str]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unsupported feature set: {feature_set}")
    return FEATURE_SETS[feature_set]


def _add_motion_features(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("frame").copy()
    group["dx_prev"] = group["x"].diff().fillna(0.0)
    group["dy_prev"] = group["y"].diff().fillna(0.0)
    group["dt"] = group["frame"].diff().fillna(1.0).clip(lower=1.0)
    return group


def make_windows(
    df: pd.DataFrame,
    seq_len: int,
    min_track_length: int,
    include_interpolated: bool,
    feature_set: str = "basic",
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    feature_columns = _feature_columns(feature_set)

    for _, group in df.groupby("track_id"):
        group = group.sort_values("frame")
        if not include_interpolated:
            group = group[~group["is_interpolated"]]
        if len(group) < max(min_track_length, seq_len + 1):
            continue

        group = _add_motion_features(group)
        values = group[feature_columns].to_numpy(dtype=np.float32)
        targets = group[STATE_COLUMNS].to_numpy(dtype=np.float32)
        for start in range(0, len(values) - seq_len):
            xs.append(values[start:start + seq_len])
            ys.append(targets[start + seq_len])

    if not xs:
        raise ValueError("No training windows created. Lower --seq-len/--min-track-length or include interpolated rows.")
    return np.stack(xs), np.stack(ys)


def fit_normalizer(values: Iterable[np.ndarray], feature_count: int | None = None) -> Normalizer:
    values = list(values)
    if feature_count is None:
        feature_count = values[0].shape[-1]
    joined = np.concatenate([v.reshape(-1, feature_count) for v in values], axis=0)
    mean = joined.mean(axis=0).astype(np.float32)
    std = joined.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return Normalizer(mean=mean.tolist(), std=std.tolist())


def make_targets(x: np.ndarray, y: np.ndarray, target_mode: str) -> np.ndarray:
    if target_mode == "absolute":
        return y.copy()
    if target_mode == "residual":
        target = y.copy()
        target[:, :2] = y[:, :2] - x[:, -1, :2]
        return target
    raise ValueError(f"Unsupported target mode: {target_mode}")


def reconstruct_absolute(x: np.ndarray, pred_target: np.ndarray, target_mode: str) -> np.ndarray:
    pred = pred_target.copy()
    if target_mode == "absolute":
        return pred
    if target_mode == "residual":
        pred[:, :2] = x[:, -1, :2] + pred_target[:, :2]
        return pred
    raise ValueError(f"Unsupported target mode: {target_mode}")


def split_train_val(x: np.ndarray, y: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    n_val = max(1, int(len(idx) * val_fraction))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    if len(train_idx) == 0:
        train_idx = val_idx
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def split_train_val_indices(n_samples: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_samples)
    n_val = max(1, int(len(idx) * val_fraction))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    if len(train_idx) == 0:
        train_idx = val_idx
    return train_idx, val_idx


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pos_err = np.linalg.norm(pred[:, :2] - target[:, :2], axis=1)
    pred_phi = np.arctan2(pred[:, 2], pred[:, 3])
    target_phi = np.arctan2(target[:, 2], target[:, 3])
    dphi = ((pred_phi - target_phi + np.pi) % (2 * np.pi)) - np.pi
    return {
        "rmse_x_px": float(np.sqrt(np.mean((pred[:, 0] - target[:, 0]) ** 2))),
        "rmse_y_px": float(np.sqrt(np.mean((pred[:, 1] - target[:, 1]) ** 2))),
        "mean_position_error_px": float(np.mean(pos_err)),
        "median_position_error_px": float(np.median(pos_err)),
        "mean_angular_error_deg": float(np.mean(np.abs(dphi)) * 180.0 / np.pi),
    }


def _parse_tags(tags: str | None) -> list[str]:
    if not tags:
        return []
    return [tag.strip() for tag in tags.split(",") if tag.strip()]


def _init_wandb(args: argparse.Namespace, windows: int, device: torch.device, model: nn.Module):
    if not args.wandb:
        return None
    if not WANDB_AVAILABLE or wandb is None:
        print("WandB requested but wandb is not available; continuing without WandB logging.")
        return None

    config = {
        "tracks": args.tracks,
        "model_out": args.model_out,
        "windows": windows,
        "feature_set": args.feature_set,
        "seq_len": args.seq_len,
        "min_track_length": args.min_track_length,
        "include_interpolated": args.include_interpolated,
        "target_mode": args.target_mode,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_size": args.hidden_size,
        "layers": args.layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "device": str(device),
        "feature_columns": _feature_columns(args.feature_set),
        "state_columns": STATE_COLUMNS,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_params": sum(p.numel() for p in model.parameters()),
    }
    try:
        os.makedirs(args.wandb_dir, exist_ok=True)
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            tags=_parse_tags(args.wandb_tags),
            notes=args.wandb_notes,
            dir=args.wandb_dir,
            mode=args.wandb_mode,
            config=config,
        )
        if args.wandb_watch:
            wandb.watch(model, log="gradients", log_freq=max(1, args.wandb_watch_freq))
        return run
    except Exception as exc:
        print(f"WandB initialization failed ({exc}); continuing without WandB logging.")
        return None


def _log_wandb(metrics: dict[str, float], step: int | None = None) -> None:
    if WANDB_AVAILABLE and wandb is not None and wandb.run is not None:
        wandb.log(metrics, step=step)


def _gradient_norm(model: nn.Module) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().data.norm(2).item()) ** 2
    return total ** 0.5


def _finish_wandb(run, model_out: str, metrics: dict[str, float], log_model: bool) -> None:
    if run is None or not WANDB_AVAILABLE or wandb is None or wandb.run is None:
        return
    for key, value in metrics.items():
        set_summary(f"final/{key}", value)
    set_summary("model_out", model_out)
    if log_model and os.path.exists(model_out):
        try:
            artifact = wandb.Artifact(os.path.basename(model_out), type="model")
            artifact.add_file(model_out)
            wandb.log_artifact(artifact)
        except Exception as exc:
            print(f"Could not log model artifact to WandB: {exc}")
    finish_run(run)


def train(args: argparse.Namespace) -> int:
    df = load_tracks(args.tracks)
    feature_columns = _feature_columns(args.feature_set)
    x, y = make_windows(df, args.seq_len, args.min_track_length, args.include_interpolated, args.feature_set)
    target_mode = args.target_mode
    target = make_targets(x, y, target_mode)
    x_normalizer = fit_normalizer([x])
    y_normalizer = fit_normalizer([target])
    x_norm = x_normalizer.transform(x)
    y_norm = y_normalizer.transform(target)
    train_idx, val_idx = split_train_val_indices(len(x_norm), args.val_fraction, args.seed)
    x_train, y_train = x_norm[train_idx], y_norm[train_idx]
    x_val, y_val = x_norm[val_idx], y_norm[val_idx]
    x_val_raw = x[val_idx]

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = LSTMTrackPredictor(
        input_size=len(feature_columns),
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        dropout=args.dropout,
        output_size=len(STATE_COLUMNS),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss()
    wandb_run = _init_wandb(args, windows=len(x), device=device, model=model)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_x = torch.from_numpy(x_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)
    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        grad_total = 0.0
        grad_batches = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            grad_total += _gradient_norm(model)
            grad_batches += 1
            optimizer.step()
            total += float(loss.item()) * len(batch_x)

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_x), val_y).item())
        train_loss = total / max(len(x_train), 1)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        grad_norm = grad_total / max(grad_batches, 1)
        _log_wandb({
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/best_loss": best_val,
            "train/grad_norm": grad_norm,
            "train/lr": optimizer.param_groups[0]["lr"],
        }, step=epoch)
        print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_norm = model(val_x).cpu().numpy()
    pred_target = y_normalizer.inverse(pred_norm)
    target_val = y_normalizer.inverse(y_val)
    pred = reconstruct_absolute(x_val_raw[:, :, :len(STATE_COLUMNS)], pred_target, target_mode)
    target = reconstruct_absolute(x_val_raw[:, :, :len(STATE_COLUMNS)], target_val, target_mode)
    metrics = compute_metrics(pred, target)
    _log_wandb({f"val/{key}": value for key, value in metrics.items()}, step=args.epochs)

    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "normalizer": asdict(x_normalizer),
        "x_normalizer": asdict(x_normalizer),
        "y_normalizer": asdict(y_normalizer),
        "feature_columns": feature_columns,
        "input_columns": feature_columns,
        "state_columns": STATE_COLUMNS,
        "feature_set": args.feature_set,
        "target_mode": target_mode,
        "seq_len": args.seq_len,
        "hidden_size": args.hidden_size,
        "layers": args.layers,
        "dropout": args.dropout,
        "metrics": metrics,
    }
    torch.save(checkpoint, args.model_out)
    _finish_wandb(wandb_run, args.model_out, metrics, args.wandb_log_model)
    print(json.dumps({"model_out": args.model_out, "windows": len(x), "metrics": metrics}, indent=2))
    return 0


def predict(args: argparse.Namespace) -> int:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(args.model, map_location=device)
    seq_len = int(checkpoint["seq_len"])
    target_mode = checkpoint.get("target_mode", "absolute")
    x_normalizer = Normalizer(**checkpoint.get("x_normalizer", checkpoint["normalizer"]))
    y_normalizer = Normalizer(**checkpoint.get("y_normalizer", checkpoint["normalizer"]))
    input_columns = checkpoint.get("input_columns", checkpoint.get("feature_columns", FEATURE_COLUMNS))
    output_size = len(checkpoint.get("state_columns", STATE_COLUMNS))
    model = LSTMTrackPredictor(
        input_size=len(input_columns),
        hidden_size=int(checkpoint["hidden_size"]),
        num_layers=int(checkpoint["layers"]),
        dropout=float(checkpoint["dropout"]),
        output_size=output_size,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    df = load_tracks(args.tracks)
    rows: list[dict] = []
    for tid, group in df.groupby("track_id"):
        group = group.sort_values("frame").reset_index(drop=True)
        group = _add_motion_features(group)
        values = group[input_columns].to_numpy(dtype=np.float32)
        states = group[STATE_COLUMNS].to_numpy(dtype=np.float32)
        if len(values) <= seq_len:
            continue
        for start in range(0, len(values) - seq_len):
            raw_seq = values[start:start + seq_len]
            state_seq = states[start:start + seq_len]
            seq = x_normalizer.transform(raw_seq)[None, ...]
            with torch.no_grad():
                pred_norm = model(torch.from_numpy(seq).to(device)).cpu().numpy()
            pred_target = y_normalizer.inverse(pred_norm)
            pred_state = reconstruct_absolute(state_seq[None, ...], pred_target, target_mode)[0]
            target_row = group.iloc[start + seq_len]
            pred_phi = float(np.arctan2(pred_state[2], pred_state[3]))
            rows.append({
                "track_id": int(tid),
                "frame": int(target_row.frame),
                "x": float(target_row.x),
                "y": float(target_row.y),
                "phi": float(target_row.phi) if pd.notna(target_row.phi) else np.nan,
                "pred_x": float(pred_state[0]),
                "pred_y": float(pred_state[1]),
                "pred_phi": pred_phi,
                "is_interpolated": bool(target_row.is_interpolated),
            })

    output = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Saved predictions -> {args.output} ({len(output)} rows)")
    if len(output):
        pred_state = output[["pred_x", "pred_y"]].to_numpy(dtype=np.float32)
        target_state = output[["x", "y"]].to_numpy(dtype=np.float32)
        pos_err = np.linalg.norm(pred_state - target_state, axis=1)
        print(json.dumps({
            "mean_position_error_px": float(np.mean(pos_err)),
            "median_position_error_px": float(np.median(pos_err)),
        }, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LSTM trajectory predictor for tracked particles")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train next-step LSTM predictor")
    p_train.add_argument("--tracks", required=True, help="Input tracks CSV")
    p_train.add_argument("--model-out", required=True, help="Output model checkpoint")
    p_train.add_argument("--seq-len", type=int, default=10, help="History length in frames")
    p_train.add_argument("--min-track-length", type=int, default=30, help="Minimum rows per track")
    p_train.add_argument("--include-interpolated", action="store_true", help="Use interpolated rows during training")
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--batch-size", type=int, default=256)
    p_train.add_argument("--hidden-size", type=int, default=64)
    p_train.add_argument("--layers", type=int, default=2)
    p_train.add_argument("--dropout", type=float, default=0.1)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--weight-decay", type=float, default=1e-4)
    p_train.add_argument("--val-fraction", type=float, default=0.2)
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--device", default=None, help="Override device, e.g. cpu or cuda")
    p_train.add_argument(
        "--feature-set",
        choices=tuple(FEATURE_SETS),
        default="motion",
        help="basic uses x/y/angle only; motion also adds dx_prev, dy_prev, and dt",
    )
    p_train.add_argument(
        "--target-mode",
        choices=TARGET_MODES,
        default="residual",
        help="absolute predicts next x/y directly; residual predicts dx/dy from the last input row",
    )
    p_train.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    p_train.add_argument("--wandb-project", default="MONA_LodeSTAR_LSTM", help="WandB project name")
    p_train.add_argument("--wandb-entity", default=None, help="WandB entity/team")
    p_train.add_argument("--wandb-run-name", default=None, help="WandB run name")
    p_train.add_argument("--wandb-tags", default="LSTM,tracking", help="Comma-separated WandB tags")
    p_train.add_argument("--wandb-notes", default="", help="WandB run notes")
    p_train.add_argument("--wandb-dir", default="wandb_logs", help="Local WandB log directory")
    p_train.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
        help="WandB mode; use offline when network/login is unavailable",
    )
    p_train.add_argument("--wandb-watch", action="store_true", help="Log model gradients with wandb.watch")
    p_train.add_argument("--wandb-watch-freq", type=int, default=100, help="wandb.watch log frequency")
    p_train.add_argument("--wandb-log-model", action="store_true", help="Upload the saved checkpoint as a WandB artifact")
    p_train.set_defaults(func=train)

    p_predict = sub.add_parser("predict", help="Write one-step predictions for a tracks CSV")
    p_predict.add_argument("--tracks", required=True, help="Input tracks CSV")
    p_predict.add_argument("--model", required=True, help="Model checkpoint from train")
    p_predict.add_argument("--output", required=True, help="Output prediction CSV")
    p_predict.add_argument("--device", default=None, help="Override device, e.g. cpu or cuda")
    p_predict.set_defaults(func=predict)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
