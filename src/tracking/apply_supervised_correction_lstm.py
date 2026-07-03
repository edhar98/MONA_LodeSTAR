#!/usr/bin/env python3
"""
Apply a supervised LodeSTAR-to-reference residual corrector to tracks.

The output preserves raw coordinates and adds refined coordinates plus a
correction magnitude. By default the core x/y columns are replaced with the
refined values for rows where the model has enough sequential context.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_supervised_correction_lstm import ResidualCorrectionLSTM


DEFAULT_TRACKS = (
    "detection_results/JP_FE/wf_2_40/JP_Fe_wf_2_40_5m4rtzfx/04/tracks/"
    "JP_Fe_wf_2_40_slm075_tracks.csv"
)
DEFAULT_MODEL = (
    "supervised_correction_outputs/JP_Fe_wf_2_40_04/"
    "supervised_lodestar_to_reference_lstm.pt"
)
DEFAULT_OUTPUT = (
    "supervised_correction_outputs/JP_Fe_wf_2_40_04/"
    "JP_Fe_wf_2_40_slm075_tracks_supervised_refined.csv"
)


@dataclass
class Normalizer:
    mean: list[float]
    std: list[float]

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - np.asarray(self.mean, dtype=np.float32)) / np.asarray(self.std, dtype=np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * np.asarray(self.std, dtype=np.float32) + np.asarray(self.mean, dtype=np.float32)


def load_tracks(path: Path, include_interpolated: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "is_interpolated" not in df.columns:
        df["is_interpolated"] = False
    if df["is_interpolated"].dtype == object:
        df["is_interpolated"] = df["is_interpolated"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        df["is_interpolated"] = df["is_interpolated"].fillna(False).astype(bool)
    if "phi" not in df.columns:
        df["phi"] = np.nan
    if "ncc" not in df.columns:
        df["ncc"] = np.nan
    if not include_interpolated:
        df = df[~df["is_interpolated"]].copy()
    df["track_id"] = pd.to_numeric(df["track_id"], errors="coerce").astype(int)
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce").astype(int)
    return df.sort_values(["track_id", "frame"]).reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for _, group in df.groupby("track_id", sort=False):
        group = group.sort_values("frame").copy()
        group["lode_x"] = group["x"]
        group["lode_y"] = group["y"]
        group["lode_ncc"] = group["ncc"]
        group["dt_prev"] = group["frame"].diff().fillna(1).clip(lower=1)
        group["dx_prev"] = group["x"].diff().fillna(0.0)
        group["dy_prev"] = group["y"].diff().fillna(0.0)
        group["speed_prev"] = np.hypot(group["dx_prev"], group["dy_prev"]) / group["dt_prev"]
        group["sin_phi"] = np.sin(group["phi"].fillna(0.0))
        group["cos_phi"] = np.cos(group["phi"].fillna(0.0))
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def build_windows(df: pd.DataFrame, feature_cols: list[str], seq_len: int):
    windows = []
    row_indices = []
    for _, group in df.groupby("track_id", sort=False):
        group = group.sort_values("frame")
        consecutive = group["frame"].diff().fillna(1).eq(1)
        segment_ids = (~consecutive).cumsum()
        for _, segment in group.groupby(segment_ids):
            if len(segment) < seq_len:
                continue
            features = segment[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
            indices = segment.index.to_numpy(dtype=int)
            for start in range(0, len(segment) - seq_len + 1):
                stop = start + seq_len
                windows.append(features[start:stop])
                row_indices.append(indices[stop - 1])
    if not windows:
        return np.empty((0, seq_len, len(feature_cols)), dtype=np.float32), np.empty((0,), dtype=int)
    return np.stack(windows), np.asarray(row_indices, dtype=int)


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply supervised LSTM correction to LodeSTAR tracks")
    parser.add_argument("--tracks", default=DEFAULT_TRACKS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--include-interpolated", action="store_true")
    parser.add_argument("--preserve-core-xy", action="store_true",
                        help="Keep x/y as raw coordinates and write only x_refined/y_refined")
    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location=args.device, weights_only=False)
    feature_cols = [str(v) for v in checkpoint["feature_cols"]]
    seq_len = int(checkpoint["seq_len"])
    input_norm = Normalizer(**checkpoint["input_normalizer"])
    target_norm = Normalizer(**checkpoint["target_normalizer"])

    model = ResidualCorrectionLSTM(
        input_size=int(checkpoint["input_size"]),
        hidden_size=int(checkpoint["hidden_size"]),
        layers=int(checkpoint["layers"]),
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    raw = load_tracks(Path(args.tracks), include_interpolated=args.include_interpolated)
    df = add_features(raw)
    windows, row_indices = build_windows(df, feature_cols, seq_len)

    df["x_raw"] = df["x"]
    df["y_raw"] = df["y"]
    df["phi_raw"] = df["phi"]
    df["x_refined"] = df["x"]
    df["y_refined"] = df["y"]
    df["supervised_dx"] = 0.0
    df["supervised_dy"] = 0.0
    df["supervised_shift_px"] = 0.0
    df["is_model_refined"] = False

    if len(windows):
        x_norm = input_norm.transform(windows).astype(np.float32)
        preds = []
        with torch.no_grad():
            for start in range(0, len(x_norm), 4096):
                batch = torch.from_numpy(x_norm[start:start + 4096]).to(device)
                pred = model(batch).cpu().numpy()
                preds.append(target_norm.inverse(pred))
        residuals = np.concatenate(preds, axis=0)
        df.loc[row_indices, "supervised_dx"] = residuals[:, 0]
        df.loc[row_indices, "supervised_dy"] = residuals[:, 1]
        df.loc[row_indices, "x_refined"] = df.loc[row_indices, "x_raw"].to_numpy() + residuals[:, 0]
        df.loc[row_indices, "y_refined"] = df.loc[row_indices, "y_raw"].to_numpy() + residuals[:, 1]
        df.loc[row_indices, "supervised_shift_px"] = np.hypot(residuals[:, 0], residuals[:, 1])
        df.loc[row_indices, "is_model_refined"] = True
        if not args.preserve_core_xy:
            df.loc[row_indices, "x"] = df.loc[row_indices, "x_refined"]
            df.loc[row_indices, "y"] = df.loc[row_indices, "y_refined"]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(
        f"wrote {output} rows={len(df)} model_refined={int(df['is_model_refined'].sum())} "
        f"mean_shift_px={df.loc[df['is_model_refined'], 'supervised_shift_px'].mean():.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
