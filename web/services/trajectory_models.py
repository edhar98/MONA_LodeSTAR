"""Optional CPU inference adapters; never replace the baseline track file."""
import hashlib
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
METHODS = {
    "causal_prediction": "Causal LSTM next-step prediction (diagnostic)",
    "bilstm_gap": "Two-sided LSTM gap refinement (experimental)",
    "supervised_correction": "Reference-calibrated LSTM correction (experimental)",
}


def catalog():
    entries = []
    for directory, pattern, method in [
        (ROOT / "lstm_outputs", "lstm_track_predictor*.pt", "causal_prediction"),
        (ROOT / "lstm_outputs", "lstm_gap_filler*.pt", "bilstm_gap"),
        (ROOT / "supervised_correction_outputs", "**/supervised_lodestar_to_reference_lstm*.pt", "supervised_correction"),
    ]:
        for path in sorted(directory.glob(pattern)):
            if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(directory.resolve()):
                continue
            relative = str(path.relative_to(ROOT))
            entries.append(dict(id=hashlib.sha256(relative.encode()).hexdigest()[:24],
                                method=method, label=f"{METHODS[method]} — {path.stem}", checkpoint=relative))
    return entries


def resolve_model(model_id):
    for item in catalog():
        if item["id"] == model_id:
            return item, ROOT / item["checkpoint"]
    raise ValueError("Unknown or unavailable trajectory checkpoint")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_tracks(df, method):
    required = ["track_id", "frame", "x", "y", "phi", "is_interpolated"]
    if method == "supervised_correction":
        required.append("ncc")
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Tracks require columns: {', '.join(sorted(missing))}; orientation cannot be fabricated")
    if any(c in df for c in ("x_raw", "original_x", "is_model_refined", "variant_refined")):
        raise ValueError("Use baseline tracks, not an already refined trajectory")
    df = df.copy()
    flags = df.is_interpolated.astype(str).str.lower()
    if not flags.isin(["true", "false", "1", "0", "1.0", "0.0"]).all():
        raise ValueError("is_interpolated must be boolean")
    df["is_interpolated"] = flags.isin(["true", "1", "1.0"])
    for col in required:
        if col == "is_interpolated":
            continue
        df[col] = pd.to_numeric(df[col], errors="raise")
        checked = df.loc[~df.is_interpolated, col] if col == "ncc" else df[col]
        if not np.isfinite(checked.to_numpy(dtype=float)).all():
            raise ValueError(f"{col} must contain finite values (orientation is required)")
    for col in ("track_id", "frame"):
        if not (df[col] == np.floor(df[col])).all() or (df[col] < 0).any():
            raise ValueError(f"{col} must contain nonnegative integers")
        df[col] = df[col].astype(np.int64)
    if df.duplicated(["track_id", "frame"]).any():
        raise ValueError("Duplicate track_id/frame rows")
    return df.sort_values(["track_id", "frame"]).reset_index(drop=True)


def infer(df, method, ck):
    """Returns all baseline rows; causal predictions live only in pred_* fields."""
    from lstm_track_predictor import (LSTMTrackPredictor, Normalizer, STATE_COLUMNS,
                                     _add_angle_features, _add_motion_features, reconstruct_absolute)
    out = df.copy()
    counts = dict(input_rows=len(df), eligible_rows=0, refined_rows=0, prediction_rows=0)
    if method != "causal_prediction":
        for col in ("x", "y", "phi"):
            out[f"{col}_raw"] = out[col]
        out["is_model_refined"] = False
    else:
        for col in ("pred_x", "pred_y", "pred_phi"):
            out[col] = np.nan
        out["has_prediction"] = False
    if method == "bilstm_gap":
        from lstm_gap_filler import BiLSTMGapFiller, predict_gap
        from build_track_variants import _iter_usable_gaps
        model = BiLSTMGapFiller(len(ck["input_columns"]), len(ck["query_columns"]),
                               int(ck["hidden_size"]), int(ck["layers"]), float(ck["dropout"]), len(ck["target_columns"]))
        model.load_state_dict(ck["model_state"])
        model.eval()
        if len(df):
            for past, gap, future, indices in _iter_usable_gaps(df, int(ck["context_len"])):
                predictions = np.asarray(predict_gap(model, ck, Normalizer(**ck["input_normalizer"]),
                    Normalizer(**ck["query_normalizer"]), Normalizer(**ck["target_normalizer"]),
                    past, future, gap, past.iloc[-1], future.iloc[0], torch.device("cpu")))
                if not np.isfinite(predictions).all():
                    raise ValueError("Model produced nonfinite predictions")
                out.loc[indices, ["x", "y", "phi"]] = predictions
                out.loc[indices, "is_model_refined"] = True
    elif method == "supervised_correction":
        from apply_supervised_correction_lstm import add_features, build_windows
        from train_supervised_correction_lstm import ResidualCorrectionLSTM
        model = ResidualCorrectionLSTM(int(ck["input_size"]), int(ck["hidden_size"]), int(ck["layers"]), float(ck["dropout"]))
        model.load_state_dict(ck["model_state"])
        model.eval()
        real = df[~df.is_interpolated].copy()
        if len(real):
            # add_features resets the index; retain explicit mapping to all-row output.
            real["web_source_row"] = real.index
            features = add_features(real)
            windows, indices = build_windows(features, ck["feature_cols"], int(ck["seq_len"]))
            for start in range(0, len(windows), 1024):
                batch = Normalizer(**ck["input_normalizer"]).transform(windows[start:start + 1024]).astype(np.float32)
                with torch.no_grad():
                    pred = model(torch.from_numpy(batch)).numpy()
                residual = Normalizer(**ck["target_normalizer"]).inverse(pred)
                if not np.isfinite(residual).all():
                    raise ValueError("Model produced nonfinite predictions")
                rows = features.loc[indices[start:start + 1024], "web_source_row"].to_numpy(dtype=int)
                out.loc[rows, ["x", "y"]] = df.loc[rows, ["x", "y"]].to_numpy() + residual
                out.loc[rows, "is_model_refined"] = True
    elif method == "causal_prediction":
        columns = ck.get("input_columns", ck.get("feature_columns", STATE_COLUMNS))
        model = LSTMTrackPredictor(len(columns), int(ck["hidden_size"]), int(ck["layers"]), float(ck["dropout"]), 4)
        model.load_state_dict(ck["model_state"])
        model.eval()
        xn = Normalizer(**(ck.get("x_normalizer") or ck["normalizer"]))
        yn = Normalizer(**(ck.get("y_normalizer") or ck["normalizer"]))
        size = int(ck["seq_len"])
        if size < 1:
            raise ValueError("Invalid checkpoint sequence length")
        for _, group in df.groupby("track_id"):
            # Never predict across missing frames or interpolated rows.
            group = group[~group.is_interpolated]
            for _, segment in group.groupby(group.frame.diff().ne(1).cumsum()):
                segment = _add_motion_features(_add_angle_features(segment))
                for stop in range(size, len(segment)):
                    context = segment.iloc[stop-size:stop]
                    batch = xn.transform(context[columns].to_numpy(dtype=np.float32))[None].astype(np.float32)
                    with torch.no_grad():
                        prediction = yn.inverse(model(torch.from_numpy(batch)).numpy())
                    absolute = reconstruct_absolute(context[STATE_COLUMNS].to_numpy(dtype=np.float32)[None], prediction, ck.get("target_mode", "absolute"))[0]
                    values = [absolute[0], absolute[1], np.arctan2(absolute[2], absolute[3])]
                    if not np.isfinite(values).all():
                        raise ValueError("Model produced nonfinite predictions")
                    out.loc[segment.index[stop], ["pred_x", "pred_y", "pred_phi"]] = values
                    out.loc[segment.index[stop], "has_prediction"] = True
    else:
        raise ValueError("Unsupported trajectory method")
    counts["prediction_rows"] = int(out.has_prediction.sum()) if "has_prediction" in out else 0
    counts["refined_rows"] = int(out.is_model_refined.sum()) if "is_model_refined" in out else 0
    counts["eligible_rows"] = counts["prediction_rows"] + counts["refined_rows"]
    if method != "causal_prediction":
        out["x_refined"] = out.x
        out["y_refined"] = out.y
        dx, dy = out.x - out.x_raw, out.y - out.y_raw
        out["refinement_shift_px"] = np.hypot(dx, dy)
        if method == "supervised_correction":
            out["supervised_dx"] = dx
            out["supervised_dy"] = dy
            out["supervised_shift_px"] = out.refinement_shift_px
        shifts = out.loc[out.is_model_refined, "refinement_shift_px"]
        counts.update(mean_shift_px=float(shifts.mean()) if len(shifts) else 0.,
                      median_shift_px=float(shifts.median()) if len(shifts) else 0.,
                      p95_shift_px=float(shifts.quantile(.95)) if len(shifts) else 0.)
    return out, counts


def run_inference(source, model_id, output, manifest):
    item, checkpoint = resolve_model(model_id)
    # Snapshot at execution (not queue submission): hashes describe the exact
    # bytes parsed/loaded even if a user replaces the source while queued.
    source_bytes, checkpoint_bytes = source.read_bytes(), checkpoint.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
    df = validate_tracks(pd.read_csv(BytesIO(source_bytes)), item["method"])
    ck = torch.load(BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    result, counts = infer(df, item["method"], ck)
    warnings = ["Experimental inference; compare with the unchanged linear baseline before physics interpretation."]
    if item["method"] == "supervised_correction":
        warnings.append("Reference-calibrated correction, not ground-truth recovery. Transfer validity depends on the checkpoint training domain.")
    if not counts["eligible_rows"]:
        warnings.append("No eligible context: output preserves the input without model predictions or refinement.")
    info = dict(method=item["method"], output_kind="predictions" if item["method"] == "causal_prediction" else "tracks",
                output_csv=output.name, output_tracks_csv=None if item["method"] == "causal_prediction" else output.name,
                manifest_file=manifest.name, counts=counts, warnings=warnings,
                source_csv=source.name, source_sha256=source_hash, checkpoint=item["checkpoint"],
                checkpoint_sha256=checkpoint_hash, device="cpu", baseline_preserved=True,
                snapshot_semantics="input and checkpoint bytes captured at job execution")
    # Jobs advertise files only after both artifacts exist. Remove only this
    # unique job's newly created files if either write fails.
    created = []
    try:
        with output.open("x") as stream:
            created.append(output)
            result.to_csv(stream, index=False)
        with manifest.open("x") as stream:
            created.append(manifest)
            stream.write(json.dumps(info, indent=2))
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return info
