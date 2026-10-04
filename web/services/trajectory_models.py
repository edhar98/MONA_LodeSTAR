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
    "bilstm_gap": "Two-sided LSTM gap refinement (experimental)",
}


def catalog():
    entries = []
    for directory, pattern, method in [
        (ROOT / "lstm_outputs", "lstm_gap_filler*.pt", "bilstm_gap"),
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
    if method not in METHODS:
        raise ValueError("Unsupported trajectory method; only optional bilstm_gap refinement is available")
    required = ["track_id", "frame", "x", "y", "phi", "is_interpolated"]
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
        if not np.isfinite(df[col].to_numpy(dtype=float)).all():
            raise ValueError(f"{col} must contain finite values (orientation is required)")
    for col in ("track_id", "frame"):
        if not (df[col] == np.floor(df[col])).all() or (df[col] < 0).any():
            raise ValueError(f"{col} must contain nonnegative integers")
        df[col] = df[col].astype(np.int64)
    if df.duplicated(["track_id", "frame"]).any():
        raise ValueError("Duplicate track_id/frame rows")
    return df.sort_values(["track_id", "frame"]).reset_index(drop=True)


def infer(df, method, ck):
    """Refine only eligible interpolated gaps; retain every measured observation."""
    if method not in METHODS:
        raise ValueError("Unsupported trajectory method; only optional bilstm_gap refinement is available")
    from lstm_track_predictor import Normalizer
    from lstm_gap_filler import BiLSTMGapFiller, predict_gap
    from build_track_variants import _iter_usable_gaps

    out = df.copy()
    for col in ("x", "y", "phi"):
        out[f"{col}_raw"] = out[col]
    out["is_model_refined"] = False
    context_len = int(ck["context_len"])
    if context_len < 1:
        raise ValueError("Invalid checkpoint context length")
    model = BiLSTMGapFiller(len(ck["input_columns"]), len(ck["query_columns"]),
                           int(ck["hidden_size"]), int(ck["layers"]), float(ck["dropout"]), len(ck["target_columns"]))
    model.load_state_dict(ck["model_state"])
    model.eval()
    if len(df):
        for past, gap, future, indices in _iter_usable_gaps(df, context_len):
            predictions = np.asarray(predict_gap(model, ck, Normalizer(**ck["input_normalizer"]),
                Normalizer(**ck["query_normalizer"]), Normalizer(**ck["target_normalizer"]),
                past, future, gap, past.iloc[-1], future.iloc[0], torch.device("cpu")))
            if not np.isfinite(predictions).all():
                raise ValueError("Model produced nonfinite predictions")
            out.loc[indices, ["x", "y", "phi"]] = predictions
            out.loc[indices, "is_model_refined"] = True
    refined = int(out.is_model_refined.sum())
    out["x_refined"] = out.x
    out["y_refined"] = out.y
    out["refinement_shift_px"] = np.hypot(out.x - out.x_raw, out.y - out.y_raw)
    shifts = out.loc[out.is_model_refined, "refinement_shift_px"]
    counts = dict(input_rows=len(df), interpolated_rows=int(df.is_interpolated.sum()),
                  eligible_rows=refined, refined_rows=refined, prediction_rows=0,
                  mean_shift_px=float(shifts.mean()) if len(shifts) else 0.,
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
    if not counts["eligible_rows"]:
        warnings.append("No eligible context: output preserves the input without model predictions or refinement.")
    info = dict(method=item["method"], output_kind="tracks",
                output_csv=output.name, output_tracks_csv=output.name,
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
