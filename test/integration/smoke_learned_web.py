#!/usr/bin/env python3
"""Opt-in real-checkpoint API smoke; temporary state, no training or deployment."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import sys
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
USER = "learned-smoke"


def main():
    def timeout(*_):
        print("FAIL: learned integration smoke exceeded 180 seconds", flush=True)
        os._exit(124)

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(180)
    with tempfile.TemporaryDirectory(prefix="mona-learned-smoke-") as directory:
        storage = Path(directory)
        os.environ.update(MONA_TRACK_JUPYTER="1", MONA_TRACK_USER=USER,
                          MONA_TRACK_PROXY_TOKEN="learned-smoke-proxy-token",
                          MONA_TRACK_HOME=str(storage / "state"),
                          MONA_TRACK_FEEDBACK_DIR=str(storage / "feedback"),
                          MPLCONFIGDIR=str(storage / "matplotlib"), CUDA_VISIBLE_DEVICES="")
        sys.path.insert(0, str(ROOT))
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        import httpx
        import numpy as np
        import pandas as pd
        from web import app as api

        async def run():
            checks = {}
            async with api.lifespan(api.app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://smoke",
                                            headers={"X-Mona-Proxy-Token": "learned-smoke-proxy-token"}) as client:
                    assert (await client.get("/health", headers={"X-Mona-Proxy-Token": "wrong"})).status_code == 403
                    async def request(method, path, **kwargs):
                        response = await client.request(method, path, **kwargs)
                        assert response.status_code == 200, (path, response.status_code, response.text[:1000])
                        return response.json()

                    async def wait(job):
                        deadline = time.monotonic() + 90
                        while time.monotonic() < deadline:
                            status = await request("GET", f"/jobs/{job['job_id']}")
                            if status["status"] in ("completed", "failed", "interrupted"):
                                assert status["status"] == "completed", status
                                return status
                            await asyncio.sleep(.1)
                        raise AssertionError(f"job timed out: {job}")

                    assert (await request("GET", f"/models/{USER}"))["models"] == []
                    # These exact real-model fixtures replace former CLI
                    # discovery; only temporary per-user copies are exposed.
                    for name, run in (("Janus", "euk2wnni"), ("Rod", "uaqqndn3")):
                        source = ROOT / "models" / run
                        fixture = api.get_user_dir(USER) / "models" / run
                        fixture.mkdir()
                        weights = fixture / f"{name}_weights.pth"
                        shutil.copyfile(source / weights.name, weights)
                        shutil.copyfile(source / "config.yaml", fixture / "config.yaml")
                        config = api.utils.load_yaml(str(fixture / "config.yaml"))
                        api.sessions[USER]["models"].append(dict(id=f"smoke-web-{name}", particle_name=name,
                            path=str(weights), config=config, source="web"))
                    api.save_user_session(USER)
                    models = (await request("GET", f"/models/{USER}"))["models"]
                    assert len(models) == 2 and all(m["source"] == "web" and not m["id"].startswith("cli:") for m in models)
                    selected = []
                    for name in ("Janus", "Rod"):
                        selected.append(next(m for m in models if m["particle_name"] == name
                                             and m.get("config", {}).get("lodestar_version", "default") == "default"))
                    frame = ROOT / "data/JP_FE/wf_2_40/04/images/JP_Fe_wf_2_40_slm075_574_001.png"
                    loaded = await request("POST", "/files/load-path", json={"username": USER, "path": str(frame)})
                    file_id = loaded["files"][0]["id"]
                    job = await request("POST", "/detect/batch", json={"username": USER,
                        "model_ids": [m["id"] for m in selected], "file_ids": [file_id], "output_name": "composite_smoke"})
                    detection = await wait(job)
                    results = api.get_user_dir(USER) / "results"
                    detected = pd.read_csv(results / detection["output_csv"])
                    assert {"particle_type", "confidence", "model_id"} <= set(detected)
                    assert len(detected) and np.isfinite(detected[["x", "y", "confidence"]]).all().all()
                    tracked = await wait(await request("POST", "/track", json={"username": USER,
                        "csv_name": detection["output_csv"], "min_track": 1, "output_name": "composite_smoke"}))
                    track_df = pd.read_csv(results / tracked["output_csv"])
                    assert "particle_type" in track_df and track_df.groupby("track_id").particle_type.nunique().max() == 1
                    analysis = await request("POST", "/analyze/abp", json={"username": USER,
                        "csv_name": tracked["output_csv"], "min_track": 1, "max_lag": 1})
                    if track_df.particle_type.nunique() > 1:
                        assert "pools multiple particle classes" in analysis["analysis_note"]
                    checks["composite"] = {"models": [m["id"] for m in selected], "detections": len(detected),
                                           "tracks": len(track_df), "classes": sorted(detected.particle_type.unique())}
                    print("PASS: real composite detection and class-aware tracking", flush=True)

                    source = ROOT / "detection_results/JP_FE/wf_2_40/JP_Fe_wf_2_40_5m4rtzfx/04/tracks/JP_Fe_wf_2_40_slm075_tracks.csv"
                    raw = pd.read_csv(source)
                    raw = raw[raw.track_id == 233].copy()
                    assert len(raw) > 20 and raw.is_interpolated.any()
                    upload = await request("POST", "/upload/csv", data={"username": USER, "file_type": "tracks"},
                        files={"file": ("real_track233_tracks.csv", raw.to_csv(index=False).encode(), "text/csv")})
                    input_path = results / upload["filename"]
                    original_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
                    catalog = await request("GET", f"/trajectory-models/{USER}")
                    assert (await client.get("/trajectory-models/someone-else")).status_code == 403
                    assert set(catalog["methods"]) == {"bilstm_gap"}
                    assert {m["method"] for m in catalog["models"]} == {"bilstm_gap"}
                    legacy = next(m for m in catalog["models"] if m["checkpoint"] == "lstm_outputs/lstm_gap_filler_jp_fe_wf_2_40_slm075.pt")
                    assert not legacy["compatible"] and "retrain" in legacy["compatibility_error"].lower()
                    legacy_path = ROOT / legacy["checkpoint"]
                    legacy_hash = hashlib.sha256(legacy_path.read_bytes()).hexdigest()
                    rejected = await client.post("/trajectory-models/run", json={"username": USER,
                        "tracks_csv": input_path.name, "model_id": legacy["id"]})
                    assert rejected.status_code == 400 and "retrain" in rejected.text.lower(), rejected.text
                    assert hashlib.sha256(legacy_path.read_bytes()).hexdigest() == legacy_hash
                    checks["legacy_checkpoint"] = "Real unversioned checkpoint explicitly rejected; retraining required; bytes preserved"
                    print("PASS: real legacy checkpoint rejection", flush=True)

                    # An explicitly synthetic, untrained v2 fixture exercises
                    # corrected inference without relabelling legacy weights.
                    from lstm_gap_filler import BiLSTMGapFiller, INPUT_COLUMNS, QUERY_COLUMNS, TARGET_COLUMNS, PREPROCESSING_VERSION
                    from services import trajectory_models as service
                    fixture_dir = storage / "lstm_outputs"
                    fixture_dir.mkdir()
                    tiny = BiLSTMGapFiller(len(INPUT_COLUMNS), len(QUERY_COLUMNS), 4, 1, 0., len(TARGET_COLUMNS))
                    for parameter in tiny.parameters():
                        torch.nn.init.zeros_(parameter)
                    normalizer = lambda columns: dict(mean=[0.] * len(columns), std=[1.] * len(columns))
                    torch.save(dict(preprocessing_version=PREPROCESSING_VERSION, context_len=10,
                        hidden_size=4, layers=1, dropout=0., input_columns=INPUT_COLUMNS,
                        query_columns=QUERY_COLUMNS, target_columns=TARGET_COLUMNS,
                        input_normalizer=normalizer(INPUT_COLUMNS), query_normalizer=normalizer(QUERY_COLUMNS),
                        target_normalizer=normalizer(TARGET_COLUMNS), model_state=tiny.state_dict()),
                        fixture_dir / "lstm_gap_filler_synthetic_v2.pt")
                    for method in ("bilstm_gap",):
                        with patch.object(service, "ROOT", storage):
                            temporary_catalog = await request("GET", f"/trajectory-models/{USER}")
                            model = next(m for m in temporary_catalog["models"] if m["method"] == method and m["compatible"])
                            job = await request("POST", "/trajectory-models/run", json={"username": USER,
                                "tracks_csv": input_path.name, "model_id": model["id"]})
                            status = await wait(job)
                        result = status["result"]
                        output = pd.read_csv(results / result["output_csv"])
                        assert len(output), (method, result)
                        assert (results / result["manifest_file"]).is_file()
                        assert result["counts"]["eligible_rows"] > 0, result
                        assert result["source_sha256"] == original_hash
                        assert result["preprocessing_version"] == PREPROCESSING_VERSION
                        assert len(output) == len(raw)
                        assert {"x_raw", "y_raw"} <= set(output)
                        assert np.isfinite(output[["x", "y"]]).all().all()
                        baseline = raw.sort_values(["track_id", "frame"]).reset_index(drop=True)
                        np.testing.assert_allclose(output[["x_raw", "y_raw"]], baseline[["x", "y"]])
                        measured = ~baseline.is_interpolated
                        np.testing.assert_allclose(output.loc[measured, ["x", "y", "phi"]],
                                                   baseline.loc[measured, ["x", "y", "phi"]])
                        assert not output.loc[measured, "is_model_refined"].any()
                        assert hashlib.sha256(input_path.read_bytes()).hexdigest() == original_hash
                        checks[method] = result
                        print(f"PASS: synthetic v2 checkpoint {method} on real tracks (not an accuracy benchmark)", flush=True)
            return checks

        checks = asyncio.run(run())
        print(json.dumps({"source": str(Path(api.__file__).resolve()), "checks": checks,
                          "limits": "API/schema smoke, not browser acceptance or scientific validation"}, indent=2))
    signal.alarm(0)


if __name__ == "__main__":
    main()
