#!/usr/bin/env python3
"""Opt-in real-model Hub API smoke; only temporary output, no training/network.

Run from repo root: /opt/mona_jupyterhub_env/bin/python -B
    test/integration/smoke_hub_pipeline.py
Requires the local 5m4rtzfx weights and dataset-04 first three PNG frames.
This is API/model verification, not browser or scientific-fit validation.
"""
import asyncio
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
SMOKE_USER = "pipeline-smoke"


def main():
    def timed_out(signum, frame):
        print("FAIL: pipeline smoke exceeded 180 seconds", flush=True)
        # Force-bound a stalled AnyIO executor during asyncio shutdown. Only
        # this run's /tmp tree may remain on timeout; normal runs remove it.
        os._exit(124)

    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(180)
    # Environment is set before importing any app module: lifespan writes only
    # under this new directory, never the installed or checkout user state.
    with tempfile.TemporaryDirectory(prefix="mona-hub-pipeline-") as temporary:
        storage = Path(temporary)
        os.environ.update(MONA_TRACK_JUPYTER="1", MONA_TRACK_USER=SMOKE_USER,
                          MONA_TRACK_HOME=str(storage / "state"),
                          MONA_TRACK_FEEDBACK_DIR=str(storage / "feedback"),
                          MPLCONFIGDIR=str(storage / "matplotlib"), CUDA_VISIBLE_DEVICES="")
        sys.path.insert(0, str(ROOT))
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        import httpx
        from web import app as api

        async def run():
            checks = {}
            async with api.lifespan(api.app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://smoke") as client:
                    async def request(method, path, **kwargs):
                        response = await client.request(method, path, **kwargs)
                        assert response.status_code == 200, (path, response.status_code, response.text[:500])
                        return response

                    async def wait_job(job):
                        deadline = time.monotonic() + 90
                        while time.monotonic() < deadline:
                            status = (await request("GET", f"/jobs/{job['job_id']}")).json()
                            if status["status"] in ("completed", "failed", "interrupted"):
                                assert status["status"] == "completed", status
                                return status
                            await asyncio.sleep(.1)
                        raise AssertionError(f"job did not finish: {job}")

                    health = (await request("GET", "/health")).json()
                    assert health["mode"] == "jupyter" and health["username"] == SMOKE_USER
                    assert Path(health["data_dir"]).is_relative_to(storage)
                    identity = (await request("GET", "/auth/me")).json()
                    assert identity["username"] == SMOKE_USER
                    assert (await client.get("/auth/check/someone-else")).status_code == 403
                    checks["identity"] = "isolated Hub identity and cross-user rejection"
                    print("PASS: lifespan and identity", flush=True)

                    images = [ROOT / f"data/JP_FE/wf_2_40/04/images/JP_Fe_wf_2_40_slm075_574_{i:03d}.png" for i in range(1, 4)]
                    assert all(path.is_file() for path in images), "required real frames unavailable"
                    uploaded = (await request("POST", "/upload", data={"username": SMOKE_USER},
                                              files={"file": (images[0].name, images[0].read_bytes(), "image/png")})).json()
                    sample = (await request("POST", "/sample", json=dict(username=SMOKE_USER, particle_name="smoke",
                                           file_id=uploaded["id"], x=0, y=0, width=64, height=64))).json()
                    assert sample["crop_count"] == 1
                    loaded_ids = []
                    for path in images:
                        loaded = (await request("POST", "/files/load-path", json={"username": SMOKE_USER, "path": str(path)})).json()
                        loaded_ids.append(loaded["files"][0]["id"])
                    checks["input"] = "multipart PNG upload, sample crop, three read-only server frames"
                    print("PASS: upload, sample and server-path input", flush=True)

                    models = (await request("GET", f"/models/{SMOKE_USER}")).json()["models"]
                    model = next(m for m in models if m.get("run_id") == "5m4rtzfx" and m["particle_name"] == "JP_Fe_wf_2_40")
                    assert model["config_source"] == "saved_run"
                    job = (await request("POST", "/detect/batch", json=dict(username=SMOKE_USER, model_id=model["id"],
                                          file_ids=loaded_ids, output_name="smoke", cutoff=.8))).json()
                    detection = await wait_job(job)
                    assert detection["frames"] == 3 and detection["total_detections"] > 0
                    checks["detection"] = {"frames": 3, "detections": detection["total_detections"], "model": model["id"], "config_source": model["config_source"]}
                    print("PASS: real model batch detection", checks["detection"], flush=True)
                    template = ROOT / "data/Samples/JP_Fe_wf_2_40/Samples/f000_d003_phi0234.0.png"
                    assert template.is_file(), "required orientation template unavailable"
                    oriented = (await request("GET", f"/detect/frame/{SMOKE_USER}/{loaded_ids[0]}/0", params=dict(
                        model_id=model["id"], detection_mode="template", template_path=str(template),
                        template_angle_step=10, template_search_radius=2))).json()
                    assert oriented["count"] > 0
                    assert len(oriented["phi"]) == len(oriented["orientation_ncc"]) == oriented["count"]
                    assert all(len(row) == 4 for row in oriented["detections"])
                    assert all(math.isfinite(v) for row in oriented["detections"] for v in row)
                    checks["orientation"] = {"detections": oriented["count"], "angle_step_degrees": 10,
                                             "search_radius_px": 2, "schema": "x,y,phi,orientation_ncc finite"}
                    print("PASS: template orientation schema", flush=True)

                    job = (await request("POST", "/track", json=dict(username=SMOKE_USER, csv_name=detection["output_csv"],
                                          output_name="smoke", min_track=2))).json()
                    tracking = await wait_job(job)
                    assert tracking["n_tracks"] > 0
                    analysis = (await request("POST", "/analyze/abp", json=dict(username=SMOKE_USER,
                                               csv_name=tracking["output_csv"], min_track=2, max_lag=2))).json()
                    assert "plot_error" not in analysis and analysis.get("plot_b64")
                    checks["tracking"] = {"tracks": tracking["n_tracks"], "rows": tracking["n_rows"]}
                    checks["analysis"] = "ABP request and plot; deliberately insufficient lags for physical inference"
                    print("PASS: tracking and ABP plot", flush=True)
                    # FileResponse ASGI streaming uses AnyIO worker threads,
                    # which stall in this restricted harness. Verify the real
                    # route's response/attachment and generated bytes directly.
                    export = await api.download_result(SMOKE_USER, tracking["output_csv"])
                    exported = Path(export.path).read_bytes()
                    assert b"track_id,frame,x,y" in exported
                    assert "attachment" in export.headers["content-disposition"]
                    checks["export"] = {"bytes": len(exported), "route_response": "attachment", "http_streaming": "not verified"}
                    print("PASS: CSV export response and bytes", flush=True)
            for name in ("users.json", "training_jobs.json", "background_jobs.json", "session.json"):
                assert (storage / "state" / name).is_file(), name
            async with api.lifespan(api.app):
                assert any(j.get("status") == "completed" for j in api.background_jobs.values())
            checks["restart"] = "completed jobs and user session survive second lifespan"
            return checks

        checks = asyncio.run(run())
        print(json.dumps({"source": str(Path(api.__file__).resolve()), "checks": checks,
                          "limitations": ["No browser automation or HTTP FileResponse streaming", "No TDMS stack", "No training", "Three frames do not validate physical estimates; coarse template setting verifies schema, not orientation accuracy"]}, indent=2))
    signal.alarm(0)


if __name__ == "__main__":
    main()
