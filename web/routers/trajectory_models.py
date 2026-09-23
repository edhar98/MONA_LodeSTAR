"""Authenticated, managed-file-only optional trajectory inference jobs."""
import threading
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import state
from services.trajectory_models import METHODS, catalog, resolve_model, run_inference

router = APIRouter(tags=["trajectory-models"])
_inference_lock = threading.Lock()


class TrajectoryRequest(BaseModel):
    username: str
    tracks_csv: str
    model_id: str


@router.get("/trajectory-models/{username}")
async def list_models(username: str):
    state.require_user(username)
    return {"models": catalog(), "methods": METHODS}


def _run(job_id, source, model_id, output, manifest):
    job = state.background_jobs[job_id]
    try:
        with _inference_lock:
            job.update(status="running", progress=5)
            state.save_background_jobs()
            result = run_inference(source, model_id, output, manifest)
            job.update(status="completed", progress=100, result=result)
    except Exception as exc:
        job.update(status="failed", error=str(exc))
    finally:
        state.save_background_jobs()


@router.post("/trajectory-models/run")
async def start_inference(request: TrajectoryRequest):
    state.require_user(request.username)
    if sum(job.get("type") == "trajectory_model" and job.get("status") in ("queued", "running")
           for job in state.background_jobs.values()) >= 4:
        raise HTTPException(429, "Trajectory inference queue is full; wait for a current job to finish")
    try:
        item, _ = resolve_model(request.model_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    results = state.get_user_dir(request.username) / "results"
    source = state.contained_path(results, request.tracks_csv)
    if source.suffix.lower() != ".csv" or not source.is_file():
        raise HTTPException(404, "Managed tracks CSV not found")
    job_id = uuid.uuid4().hex
    suffix = "predictions" if item["method"] == "causal_prediction" else "tracks"
    output = state.contained_path(results, f"trajectory_{job_id}_{suffix}.csv")
    manifest = state.contained_path(results, f"trajectory_{job_id}_manifest.json")
    state.background_jobs[job_id] = dict(id=job_id, type="trajectory_model", username=request.username,
        status="queued", progress=0, input_csv=source.name, method=item["method"], created_at=datetime.now().isoformat())
    state.save_background_jobs()
    threading.Thread(target=_run, args=(job_id, source, request.model_id, output, manifest), daemon=True).start()
    return {"job_id": job_id, "status": "queued"}


@router.get("/trajectory-models/jobs/{username}/{job_id}")
async def job_status(username: str, job_id: str):
    state.require_user(username)
    job = state.background_jobs.get(job_id)
    if not job or job.get("username") != username or job.get("type") != "trajectory_model":
        raise HTTPException(404, "Trajectory job not found")
    return job
