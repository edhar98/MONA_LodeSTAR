import os
import sys
import json
import uuid
import time
import shutil
import asyncio
import threading
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Any
from contextlib import asynccontextmanager
from io import BytesIO
import base64

import numpy as np
import pandas as pd
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from pydantic import BaseModel

from config import SRC_DIR as _CFG_SRC
sys.path.insert(0, str(_CFG_SRC))
sys.path.insert(0, str(_CFG_SRC / "tracking"))
sys.path.insert(0, str(_CFG_SRC / "analysis"))

from services.tdms_cache import get_images
import utils

try:
    from track_particles import apply_nms, link_tracks, interpolate_gaps
    _tracking_available = True
except ImportError as _e:
    print(f"[warn] tracking module unavailable: {_e}")
    _tracking_available = False

try:
    from analyze_tracks import (
        compute_msd, compute_angular_msd, fit_msd, fit_angular_msd, plot_msd
    )
    _analysis_available = True
except ImportError as _e:
    print(f"[warn] analysis module unavailable: {_e}")
    _analysis_available = False

# ---------------------------------------------------------------------------
# Shared modules
# ---------------------------------------------------------------------------

from config import WEB_DIR, DATA_DIR, SRC_DIR, ALLOWED_UPLOAD_EXT, JUPYTER_MODE, resolve_identity
import state
from state import (
    users, sessions, training_jobs, background_jobs, jobs_lock as _jobs_lock,
    hash_password, save_users, load_users, save_training_jobs, load_training_jobs,
    save_background_jobs, load_background_jobs, get_user_dir, save_user_session,
    load_user_session, merged_dir as _merged_dir, safe_merged_name as _safe_merged_name,
    require_user, ensure_jupyter_user,
)
from services.frames import (
    extract_frame, parse_tdms_info as _parse_tdms_info,
    normalize_tdms_frame as _normalize_tdms_frame,
    normalize_tdms_stack as _normalize_tdms_stack,
    load_file_stack as _load_file_stack,
    build_session_frame_getter as _build_session_frame_getter,
)

_ALLOWED_UPLOAD_EXT = ALLOWED_UPLOAD_EXT


# ---------------------------------------------------------------------------
# App + lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _event_loop
    _event_loop = asyncio.get_event_loop()
    load_users()
    load_training_jobs()
    load_background_jobs()
    if JUPYTER_MODE:
        ensure_jupyter_user()
    yield
    save_users()
    save_training_jobs()
    save_background_jobs()


app = FastAPI(title="MONA Track", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    print(f"Validation error: {exc}")
    return JSONResponse(status_code=400, content={"error": "Validation error", "detail": str(exc)})


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    print(f"HTTP error: {exc}")
    return JSONResponse(status_code=exc.status_code, content={"error": str(exc.detail)})


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    import traceback
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"error": str(exc)})


from auth import router as auth_router
from routers.files import router as files_router
from routers.tdms_explorer import router as tdms_router

app.include_router(auth_router)
app.include_router(files_router)
app.include_router(tdms_router)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class UserLogin(BaseModel):
    username: str
    password: str


class TrainRequest(BaseModel):
    username: str
    particle_name: str
    n_transforms: int = 4
    max_epochs: int = 100
    batch_size: int = 8
    lr: float = 0.0001
    length: int = 400
    mul_min: float = 0.9
    mul_max: float = 1.1
    add_min: float = -0.1
    add_max: float = 0.1
    scale_min: float = 0.9
    scale_max: float = 1.1
    rotation_min: float = 0.0
    rotation_max: float = 1.0
    translate_min: float = -5.0
    translate_max: float = 5.0
    use_affine: bool = False


class SampleRequest(BaseModel):
    username: str
    particle_name: str
    x: int
    y: int
    width: int
    height: int
    file_id: str
    frame_index: int = 0


class MaskRequest(BaseModel):
    username: str
    particle_name: str
    mask_data: str  # base64 PNG of drawn polygon
    file_id: Optional[str] = None
    frame_index: int = 0


class TdmsSettings(BaseModel):
    normalize: bool = True


class BatchDetectRequest(BaseModel):
    username: str
    model_id: str
    file_id: Optional[str] = None
    file_ids: Optional[List[str]] = None
    alpha: float = 1.0
    beta: float = 0.0
    cutoff: float = 0.8
    output_name: Optional[str] = None


class TrackRequest(BaseModel):
    username: str
    csv_name: str        # filename in results/ dir (e.g. "foo_detections.csv")
    output_name: Optional[str] = None
    min_dist: float = 20.0
    max_link: float = 30.0
    min_track: int = 5
    max_gap: int = 10


class AbpRequest(BaseModel):
    username: str
    csv_name: str        # filename in results/ dir (e.g. "foo_tracks.csv")
    dt: float = 1 / 30
    max_lag: int = 100
    min_track: int = 50
    px_size: float = 0.078
    include_interpolated: bool = False


class CircularMaskRequest(BaseModel):
    username: str
    file_id: str
    frame_index: int = 0
    roi_center_x: float
    roi_center_y: float
    roi_radius: float
    particle_name: str


class VideoMergeRequest(BaseModel):
    username: str
    file_ids: List[str]
    output_name: str
    fps: float = 30.0


class MergeFromFilesRequest(BaseModel):
    username: str
    file_ids: List[str]
    output_name: str = "merged"
    fps: float = 30.0
    normalize: bool = True


class TrackVisualizeRequest(BaseModel):
    username: str
    tracks_csv: str
    bg_dir: Optional[str] = None
    file_ids: Optional[List[str]] = None
    fps: int = 10
    trail: int = 20


class TdmsExportRequest(BaseModel):
    username: str
    file_id: str
    output_format: str = "png"
    dtype: str = "uint8"
    normalize: bool = True
    fps: float = 30.0
    start_frame: int = 0
    end_frame: Optional[int] = None
    save_to_server: bool = False
    output_name: Optional[str] = None


class ChunkUploadStart(BaseModel):
    username: str
    filename: str
    total_size: int
    normalize: bool = True


class ChunkUploadComplete(BaseModel):
    username: str
    upload_id: str
    filename: str
    normalize: bool = True


class RenameModelRequest(BaseModel):
    new_name: str



# ---------------------------------------------------------------------------
# Static routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    template = WEB_DIR / "templates" / "index.html"
    if not template.exists():
        return HTMLResponse("<h1>MONA Track</h1>")
    return HTMLResponse(
        content=template.read_text(),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
    )


@app.get("/favicon.ico")
async def favicon():
    icon = WEB_DIR / "icon.svg"
    if icon.exists():
        return FileResponse(icon, media_type="image/svg+xml")
    raise HTTPException(status_code=404)


@app.get("/icon.svg")
async def icon():
    icon = WEB_DIR / "icon.svg"
    if icon.exists():
        return FileResponse(icon, media_type="image/svg+xml")
    raise HTTPException(status_code=404)


@app.get("/health")
async def health():
    out = {
        "status": "ok",
        "version": "beta",
        "gpu": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "tracking_available": _tracking_available,
        "analysis_available": _analysis_available,
        "mode": "jupyter" if JUPYTER_MODE else "standalone",
        "data_dir": str(DATA_DIR),
    }
    if JUPYTER_MODE:
        out["username"] = resolve_identity()
    return out


class FeedbackRequest(BaseModel):
    username: str
    message: str
    contact: str = ""
    page: str = ""


@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    username = require_user(request.username)
    message = (request.message or "").strip()
    if len(message) < 3:
        raise HTTPException(status_code=400, detail="Feedback is too short")
    if len(message) > 4000:
        raise HTTPException(status_code=400, detail="Feedback is too long (max 4000 chars)")
    entry = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "username": username,
        "contact": (request.contact or "").strip()[:200],
        "page": (request.page or "").strip()[:120],
        "message": message,
        "mode": "jupyter" if JUPYTER_MODE else "standalone",
    }
    user_dir = get_user_dir(username)
    user_path = user_dir / "feedback.jsonl"
    paths = [user_path]
    shared = os.environ.get("MONA_TRACK_FEEDBACK_DIR", "").strip()
    if shared:
        sp = Path(shared).expanduser()
        try:
            sp.mkdir(parents=True, exist_ok=True)
            paths.append(sp / "feedback.jsonl")
        except Exception:
            pass
    line = json.dumps(entry, ensure_ascii=False) + "\n"
    for fp in paths:
        with open(fp, "a", encoding="utf-8") as f:
            f.write(line)
    return {"status": "ok", "saved_to": str(user_path)}




# ---------------------------------------------------------------------------
# Sample creation
# ---------------------------------------------------------------------------

@app.post("/sample")
async def save_sample(request: SampleRequest):
    if request.username not in sessions:
        load_user_session(request.username)
    if request.file_id not in sessions[request.username]["files"]:
        raise HTTPException(status_code=404, detail="File not found")
    file_info = sessions[request.username]["files"][request.file_id]
    img, _ = extract_frame(Path(file_info["path"]), file_info, request.frame_index)
    x1, y1 = max(0, request.x), max(0, request.y)
    x2 = min(img.width, request.x + request.width)
    y2 = min(img.height, request.y + request.height)
    cropped = img.crop((x1, y1, x2, y2))
    existing = sessions[request.username]["samples"].get(request.particle_name, [])
    if existing:
        ref = existing[0]
        cropped = cropped.resize((ref["width"], ref["height"]), Image.LANCZOS)
    sample_dir = get_user_dir(request.username) / "samples" / request.particle_name
    sample_dir.mkdir(parents=True, exist_ok=True)
    crop_id = uuid.uuid4().hex[:8]
    sample_path = sample_dir / f"crop_{crop_id}.jpg"
    cropped.save(sample_path, format="JPEG")
    sample_info = {"id": crop_id, "path": str(sample_path), "width": cropped.width, "height": cropped.height}
    sessions[request.username]["samples"].setdefault(request.particle_name, []).append(sample_info)
    buf = BytesIO()
    cropped.save(buf, format="PNG")
    save_user_session(request.username)
    pool = sessions[request.username]["samples"][request.particle_name]
    return {"particle_name": request.particle_name, "sample": sample_info,
            "preview": f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}",
            "crop_count": len(pool),
            "pool_size": {"width": pool[0]["width"], "height": pool[0]["height"]}}


@app.post("/mask")
async def save_mask(request: MaskRequest):
    if request.username not in sessions:
        load_user_session(request.username)

    mask_dir = get_user_dir(request.username) / "masks" / request.particle_name
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_path = mask_dir / f"{request.particle_name}_mask.png"

    raw = request.mask_data.split(",")[1] if "," in request.mask_data else request.mask_data
    mask_bytes = base64.b64decode(raw)
    mask_img = Image.open(BytesIO(mask_bytes))
    mask_img.save(mask_path, format="PNG")

    sessions[request.username].setdefault("masks", {})[request.particle_name] = {"path": str(mask_path)}

    # Create sample from mask if a source file is given
    preview = None
    if request.file_id and request.file_id in sessions[request.username].get("files", {}):
        try:
            file_info = sessions[request.username]["files"][request.file_id]
            src_img, _ = extract_frame(Path(file_info["path"]), file_info, request.frame_index)
            src_arr = np.array(src_img)

            mask_arr = np.array(mask_img.convert("L"))
            if mask_arr.shape != src_arr.shape:
                mask_arr = cv2.resize(mask_arr, (src_arr.shape[1], src_arr.shape[0]))
            binary = (mask_arr > 128).astype(np.uint8)

            masked = (src_arr * binary).astype(np.uint8)
            ys, xs = np.where(binary)
            if len(xs) > 0:
                x1, x2 = int(xs.min()), int(xs.max()) + 1
                y1, y2 = int(ys.min()), int(ys.max()) + 1
                cropped = masked[y1:y2, x1:x2]
            else:
                cropped = masked

            cropped_img = Image.fromarray(cropped, mode="L")
            existing_mask = sessions[request.username]["samples"].get(request.particle_name, [])
            if existing_mask:
                ref = existing_mask[0]
                cropped_img = cropped_img.resize((ref["width"], ref["height"]), Image.LANCZOS)
            sample_dir = get_user_dir(request.username) / "samples" / request.particle_name
            sample_dir.mkdir(parents=True, exist_ok=True)
            crop_id = uuid.uuid4().hex[:8]
            sample_path = sample_dir / f"crop_{crop_id}.jpg"
            cropped_img.save(sample_path, format="JPEG")

            sample_info = {"id": crop_id, "path": str(sample_path), "width": cropped_img.width, "height": cropped_img.height,
                           "mask_type": "polygon"}
            sessions[request.username]["samples"].setdefault(request.particle_name, []).append(sample_info)

            buf = BytesIO()
            cropped_img.save(buf, format="PNG")
            preview = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
        except Exception as e:
            print(f"[mask] sample creation failed: {e}")

    save_user_session(request.username)
    result = {"status": "saved", "particle_name": request.particle_name}
    if preview:
        result["preview"] = preview
    return result


@app.post("/mask/circular")
async def apply_circular_mask(request: CircularMaskRequest):
    if request.username not in sessions:
        load_user_session(request.username)
    if request.file_id not in sessions[request.username]["files"]:
        raise HTTPException(status_code=404, detail="File not found")
    file_info = sessions[request.username]["files"][request.file_id]
    img, _ = extract_frame(Path(file_info["path"]), file_info, request.frame_index)
    img_arr = np.array(img)
    h, w = img_arr.shape[:2]
    y_c, x_c = np.ogrid[:h, :w]
    dist = np.sqrt((x_c - request.roi_center_x) ** 2 + (y_c - request.roi_center_y) ** 2)
    mask = dist <= request.roi_radius
    masked = np.where(mask, img_arr, 0).astype(np.uint8)
    x1 = max(0, int(request.roi_center_x - request.roi_radius))
    y1 = max(0, int(request.roi_center_y - request.roi_radius))
    x2 = min(w, int(request.roi_center_x + request.roi_radius))
    y2 = min(h, int(request.roi_center_y + request.roi_radius))
    cropped = Image.fromarray(masked[y1:y2, x1:x2], mode="L")
    existing_circ = sessions[request.username]["samples"].get(request.particle_name, [])
    if existing_circ:
        ref = existing_circ[0]
        cropped = cropped.resize((ref["width"], ref["height"]), Image.LANCZOS)
    sample_dir = get_user_dir(request.username) / "samples" / request.particle_name
    sample_dir.mkdir(parents=True, exist_ok=True)
    crop_id = uuid.uuid4().hex[:8]
    sample_path = sample_dir / f"crop_{crop_id}.jpg"
    cropped.save(sample_path, format="JPEG")
    sample_info = {"id": crop_id, "path": str(sample_path), "width": cropped.width, "height": cropped.height, "mask_type": "circular"}
    sessions[request.username]["samples"].setdefault(request.particle_name, []).append(sample_info)
    buf = BytesIO()
    cropped.save(buf, format="PNG")
    save_user_session(request.username)
    pool = sessions[request.username]["samples"][request.particle_name]
    return {"particle_name": request.particle_name, "sample": sample_info,
            "preview": f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}",
            "crop_count": len(pool),
            "pool_size": {"width": pool[0]["width"], "height": pool[0]["height"]}}


@app.get("/samples/{username}")
async def get_samples(username: str):
    if username not in sessions:
        load_user_session(username)
    return {"samples": sessions[username].get("samples", {})}


@app.delete("/sample/{username}/{particle_name}")
async def delete_sample(username: str, particle_name: str):
    if username not in sessions:
        load_user_session(username)
    if particle_name in sessions[username]["samples"]:
        sample_dir = get_user_dir(username) / "samples" / particle_name
        if sample_dir.exists():
            shutil.rmtree(sample_dir)
        del sessions[username]["samples"][particle_name]
        save_user_session(username)
    return {"status": "deleted"}


@app.delete("/sample/{username}/{particle_name}/{crop_id}")
async def delete_crop(username: str, particle_name: str, crop_id: str):
    if username not in sessions:
        load_user_session(username)
    samples = sessions[username]["samples"].get(particle_name, [])
    crop = next((s for s in samples if s.get("id") == crop_id), None)
    if crop:
        Path(crop["path"]).unlink(missing_ok=True)
        sessions[username]["samples"][particle_name] = [s for s in samples if s.get("id") != crop_id]
        if not sessions[username]["samples"][particle_name]:
            del sessions[username]["samples"][particle_name]
        save_user_session(username)
    return {"status": "deleted"}


@app.get("/sample/preview/{username}/{particle_name}/{crop_id}")
async def get_crop_preview(username: str, particle_name: str, crop_id: str):
    if username not in sessions:
        load_user_session(username)
    samples = sessions[username]["samples"].get(particle_name, [])
    crop = next((s for s in samples if s.get("id") == crop_id), None)
    if not crop:
        raise HTTPException(status_code=404)
    p = Path(crop["path"])
    if not p.exists():
        raise HTTPException(status_code=404)
    return FileResponse(str(p), media_type="image/jpeg")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training(job_id: str, username: str, particle_name: str, config: dict):
    try:
        import lightning as L
        import deeptrack as dt
        import deeptrack.deeplay as dl

        start_time = time.time()
        with _jobs_lock:
            training_jobs[job_id]["status"] = "running"
            training_jobs[job_id]["progress"] = 0
            training_jobs[job_id]["start_time"] = start_time
            training_jobs[job_id]["losses"] = []

        sample_dir = get_user_dir(username) / "samples" / particle_name
        crop_paths = sorted(sample_dir.glob("crop_*.jpg")) if sample_dir.exists() else []
        if not crop_paths:
            legacy = sample_dir / f"{particle_name}.jpg"
            if legacy.exists():
                crop_paths = [legacy]
        if not crop_paths:
            raise FileNotFoundError(f"No samples found for {particle_name}")

        def _load_crop(path):
            arr = np.array(dt.LoadImage(str(path)).resolve()).astype(np.float32)
            if len(arr.shape) == 3 and arr.shape[-1] == 3:
                arr = np.dot(arr[..., :3], [0.299, 0.587, 0.114])
            if len(arr.shape) == 2:
                arr = arr[..., np.newaxis]
            return arr

        import random as _rnd
        sample_pool = [_load_crop(p) for p in crop_paths]

        sample_fn = lambda: _rnd.choice(sample_pool).copy()
        pipeline = dt.Value(sample_fn)
        use_affine = bool(config.get("use_affine", False))
        if use_affine:
            smin = float(config.get("scale_min", 0.9))
            smax = float(config.get("scale_max", 1.1))
            rmin = float(config.get("rotation_min", 0.0))
            rmax = float(config.get("rotation_max", 1.0))
            tmin = float(config.get("translate_min", -5.0))
            tmax = float(config.get("translate_max", 5.0))
            if smax < smin:
                smin, smax = smax, smin
            if rmax < rmin:
                rmin, rmax = rmax, rmin
            if tmax < tmin:
                tmin, tmax = tmax, tmin
            pipeline = pipeline >> dt.Affine(
                scale=lambda: np.random.uniform(smin, smax),
                rotate=lambda: 2 * np.pi * np.random.uniform(rmin, rmax),
                translate=lambda: np.random.uniform(tmin, tmax, 2),
                mode="constant",
            )
        pipeline = (
            pipeline
            >> dt.Multiply(lambda: np.random.uniform(config["mul_min"], config["mul_max"]))
            >> dt.Add(lambda: np.random.uniform(config["add_min"], config["add_max"]))
            >> dt.MoveAxis(-1, 0)
            >> dt.pytorch.ToTensor(dtype=torch.float32)
        )

        dataset = dt.pytorch.Dataset(pipeline, length=config["length"], replace=False)
        loader = dl.DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0)

        lodestar = dl.LodeSTAR(n_transforms=config["n_transforms"], optimizer=dl.Adam(lr=config["lr"])).build()
        with torch.no_grad():
            _ = lodestar(torch.randn(1, 1, 64, 64))

        max_epochs = config["max_epochs"]

        class LossCallback(L.Callback):
            def __init__(self):
                self._batch_losses = []

            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                if outputs is None:
                    return
                if isinstance(outputs, dict) and "loss" in outputs:
                    self._batch_losses.append(float(outputs["loss"].detach().cpu()))
                elif isinstance(outputs, torch.Tensor):
                    self._batch_losses.append(float(outputs.detach().cpu()))

            def on_train_epoch_end(self, trainer, pl_module):
                # Cancel check
                with _jobs_lock:
                    cancel = training_jobs.get(job_id, {}).get("cancel_requested", False)
                if cancel:
                    trainer.should_stop = True
                    with _jobs_lock:
                        training_jobs[job_id]["status"] = "cancelled"
                    _push_ws(job_id, {"status": "cancelled", "job_id": job_id})
                    return

                epoch = trainer.current_epoch + 1
                avg_loss = sum(self._batch_losses) / len(self._batch_losses) if self._batch_losses else None
                self._batch_losses = []

                with _jobs_lock:
                    if avg_loss is not None:
                        training_jobs[job_id]["losses"].append(avg_loss)
                        training_jobs[job_id]["current_loss"] = avg_loss
                    training_jobs[job_id]["current_epoch"] = epoch
                    training_jobs[job_id]["progress"] = int(epoch / max_epochs * 100)
                    training_jobs[job_id]["elapsed_time"] = time.time() - start_time

                _push_ws(job_id, {
                    "status": "running",
                    "epoch": epoch,
                    "max_epochs": max_epochs,
                    "progress": int(epoch / max_epochs * 100),
                    "current_loss": avg_loss,
                    "losses": training_jobs[job_id].get("losses", [])[-20:],
                })
                save_training_jobs()

        trainer = dl.Trainer(
            max_epochs=max_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            precision="16-mixed" if torch.cuda.is_available() else "32",
            enable_progress_bar=False,
            logger=False,
            enable_checkpointing=False,
            callbacks=[LossCallback()],
        )
        trainer.fit(lodestar, loader)

        # Check if cancelled before saving
        if training_jobs.get(job_id, {}).get("status") == "cancelled":
            return

        runtime = time.time() - start_time
        model_dir = get_user_dir(username) / "models"
        model_path = model_dir / f"{particle_name}_weights.pth"
        torch.save(lodestar.state_dict(), model_path)

        losses = training_jobs[job_id].get("losses", [])
        summary = {
            "runtime_seconds": round(runtime, 2),
            "runtime_formatted": f"{int(runtime // 60)}m {int(runtime % 60)}s",
            "total_epochs": max_epochs,
            "final_loss": losses[-1] if losses else None,
            "min_loss": min(losses) if losses else None,
            "loss_history": losses[-10:] if len(losses) > 10 else losses,
            "device": "GPU" if torch.cuda.is_available() else "CPU",
        }
        model_info = {
            "id": job_id, "particle_name": particle_name, "path": str(model_path),
            "config": config, "created_at": datetime.now().isoformat(), "summary": summary,
        }
        sessions[username]["models"].append(model_info)
        save_user_session(username)

        with _jobs_lock:
            training_jobs[job_id]["status"] = "completed"
            training_jobs[job_id]["progress"] = 100
            training_jobs[job_id]["model_info"] = model_info
            training_jobs[job_id]["summary"] = summary
        _push_ws(job_id, {"status": "completed", "summary": summary})
        save_training_jobs()

    except Exception as e:
        with _jobs_lock:
            training_jobs[job_id]["status"] = "failed"
            training_jobs[job_id]["error"] = str(e)
            training_jobs[job_id]["elapsed_time"] = time.time() - training_jobs[job_id].get("start_time", time.time())
        _push_ws(job_id, {"status": "failed", "error": str(e)})
        save_training_jobs()


@app.post("/train")
async def start_training(request: TrainRequest):
    if request.username not in sessions:
        load_user_session(request.username)
    if request.particle_name not in sessions[request.username]["samples"]:
        raise HTTPException(status_code=400, detail="No sample found")

    job_id = str(uuid.uuid4())[:8]
    config = request.model_dump()
    del config["username"]
    del config["particle_name"]

    with _jobs_lock:
        training_jobs[job_id] = {
            "id": job_id, "username": request.username,
            "particle_name": request.particle_name,
            "status": "queued", "progress": 0, "config": config,
            "created_at": datetime.now().isoformat(),
        }
    save_training_jobs()

    t = threading.Thread(target=run_training, args=(job_id, request.username, request.particle_name, config), daemon=True)
    t.start()
    return {"job_id": job_id, "status": "queued"}


@app.post("/train/{job_id}/cancel")
async def cancel_training(job_id: str):
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    with _jobs_lock:
        training_jobs[job_id]["cancel_requested"] = True
    return {"status": "cancel_requested", "job_id": job_id}


@app.get("/train/active/{username}")
async def get_active_jobs(username: str):
    active = [j for j in training_jobs.values()
              if j.get("username") == username and j.get("status") in ("queued", "running")]
    return {"jobs": active}


@app.get("/train/{job_id}")
async def get_training_status(job_id: str):
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return training_jobs[job_id]


@app.websocket("/ws/train/{job_id}")
async def ws_training(websocket: WebSocket, job_id: str):
    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue()
    _ws_queues[job_id] = queue
    try:
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=30.0)
                await websocket.send_json(msg)
                if msg.get("status") in ("completed", "failed", "cancelled"):
                    break
            except asyncio.TimeoutError:
                job = training_jobs.get(job_id, {})
                if job.get("status") in ("completed", "failed", "cancelled"):
                    await websocket.send_json({"status": job["status"], "job_id": job_id})
                    break
                safe = {k: v for k, v in job.items() if isinstance(v, (str, int, float, bool, type(None)))}
                await websocket.send_json({"heartbeat": True, **safe})
    except Exception:
        pass
    finally:
        _ws_queues.pop(job_id, None)
        try:
            await websocket.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def load_model(username: str, model_id: str):
    if username not in sessions:
        load_user_session(username)
    model_info = next((m for m in sessions[username]["models"] if m["id"] == model_id), None)
    if not model_info:
        raise HTTPException(status_code=404, detail="Model not found")
    model_path = Path(model_info["path"])
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model file not found")

    import deeptrack.deeplay as dl
    config = model_info.get("config", {})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lodestar = dl.LodeSTAR(n_transforms=config.get("n_transforms", 4), optimizer=dl.Adam(lr=config.get("lr", 0.0001))).build()
    lodestar.load_state_dict(torch.load(model_path, map_location=device))
    lodestar.eval()
    lodestar = lodestar.to(device)
    return lodestar


@app.get("/models/{username}")
async def get_models(username: str):
    if username not in sessions:
        load_user_session(username)
    return {"models": sessions[username].get("models", [])}


@app.delete("/models/{username}/{model_id}")
async def delete_model(username: str, model_id: str):
    if username not in sessions:
        load_user_session(username)
    models = sessions[username].get("models", [])
    model_info = next((m for m in models if m["id"] == model_id), None)
    if not model_info:
        raise HTTPException(status_code=404, detail="Model not found")
    mp = Path(model_info["path"])
    if mp.exists():
        mp.unlink()
    sessions[username]["models"] = [m for m in models if m["id"] != model_id]
    save_user_session(username)
    return {"status": "deleted", "model_id": model_id}


@app.put("/models/{username}/{model_id}/rename")
async def rename_model(username: str, model_id: str, request: RenameModelRequest):
    if username not in sessions:
        load_user_session(username)
    models = sessions[username].get("models", [])
    model_info = next((m for m in models if m["id"] == model_id), None)
    if not model_info:
        raise HTTPException(status_code=404, detail="Model not found")
    old = Path(model_info["path"])
    new = old.parent / f"{request.new_name}_weights.pth"
    if old.exists():
        old.rename(new)
    model_info["particle_name"] = request.new_name
    model_info["path"] = str(new)
    save_user_session(username)
    return {"status": "renamed", "model_id": model_id, "new_name": request.new_name}


# ---------------------------------------------------------------------------
# Detection (single-frame and batch)
# ---------------------------------------------------------------------------

def run_detection_on_image(lodestar, img: Image.Image, alpha: float, beta: float, cutoff: float, return_weightmap: bool):
    if img.mode != "L":
        img = img.convert("L")
    image = np.array(img).astype(np.float32)
    device = next(lodestar.parameters()).device
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        model_output = lodestar(image_tensor)
        detections = lodestar.detect(image_tensor, alpha=alpha, beta=beta, mode="constant", cutoff=cutoff)[0]

    if len(detections) > 0:
        d = detections[:, [1, 0]]
        detections_list = d.cpu().tolist() if hasattr(d, 'cpu') else d.tolist()
    else:
        detections_list = []

    buf = BytesIO()
    img.save(buf, format="PNG")
    result = {
        "detections": detections_list,
        "count": len(detections_list),
        "image": f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}",
        "width": img.width,
        "height": img.height,
    }

    if return_weightmap and model_output is not None:
        if model_output.shape[1] >= 3:
            weights = model_output[0, -1].detach().cpu().numpy()
        else:
            weights = model_output[0, 0].detach().cpu().numpy()
        h, w = result["height"], result["width"]
        if weights.shape != (h, w):
            weights = cv2.resize(weights, (w, h), interpolation=cv2.INTER_LINEAR)
        wnorm = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
        wcol = (plt.cm.hot(wnorm)[:, :, :3] * 255).astype(np.uint8)
        wbuf = BytesIO()
        Image.fromarray(wcol).save(wbuf, format="PNG")
        result["weightmap"] = f"data:image/png;base64,{base64.b64encode(wbuf.getvalue()).decode()}"

    return result


@app.post("/detect/upload")
async def upload_detect_file(
    username: str = Form(None),
    file: UploadFile = File(None),
    normalize: str = Form("true"),
):
    if not username or not file:
        return JSONResponse(status_code=400, content={"error": "Missing username or file"})
    try:
        username = require_user(username)
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": str(e.detail)})

    normalize_bool = normalize.lower() in ("true", "1", "yes")
    file_id = str(uuid.uuid4())[:8]
    filename = file.filename or f"detect_{file_id}"
    ext = Path(filename).suffix.lower()
    if ext not in _ALLOWED_UPLOAD_EXT:
        return JSONResponse(status_code=400, content={"error": "Unsupported file type"})

    file_path = get_user_dir(username) / "uploads" / f"detect_{file_id}{ext}"
    file_path.write_bytes(await file.read())

    file_info = {
        "id": file_id, "filename": filename, "path": str(file_path),
        "type": "tdms" if ext == ".tdms" else "image",
        "frame_count": 1, "tdms_settings": {"normalize": normalize_bool},
    }
    if ext == ".tdms":
        _parse_tdms_info(file_path, file_info)
    else:
        img = Image.open(file_path)
        file_info["width"] = img.width
        file_info["height"] = img.height

    # Persist in session so it survives restarts
    sessions[username].setdefault("detect_files", {})[file_id] = file_info
    save_user_session(username)
    return file_info


@app.get("/detect/frame/{username}/{file_id}/{index}")
async def get_detect_frame(
    username: str, file_id: str, index: int,
    model_id: str,
    alpha: float = 1.0, beta: float = 0.0, cutoff: float = 0.8,
    return_weightmap: bool = False,
):
    if username not in sessions:
        load_user_session(username)
    detect_files = sessions[username].get("detect_files", {})
    file_info = detect_files.get(file_id) or sessions[username].get("files", {}).get(file_id)
    if not file_info:
        raise HTTPException(status_code=404, detail="File not found")
    img, frame_count = extract_frame(Path(file_info["path"]), file_info, index)
    if file_info.get("frame_count", 1) != frame_count:
        file_info["frame_count"] = frame_count
        save_user_session(username)
    lodestar = load_model(username, model_id)
    result = run_detection_on_image(lodestar, img, alpha, beta, cutoff, return_weightmap)
    result["frame_index"] = index
    result["frame_count"] = frame_count
    return result


@app.post("/detect")
async def run_detection(
    username: str = Form(...),
    model_id: str = Form(...),
    file: UploadFile = File(...),
    alpha: float = Form(1.0),
    beta: float = Form(0.0),
    cutoff: float = Form(0.8),
    return_weightmap: bool = Form(False),
):
    lodestar = load_model(username, model_id)
    content = await file.read()
    img = Image.open(BytesIO(content))
    result = run_detection_on_image(lodestar, img, alpha, beta, cutoff, return_weightmap)
    result["params"] = {"alpha": alpha, "beta": beta, "cutoff": cutoff}
    return result


def _iter_detection_frames(file_info: dict):
    stack = _load_file_stack(file_info)
    for i in range(stack.shape[0]):
        yield i, Image.fromarray(stack[i])



def run_batch_detection(job_id: str, username: str, file_infos: List[dict],
                        model_id: str, params: dict, output_csv: Path):
    try:
        background_jobs[job_id]["status"] = "running"
        save_background_jobs()
        lodestar = load_model(username, model_id)

        rows = []
        global_frame = 0
        n_files = len(file_infos)
        for fi, file_info in enumerate(file_infos):
            background_jobs[job_id]["files_done"] = fi
            background_jobs[job_id]["current_file"] = file_info.get("filename", "")
            first_frame = True
            for local_i, img in _iter_detection_frames(file_info):
                if first_frame:
                    remaining = sum(int(f.get("frame_count") or 1) for f in file_infos[fi + 1:])
                    background_jobs[job_id]["frames_total"] = (
                        global_frame + int(file_info["frame_count"]) + remaining
                    )
                    first_frame = False
                result = run_detection_on_image(
                    lodestar, img, params["alpha"], params["beta"], params["cutoff"], False
                )
                for det in result["detections"]:
                    rows.append({
                        "x": det[0], "y": det[1], "phi": np.nan,
                        "frame": global_frame,
                        "frame_local": local_i,
                        "stack": fi,
                        "source_file": file_info.get("filename", ""),
                    })
                global_frame += 1
                total = max(1, int(background_jobs[job_id].get("frames_total") or global_frame))
                background_jobs[job_id]["frames_done"] = global_frame
                background_jobs[job_id]["progress"] = int(global_frame / total * 100)
                if global_frame == 1 or global_frame % 10 == 0:
                    save_background_jobs()

            if username in sessions:
                for store in ("files", "detect_files"):
                    bucket = sessions[username].get(store, {})
                    if file_info.get("id") in bucket:
                        bucket[file_info["id"]]["frame_count"] = file_info["frame_count"]
                        bucket[file_info["id"]]["width"] = file_info.get("width")
                        bucket[file_info["id"]]["height"] = file_info.get("height")
                save_user_session(username)

        background_jobs[job_id]["files_done"] = n_files
        background_jobs[job_id]["frames_total"] = global_frame
        background_jobs[job_id]["progress"] = 100

        cols = ["x", "y", "phi", "frame", "frame_local", "stack", "source_file"]
        df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)
        df.to_csv(output_csv)

        background_jobs[job_id]["status"] = "completed"
        background_jobs[job_id]["output_csv"] = output_csv.name
        background_jobs[job_id]["total_detections"] = len(df)
        background_jobs[job_id]["frames"] = global_frame
        background_jobs[job_id]["files"] = n_files
        save_background_jobs()
    except Exception as e:
        import traceback; traceback.print_exc()
        background_jobs[job_id]["status"] = "failed"
        background_jobs[job_id]["error"] = str(e)
        save_background_jobs()


@app.post("/detect/batch")
async def detect_batch(request: BatchDetectRequest):
    if request.username not in sessions:
        load_user_session(request.username)

    file_ids: List[str] = []
    if request.file_ids:
        file_ids.extend(request.file_ids)
    if request.file_id and request.file_id not in file_ids:
        file_ids.append(request.file_id)
    if not file_ids:
        raise HTTPException(status_code=400, detail="Select at least one file")

    file_infos: List[dict] = []
    for fid in file_ids:
        file_info = sessions[request.username].get("files", {}).get(fid)
        if not file_info:
            file_info = sessions[request.username].get("detect_files", {}).get(fid)
        if not file_info:
            raise HTTPException(status_code=404, detail=f"File not found in session: {fid}")
        file_infos.append(dict(file_info))

    base_name = request.output_name or Path(file_infos[0]["filename"]).stem
    if len(file_infos) > 1 and not request.output_name:
        base_name = f"{base_name}_plus{len(file_infos) - 1}"
    output_csv = get_user_dir(request.username) / "results" / f"{base_name}_detections.csv"

    job_id = str(uuid.uuid4())[:8]
    background_jobs[job_id] = {
        "id": job_id, "type": "batch_detect",
        "username": request.username,
        "status": "queued", "progress": 0,
        "frames_done": 0,
        "frames_total": sum(int(f.get("frame_count") or 1) for f in file_infos),
        "files_done": 0,
        "files_total": len(file_infos),
        "output_csv": output_csv.name,
        "created_at": datetime.now().isoformat(),
    }
    save_background_jobs()

    params = {"alpha": request.alpha, "beta": request.beta, "cutoff": request.cutoff}
    t = threading.Thread(
        target=run_batch_detection,
        args=(job_id, request.username, file_infos, request.model_id, params, output_csv),
        daemon=True,
    )
    t.start()
    return {"job_id": job_id, "status": "queued", "output_csv": output_csv.name, "files": len(file_infos)}


# ---------------------------------------------------------------------------
# Tracking pipeline
# ---------------------------------------------------------------------------

def run_tracking_job(job_id: str, username: str, csv_path: Path, params: dict, output_csv: Path):
    try:
        if not _tracking_available:
            raise RuntimeError("Tracking module not available")
        background_jobs[job_id]["status"] = "running"

        df = pd.read_csv(csv_path, index_col=0)
        if "orientation_ncc" in df.columns and "ncc" not in df.columns:
            df = df.rename(columns={"orientation_ncc": "ncc"})
        if "ncc" not in df.columns:
            df["ncc"] = np.nan
        if "phi" not in df.columns:
            df["phi"] = np.nan
        df["frame"] = df["frame"].astype(int)

        background_jobs[job_id]["n_raw_detections"] = len(df)
        background_jobs[job_id]["n_frames"] = df["frame"].nunique()

        # NMS
        df = apply_nms(df, params["min_dist"])
        background_jobs[job_id]["n_after_nms"] = len(df)

        # Link
        tracks = link_tracks(df, params["max_link"], params["max_gap"])

        # Filter short tracks
        lengths = tracks.groupby("track_id").size()
        valid_ids = lengths[lengths >= params["min_track"]].index
        tracks = tracks[tracks["track_id"].isin(valid_ids)].reset_index(drop=True)

        # Interpolate gaps
        tracks = interpolate_gaps(tracks, params["max_gap"])

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        tracks.to_csv(output_csv, index=False)

        background_jobs[job_id]["status"] = "completed"
        background_jobs[job_id]["output_csv"] = output_csv.name
        background_jobs[job_id]["n_tracks"] = int(tracks["track_id"].nunique())
        background_jobs[job_id]["n_rows"] = len(tracks)
        background_jobs[job_id]["n_interpolated"] = int(tracks["is_interpolated"].sum())
        save_background_jobs()
    except Exception as e:
        import traceback; traceback.print_exc()
        background_jobs[job_id]["status"] = "failed"
        background_jobs[job_id]["error"] = str(e)
        save_background_jobs()


@app.post("/track")
async def run_tracking(request: TrackRequest):
    if not _tracking_available:
        raise HTTPException(status_code=503, detail="Tracking module unavailable")
    user_dir = get_user_dir(request.username)
    csv_path = user_dir / "results" / request.csv_name
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"Detection CSV not found: {request.csv_name}")

    stem = Path(request.csv_name).stem.replace("_detections", "")
    output_name = request.output_name or stem
    output_csv = user_dir / "results" / f"{output_name}_tracks.csv"

    job_id = str(uuid.uuid4())[:8]
    background_jobs[job_id] = {
        "id": job_id, "type": "tracking",
        "username": request.username,
        "status": "queued", "progress": 0,
        "input_csv": request.csv_name,
        "output_csv": output_csv.name,
        "created_at": datetime.now().isoformat(),
    }
    save_background_jobs()

    params = {
        "min_dist": request.min_dist, "max_link": request.max_link,
        "min_track": request.min_track, "max_gap": request.max_gap,
    }
    t = threading.Thread(
        target=run_tracking_job,
        args=(job_id, request.username, csv_path, params, output_csv),
        daemon=True,
    )
    t.start()
    return {"job_id": job_id, "status": "queued", "output_csv": output_csv.name}


# ---------------------------------------------------------------------------
# Track visualization
# ---------------------------------------------------------------------------

try:
    from visualize_tracks import make_overview, make_video as _viz_make_video
    _viz_available = True
except ImportError as _e:
    print(f"[warn] visualize_tracks unavailable: {_e}")
    _viz_available = False


def _resolve_bg_dir(bg_dir: Optional[str]):
    import tempfile
    if bg_dir and Path(bg_dir).is_dir():
        return bg_dir, None
    tmpdir = tempfile.mkdtemp(prefix="mona_nobg_")
    return tmpdir, tmpdir


def _session_file_infos(username: str, file_ids: List[str]) -> List[dict]:
    if username not in sessions:
        load_user_session(username)
    out: List[dict] = []
    for fid in file_ids:
        info = sessions[username].get("files", {}).get(fid)
        if not info:
            info = sessions[username].get("detect_files", {}).get(fid)
        if not info:
            raise HTTPException(status_code=404, detail=f"File not found in session: {fid}")
        out.append(dict(info))
    return out


def _resolve_viz_source(request: TrackVisualizeRequest, frames_needed: Optional[set] = None):
    if request.file_ids:
        file_infos = _session_file_infos(request.username, list(request.file_ids))
        get_frame = _build_session_frame_getter(file_infos, frames_needed=frames_needed)
        return None, None, get_frame
    images_dir, tmpdir = _resolve_bg_dir(request.bg_dir)
    return images_dir, tmpdir, None


@app.post("/tracks/visualize-overview")
def visualize_tracks_overview(request: TrackVisualizeRequest):
    if not _viz_available:
        raise HTTPException(status_code=503, detail="visualize_tracks module unavailable")
    results_dir = get_user_dir(request.username) / "results"
    tracks_path = results_dir / request.tracks_csv
    if not tracks_path.exists():
        raise HTTPException(status_code=404, detail=f"Tracks CSV not found: {request.tracks_csv}")

    df = pd.read_csv(tracks_path)
    if "is_interpolated" not in df.columns:
        df["is_interpolated"] = False
    df["is_interpolated"] = df["is_interpolated"].astype(bool)

    mid_frame = int(df["frame"].median())
    images_dir, tmpdir, get_frame = _resolve_viz_source(request, frames_needed={mid_frame})
    base = tracks_path.stem
    output_path = results_dir / f"{base}_overview.png"
    try:
        make_overview(df, images_dir or "", str(output_path), get_frame=get_frame)
    finally:
        if tmpdir:
            import shutil as _shutil
            _shutil.rmtree(tmpdir, ignore_errors=True)

    png_b64 = base64.b64encode(output_path.read_bytes()).decode()
    return {
        "status": "done",
        "image": f"data:image/png;base64,{png_b64}",
        "filename": f"{base}_overview.png",
    }


def _run_visualize_video(job_id: str, request: TrackVisualizeRequest, results_dir: Path):
    try:
        background_jobs[job_id]["status"] = "running"
        tracks_path = results_dir / request.tracks_csv
        df = pd.read_csv(tracks_path)
        if "is_interpolated" not in df.columns:
            df["is_interpolated"] = False
        df["is_interpolated"] = df["is_interpolated"].astype(bool)

        needed = set(int(f) for f in df["frame"].unique())
        images_dir, tmpdir, get_frame = _resolve_viz_source(request, frames_needed=needed)
        base = tracks_path.stem
        output_path = results_dir / f"{base}_video.mp4"
        try:
            _viz_make_video(
                df, images_dir or "", str(output_path),
                fps=request.fps, trail_frames=request.trail, get_frame=get_frame,
            )
        finally:
            if tmpdir:
                import shutil as _shutil
                _shutil.rmtree(tmpdir, ignore_errors=True)

        background_jobs[job_id]["status"] = "completed"
        background_jobs[job_id]["output_mp4"] = f"{base}_video.mp4"
        save_background_jobs()
    except Exception as e:
        background_jobs[job_id]["status"] = "failed"
        background_jobs[job_id]["error"] = str(e)
        save_background_jobs()


@app.post("/tracks/visualize-video")
async def start_visualize_video(request: TrackVisualizeRequest):
    if not _viz_available:
        raise HTTPException(status_code=503, detail="visualize_tracks module unavailable")
    results_dir = get_user_dir(request.username) / "results"
    tracks_path = results_dir / request.tracks_csv
    if not tracks_path.exists():
        raise HTTPException(status_code=404, detail=f"Tracks CSV not found: {request.tracks_csv}")

    job_id = str(uuid.uuid4())[:8]
    background_jobs[job_id] = {
        "id": job_id, "type": "visualize_video",
        "username": request.username,
        "status": "queued", "progress": 0,
        "created_at": datetime.now().isoformat(),
    }
    save_background_jobs()
    threading.Thread(target=_run_visualize_video, args=(job_id, request, results_dir), daemon=True).start()
    return {"job_id": job_id, "status": "queued"}


# ---------------------------------------------------------------------------
# Physics analysis (ABP/MSD)
# ---------------------------------------------------------------------------

@app.post("/analyze/abp")
async def analyze_abp(request: AbpRequest):
    if not _analysis_available:
        raise HTTPException(status_code=503, detail="Analysis module unavailable")
    user_dir = get_user_dir(request.username)
    csv_path = user_dir / "results" / request.csv_name
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"Tracks CSV not found: {request.csv_name}")

    tracks = pd.read_csv(csv_path)
    if "is_interpolated" not in tracks.columns:
        tracks["is_interpolated"] = False
    tracks["is_interpolated"] = tracks["is_interpolated"].astype(bool)
    if "phi" not in tracks.columns:
        tracks["phi"] = np.nan

    n_tracks = int(tracks["track_id"].nunique())
    n_real = int((~tracks["is_interpolated"]).sum())

    msd_df = compute_msd(tracks, request.max_lag, request.min_track, request.include_interpolated)
    amsd_df = compute_angular_msd(tracks, request.max_lag, request.min_track, request.include_interpolated)

    fit_params_px = fit_msd(msd_df, request.dt)
    D_r_angular = fit_angular_msd(amsd_df, request.dt)

    px = request.px_size
    msd_df["msd_um2"] = msd_df["msd"] * px ** 2

    stem = Path(request.csv_name).stem
    plot_base = stem + "_abp"
    output_dir = str(user_dir / "results")

    fit_params_phys = None
    result: Dict[str, Any] = {
        "n_tracks": n_tracks,
        "n_real_rows": n_real,
        "D_r_angular": float(D_r_angular) if D_r_angular is not None else None,
        "plot_name": f"{plot_base}_msd.png",
    }

    if fit_params_px is not None:
        D_t_px, v0_px, D_r_msd = fit_params_px
        D_t = float(D_t_px * px ** 2)
        v0 = float(v0_px * px)
        D_r_msd_f = float(D_r_msd)
        fit_params_phys = (D_t, v0, D_r_msd_f)
        result["D_t"] = D_t
        result["v0"] = v0
        result["D_r_msd"] = D_r_msd_f

    try:
        plot_msd(msd_df, amsd_df, request.dt, fit_params_phys, D_r_angular, output_dir, plot_base, px_um=px)
        plot_path = user_dir / "results" / f"{plot_base}_msd.png"
        if plot_path.exists():
            result["plot_b64"] = f"data:image/png;base64,{base64.b64encode(plot_path.read_bytes()).decode()}"
    except Exception as e:
        result["plot_error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Background job status (batch detect, tracking, etc.)
# ---------------------------------------------------------------------------

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    if job_id in background_jobs:
        return background_jobs[job_id]
    if job_id in training_jobs:
        return training_jobs[job_id]
    raise HTTPException(status_code=404, detail="Job not found")


@app.get("/jobs/user/{username}")
async def list_user_jobs(username: str):
    bg = [j for j in background_jobs.values() if j.get("username") == username]
    tr = [j for j in training_jobs.values() if j.get("username") == username]
    return {"background_jobs": bg, "training_jobs": tr}


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

@app.get("/config/defaults")
async def get_default_config():
    config_path = SRC_DIR / "config.yaml"
    defaults = {
        "n_transforms": 4, "max_epochs": 100, "batch_size": 8, "lr": 0.0001, "length": 400,
        "alpha": 1.0, "beta": 0.0, "cutoff": 0.8,
        "mul_min": 0.9, "mul_max": 1.1, "add_min": -0.1, "add_max": 0.1,
        "scale_min": 0.9, "scale_max": 1.1, "rotation_min": 0.0, "rotation_max": 1.0,
        "translate_min": -5.0, "translate_max": 5.0,
        "min_dist": 20.0, "max_link": 30.0, "min_track": 5, "max_gap": 10,
        "dt": round(1 / 30, 6), "max_lag": 100, "min_track_abp": 50, "px_size": 0.078,
    }
    if config_path.exists():
        try:
            config = utils.load_yaml(str(config_path))
            for key in defaults:
                if key in config:
                    defaults[key] = config[key]
        except Exception:
            pass
    return defaults



@app.get("/results/{username}")
async def list_results(username: str):
    username = require_user(username)
    results_dir = get_user_dir(username) / "results"
    results = []
    if results_dir.exists():
        for f in results_dir.iterdir():
            if f.is_file():
                results.append({"name": f.name, "path": str(f), "size": f.stat().st_size, "type": f.suffix})
            elif f.is_dir():
                results.append({"name": f.name, "path": str(f), "type": "folder",
                                 "png_count": len(list(f.glob("*.png")))})
    return {"results": results}


@app.get("/results/{username}/download/{filename}")
async def download_result(username: str, filename: str, inline: bool = False):
    results_dir = get_user_dir(username) / "results"
    fp = results_dir / filename
    if not fp.exists() or not fp.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    media = None
    suffix = fp.suffix.lower()
    if suffix == ".mp4":
        media = "video/mp4"
    elif suffix == ".png":
        media = "image/png"
    elif suffix in (".jpg", ".jpeg"):
        media = "image/jpeg"
    disposition = "inline" if inline or suffix == ".mp4" else "attachment"
    return FileResponse(fp, filename=filename, media_type=media, content_disposition_type=disposition)


@app.get("/results/{username}/merged")
async def list_merged_videos(username: str):
    username = require_user(username)
    merged = _merged_dir(username)
    videos = []
    for f in sorted(merged.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True):
        st = f.stat()
        videos.append({
            "name": f.name,
            "size": st.st_size,
            "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(),
        })
    return {"videos": videos}


@app.get("/results/{username}/merged/{filename}")
async def get_merged_video(username: str, filename: str):
    username = require_user(username)
    name = _safe_merged_name(filename)
    fp = _merged_dir(username) / name
    if not fp.exists() or not fp.is_file():
        raise HTTPException(status_code=404, detail="Merged video not found")
    return FileResponse(
        fp, filename=name, media_type="video/mp4", content_disposition_type="inline",
    )


@app.delete("/results/{username}/merged/{filename}")
async def delete_merged_video(username: str, filename: str):
    username = require_user(username)
    name = _safe_merged_name(filename)
    fp = _merged_dir(username) / name
    if not fp.exists() or not fp.is_file():
        raise HTTPException(status_code=404, detail="Merged video not found")
    fp.unlink()
    return {"status": "deleted", "name": name}


# ---------------------------------------------------------------------------
# Video merge
# ---------------------------------------------------------------------------

@app.post("/video/merge")
async def merge_videos(request: VideoMergeRequest):
    request.username = require_user(request.username)
    results_dir = get_user_dir(request.username) / "results"
    mp4_files = []
    for fid in request.file_ids:
        mp4 = results_dir / f"{fid}.mp4"
        if mp4.exists():
            mp4_files.append(mp4)
        else:
            for mp4 in results_dir.glob(f"*{fid}*.mp4"):
                mp4_files.append(mp4)
                break
    if not mp4_files:
        raise HTTPException(status_code=404, detail="No video files found")
    try:
        import imageio
        all_frames = []
        for mp4_file in mp4_files:
            reader = imageio.get_reader(str(mp4_file))
            for frame in reader:
                all_frames.append(frame)
            reader.close()
        output_path = results_dir / f"{request.output_name}.mp4"
        imageio.mimwrite(str(output_path), all_frames, fps=request.fps, codec="libx264", quality=8, macro_block_size=1)
        return {"status": "merged", "path": str(output_path),
                "total_frames": len(all_frames), "source_files": len(mp4_files)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/video/merge-from-files")
async def merge_from_files(request: MergeFromFilesRequest):
    request.username = require_user(request.username)
    if not request.file_ids:
        raise HTTPException(status_code=400, detail="No files selected")

    out_name = _safe_merged_name(request.output_name or "merged")
    output_path = _merged_dir(request.username) / out_name

    job_id = str(uuid.uuid4())[:8]
    background_jobs[job_id] = {
        "id": job_id, "type": "merge_video",
        "username": request.username,
        "status": "queued", "progress": 0,
        "files_done": 0, "files_total": len(request.file_ids),
        "frames_done": 0, "output_mp4": out_name, "filename": out_name,
        "created_at": datetime.now().isoformat(),
    }
    save_background_jobs()

    t = threading.Thread(
        target=_run_merge_job,
        args=(job_id, request.username, list(request.file_ids),
              output_path, request.fps, request.normalize),
        daemon=True,
    )
    t.start()
    return {"job_id": job_id, "status": "queued", "filename": out_name}


def _run_merge_job(job_id: str, username: str, file_ids: List[str],
                   output_path: Path, fps: float, normalize: bool):
    try:
        import imageio
        background_jobs[job_id]["status"] = "running"
        save_background_jobs()

        if username not in sessions:
            load_user_session(username)

        writer = None
        total_frames = 0
        skipped = []
        n_files = len(file_ids)

        for fi, fid in enumerate(file_ids):
            file_info = (sessions[username].get("files", {}).get(fid) or
                         sessions[username].get("detect_files", {}).get(fid))
            if not file_info:
                skipped.append(fid)
                continue
            try:
                if file_info["type"] == "tdms":
                    images = get_images(str(file_info["path"]))
                    if images is None:
                        skipped.append(fid)
                        continue
                    for raw in images:
                        frame = raw.astype(np.float32)
                        if normalize:
                            lo, hi = frame.min(), frame.max()
                            frame = ((frame - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8) if hi > lo \
                                else np.zeros_like(frame, dtype=np.uint8)
                        else:
                            frame = np.clip(frame, 0, 255).astype(np.uint8)
                        rgb = np.stack([frame, frame, frame], axis=-1)
                        if writer is None:
                            writer = imageio.get_writer(
                                str(output_path), fps=fps, codec="libx264",
                                quality=8, macro_block_size=1, format="FFMPEG",
                            )
                        writer.append_data(rgb)
                        total_frames += 1
                        if total_frames % 25 == 0:
                            background_jobs[job_id]["frames_done"] = total_frames
                else:
                    img, _ = extract_frame(Path(file_info["path"]), file_info, 0)
                    rgb = np.array(img.convert("RGB"))
                    if writer is None:
                        writer = imageio.get_writer(
                            str(output_path), fps=fps, codec="libx264",
                            quality=8, macro_block_size=1, format="FFMPEG",
                        )
                    writer.append_data(rgb)
                    total_frames += 1
            except Exception as e:
                skipped.append(f"{fid}:{e}")

            background_jobs[job_id]["files_done"] = fi + 1
            background_jobs[job_id]["progress"] = int((fi + 1) / n_files * 100)
            background_jobs[job_id]["frames_done"] = total_frames
            save_background_jobs()

        if writer is not None:
            writer.close()

        if total_frames == 0 or not output_path.exists():
            raise RuntimeError(f"No frames written. Skipped: {skipped}")

        background_jobs[job_id]["status"] = "completed"
        background_jobs[job_id]["progress"] = 100
        background_jobs[job_id]["total_frames"] = total_frames
        background_jobs[job_id]["source_files"] = n_files - len(skipped)
        background_jobs[job_id]["filename"] = output_path.name
        background_jobs[job_id]["path"] = str(output_path)
        save_background_jobs()
    except Exception as e:
        import traceback
        traceback.print_exc()
        background_jobs[job_id]["status"] = "failed"
        background_jobs[job_id]["error"] = str(e)
        save_background_jobs()
        try:
            if output_path.exists():
                output_path.unlink()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Test upload (dev/debug)
# ---------------------------------------------------------------------------

@app.post("/test-upload")
async def test_upload(file: UploadFile = File(None)):
    if file:
        content = await file.read()
        return {"filename": file.filename, "size": len(content)}
    return {"error": "no file"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    try:
        uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
    except Exception:
        uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
