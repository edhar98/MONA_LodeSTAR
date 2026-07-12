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

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR / "tracking"))
sys.path.insert(0, str(SRC_DIR / "analysis"))

from tdms_explorer import TDMSFileExplorer
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
# Paths and global state
# ---------------------------------------------------------------------------

WEB_DIR = Path(__file__).parent
DATA_DIR = WEB_DIR / "data"
USERS_FILE = WEB_DIR / "users.json"
JOBS_FILE = WEB_DIR / "training_jobs.json"
BG_JOBS_FILE = WEB_DIR / "background_jobs.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)

users: Dict[str, Dict[str, Any]] = {}
sessions: Dict[str, Dict[str, Any]] = {}
training_jobs: Dict[str, Dict[str, Any]] = {}
background_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()

# For WebSocket push from training thread
_event_loop: Optional[asyncio.AbstractEventLoop] = None
_ws_queues: Dict[str, asyncio.Queue] = {}


def _push_ws(job_id: str, data: dict):
    """Thread-safe push to WebSocket queue for a training job."""
    if _event_loop is not None and job_id in _ws_queues:
        asyncio.run_coroutine_threadsafe(
            _ws_queues[job_id].put(data), _event_loop
        )


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_training_jobs():
    with open(JOBS_FILE, "w") as f:
        json.dump(training_jobs, f, indent=2, default=str)


def load_training_jobs():
    global training_jobs
    if JOBS_FILE.exists():
        with open(JOBS_FILE) as f:
            training_jobs = json.load(f)
        for job in training_jobs.values():
            if job.get("status") == "running":
                job["status"] = "interrupted"


def save_background_jobs():
    with open(BG_JOBS_FILE, "w") as f:
        json.dump(background_jobs, f, indent=2, default=str)


def load_background_jobs():
    global background_jobs
    if BG_JOBS_FILE.exists():
        with open(BG_JOBS_FILE) as f:
            background_jobs = json.load(f)
        for job in background_jobs.values():
            if job.get("status") == "running":
                job["status"] = "interrupted"


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def save_users():
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2, default=str)


def load_users():
    global users
    if USERS_FILE.exists():
        with open(USERS_FILE) as f:
            users = json.load(f)


def get_user_dir(username: str) -> Path:
    user_dir = DATA_DIR / username
    for sub in ["uploads", "samples", "models", "results", "masks"]:
        (user_dir / sub).mkdir(parents=True, exist_ok=True)
    return user_dir


def save_user_session(username: str):
    user_dir = get_user_dir(username)
    if username in sessions:
        with open(user_dir / "session.json", "w") as f:
            json.dump(sessions[username], f, indent=2, default=str)


def load_user_session(username: str):
    session_file = get_user_dir(username) / "session.json"
    if session_file.exists():
        with open(session_file) as f:
            sessions[username] = json.load(f)
    else:
        sessions[username] = {
            "files": {}, "samples": {}, "models": [], "masks": {}, "detect_files": {}
        }
    # Ensure detect_files key always exists
    sessions[username].setdefault("detect_files", {})


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
    file_id: str
    model_id: str
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
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frame(file_path: Path, file_info: dict, index: int):
    if file_info["type"] == "tdms":
        normalize = file_info.get("tdms_settings", {}).get("normalize", True)
        explorer = TDMSFileExplorer(str(file_path))
        images = explorer.extract_images()
        if images is None:
            raise ValueError(f"No image data in {file_path}")
        if index < 0 or index >= len(images):
            raise ValueError(f"Frame {index} out of range")
        frame = images[index].astype(np.float32)
        if normalize:
            fmin, fmax = frame.min(), frame.max()
            frame = ((frame - fmin) / (fmax - fmin + 1e-8) * 255).astype(np.uint8) if fmax > fmin \
                else np.zeros_like(frame, dtype=np.uint8)
        else:
            if images[index].dtype == np.uint16:
                frame = (images[index] >> 8).astype(np.uint8)
            else:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
        return Image.fromarray(frame, mode="L"), len(images)
    else:
        img = Image.open(file_path)
        if img.mode != "L":
            img = img.convert("L")
        return img, 1


# ---------------------------------------------------------------------------
# Static routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    template = WEB_DIR / "templates" / "index.html"
    return template.read_text() if template.exists() else HTMLResponse("<h1>MONA Track</h1>")


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
    return {
        "status": "ok",
        "gpu": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "tracking_available": _tracking_available,
        "analysis_available": _analysis_available,
    }


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

@app.post("/auth/register")
async def register(data: UserLogin):
    if data.username in users:
        return JSONResponse(status_code=400, content={"error": "Username already exists"})
    if len(data.username) < 3:
        return JSONResponse(status_code=400, content={"error": "Username must be at least 3 characters"})
    if len(data.password) < 4:
        return JSONResponse(status_code=400, content={"error": "Password must be at least 4 characters"})
    users[data.username] = {
        "password_hash": hash_password(data.password),
        "created_at": datetime.now().isoformat(),
    }
    get_user_dir(data.username)
    sessions[data.username] = {"files": {}, "samples": {}, "models": [], "masks": {}, "detect_files": {}}
    save_users()
    save_user_session(data.username)
    return {"status": "registered", "username": data.username}


@app.post("/auth/login")
async def login(data: UserLogin):
    if data.username not in users:
        return JSONResponse(status_code=401, content={"error": "Invalid username or password"})
    if users[data.username]["password_hash"] != hash_password(data.password):
        return JSONResponse(status_code=401, content={"error": "Invalid username or password"})
    load_user_session(data.username)
    return {"status": "logged_in", "username": data.username}


@app.get("/auth/check/{username}")
async def check_user(username: str):
    if username not in users:
        return {"exists": False}
    load_user_session(username)
    return {"exists": True, "session": sessions.get(username, {})}


# ---------------------------------------------------------------------------
# Chunked upload
# ---------------------------------------------------------------------------

_ALLOWED_UPLOAD_EXT = {".tdms", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@app.post("/upload/start")
async def upload_start(data: ChunkUploadStart):
    if data.username not in users:
        return JSONResponse(status_code=401, content={"error": "User not found"})
    file_id = str(uuid.uuid4())[:8]
    ext = Path(data.filename).suffix.lower()
    if ext not in _ALLOWED_UPLOAD_EXT:
        return JSONResponse(status_code=400, content={"error": "Unsupported file type"})
    file_path = get_user_dir(data.username) / "uploads" / f"{file_id}{ext}"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.touch()
    return {"upload_id": file_id, "file_path": str(file_path), "settings": {"normalize": data.normalize}}


@app.post("/upload/chunk/{upload_id}")
async def upload_chunk(upload_id: str, request: Request, offset: int = 0):
    body = await request.body()
    # Search in sessions first
    for session in sessions.values():
        for finfo in session.get("files", {}).values():
            if finfo.get("id") == upload_id:
                with open(finfo["path"], "r+b") as f:
                    f.seek(offset)
                    f.write(body)
                return {"received": len(body), "offset": offset}
    # Fallback: scan upload dirs
    for username in users:
        user_dir = get_user_dir(username)
        for ext in _ALLOWED_UPLOAD_EXT:
            fp = user_dir / "uploads" / f"{upload_id}{ext}"
            if fp.exists():
                with open(fp, "r+b") as f:
                    f.seek(offset)
                    f.write(body)
                return {"received": len(body), "offset": offset}
    return JSONResponse(status_code=404, content={"error": "Upload not found"})


def _parse_tdms_info(file_path: Path, file_info: dict):
    try:
        explorer = TDMSFileExplorer(str(file_path))
        images = explorer.extract_images()
        if images is not None:
            file_info["frame_count"] = images.shape[0]
            file_info["width"] = images.shape[2]
            file_info["height"] = images.shape[1]
            file_info["dtype"] = str(images.dtype)
        else:
            file_info["error"] = "Could not auto-detect image dimensions"
    except Exception as e:
        file_info["error"] = str(e)
    return file_info


@app.post("/upload/complete")
async def upload_complete(data: ChunkUploadComplete):
    if data.username not in users:
        return JSONResponse(status_code=401, content={"error": "User not found"})
    if data.username not in sessions:
        load_user_session(data.username)
    ext = Path(data.filename).suffix.lower()
    file_path = get_user_dir(data.username) / "uploads" / f"{data.upload_id}{ext}"
    if not file_path.exists():
        return JSONResponse(status_code=404, content={"error": "Upload file not found"})
    file_info = {
        "id": data.upload_id, "filename": data.filename, "path": str(file_path),
        "type": "tdms" if ext == ".tdms" else "image",
        "frame_count": 1, "tdms_settings": {"normalize": data.normalize},
    }
    if ext == ".tdms":
        _parse_tdms_info(file_path, file_info)
    else:
        img = Image.open(file_path)
        file_info["width"] = img.width
        file_info["height"] = img.height
    sessions[data.username]["files"][data.upload_id] = file_info
    save_user_session(data.username)
    return file_info


@app.post("/upload")
async def upload_file(request: Request):
    try:
        form = await request.form()
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Form parsing failed: {e}"})

    username = form.get("username")
    file = form.get("file")
    normalize = form.get("normalize", "true")

    if not username:
        return JSONResponse(status_code=400, content={"error": "Missing username"})
    if not file:
        return JSONResponse(status_code=400, content={"error": "Missing file"})
    if username not in users:
        return JSONResponse(status_code=401, content={"error": "User not found"})
    if username not in sessions:
        load_user_session(username)

    normalize_bool = str(normalize).lower() in ("true", "1", "yes")
    file_id = str(uuid.uuid4())[:8]
    filename = file.filename or f"upload_{file_id}"
    ext = Path(filename).suffix.lower()
    if ext not in _ALLOWED_UPLOAD_EXT:
        return JSONResponse(status_code=400, content={"error": "Unsupported file type"})

    file_path = get_user_dir(username) / "uploads" / f"{file_id}{ext}"
    content = await file.read()
    file_path.write_bytes(content)

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

    sessions[username]["files"][file_id] = file_info
    save_user_session(username)
    return file_info


@app.post("/upload/csv")
async def upload_csv(
    username: str = Form(...),
    file: UploadFile = File(...),
    file_type: str = Form("detection"),  # "detection" | "tracks"
):
    if username not in users:
        return JSONResponse(status_code=401, content={"error": "User not found"})
    if not file.filename or not file.filename.endswith(".csv"):
        return JSONResponse(status_code=400, content={"error": "Only CSV files are accepted"})

    content = await file.read()
    user_dir = get_user_dir(username)
    save_path = user_dir / "results" / file.filename
    save_path.write_bytes(content)

    return {
        "status": "uploaded",
        "filename": file.filename,
        "size": len(content),
        "file_type": file_type,
    }


# ---------------------------------------------------------------------------
# Frame access
# ---------------------------------------------------------------------------

@app.get("/frame/{username}/{file_id}/{index}")
async def get_frame(username: str, file_id: str, index: int):
    if username not in sessions:
        load_user_session(username)
    if file_id not in sessions[username]["files"]:
        raise HTTPException(status_code=404, detail="File not found")
    file_info = sessions[username]["files"][file_id]
    img, _ = extract_frame(Path(file_info["path"]), file_info, index)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return {
        "image": f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}",
        "width": img.width,
        "height": img.height,
    }


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
    sample_dir = get_user_dir(request.username) / "samples" / request.particle_name
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_path = sample_dir / f"{request.particle_name}.jpg"
    cropped.save(sample_path, format="JPEG")
    sample_info = {"path": str(sample_path), "width": cropped.width, "height": cropped.height}
    sessions[request.username]["samples"][request.particle_name] = [sample_info]
    buf = BytesIO()
    cropped.save(buf, format="PNG")
    save_user_session(request.username)
    return {"particle_name": request.particle_name, "sample": sample_info,
            "preview": f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"}


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
            sample_dir = get_user_dir(request.username) / "samples" / request.particle_name
            sample_dir.mkdir(parents=True, exist_ok=True)
            sample_path = sample_dir / f"{request.particle_name}.jpg"
            cropped_img.save(sample_path, format="JPEG")

            sample_info = {"path": str(sample_path), "width": cropped_img.width, "height": cropped_img.height,
                           "mask_type": "polygon"}
            sessions[request.username]["samples"][request.particle_name] = [sample_info]

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
    sample_dir = get_user_dir(request.username) / "samples" / request.particle_name
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_path = sample_dir / f"{request.particle_name}.jpg"
    cropped.save(sample_path, format="JPEG")
    sample_info = {"path": str(sample_path), "width": cropped.width, "height": cropped.height, "mask_type": "circular"}
    sessions[request.username]["samples"][request.particle_name] = [sample_info]
    buf = BytesIO()
    cropped.save(buf, format="PNG")
    save_user_session(request.username)
    return {"particle_name": request.particle_name, "sample": sample_info,
            "preview": f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"}


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

        sample_path = get_user_dir(username) / "samples" / particle_name / f"{particle_name}.jpg"
        if not sample_path.exists():
            raise FileNotFoundError(f"Sample not found: {sample_path}")

        training_image = np.array(dt.LoadImage(str(sample_path)).resolve()).astype(np.float32)
        if len(training_image.shape) == 3 and training_image.shape[-1] == 3:
            training_image = np.dot(training_image[..., :3], [0.299, 0.587, 0.114])
        if len(training_image.shape) == 2:
            training_image = training_image[..., np.newaxis]

        pipeline_ops = [dt.Value(training_image)]
        if config.get("use_affine", False):
            pipeline_ops.append(dt.Affine(
                scale=lambda: np.random.uniform(config["scale_min"], config["scale_max"]),
                rotate=lambda: 2 * np.pi * np.random.uniform(config["rotation_min"], config["rotation_max"]),
                translate=lambda: np.random.uniform(config["translate_min"], config["translate_max"], 2),
                mode="constant",
            ))
        pipeline_ops.extend([
            dt.Multiply(lambda: np.random.uniform(config["mul_min"], config["mul_max"])),
            dt.Add(lambda: np.random.uniform(config["add_min"], config["add_max"])),
            dt.MoveAxis(-1, 0),
            dt.pytorch.ToTensor(dtype=torch.float32),
        ])
        pipeline = pipeline_ops[0]
        for op in pipeline_ops[1:]:
            pipeline = pipeline >> op

        dataset = dt.pytorch.Dataset(pipeline, length=config["length"], replace=False)
        loader = dl.DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)

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
    config = request.dict()
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

    detections_list = detections[:, [1, 0]].cpu().tolist() if len(detections) > 0 else []

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
        Image.fromarray(wcol, mode="RGB").save(wbuf, format="PNG")
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
    if username not in users:
        return JSONResponse(status_code=401, content={"error": "User not found"})
    if username not in sessions:
        load_user_session(username)

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
    if file_id not in detect_files:
        raise HTTPException(status_code=404, detail="File not found")
    file_info = detect_files[file_id]
    img, _ = extract_frame(Path(file_info["path"]), file_info, index)
    lodestar = load_model(username, model_id)
    result = run_detection_on_image(lodestar, img, alpha, beta, cutoff, return_weightmap)
    result["frame_index"] = index
    result["frame_count"] = file_info["frame_count"]
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


def run_batch_detection(job_id: str, username: str, file_info: dict,
                        model_id: str, params: dict, output_csv: Path):
    try:
        background_jobs[job_id]["status"] = "running"
        lodestar = load_model(username, model_id)

        frame_count = file_info.get("frame_count", 1)
        rows = []
        for i in range(frame_count):
            img, _ = extract_frame(Path(file_info["path"]), file_info, i)
            result = run_detection_on_image(
                lodestar, img, params["alpha"], params["beta"], params["cutoff"], False
            )
            for det in result["detections"]:
                rows.append({"x": det[0], "y": det[1], "phi": np.nan, "frame": i})
            background_jobs[job_id]["progress"] = int((i + 1) / frame_count * 100)
            background_jobs[job_id]["frames_done"] = i + 1

        df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["x", "y", "phi", "frame"])
        df.to_csv(output_csv)  # with default integer index — matches what test_single_particle.py writes

        background_jobs[job_id]["status"] = "completed"
        background_jobs[job_id]["output_csv"] = output_csv.name
        background_jobs[job_id]["total_detections"] = len(df)
        background_jobs[job_id]["frames"] = frame_count
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
    file_info = sessions[request.username].get("files", {}).get(request.file_id)
    if not file_info:
        file_info = sessions[request.username].get("detect_files", {}).get(request.file_id)
    if not file_info:
        raise HTTPException(status_code=404, detail="File not found in session")

    base_name = request.output_name or Path(file_info["filename"]).stem
    output_csv = get_user_dir(request.username) / "results" / f"{base_name}_detections.csv"

    job_id = str(uuid.uuid4())[:8]
    background_jobs[job_id] = {
        "id": job_id, "type": "batch_detect",
        "username": request.username,
        "status": "queued", "progress": 0,
        "frames_done": 0,
        "frames_total": file_info.get("frame_count", 1),
        "output_csv": output_csv.name,
        "created_at": datetime.now().isoformat(),
    }
    save_background_jobs()

    params = {"alpha": request.alpha, "beta": request.beta, "cutoff": request.cutoff}
    t = threading.Thread(
        target=run_batch_detection,
        args=(job_id, request.username, file_info, request.model_id, params, output_csv),
        daemon=True,
    )
    t.start()
    return {"job_id": job_id, "status": "queued", "output_csv": output_csv.name}


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


# ---------------------------------------------------------------------------
# TDMS
# ---------------------------------------------------------------------------

@app.get("/tdms/structure/{username}/{file_id}")
async def get_tdms_structure(username: str, file_id: str):
    if username not in sessions:
        load_user_session(username)
    if file_id not in sessions[username]["files"]:
        raise HTTPException(status_code=404, detail="File not found")
    file_info = sessions[username]["files"][file_id]
    if file_info["type"] != "tdms":
        raise HTTPException(status_code=400, detail="Not a TDMS file")
    explorer = TDMSFileExplorer(str(file_info["path"]))
    return {"structure": explorer.list_contents()}


@app.post("/tdms/export")
async def export_tdms(request: TdmsExportRequest):
    if request.username not in sessions:
        load_user_session(request.username)
    if request.file_id not in sessions[request.username]["files"]:
        raise HTTPException(status_code=404, detail="File not found")
    file_info = sessions[request.username]["files"][request.file_id]
    if file_info["type"] != "tdms":
        raise HTTPException(status_code=400, detail="Not a TDMS file")

    import zipfile
    explorer = TDMSFileExplorer(str(file_info["path"]))
    if explorer.extract_images() is None:
        raise HTTPException(status_code=400, detail="Could not extract images from TDMS file")

    start = request.start_frame
    end = request.end_frame
    dtype_map = {"uint8": np.uint8, "uint16": np.uint16}
    dtype = dtype_map.get(request.dtype, np.uint8)
    user_dir = get_user_dir(request.username)
    base_name = request.output_name or Path(file_info["filename"]).stem
    frame_count = (end if end is not None else explorer.extract_images().shape[0]) - start

    if request.output_format == "mp4":
        output_path = user_dir / "results" / f"{base_name}.mp4"
        explorer.write_video(str(output_path), start_frame=start, end_frame=end,
                             fps=request.fps, dtype=dtype, force=True, normed=request.normalize)
        if request.save_to_server:
            return {"status": "saved", "path": str(output_path), "frames": frame_count}
        return {"status": "ready",
                "data": base64.b64encode(output_path.read_bytes()).decode(),
                "filename": f"{base_name}.mp4", "frames": frame_count}
    else:
        export_dir = user_dir / "results" / base_name
        explorer.write_images(str(export_dir), base_name=base_name, start_frame=start,
                              end_frame=end, dtype=dtype, force=True, normed=request.normalize)
        if request.save_to_server:
            return {"status": "saved", "path": str(export_dir), "frames": frame_count}
        zip_path = user_dir / "results" / f"{base_name}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for img_file in export_dir.glob("*.png"):
                zf.write(img_file, img_file.name)
        return {"status": "ready",
                "data": base64.b64encode(zip_path.read_bytes()).decode(),
                "filename": f"{base_name}.zip", "frames": frame_count}


# ---------------------------------------------------------------------------
# File management
# ---------------------------------------------------------------------------

@app.get("/files/{username}")
async def list_user_files(username: str, file_type: Optional[str] = None):
    if username not in sessions:
        load_user_session(username)
    files = sessions[username].get("files", {})
    if file_type:
        files = {k: v for k, v in files.items() if v.get("type") == file_type}
    return {"files": files}


@app.delete("/files/{username}/{file_id}")
async def delete_file(username: str, file_id: str):
    if username not in users:
        return JSONResponse(status_code=401, content={"error": "User not found"})
    if username not in sessions:
        load_user_session(username)
    if file_id not in sessions[username].get("files", {}):
        return JSONResponse(status_code=404, content={"error": "File not found"})
    fp = Path(sessions[username]["files"][file_id]["path"])
    if fp.exists():
        try:
            fp.unlink()
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Failed to delete: {e}"})
    del sessions[username]["files"][file_id]
    save_user_session(username)
    return {"status": "deleted", "file_id": file_id}


@app.get("/results/{username}")
async def list_results(username: str):
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
async def download_result(username: str, filename: str):
    results_dir = get_user_dir(username) / "results"
    fp = results_dir / filename
    if not fp.exists() or not fp.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(fp, filename=filename)


# ---------------------------------------------------------------------------
# Video merge
# ---------------------------------------------------------------------------

@app.post("/video/merge")
async def merge_videos(request: VideoMergeRequest):
    if request.username not in users:
        raise HTTPException(status_code=401, detail="User not found")
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
        imageio.mimwrite(str(output_path), all_frames, fps=request.fps, codec="libx264", quality=8)
        return {"status": "merged", "path": str(output_path),
                "total_frames": len(all_frames), "source_files": len(mp4_files)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
        uvicorn.run(app, host="0.0.0.0", port=args.port, http="httptools", log_level="info")
    except Exception:
        uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
