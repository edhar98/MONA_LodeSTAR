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
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends, Cookie
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from tdms_to_png import extract_images_from_tdms
import utils

WEB_DIR = Path(__file__).parent
DATA_DIR = WEB_DIR / "data"
USERS_FILE = WEB_DIR / "users.json"

for d in [DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

users: Dict[str, Dict[str, Any]] = {}
sessions: Dict[str, Dict[str, Any]] = {}
training_jobs: Dict[str, Dict[str, Any]] = {}
JOBS_FILE = WEB_DIR / "training_jobs.json"

def save_training_jobs():
    with open(JOBS_FILE, "w") as f:
        json.dump(training_jobs, f, indent=2, default=str)

def load_training_jobs():
    global training_jobs
    if JOBS_FILE.exists():
        with open(JOBS_FILE, "r") as f:
            training_jobs = json.load(f)
        for job_id in list(training_jobs.keys()):
            if training_jobs[job_id].get("status") == "running":
                training_jobs[job_id]["status"] = "interrupted"

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def save_users():
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2, default=str)

def load_users():
    global users
    if USERS_FILE.exists():
        with open(USERS_FILE, "r") as f:
            users = json.load(f)

def get_user_dir(username: str) -> Path:
    user_dir = DATA_DIR / username
    for subdir in ["uploads", "samples", "models", "results", "masks"]:
        (user_dir / subdir).mkdir(parents=True, exist_ok=True)
    return user_dir

def save_user_session(username: str):
    user_dir = get_user_dir(username)
    session_file = user_dir / "session.json"
    if username in sessions:
        with open(session_file, "w") as f:
            json.dump(sessions[username], f, indent=2, default=str)

def load_user_session(username: str):
    user_dir = get_user_dir(username)
    session_file = user_dir / "session.json"
    if session_file.exists():
        with open(session_file, "r") as f:
            sessions[username] = json.load(f)
    else:
        sessions[username] = {"files": {}, "samples": {}, "models": [], "masks": {}}

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
    mask_data: str

class TdmsSettings(BaseModel):
    image_width: int = 1024
    image_height: int = 1024
    channel_index: int = 0
    group_name: Optional[str] = None
    normalize: bool = True

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_users()
    load_training_jobs()
    yield
    save_users()
    save_training_jobs()

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
    print(f"General error: {exc}")
    import traceback
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"error": str(exc)})

def extract_frame(file_path: Path, file_info: dict, index: int):
    if file_info["type"] == "tdms":
        settings = file_info.get("tdms_settings", {})
        width = settings.get("image_width", 1024)
        height = settings.get("image_height", 1024)
        channel = settings.get("channel_index", 0)
        group = settings.get("group_name")
        normalize = settings.get("normalize", True)
        
        images = extract_images_from_tdms(file_path, image_width=width, image_height=height, channel_index=channel, group_name=group)
        
        if index < 0 or index >= len(images):
            raise ValueError(f"Frame index {index} out of range")
        
        frame = images[index].astype(np.float32)
        if normalize:
            fmin, fmax = frame.min(), frame.max()
            if fmax > fmin:
                frame = ((frame - fmin) / (fmax - fmin) * 255).astype(np.uint8)
            else:
                frame = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        else:
            if images[index].dtype == np.uint16:
                frame = (images[index] >> 8).astype(np.uint8)
            else:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
        
        return Image.fromarray(frame, mode='L'), len(images)
    else:
        img = Image.open(file_path)
        if img.mode != 'L':
            img = img.convert('L')
        return img, 1

@app.get("/", response_class=HTMLResponse)
async def root():
    template_path = WEB_DIR / "templates" / "index.html"
    if template_path.exists():
        return template_path.read_text()
    return HTMLResponse("<h1>MONA Track - Template not found</h1>")

@app.post("/test-upload")
async def test_upload(file: UploadFile = File(None)):
    if file:
        content = await file.read()
        return {"filename": file.filename, "size": len(content)}
    return {"error": "no file"}

@app.get("/favicon.ico")
async def favicon():
    icon_path = WEB_DIR / "icon.svg"
    if icon_path.exists():
        return FileResponse(icon_path, media_type="image/svg+xml")
    raise HTTPException(status_code=404)

@app.get("/icon.svg")
async def icon():
    icon_path = WEB_DIR / "icon.svg"
    if icon_path.exists():
        return FileResponse(icon_path, media_type="image/svg+xml")
    raise HTTPException(status_code=404)

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
        "created_at": datetime.now().isoformat()
    }
    get_user_dir(data.username)
    sessions[data.username] = {"files": {}, "samples": {}, "models": [], "masks": {}}
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

class ChunkUploadStart(BaseModel):
    username: str
    filename: str
    total_size: int
    image_width: int = 1024
    image_height: int = 1024
    channel_index: int = 0
    normalize: bool = True

@app.post("/upload/start")
async def upload_start(data: ChunkUploadStart):
    if data.username not in users:
        return JSONResponse(status_code=401, content={"error": "User not found"})
    
    file_id = str(uuid.uuid4())[:8]
    ext = Path(data.filename).suffix.lower()
    
    if ext not in [".tdms", ".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
        return JSONResponse(status_code=400, content={"error": "Unsupported file type"})
    
    user_dir = get_user_dir(data.username)
    file_path = user_dir / "uploads" / f"{file_id}{ext}"
    
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.touch()
    
    return {
        "upload_id": file_id,
        "file_path": str(file_path),
        "settings": {
            "image_width": data.image_width,
            "image_height": data.image_height,
            "channel_index": data.channel_index,
            "normalize": data.normalize
        }
    }

@app.post("/upload/chunk/{upload_id}")
async def upload_chunk(upload_id: str, request: Request, offset: int = 0):
    body = await request.body()
    
    for username, session in sessions.items():
        for fid, finfo in session.get("files", {}).items():
            if fid == upload_id:
                file_path = Path(finfo["path"])
                with open(file_path, "r+b") as f:
                    f.seek(offset)
                    f.write(body)
                return {"received": len(body), "offset": offset}
    
    for username in users:
        user_dir = get_user_dir(username)
        for ext in [".tdms", ".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
            file_path = user_dir / "uploads" / f"{upload_id}{ext}"
            if file_path.exists():
                with open(file_path, "r+b") as f:
                    f.seek(offset)
                    f.write(body)
                return {"received": len(body), "offset": offset}
    
    return JSONResponse(status_code=404, content={"error": "Upload not found"})

class ChunkUploadComplete(BaseModel):
    username: str
    upload_id: str
    filename: str
    image_width: int = 1024
    image_height: int = 1024
    channel_index: int = 0
    normalize: bool = True

@app.post("/upload/complete")
async def upload_complete(data: ChunkUploadComplete):
    if data.username not in users:
        return JSONResponse(status_code=401, content={"error": "User not found"})
    
    if data.username not in sessions:
        load_user_session(data.username)
    
    user_dir = get_user_dir(data.username)
    ext = Path(data.filename).suffix.lower()
    file_path = user_dir / "uploads" / f"{data.upload_id}{ext}"
    
    if not file_path.exists():
        return JSONResponse(status_code=404, content={"error": "Upload file not found"})
    
    file_info = {
        "id": data.upload_id,
        "filename": data.filename,
        "path": str(file_path),
        "type": "tdms" if ext == ".tdms" else "image",
        "frame_count": 1,
        "tdms_settings": {
            "image_width": data.image_width,
            "image_height": data.image_height,
            "channel_index": data.channel_index,
            "normalize": data.normalize
        }
    }
    
    if ext == ".tdms":
        try:
            from nptdms import TdmsFile
            tdms_file = TdmsFile.read(str(file_path))
            groups = tdms_file.groups()
            if groups:
                channels = groups[0].channels()
                if data.channel_index < len(channels):
                    raw_data = channels[data.channel_index][:]
                    total_pixels = raw_data.size
                    image_size = data.image_width * data.image_height
                    if total_pixels % image_size == 0:
                        file_info["frame_count"] = total_pixels // image_size
                        file_info["width"] = data.image_width
                        file_info["height"] = data.image_height
        except Exception as e:
            print(f"TDMS parse error: {e}")
    else:
        img = Image.open(file_path)
        file_info["width"] = img.width
        file_info["height"] = img.height
    
    sessions[data.username]["files"][data.upload_id] = file_info
    save_user_session(data.username)
    
    return file_info

@app.post("/upload")
async def upload_file(request: Request):
    from starlette.datastructures import UploadFile as StarletteUploadFile
    import aiofiles
    
    try:
        form = await request.form()
    except Exception as e:
        print(f"Form parsing error: {e}")
        return JSONResponse(status_code=400, content={"error": f"Form parsing failed: {str(e)}"})
    
    username = form.get("username")
    file = form.get("file")
    image_width = form.get("image_width", "1024")
    image_height = form.get("image_height", "1024")
    channel_index = form.get("channel_index", "0")
    normalize = form.get("normalize", "true")
    
    if not username:
        return JSONResponse(status_code=400, content={"error": "Missing username"})
    if not file:
        return JSONResponse(status_code=400, content={"error": "Missing file"})
    
    try:
        image_width = int(image_width)
        image_height = int(image_height)
        channel_index = int(channel_index)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid number: {e}"})
    
    normalize_bool = str(normalize).lower() in ("true", "1", "yes")
    try:
        if username not in users:
            return JSONResponse(status_code=401, content={"error": "User not found"})
        
        if username not in sessions:
            load_user_session(username)
        
        file_id = str(uuid.uuid4())[:8]
        filename = file.filename or f"upload_{file_id}"
        ext = Path(filename).suffix.lower()
        
        if ext not in [".tdms", ".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
            return JSONResponse(status_code=400, content={"error": "Unsupported file type"})
        
        user_dir = get_user_dir(username)
        file_path = user_dir / "uploads" / f"{file_id}{ext}"
        
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        file_info = {
            "id": file_id,
            "filename": filename,
            "path": str(file_path),
            "type": "tdms" if ext == ".tdms" else "image",
            "frame_count": 1,
            "tdms_settings": {"image_width": image_width, "image_height": image_height, "channel_index": channel_index, "normalize": normalize_bool}
        }
        
        if ext == ".tdms":
            try:
                from nptdms import TdmsFile
                tdms_file = TdmsFile.read(str(file_path))
                groups = tdms_file.groups()
                if groups:
                    channels = groups[0].channels()
                    if channel_index < len(channels):
                        data = channels[channel_index][:]
                        total_pixels = data.size
                        image_size = image_width * image_height
                        if total_pixels % image_size == 0:
                            file_info["frame_count"] = total_pixels // image_size
                            file_info["width"] = image_width
                            file_info["height"] = image_height
                            file_info["dtype"] = str(data.dtype)
                        else:
                            file_info["error"] = f"Data size {total_pixels} not divisible by {image_size}"
            except Exception as e:
                file_info["error"] = str(e)
        else:
            img = Image.open(file_path)
            file_info["width"] = img.width
            file_info["height"] = img.height
        
        sessions[username]["files"][file_id] = file_info
        save_user_session(username)
        return file_info
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/frame/{username}/{file_id}/{index}")
async def get_frame(username: str, file_id: str, index: int):
    if username not in sessions:
        load_user_session(username)
    if file_id not in sessions[username]["files"]:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_info = sessions[username]["files"][file_id]
    img, _ = extract_frame(Path(file_info["path"]), file_info, index)
    
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return {"image": f"data:image/png;base64,{img_base64}", "width": img.width, "height": img.height}

@app.post("/sample")
async def save_sample(request: SampleRequest):
    if request.username not in sessions:
        load_user_session(request.username)
    if request.file_id not in sessions[request.username]["files"]:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_info = sessions[request.username]["files"][request.file_id]
    img, _ = extract_frame(Path(file_info["path"]), file_info, request.frame_index)
    
    x1, y1 = max(0, request.x), max(0, request.y)
    x2, y2 = min(img.width, request.x + request.width), min(img.height, request.y + request.height)
    cropped = img.crop((x1, y1, x2, y2))
    
    sample_dir = get_user_dir(request.username) / "samples" / request.particle_name
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_path = sample_dir / f"{request.particle_name}.jpg"
    cropped.save(sample_path, format="JPEG")
    
    sample_info = {"path": str(sample_path), "width": cropped.width, "height": cropped.height}
    sessions[request.username]["samples"][request.particle_name] = [sample_info]
    
    buffer = BytesIO()
    cropped.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    save_user_session(request.username)
    return {"particle_name": request.particle_name, "sample": sample_info, "preview": f"data:image/png;base64,{img_base64}"}

@app.post("/mask")
async def save_mask(request: MaskRequest):
    if request.username not in sessions:
        load_user_session(request.username)
    
    mask_dir = get_user_dir(request.username) / "masks" / request.particle_name
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_path = mask_dir / f"{request.particle_name}_mask.png"
    
    mask_data = request.mask_data.split(",")[1] if "," in request.mask_data else request.mask_data
    mask_bytes = base64.b64decode(mask_data)
    mask_img = Image.open(BytesIO(mask_bytes))
    mask_img.save(mask_path, format="PNG")
    
    if "masks" not in sessions[request.username]:
        sessions[request.username]["masks"] = {}
    sessions[request.username]["masks"][request.particle_name] = {"path": str(mask_path)}
    
    save_user_session(request.username)
    return {"status": "saved", "particle_name": request.particle_name}

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

def run_training(job_id: str, username: str, particle_name: str, config: dict):
    try:
        import lightning as L
        
        start_time = time.time()
        training_jobs[job_id]["status"] = "running"
        training_jobs[job_id]["progress"] = 0
        training_jobs[job_id]["start_time"] = start_time
        training_jobs[job_id]["losses"] = []
        
        import deeptrack as dt
        import deeptrack.deeplay as dl
        
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
                mode='constant'
            ))
        
        pipeline_ops.extend([
            dt.Multiply(lambda: np.random.uniform(config["mul_min"], config["mul_max"])),
            dt.Add(lambda: np.random.uniform(config["add_min"], config["add_max"])),
            dt.MoveAxis(-1, 0),
            dt.pytorch.ToTensor(dtype=torch.float32)
        ])
        
        training_pipeline = pipeline_ops[0]
        for op in pipeline_ops[1:]:
            training_pipeline = training_pipeline >> op
        
        training_dataset = dt.pytorch.Dataset(training_pipeline, length=config["length"], replace=False)
        train_dataloader = dl.DataLoader(training_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
        
        lodestar = dl.LodeSTAR(n_transforms=config["n_transforms"], optimizer=dl.Adam(lr=config["lr"])).build()
        
        with torch.no_grad():
            _ = lodestar(torch.randn(1, 1, 64, 64))
        
        class LossCallback(L.Callback):
            def __init__(self, job_id):
                self.job_id = job_id
                self.epoch_losses = []
            
            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                if outputs is not None:
                    if isinstance(outputs, dict) and "loss" in outputs:
                        loss_val = float(outputs["loss"].detach().cpu().item())
                    elif isinstance(outputs, torch.Tensor):
                        loss_val = float(outputs.detach().cpu().item())
                    else:
                        return
                    self.epoch_losses.append(loss_val)
            
            def on_train_epoch_end(self, trainer, pl_module):
                if self.epoch_losses:
                    avg_loss = sum(self.epoch_losses) / len(self.epoch_losses)
                    training_jobs[self.job_id]["losses"].append(avg_loss)
                    training_jobs[self.job_id]["current_loss"] = avg_loss
                    self.epoch_losses = []
        
        loss_callback = LossCallback(job_id)
        
        trainer = dl.Trainer(
            max_epochs=config["max_epochs"],
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            precision="16-mixed" if torch.cuda.is_available() else "32",
            enable_progress_bar=False,
            logger=False,
            enable_checkpointing=False,
            callbacks=[loss_callback]
        )
        
        original_advance = trainer.fit_loop.on_advance_end
        def patched_advance():
            original_advance()
            epoch = trainer.current_epoch + 1
            training_jobs[job_id]["progress"] = int(epoch / config["max_epochs"] * 100)
            training_jobs[job_id]["current_epoch"] = epoch
            training_jobs[job_id]["elapsed_time"] = time.time() - start_time
            save_training_jobs()
        trainer.fit_loop.on_advance_end = patched_advance
        
        trainer.fit(lodestar, train_dataloader)
        
        end_time = time.time()
        runtime = end_time - start_time
        
        model_dir = get_user_dir(username) / "models"
        model_path = model_dir / f"{particle_name}_weights.pth"
        torch.save(lodestar.state_dict(), model_path)
        
        losses = training_jobs[job_id].get("losses", [])
        summary = {
            "runtime_seconds": round(runtime, 2),
            "runtime_formatted": f"{int(runtime // 60)}m {int(runtime % 60)}s",
            "total_epochs": config["max_epochs"],
            "final_loss": losses[-1] if losses else None,
            "min_loss": min(losses) if losses else None,
            "loss_history": losses[-10:] if len(losses) > 10 else losses,
            "device": "GPU" if torch.cuda.is_available() else "CPU"
        }
        
        model_info = {"id": job_id, "particle_name": particle_name, "path": str(model_path), "config": config, "created_at": datetime.now().isoformat(), "summary": summary}
        sessions[username]["models"].append(model_info)
        save_user_session(username)
        
        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["progress"] = 100
        training_jobs[job_id]["model_info"] = model_info
        training_jobs[job_id]["summary"] = summary
        save_training_jobs()
    except Exception as e:
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["error"] = str(e)
        training_jobs[job_id]["elapsed_time"] = time.time() - training_jobs[job_id].get("start_time", time.time())
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
    
    training_jobs[job_id] = {"id": job_id, "username": request.username, "particle_name": request.particle_name, "status": "queued", "progress": 0, "config": config, "created_at": datetime.now().isoformat()}
    save_training_jobs()
    
    thread = threading.Thread(target=run_training, args=(job_id, request.username, request.particle_name, config))
    thread.start()
    
    return {"job_id": job_id, "status": "queued"}

@app.get("/train/active/{username}")
async def get_active_jobs(username: str):
    active = [job for job in training_jobs.values() if job.get("username") == username and job.get("status") in ["queued", "running"]]
    return {"jobs": active}

@app.get("/train/{job_id}")
async def get_training_status(job_id: str):
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return training_jobs[job_id]

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
    
    model_path = Path(model_info["path"])
    if model_path.exists():
        model_path.unlink()
    
    sessions[username]["models"] = [m for m in models if m["id"] != model_id]
    save_user_session(username)
    
    return {"status": "deleted", "model_id": model_id}

class RenameModelRequest(BaseModel):
    new_name: str

@app.put("/models/{username}/{model_id}/rename")
async def rename_model(username: str, model_id: str, request: RenameModelRequest):
    if username not in sessions:
        load_user_session(username)
    
    models = sessions[username].get("models", [])
    model_info = next((m for m in models if m["id"] == model_id), None)
    if not model_info:
        raise HTTPException(status_code=404, detail="Model not found")
    
    old_path = Path(model_info["path"])
    new_path = old_path.parent / f"{request.new_name}_weights.pth"
    
    if old_path.exists():
        old_path.rename(new_path)
    
    model_info["particle_name"] = request.new_name
    model_info["path"] = str(new_path)
    save_user_session(username)
    
    return {"status": "renamed", "model_id": model_id, "new_name": request.new_name}

detect_files: Dict[str, Dict[str, Any]] = {}

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
    lodestar = dl.LodeSTAR(n_transforms=config.get("n_transforms", 4), optimizer=dl.Adam(lr=config.get("lr", 0.0001))).build()
    lodestar.load_state_dict(torch.load(model_path, map_location="cpu"))
    lodestar.eval()
    return lodestar

def run_detection_on_image(lodestar, img: Image.Image, alpha: float, beta: float, cutoff: float, return_weightmap: bool):
    if img.mode != 'L':
        img = img.convert('L')
    image = np.array(img).astype(np.float32)
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
    
    with torch.no_grad():
        model_output = lodestar(image_tensor)
        detections = lodestar.detect(image_tensor, alpha=alpha, beta=beta, mode="constant", cutoff=cutoff)[0]
    
    detections_list = detections[:, [1, 0]].tolist() if len(detections) > 0 else []
    
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    result = {
        "detections": detections_list,
        "count": len(detections_list),
        "image": f"data:image/png;base64,{img_base64}",
        "width": img.width,
        "height": img.height
    }
    
    if return_weightmap and model_output is not None:
        if len(model_output.shape) == 4 and model_output.shape[1] >= 3:
            weights = model_output[0, -1].detach().cpu().numpy()
        else:
            weights = model_output[0, 0].detach().cpu().numpy() if len(model_output.shape) == 4 else model_output.detach().cpu().numpy()
        
        h, w = result["height"], result["width"]
        if weights.shape != (h, w):
            weights = cv2.resize(weights, (w, h), interpolation=cv2.INTER_LINEAR)
        
        weights_norm = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
        weights_colored = (plt.cm.hot(weights_norm)[:, :, :3] * 255).astype(np.uint8)
        weights_img = Image.fromarray(weights_colored, mode='RGB')
        weights_buffer = BytesIO()
        weights_img.save(weights_buffer, format="PNG")
        result["weightmap"] = f"data:image/png;base64,{base64.b64encode(weights_buffer.getvalue()).decode()}"
    
    return result

@app.post("/detect/upload")
async def upload_detect_file(
    username: str = Form(None),
    file: UploadFile = File(None),
    image_width: str = Form("1024"),
    image_height: str = Form("1024"),
    channel_index: str = Form("0"),
    normalize: str = Form("true")
):
    if not username:
        return JSONResponse(status_code=400, content={"error": "Missing username"})
    if not file:
        return JSONResponse(status_code=400, content={"error": "Missing file"})
    
    try:
        image_width = int(image_width)
        image_height = int(image_height)
        channel_index = int(channel_index)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid number: {e}"})
    
    normalize_bool = normalize.lower() in ("true", "1", "yes")
    if username not in users:
        return JSONResponse(status_code=401, content={"error": "User not found"})
    
    file_id = str(uuid.uuid4())[:8]
    filename = file.filename or f"detect_{file_id}"
    ext = Path(filename).suffix.lower()
    
    if ext not in [".tdms", ".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
        return JSONResponse(status_code=400, content={"error": "Unsupported file type"})
    
    user_dir = get_user_dir(username)
    file_path = user_dir / "uploads" / f"detect_{file_id}{ext}"
    
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    
    file_info = {
        "id": file_id,
        "filename": filename,
        "path": str(file_path),
        "type": "tdms" if ext == ".tdms" else "image",
        "frame_count": 1,
        "tdms_settings": {"image_width": image_width, "image_height": image_height, "channel_index": channel_index, "normalize": normalize_bool}
    }
    
    if ext == ".tdms":
        try:
            from nptdms import TdmsFile
            tdms_file = TdmsFile.read(str(file_path))
            groups = tdms_file.groups()
            if groups:
                channels = groups[0].channels()
                if channel_index < len(channels):
                    data = channels[channel_index][:]
                    total_pixels = data.size
                    image_size = image_width * image_height
                    if total_pixels % image_size == 0:
                        file_info["frame_count"] = total_pixels // image_size
                        file_info["width"] = image_width
                        file_info["height"] = image_height
                    else:
                        file_info["error"] = f"Data size {total_pixels} not divisible by {image_size}"
        except Exception as e:
            file_info["error"] = str(e)
    else:
        img = Image.open(file_path)
        file_info["width"] = img.width
        file_info["height"] = img.height
    
    if username not in detect_files:
        detect_files[username] = {}
    detect_files[username][file_id] = file_info
    
    return file_info

@app.get("/detect/frame/{username}/{file_id}/{index}")
async def get_detect_frame(
    username: str,
    file_id: str,
    index: int,
    model_id: str,
    alpha: float = 1.0,
    beta: float = 0.0,
    cutoff: float = 0.8,
    return_weightmap: bool = False
):
    if username not in detect_files or file_id not in detect_files[username]:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_info = detect_files[username][file_id]
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
    return_weightmap: bool = Form(False)
):
    lodestar = load_model(username, model_id)
    
    content = await file.read()
    img = Image.open(BytesIO(content))
    
    result = run_detection_on_image(lodestar, img, alpha, beta, cutoff, return_weightmap)
    result["params"] = {"alpha": alpha, "beta": beta, "cutoff": cutoff}
    
    return result

@app.get("/config/defaults")
async def get_default_config():
    config_path = Path(__file__).parent.parent / "src" / "config.yaml"
    defaults = {
        "n_transforms": 4, "max_epochs": 100, "batch_size": 8, "lr": 0.0001, "length": 400,
        "alpha": 1.0, "beta": 0.0, "cutoff": 0.8,
        "mul_min": 0.9, "mul_max": 1.1, "add_min": -0.1, "add_max": 0.1,
        "scale_min": 0.9, "scale_max": 1.1, "rotation_min": 0.0, "rotation_max": 1.0,
        "translate_min": -5.0, "translate_max": 5.0
    }
    if config_path.exists():
        config = utils.load_yaml(str(config_path))
        for key in defaults:
            if key in config:
                defaults[key] = config[key]
    return defaults

@app.get("/tdms/structure/{username}/{file_id}")
async def get_tdms_structure(username: str, file_id: str):
    if username not in sessions:
        load_user_session(username)
    if file_id not in sessions[username]["files"]:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_info = sessions[username]["files"][file_id]
    if file_info["type"] != "tdms":
        raise HTTPException(status_code=400, detail="Not a TDMS file")
    
    from tdms_to_png import list_tdms_structure
    structure = list_tdms_structure(Path(file_info["path"]))
    return {"structure": structure}

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

@app.post("/tdms/export")
async def export_tdms(request: TdmsExportRequest):
    if request.username not in sessions:
        load_user_session(request.username)
    if request.file_id not in sessions[request.username]["files"]:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_info = sessions[request.username]["files"][request.file_id]
    if file_info["type"] != "tdms":
        raise HTTPException(status_code=400, detail="Not a TDMS file")
    
    from tdms_to_png import extract_images_from_tdms, save_images, save_video
    import zipfile
    
    settings = file_info.get("tdms_settings", {})
    images = extract_images_from_tdms(
        Path(file_info["path"]),
        image_width=settings.get("image_width", 1024),
        image_height=settings.get("image_height", 1024),
        channel_index=settings.get("channel_index", 0)
    )
    
    start = request.start_frame
    end = request.end_frame if request.end_frame is not None else len(images)
    images = images[start:end]
    
    dtype_map = {"uint8": np.uint8, "uint16": np.uint16}
    dtype = dtype_map.get(request.dtype, np.uint8)
    
    user_dir = get_user_dir(request.username)
    base_name = request.output_name or Path(file_info["filename"]).stem
    
    if request.output_format == "mp4":
        output_path = user_dir / "results" / f"{base_name}.mp4"
        save_video(images, output_path, fps=request.fps, dtype=dtype, force=True, normed=request.normalize)
        
        if request.save_to_server:
            return {"status": "saved", "path": str(output_path), "frames": len(images)}
        
        with open(output_path, "rb") as f:
            video_data = base64.b64encode(f.read()).decode()
        return {"status": "ready", "data": video_data, "filename": f"{base_name}.mp4", "frames": len(images)}
    
    else:
        export_dir = user_dir / "results" / base_name
        export_dir.mkdir(parents=True, exist_ok=True)
        save_images(images, export_dir, base_name, dtype=dtype, force=True, normed=request.normalize)
        
        if request.save_to_server:
            return {"status": "saved", "path": str(export_dir), "frames": len(images)}
        
        zip_path = user_dir / "results" / f"{base_name}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for img_file in export_dir.glob("*.png"):
                zf.write(img_file, img_file.name)
        
        with open(zip_path, "rb") as f:
            zip_data = base64.b64encode(f.read()).decode()
        return {"status": "ready", "data": zip_data, "filename": f"{base_name}.zip", "frames": len(images)}

class CircularMaskRequest(BaseModel):
    username: str
    file_id: str
    frame_index: int = 0
    roi_center_x: float
    roi_center_y: float
    roi_radius: float
    particle_name: str

@app.post("/mask/circular")
async def apply_circular_mask(request: CircularMaskRequest):
    if request.username not in sessions:
        load_user_session(request.username)
    if request.file_id not in sessions[request.username]["files"]:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_info = sessions[request.username]["files"][request.file_id]
    img, _ = extract_frame(Path(file_info["path"]), file_info, request.frame_index)
    img_array = np.array(img)
    
    h, w = img_array.shape[:2]
    y_coords, x_coords = np.ogrid[:h, :w]
    dist = np.sqrt((x_coords - request.roi_center_x)**2 + (y_coords - request.roi_center_y)**2)
    mask = dist <= request.roi_radius
    
    masked_array = np.where(mask, img_array, 0).astype(np.uint8)
    
    x1 = max(0, int(request.roi_center_x - request.roi_radius))
    y1 = max(0, int(request.roi_center_y - request.roi_radius))
    x2 = min(w, int(request.roi_center_x + request.roi_radius))
    y2 = min(h, int(request.roi_center_y + request.roi_radius))
    
    cropped = masked_array[y1:y2, x1:x2]
    cropped_img = Image.fromarray(cropped, mode='L')
    
    sample_dir = get_user_dir(request.username) / "samples" / request.particle_name
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_path = sample_dir / f"{request.particle_name}.jpg"
    cropped_img.save(sample_path, format="JPEG")
    
    sample_info = {"path": str(sample_path), "width": cropped_img.width, "height": cropped_img.height, "mask_type": "circular"}
    sessions[request.username]["samples"][request.particle_name] = [sample_info]
    
    buffer = BytesIO()
    cropped_img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    save_user_session(request.username)
    return {"particle_name": request.particle_name, "sample": sample_info, "preview": f"data:image/png;base64,{img_base64}"}

class VideoMergeRequest(BaseModel):
    username: str
    file_ids: List[str]
    output_name: str
    fps: float = 30.0

@app.post("/video/merge")
async def merge_videos(request: VideoMergeRequest):
    if request.username not in users:
        raise HTTPException(status_code=401, detail="User not found")
    
    user_dir = get_user_dir(request.username)
    results_dir = user_dir / "results"
    
    mp4_files = []
    for file_id in request.file_ids:
        mp4_path = results_dir / f"{file_id}.mp4"
        if mp4_path.exists():
            mp4_files.append(mp4_path)
        else:
            for mp4 in results_dir.glob(f"*{file_id}*.mp4"):
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
        imageio.mimwrite(str(output_path), all_frames, fps=request.fps, codec='libx264', quality=8)
        
        return {"status": "merged", "path": str(output_path), "total_frames": len(all_frames), "source_files": len(mp4_files)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
    
    file_info = sessions[username]["files"][file_id]
    file_path = Path(file_info["path"])
    
    if file_path.exists():
        try:
            file_path.unlink()
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Failed to delete file: {e}"})
    
    del sessions[username]["files"][file_id]
    save_user_session(username)
    
    return {"status": "deleted", "file_id": file_id}

@app.get("/results/{username}")
async def list_results(username: str):
    user_dir = get_user_dir(username)
    results_dir = user_dir / "results"
    
    results = []
    if results_dir.exists():
        for f in results_dir.iterdir():
            if f.is_file():
                results.append({"name": f.name, "path": str(f), "size": f.stat().st_size, "type": f.suffix})
            elif f.is_dir():
                png_count = len(list(f.glob("*.png")))
                results.append({"name": f.name, "path": str(f), "type": "folder", "png_count": png_count})
    
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    try:
        uvicorn.run(app, host="0.0.0.0", port=args.port, http="httptools", log_level="info")
    except Exception:
        uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
