import base64
import zipfile
import tempfile
import uuid
import math
from pathlib import Path
from typing import Optional

import imageio
import numpy as np
from fastapi import APIRouter, HTTPException
from PIL import Image
from pydantic import BaseModel
from services.frames import normalize_tdms_frame
from services.tdms_cache import get_images

import state
from services import tdms_ops

router = APIRouter(prefix="/tdms", tags=["tdms"])


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


class AnalyzeRequest(BaseModel):
    username: str
    file_id: str
    frame: int = 0
    bins: int = 256
    filter_type: str = "gaussian"
    sigma: float = 1.0
    method: str = "canny"
    direction: str = "horizontal"
    position: Optional[int] = None
    frame_a: int = 0
    frame_b: int = 1
    compare_method: str = "difference"


def _tdms_file(username: str, file_id: str) -> dict:
    info = state.get_session_file(username, file_id)
    if info.get("type") != "tdms":
        raise HTTPException(status_code=400, detail="Not a TDMS file")
    return info


@router.get("/structure/{username}/{file_id}")
async def get_tdms_structure(username: str, file_id: str):
    info = _tdms_file(username, file_id)
    try:
        structure = tdms_ops.list_structure(info["path"])
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"structure": structure, "file_id": file_id, "filename": info.get("filename")}


@router.get("/frame/{username}/{file_id}/{index}")
async def get_tdms_frame(
    username: str,
    file_id: str,
    index: int,
    cmap: str = "gray",
    normalize: Optional[bool] = None,
):
    info = _tdms_file(username, file_id)
    effective_normalize = info.get("tdms_settings", {}).get("normalize", True) if normalize is None else normalize
    try:
        image, frame_count, width, height = tdms_ops.get_frame_image(
            info["path"], index, normalize=effective_normalize, cmap=cmap
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    if info.get("frame_count", 1) != frame_count:
        info["frame_count"] = frame_count
        info["width"] = width
        info["height"] = height
        state.save_user_session(username)
    return {
        "image": image,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "frame_index": index,
        "cmap": cmap,
        "normalized": effective_normalize,
    }


@router.get("/channel/{username}/{file_id}")
async def get_channel(username: str, file_id: str, group: str, channel: str):
    info = _tdms_file(username, file_id)
    try:
        return tdms_ops.get_channel_series(info["path"], group, channel)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/analyze/histogram")
async def analyze_histogram(req: AnalyzeRequest):
    info = _tdms_file(req.username, req.file_id)
    try:
        return tdms_ops.analyze_histogram(info["path"], req.frame, bins=req.bins)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/analyze/filter")
async def analyze_filter(req: AnalyzeRequest):
    info = _tdms_file(req.username, req.file_id)
    try:
        return tdms_ops.analyze_filter(info["path"], req.frame, req.filter_type, sigma=req.sigma)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/analyze/edges")
async def analyze_edges(req: AnalyzeRequest):
    info = _tdms_file(req.username, req.file_id)
    try:
        return tdms_ops.analyze_edges(info["path"], req.frame, method=req.method)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/analyze/profile")
async def analyze_profile(req: AnalyzeRequest):
    info = _tdms_file(req.username, req.file_id)
    try:
        return tdms_ops.analyze_profile(info["path"], req.frame, direction=req.direction, position=req.position)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/compare")
async def compare_frames(req: AnalyzeRequest):
    info = _tdms_file(req.username, req.file_id)
    try:
        return tdms_ops.compare_frames(
            info["path"], req.frame_a, req.frame_b, method=req.compare_method
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def _export_frame(raw: np.ndarray, normalize: bool, dtype: np.dtype) -> np.ndarray:
    if dtype == np.uint16:
        f = raw.astype(np.float32)
        if normalize:
            lo, hi = float(f.min()), float(f.max())
            if hi > lo:
                return ((f - lo) / (hi - lo) * 65535.0).astype(np.uint16)
            return np.zeros_like(raw, dtype=np.uint16)
        return np.clip(f, 0, 65535).astype(np.uint16)
    return normalize_tdms_frame(raw, normalize)


@router.post("/export")
async def export_tdms(request: TdmsExportRequest):
    info = _tdms_file(request.username, request.file_id)
    if request.output_format not in {"png", "mp4"}:
        raise HTTPException(status_code=400, detail="Unsupported export format")
    if request.dtype not in {"uint8", "uint16"}:
        raise HTTPException(status_code=400, detail="Unsupported export dtype")
    if not math.isfinite(request.fps) or request.fps <= 0:
        raise HTTPException(status_code=400, detail="FPS must be finite and positive")
    try:
        images = get_images(str(info["path"]))
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if images is None:
        raise HTTPException(status_code=400, detail="Could not extract images from TDMS file")

    start = request.start_frame
    end = request.end_frame if request.end_frame is not None else int(images.shape[0])
    if start < 0 or end > images.shape[0] or start >= end:
        raise HTTPException(status_code=400, detail=f"Invalid frame range {start}:{end}")
    dtype = {"uint8": np.uint8, "uint16": np.uint16}.get(request.dtype, np.uint8)
    user_dir = state.get_user_dir(request.username)
    base_name = state.safe_name(request.output_name or Path(info["filename"]).stem)
    frame_count = end - start

    results_dir = state.contained_path(user_dir, "results")
    results_dir.mkdir(parents=True, exist_ok=True)
    published_dir = state.contained_path(results_dir, f"{base_name}-{uuid.uuid4().hex}")
    # A completed export is published in one rename. Failure removes only this
    # request's private staging, never an older same-name export.
    with tempfile.TemporaryDirectory(prefix=".tdms-export-", dir=results_dir) as tmp:
        staging = Path(tmp)
        if request.output_format == "mp4":
            filename = f"{base_name}.mp4"
            with imageio.get_writer(
                str(staging / filename), fps=request.fps,
                codec="libx264", quality=8, macro_block_size=1,
            ) as writer:
                for i in range(start, end):
                    writer.append_data(normalize_tdms_frame(images[i], request.normalize))
        else:
            generated = []
            for i in range(start, end):
                frame = _export_frame(images[i], request.normalize, dtype)
                mode = "I;16" if frame.dtype == np.uint16 else "L"
                frame_path = staging / f"{base_name}_{i + 1:03d}.png"
                Image.fromarray(frame, mode=mode).save(frame_path)
                generated.append(frame_path)
            filename = f"{base_name}.zip"
            if not request.save_to_server:
                with zipfile.ZipFile(staging / filename, "w", zipfile.ZIP_DEFLATED) as zf:
                    for frame_path in generated:
                        zf.write(frame_path, frame_path.name)
        staging.rename(published_dir)
    if request.save_to_server:
        output_path = published_dir / filename if request.output_format == "mp4" else published_dir
        return {"status": "saved", "path": str(output_path), "frames": frame_count}
    return {
        "status": "ready",
        "data": base64.b64encode((published_dir / filename).read_bytes()).decode(),
        "filename": filename,
        "frames": frame_count,
    }
