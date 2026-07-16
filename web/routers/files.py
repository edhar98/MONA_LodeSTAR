import base64
import glob as _glob
import uuid
from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from starlette.requests import Request

import state
from config import ALLOWED_UPLOAD_EXT
from services.frames import extract_frame, parse_tdms_info

router = APIRouter(tags=["files"])


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


class PathLoadRequest(BaseModel):
    username: str
    path: str
    normalize: bool = True


@router.post("/upload/start")
async def upload_start(data: ChunkUploadStart):
    try:
        data.username = state.require_user(data.username)
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": str(e.detail)})
    file_id = str(uuid.uuid4())[:8]
    ext = Path(data.filename).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXT:
        return JSONResponse(status_code=400, content={"error": "Unsupported file type"})
    file_path = state.get_user_dir(data.username) / "uploads" / f"{file_id}{ext}"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.touch()
    return {"upload_id": file_id, "file_path": str(file_path), "settings": {"normalize": data.normalize}}


@router.post("/upload/chunk/{upload_id}")
async def upload_chunk(upload_id: str, request: Request, offset: int = 0):
    body = await request.body()
    for session in state.sessions.values():
        for finfo in session.get("files", {}).values():
            if finfo.get("id") == upload_id:
                with open(finfo["path"], "r+b") as f:
                    f.seek(offset)
                    f.write(body)
                return {"received": len(body), "offset": offset}
    for username in state.users:
        user_dir = state.get_user_dir(username)
        for ext in ALLOWED_UPLOAD_EXT:
            fp = user_dir / "uploads" / f"{upload_id}{ext}"
            if fp.exists():
                with open(fp, "r+b") as f:
                    f.seek(offset)
                    f.write(body)
                return {"received": len(body), "offset": offset}
    return JSONResponse(status_code=404, content={"error": "Upload not found"})


@router.post("/upload/complete")
async def upload_complete(data: ChunkUploadComplete):
    try:
        data.username = state.require_user(data.username)
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": str(e.detail)})
    ext = Path(data.filename).suffix.lower()
    file_path = state.get_user_dir(data.username) / "uploads" / f"{data.upload_id}{ext}"
    if not file_path.exists():
        return JSONResponse(status_code=404, content={"error": "Upload file not found"})
    file_info = {
        "id": data.upload_id, "filename": data.filename, "path": str(file_path),
        "type": "tdms" if ext == ".tdms" else "image",
        "frame_count": 1, "tdms_settings": {"normalize": data.normalize},
    }
    if ext == ".tdms":
        parse_tdms_info(file_path, file_info)
    else:
        img = Image.open(file_path)
        file_info["width"] = img.width
        file_info["height"] = img.height
    state.sessions[data.username]["files"][data.upload_id] = file_info
    state.save_user_session(data.username)
    return file_info


@router.post("/upload")
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
    try:
        username = state.require_user(str(username))
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": str(e.detail)})

    normalize_bool = str(normalize).lower() in ("true", "1", "yes")
    file_id = str(uuid.uuid4())[:8]
    filename = file.filename or f"upload_{file_id}"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXT:
        return JSONResponse(status_code=400, content={"error": "Unsupported file type"})

    file_path = state.get_user_dir(username) / "uploads" / f"{file_id}{ext}"
    file_path.write_bytes(await file.read())

    file_info = {
        "id": file_id, "filename": filename, "path": str(file_path),
        "type": "tdms" if ext == ".tdms" else "image",
        "frame_count": 1, "tdms_settings": {"normalize": normalize_bool},
    }
    if ext == ".tdms":
        parse_tdms_info(file_path, file_info)
    else:
        img = Image.open(file_path)
        file_info["width"] = img.width
        file_info["height"] = img.height

    state.sessions[username]["files"][file_id] = file_info
    state.save_user_session(username)
    return file_info


@router.post("/files/load-path")
async def load_from_path(data: PathLoadRequest):
    data.username = state.require_user(data.username)

    raw = data.path.strip()
    if any(c in raw for c in ("*", "?", "[")):
        candidates = [Path(p) for p in sorted(_glob.glob(raw, recursive=True))]
    else:
        p = Path(raw)
        if p.is_dir():
            candidates = sorted(p.iterdir())
        elif p.is_file():
            candidates = [p]
        else:
            raise HTTPException(status_code=404, detail=f"Path not found: {raw}")

    paths = [f for f in candidates if f.is_file() and f.suffix.lower() in ALLOWED_UPLOAD_EXT]
    if not paths:
        raise HTTPException(status_code=400, detail="No supported files found at the given path")

    registered = []
    for file_path in paths:
        file_id = uuid.uuid4().hex[:8]
        ext = file_path.suffix.lower()
        file_info = {
            "id": file_id,
            "filename": file_path.name,
            "path": str(file_path),
            "type": "tdms" if ext == ".tdms" else "image",
            "frame_count": 1,
            "tdms_settings": {"normalize": data.normalize},
            "server_path": True,
        }
        if ext != ".tdms":
            try:
                img = Image.open(file_path)
                file_info["width"] = img.width
                file_info["height"] = img.height
            except Exception:
                pass
        state.sessions[data.username]["files"][file_id] = file_info
        registered.append(file_info)

    state.save_user_session(data.username)
    return {"files": registered, "count": len(registered)}


@router.post("/upload/csv")
async def upload_csv(
    username: str = Form(...),
    file: UploadFile = File(...),
    file_type: str = Form("detection"),
):
    try:
        username = state.require_user(username)
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": str(e.detail)})
    if not file.filename or not file.filename.endswith(".csv"):
        return JSONResponse(status_code=400, content={"error": "Only CSV files are accepted"})

    content = await file.read()
    save_path = state.get_user_dir(username) / "results" / file.filename
    save_path.write_bytes(content)
    return {
        "status": "uploaded",
        "filename": file.filename,
        "size": len(content),
        "file_type": file_type,
    }


@router.get("/frame/{username}/{file_id}/{index}")
async def get_frame(username: str, file_id: str, index: int):
    file_info = state.get_session_file(username, file_id)
    if file_id not in state.sessions[username].get("files", {}):
        raise HTTPException(status_code=404, detail="File not found")
    file_info = state.sessions[username]["files"][file_id]
    img, frame_count = extract_frame(Path(file_info["path"]), file_info, index)
    if file_info.get("frame_count", 1) != frame_count:
        state.sessions[username]["files"][file_id]["frame_count"] = frame_count
        state.sessions[username]["files"][file_id]["width"] = img.width
        state.sessions[username]["files"][file_id]["height"] = img.height
        state.save_user_session(username)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return {
        "image": f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}",
        "width": img.width,
        "height": img.height,
        "frame_count": frame_count,
    }


@router.get("/files/{username}")
async def list_files(username: str, file_type: str = None):
    sess = state.require_session(username)
    files = sess.get("files", {})
    if file_type:
        files = {k: v for k, v in files.items() if v.get("type") == file_type}
    return {"files": files}


@router.delete("/files/{username}/{file_id}")
async def delete_file(username: str, file_id: str):
    sess = state.require_session(username)
    if file_id not in sess.get("files", {}):
        raise HTTPException(status_code=404, detail="File not found")
    finfo = sess["files"][file_id]
    from services.tdms_cache import invalidate
    invalidate(finfo.get("path"))
    if not finfo.get("server_path"):
        try:
            Path(finfo["path"]).unlink(missing_ok=True)
        except Exception:
            pass
    del sess["files"][file_id]
    state.save_user_session(username)
    return {"status": "deleted", "id": file_id}
