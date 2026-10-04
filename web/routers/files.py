import base64
import glob as _glob
import uuid
import secrets
import threading
import time
import os
from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from starlette.requests import Request
from starlette.datastructures import UploadFile as StarletteUploadFile

import state
from config import ALLOWED_UPLOAD_EXT
from services.frames import extract_frame, parse_tdms_info, _file_signature

router = APIRouter(tags=["files"])

MAX_UPLOAD_SIZE = 8 * 1024 ** 3
MAX_CHUNK_SIZE = 4 * 1024 ** 2
UPLOAD_TTL = 3600
_uploads = {}
_upload_lock = threading.RLock()


def _publish_upload(temporary, destination, file_info):
    # A hard link creates the destination atomically and never replaces an
    # existing name, unlike exists()+rename(). Both paths share one directory.
    try:
        os.link(temporary, destination, follow_symlinks=False)
    except FileExistsError as exc:
        raise HTTPException(409, "Upload destination already exists") from exc
    temporary.unlink()
    file_info["source_version"] = dict(zip(
        ("resolved_path", "device", "inode", "size", "mtime_ns", "ctime_ns"),
        _file_signature(destination)))


def _upload_path(record):
    path = record["path"]
    if path.is_symlink():
        raise HTTPException(409, "Upload file changed")
    try:
        stat = path.stat()
    except FileNotFoundError:
        raise HTTPException(409, "Upload file missing")
    if (stat.st_dev, stat.st_ino) != record["identity"]:
        raise HTTPException(409, "Upload file changed")
    return path


def _expire_uploads():
    # Only delete this process's tracked, still-identical temporary files.
    for upload_id, record in list(_uploads.items()):
        if time.monotonic() - record["updated"] >= UPLOAD_TTL:
            try:
                _upload_path(record).unlink()
            except (HTTPException, OSError):
                pass
            del _uploads[upload_id]


def _require_upload(upload_id, username, token):
    state.require_user(username)
    _expire_uploads()
    record = _uploads.get(upload_id)
    if (not record or record["username"] != username or not token
            or not secrets.compare_digest(record["token"].encode(), token.encode())):
        raise HTTPException(404, "Upload session not found")
    return record


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
    upload_token: str = ""


class PathLoadRequest(BaseModel):
    username: str
    path: str
    normalize: bool = True


class BulkDeleteFilesRequest(BaseModel):
    username: str
    file_ids: list[str]


class ReorderFilesRequest(BaseModel):
    username: str
    file_ids: list[str]


@router.post("/upload/start")
async def upload_start(data: ChunkUploadStart):
    try:
        data.username = state.require_user(data.username)
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": str(e.detail)})
    ext = Path(data.filename).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXT:
        return JSONResponse(status_code=400, content={"error": "Unsupported file type"})
    if not 0 < data.total_size <= MAX_UPLOAD_SIZE:
        raise HTTPException(413, f"Upload size must be between 1 and {MAX_UPLOAD_SIZE} bytes")
    with _upload_lock:
        _expire_uploads()
        if len(_uploads) >= 32 or sum(r["username"] == data.username for r in _uploads.values()) >= 8:
            raise HTTPException(429, "Too many unfinished uploads")
        file_id = uuid.uuid4().hex
        file_path = state.contained_path(state.get_user_dir(data.username) / "uploads", f".upload_{file_id}.part")
        file_path.touch(exist_ok=False)
        stat = file_path.stat()
        token = secrets.token_urlsafe(32)
        _uploads[file_id] = dict(username=data.username, token=token, path=file_path,
            identity=(stat.st_dev, stat.st_ino), filename=data.filename, ext=ext,
            normalize=data.normalize, total=data.total_size, received=0, updated=time.monotonic())
    return {"upload_id": file_id, "upload_token": token, "chunk_size_limit": MAX_CHUNK_SIZE,
            "settings": {"normalize": data.normalize}}


@router.post("/upload/chunk/{upload_id}")
async def upload_chunk(upload_id: str, request: Request, offset: int = 0, username: str = ""):
    state.safe_name(upload_id)
    if offset < 0:
        raise HTTPException(status_code=400, detail="Offset must be nonnegative")
    token = request.headers.get("x-upload-token", "")
    with _upload_lock:
        record = _require_upload(upload_id, username, token)
        if offset != record["received"]:
            raise HTTPException(409, "Chunk offset must equal received byte count")
        remaining = record["total"] - offset
    body = bytearray()
    async for block in request.stream():
        if len(body) + len(block) > min(MAX_CHUNK_SIZE, remaining):
            raise HTTPException(413, "Chunk exceeds allowed size")
        body.extend(block)
    if not body:
        raise HTTPException(400, "Empty chunk")
    with _upload_lock:
        record = _require_upload(upload_id, username, token)
        path = _upload_path(record)
        if offset != record["received"] or path.stat().st_size != offset:
            raise HTTPException(409, "Chunk offset or upload file changed")
        with path.open("r+b") as stream:
            stream.seek(offset)
            stream.write(body)
        record.update(received=offset + len(body), updated=time.monotonic())
    return {"received": len(body), "offset": offset}


@router.post("/upload/complete")
async def upload_complete(data: ChunkUploadComplete):
    try:
        data.username = state.require_user(data.username)
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": str(e.detail)})
    state.safe_name(data.upload_id)
    with _upload_lock:
        record = _require_upload(data.upload_id, data.username, data.upload_token)
        temporary = _upload_path(record)
        if data.filename != record["filename"] or data.normalize != record["normalize"]:
            raise HTTPException(400, "Upload metadata does not match upload start")
        if record["received"] != record["total"] or temporary.stat().st_size != record["total"]:
            raise HTTPException(409, "Upload is incomplete or file size changed")
        file_path = state.contained_path(temporary.parent, f"{data.upload_id}{record['ext']}")
        if file_path.exists():
            raise HTTPException(409, "Upload destination already exists")
        file_info = dict(id=data.upload_id, filename=data.filename, path=str(file_path),
            type="tdms" if record["ext"] == ".tdms" else "image", frame_count=1,
            tdms_settings={"normalize": data.normalize})
        if record["ext"] == ".tdms":
            parse_tdms_info(temporary, file_info)
            if file_info.get("error"):
                raise HTTPException(400, "Uploaded TDMS file is invalid or has no usable images")
        else:
            try:
                with Image.open(temporary) as img:
                    img.verify()
                    file_info.update(width=img.width, height=img.height)
            except (OSError, ValueError, SyntaxError) as exc:
                raise HTTPException(400, "Uploaded image is invalid") from exc
        _publish_upload(temporary, file_path, file_info)
        # Revoke the capability before registering the completed file.
        del _uploads[data.upload_id]
        state.sessions[data.username]["files"][data.upload_id] = file_info
        state.save_user_session(data.username)
        return file_info


@router.post("/upload")
async def upload_file(request: Request):
    try:
        form = await request.form(max_files=1, max_fields=3)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Form parsing failed: {e}"})

    username = form.get("username")
    file = form.get("file")
    normalize = form.get("normalize", "true")

    if not username:
        return JSONResponse(status_code=400, content={"error": "Missing username"})
    if not isinstance(file, StarletteUploadFile):
        return JSONResponse(status_code=400, content={"error": "Missing file"})
    try:
        username = state.require_user(str(username))
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": str(e.detail)})

    normalize_bool = str(normalize).lower() in ("true", "1", "yes")
    file_id = uuid.uuid4().hex
    filename = file.filename or f"upload_{file_id}"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXT:
        return JSONResponse(status_code=400, content={"error": "Unsupported file type"})

    file_path = state.contained_path(state.get_user_dir(username) / "uploads", f"{file_id}{ext}")
    temporary = state.contained_path(file_path.parent, f".upload_{file_id}.part")
    file_info = {
        "id": file_id, "filename": filename, "path": str(file_path),
        "type": "tdms" if ext == ".tdms" else "image",
        "frame_count": 1, "tdms_settings": {"normalize": normalize_bool},
    }
    created = False
    try:
        with temporary.open("xb") as stream:
            created = True
            total = 0
            while block := await file.read(MAX_CHUNK_SIZE):
                total += len(block)
                if total > MAX_UPLOAD_SIZE:
                    raise HTTPException(413, f"Upload exceeds {MAX_UPLOAD_SIZE} bytes")
                stream.write(block)
        if not total:
            raise HTTPException(400, "Empty upload")
        if ext == ".tdms":
            parse_tdms_info(temporary, file_info)
            if file_info.get("error"):
                raise HTTPException(400, "Uploaded TDMS file is invalid or has no usable images")
        else:
            try:
                with Image.open(temporary) as img:
                    img.verify()
                    file_info.update(width=img.width, height=img.height)
            except (OSError, ValueError, SyntaxError) as exc:
                raise HTTPException(400, "Uploaded image is invalid") from exc
        _publish_upload(temporary, file_path, file_info)
        state.sessions[username]["files"][file_id] = file_info
        state.save_user_session(username)
        return file_info
    finally:
        await file.close()
        if created:
            temporary.unlink(missing_ok=True)


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

    session_files = state.sessions[data.username]["files"]
    by_path = {}
    for file_id, info in session_files.items():
        if info.get("server_path") and info.get("path"):
            # Keep the first registration when old duplicate IDs exist: those
            # IDs may still be referenced by crops/jobs, so never remove them.
            by_path.setdefault(str(Path(info["path"]).resolve()), file_id)
    registered, seen = [], set()
    added = updated = 0
    for file_path in paths:
        filename, ext = file_path.name, file_path.suffix.lower()
        file_path = file_path.resolve()
        canonical = str(file_path)
        if canonical in seen:
            continue
        seen.add(canonical)
        existing_id = by_path.get(canonical)
        file_id = existing_id or uuid.uuid4().hex
        file_info = dict(session_files.get(file_id, {}))
        for key in ("frame_count", "width", "height", "dtype", "source_version", "error"):
            file_info.pop(key, None)
        file_info.update({
            "id": file_id,
            "filename": session_files.get(file_id, {}).get("filename", filename),
            "path": canonical,
            "type": "tdms" if ext == ".tdms" else "image",
            "tdms_settings": {"normalize": data.normalize},
            "server_path": True,
        })
        if ext == ".tdms":
            parse_tdms_info(file_path, file_info)
        else:
            try:
                image, _ = extract_frame(file_path, file_info, 0)
                image.close()
            except Exception as exc:
                file_info["error"] = str(exc)
        session_files[file_id] = file_info
        by_path[canonical] = file_id
        if existing_id:
            updated += 1
        else:
            added += 1
        registered.append(file_info)

    state.save_user_session(data.username)
    return {"files": registered, "count": len(registered), "added": added, "updated": updated}


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

    save_path = state.contained_path(state.get_user_dir(username) / "results", file.filename)
    content = await file.read()
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
    previous = dict(file_info)
    img, frame_count = extract_frame(Path(file_info["path"]), file_info, index)
    if file_info != previous:
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


def _remove_session_file(sess: dict, file_id: str) -> bool:
    if file_id not in sess.get("files", {}):
        return False
    finfo = sess["files"][file_id]
    from services.tdms_cache import invalidate
    invalidate(finfo.get("path"))
    if not finfo.get("server_path"):
        try:
            Path(finfo["path"]).unlink(missing_ok=True)
        except Exception:
            pass
    del sess["files"][file_id]
    return True


@router.delete("/files/{username}/{file_id}")
async def delete_file(username: str, file_id: str):
    sess = state.require_session(username)
    if file_id not in sess.get("files", {}):
        raise HTTPException(status_code=404, detail="File not found")
    _remove_session_file(sess, file_id)
    state.save_user_session(username)
    return {"status": "deleted", "id": file_id}


@router.post("/files/bulk-delete")
async def bulk_delete_files(data: BulkDeleteFilesRequest):
    sess = state.require_session(data.username)
    if not data.file_ids:
        raise HTTPException(status_code=400, detail="Select at least one file")
    deleted = []
    missing = []
    for file_id in data.file_ids:
        if _remove_session_file(sess, file_id):
            deleted.append(file_id)
        else:
            missing.append(file_id)
    state.save_user_session(data.username)
    return {"status": "deleted", "deleted": deleted, "missing": missing}


@router.post("/files/reorder")
async def reorder_files(data: ReorderFilesRequest):
    sess = state.require_session(data.username)
    files = sess.get("files", {})
    if set(data.file_ids) != set(files.keys()):
        raise HTTPException(status_code=400, detail="Reorder list must contain exactly the current session file ids")
    sess["files"] = {file_id: files[file_id] for file_id in data.file_ids}
    state.save_user_session(data.username)
    return {"status": "reordered", "file_ids": data.file_ids}
