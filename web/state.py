import json
import hashlib
import threading
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import HTTPException

from config import (
    DATA_DIR, USERS_FILE, JOBS_FILE, BG_JOBS_FILE,
    JUPYTER_MODE, resolve_identity,
)

users: Dict[str, Dict[str, Any]] = {}
sessions: Dict[str, Dict[str, Any]] = {}
training_jobs: Dict[str, Dict[str, Any]] = {}
background_jobs: Dict[str, Dict[str, Any]] = {}
jobs_lock = threading.Lock()
_save_lock = threading.RLock()


def safe_name(name: str) -> str:
    if (not name or name.strip() != name or name in (".", "..")
            or any(c in name for c in ("/", "\\", "\x00", ":"))):
        raise HTTPException(status_code=400, detail="Invalid filename")
    return name


def contained_path(root: Path, name: str) -> Path:
    candidate = root / safe_name(name)
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise HTTPException(status_code=400, detail="Path escapes storage directory")
    return candidate


def _save_json(path: Path, data: dict):
    # Serialize concurrent writers and replace atomically, retaining the last
    # valid file if serialization or writing fails.
    with _save_lock:
        payload = json.dumps(data, indent=2, default=str)
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        try:
            with os.fdopen(fd, "w") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def get_user_dir(username: str) -> Path:
    safe_name(username)
    if JUPYTER_MODE and username != resolve_identity():
        raise HTTPException(status_code=403, detail="Username does not match Jupyter user")
    user_dir = DATA_DIR if JUPYTER_MODE else contained_path(DATA_DIR, username)
    for sub in ["uploads", "samples", "models", "results", "masks"]:
        contained_path(user_dir, sub).mkdir(parents=True, exist_ok=True)
    contained_path(user_dir / "results", "merged").mkdir(parents=True, exist_ok=True)
    return user_dir


def ensure_jupyter_user(username: Optional[str] = None) -> str:
    name = username or resolve_identity()
    expected = resolve_identity()
    if name != expected:
        raise HTTPException(status_code=403, detail="Username does not match Jupyter user")
    if name not in users:
        users[name] = {
            "password_hash": "",
            "created_at": datetime.now().isoformat(),
            "jupyter": True,
        }
    if name not in sessions:
        load_user_session(name)
    return name


def require_user(username: str) -> str:
    if JUPYTER_MODE:
        return ensure_jupyter_user(username)
    if username not in users:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if username not in sessions:
        load_user_session(username)
    return username


def merged_dir(username: str) -> Path:
    d = contained_path(get_user_dir(username) / "results", "merged")
    d.mkdir(parents=True, exist_ok=True)
    return d


def safe_merged_name(name: str) -> str:
    base = safe_name(name)
    if not base or base in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not base.lower().endswith(".mp4"):
        base = f"{base}.mp4"
    return base


def save_users():
    _save_json(USERS_FILE, users)


def load_users():
    if USERS_FILE.exists():
        with open(USERS_FILE) as f:
            loaded = json.load(f)
        users.clear()
        users.update(loaded)


def save_training_jobs():
    _save_json(JOBS_FILE, training_jobs)


def load_training_jobs():
    if JOBS_FILE.exists():
        with open(JOBS_FILE) as f:
            loaded = json.load(f)
        training_jobs.clear()
        training_jobs.update(loaded)
        for job in training_jobs.values():
            if job.get("status") in ("running", "queued"):
                job["status"] = "interrupted"
        save_training_jobs()


def save_background_jobs():
    _save_json(BG_JOBS_FILE, background_jobs)


def load_background_jobs():
    if BG_JOBS_FILE.exists():
        with open(BG_JOBS_FILE) as f:
            loaded = json.load(f)
        background_jobs.clear()
        background_jobs.update(loaded)
        for job in background_jobs.values():
            if job.get("status") in ("running", "queued"):
                job["status"] = "interrupted"
        save_background_jobs()


def save_user_session(username: str):
    user_dir = get_user_dir(username)
    if username in sessions:
        _save_json(contained_path(user_dir, "session.json"), sessions[username])


def load_user_session(username: str):
    session_file = contained_path(get_user_dir(username), "session.json")
    if session_file.exists():
        with open(session_file) as f:
            sessions[username] = json.load(f)
    else:
        sessions[username] = {
            "files": {}, "samples": {}, "models": [], "masks": {}, "detect_files": {}
        }
    sessions[username].setdefault("detect_files", {})


def require_session(username: str) -> dict:
    require_user(username)
    if username not in sessions:
        load_user_session(username)
    return sessions[username]


def get_session_file(username: str, file_id: str) -> dict:
    sess = require_session(username)
    info = sess.get("files", {}).get(file_id)
    if not info:
        info = sess.get("detect_files", {}).get(file_id)
    if not info:
        raise HTTPException(status_code=404, detail="File not found")
    return info
