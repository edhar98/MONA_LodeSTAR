import json
import hashlib
import threading
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


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def get_user_dir(username: str) -> Path:
    user_dir = DATA_DIR if JUPYTER_MODE else (DATA_DIR / username)
    for sub in ["uploads", "samples", "models", "results", "masks"]:
        (user_dir / sub).mkdir(parents=True, exist_ok=True)
    (user_dir / "results" / "merged").mkdir(parents=True, exist_ok=True)
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
    d = get_user_dir(username) / "results" / "merged"
    d.mkdir(parents=True, exist_ok=True)
    return d


def safe_merged_name(name: str) -> str:
    base = Path(name).name.strip()
    if not base or base in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not base.lower().endswith(".mp4"):
        base = f"{base}.mp4"
    return base


def save_users():
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2, default=str)


def load_users():
    global users
    if USERS_FILE.exists():
        with open(USERS_FILE) as f:
            users = json.load(f)


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
    sessions[username].setdefault("detect_files", {})


def require_session(username: str) -> dict:
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
