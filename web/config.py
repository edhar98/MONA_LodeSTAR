import os
from pathlib import Path

WEB_DIR = Path(__file__).parent


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


JUPYTER_MODE = _env_truthy("MONA_TRACK_JUPYTER")


def resolve_identity() -> str:
    for key in ("MONA_TRACK_USER", "JUPYTERHUB_USER", "USER"):
        val = os.environ.get(key, "").strip()
        if val:
            return val
    return "default"


def resolve_data_dir() -> Path:
    override = os.environ.get("MONA_TRACK_HOME", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if JUPYTER_MODE:
        return (Path.home() / "mona_track").resolve()
    return (WEB_DIR / "data").resolve()


DATA_DIR = resolve_data_dir()
DATA_DIR.mkdir(parents=True, exist_ok=True)

if JUPYTER_MODE:
    USERS_FILE = DATA_DIR / "users.json"
    JOBS_FILE = DATA_DIR / "training_jobs.json"
    BG_JOBS_FILE = DATA_DIR / "background_jobs.json"
else:
    USERS_FILE = WEB_DIR / "users.json"
    JOBS_FILE = WEB_DIR / "training_jobs.json"
    BG_JOBS_FILE = WEB_DIR / "background_jobs.json"

SRC_DIR = WEB_DIR.parent / "src"

DEFAULT_FEEDBACK_DIR = Path("/home/mona/mona_track_feedback")


def resolve_feedback_dir() -> Path:
    override = os.environ.get("MONA_TRACK_FEEDBACK_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if JUPYTER_MODE:
        return DEFAULT_FEEDBACK_DIR
    return (WEB_DIR / "feedback").resolve()


FEEDBACK_DIR = resolve_feedback_dir()
FEEDBACK_FILE = FEEDBACK_DIR / "feedback.jsonl"

ALLOWED_UPLOAD_EXT = {".tdms", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
