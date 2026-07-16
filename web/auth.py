from datetime import datetime

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import state
from config import DATA_DIR, JUPYTER_MODE


router = APIRouter(prefix="/auth", tags=["auth"])


class UserLogin(BaseModel):
    username: str
    password: str


@router.get("/me")
async def me():
    if JUPYTER_MODE:
        username = state.ensure_jupyter_user()
        return {
            "mode": "jupyter",
            "username": username,
            "data_dir": str(get_user_data_dir(username)),
            "session": state.sessions.get(username, {}),
        }
    return {
        "mode": "standalone",
        "username": None,
        "data_dir": str(DATA_DIR),
        "session": None,
    }


def get_user_data_dir(username: str) -> str:
    return str(state.get_user_dir(username))


@router.post("/register")
async def register(data: UserLogin):
    if JUPYTER_MODE:
        return JSONResponse(
            status_code=400,
            content={"error": "Registration disabled in Jupyter mode"},
        )
    if data.username in state.users:
        return JSONResponse(status_code=400, content={"error": "Username already exists"})
    if len(data.username) < 3:
        return JSONResponse(status_code=400, content={"error": "Username must be at least 3 characters"})
    if len(data.password) < 4:
        return JSONResponse(status_code=400, content={"error": "Password must be at least 4 characters"})
    state.users[data.username] = {
        "password_hash": state.hash_password(data.password),
        "created_at": datetime.now().isoformat(),
    }
    state.get_user_dir(data.username)
    state.sessions[data.username] = {
        "files": {}, "samples": {}, "models": [], "masks": {}, "detect_files": {}
    }
    state.save_users()
    state.save_user_session(data.username)
    return {"status": "registered", "username": data.username}


@router.post("/login")
async def login(data: UserLogin):
    if JUPYTER_MODE:
        username = state.ensure_jupyter_user()
        return {"status": "logged_in", "username": username, "mode": "jupyter"}
    if data.username not in state.users:
        return JSONResponse(status_code=401, content={"error": "Invalid username or password"})
    if state.users[data.username]["password_hash"] != state.hash_password(data.password):
        return JSONResponse(status_code=401, content={"error": "Invalid username or password"})
    state.load_user_session(data.username)
    return {"status": "logged_in", "username": data.username, "mode": "standalone"}


@router.get("/check/{username}")
async def check_user(username: str):
    if JUPYTER_MODE:
        name = state.ensure_jupyter_user(username)
        return {"exists": True, "mode": "jupyter", "session": state.sessions.get(name, {})}
    if username not in state.users:
        return {"exists": False}
    state.load_user_session(username)
    return {"exists": True, "mode": "standalone", "session": state.sessions.get(username, {})}
