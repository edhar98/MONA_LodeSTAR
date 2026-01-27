import os
import sys

def setup_mona_track():
    return {
        "command": [
            sys.executable,
            "-m", "uvicorn",
            "app:app",
            "--port", "{port}",
            "--host", "127.0.0.1"
        ],
        "timeout": 60,
        "launcher_entry": {
            "title": "MONA Track",
            "icon_path": os.path.join(os.path.dirname(__file__), "icon.svg")
        },
        "cwd": os.path.dirname(__file__),
        "new_browser_tab": True
    }
