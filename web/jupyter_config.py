import os
import sys


def setup_mona_track():
    web_dir = os.path.dirname(os.path.abspath(__file__))
    home = os.path.expanduser("~/mona_track")
    python = os.environ.get("MONA_TRACK_PYTHON") or sys.executable
    launch = os.path.join(web_dir, "jupyter_launch.py")
    hub_user = (
        os.environ.get("JUPYTERHUB_USER")
        or os.environ.get("USER")
        or os.environ.get("LOGNAME")
        or "default"
    )
    return {
        "command": [python, launch, "{port}"],
        "timeout": 120,
        "launcher_entry": {
            "title": "MONA Track",
            "icon_path": os.path.join(web_dir, "icon.svg"),
        },
        "cwd": web_dir,
        "new_browser_tab": True,
        "environment": {
            "MONA_TRACK_JUPYTER": "1",
            "MONA_TRACK_HOME": home,
            "MONA_TRACK_FEEDBACK_DIR": "/home/mona/mona_track_feedback",
            "MONA_TRACK_USER": hub_user,
            "JUPYTERHUB_USER": hub_user,
            "USER": hub_user,
            "PYTHONUNBUFFERED": "1",
        },
    }
