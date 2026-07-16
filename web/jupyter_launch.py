import os
import sys


def main() -> None:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: jupyter_launch.py PORT\n")
        sys.exit(2)
    port = sys.argv[1]
    web_dir = os.path.dirname(os.path.abspath(__file__))
    user = (
        os.environ.get("JUPYTERHUB_USER")
        or os.environ.get("USER")
        or os.environ.get("LOGNAME")
        or "default"
    )
    os.environ["MONA_TRACK_JUPYTER"] = "1"
    os.environ["MONA_TRACK_HOME"] = os.environ.get(
        "MONA_TRACK_HOME", os.path.expanduser("~/mona_track")
    )
    os.environ["MONA_TRACK_USER"] = user
    os.environ.setdefault("USER", user)
    os.environ.setdefault("JUPYTERHUB_USER", user)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.chdir(web_dir)
    sys.argv = [
        "uvicorn",
        "app:app",
        "--app-dir",
        web_dir,
        "--host",
        "127.0.0.1",
        "--port",
        port,
    ]
    from uvicorn.main import main as uvicorn_main
    uvicorn_main()


if __name__ == "__main__":
    main()
