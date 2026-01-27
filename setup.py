from setuptools import setup, find_packages

setup(
    name="mona-track",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "fastapi",
        "uvicorn",
        "python-multipart",
        "jupyter-server-proxy"
    ],
    entry_points={
        "jupyter_serverproxy_servers": [
            "mona-track = web.jupyter_config:setup_mona_track"
        ]
    },
    package_data={
        "web": ["templates/*.html", "icon.svg"]
    }
)