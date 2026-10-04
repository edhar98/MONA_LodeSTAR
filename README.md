# MONA LodeSTAR

MONA LodeSTAR detects particles in microscopy images, links detections into
trajectories, and provides motion analysis. It includes a FastAPI web interface
for image and TDMS workflows and command-line tools for training and research.

## Start here

- [Web guide](docs/WEB_GUIDE.md): loading data, training, detection, tracking, and export.
- [Deployment](docs/DEPLOYMENT.md): JupyterHub setup, updates, and troubleshooting.
- [Command reference](docs/QUICK_REFERENCE.md): core CLI workflows.
- [Architecture](docs/ARCHITECTURE.md): code structure and data flow.
- [Known limitations](docs/KNOWN_ISSUES.md): security, scientific validity, and pending improvements.
- [Tests](test/README.md): automated checks and fixture requirements.

## Run the web app

Run from the repository root using the existing MONA environment:

```bash
/opt/mona_jupyterhub_env/bin/python -m uvicorn web.app:app --host 127.0.0.1 --port 8002
```

Open http://localhost:8002 on the same machine. For remote access, use the
authenticated JupyterHub launcher. A manually started app can also be reached
through your Jupyter user server at `/user/YOUR_USERNAME/proxy/8002/` if that
server can reach the app. This proxy route does not switch standalone mode to
automatic Hub identity.

Binding to `0.0.0.0` exposes the standalone app to the network. Use it only for
trusted-network testing; standalone accounts are not a secure multi-user
authentication boundary. See the deployment guide before exposing a service.

## Environment

The current MONA environment uses Python 3.10 at
`/opt/mona_jupyterhub_env/bin/python`. Core dependencies are listed in
`src/requirements.txt`; web packaging is defined in `setup.py`. CUDA is optional
but useful for detector training and inference. TDMS support comes from the
installed `tdms_explorer` package.

For a new environment, install dependencies in an isolated environment and run
the tests before use. Requirements are not fully pinned, so installation alone
does not reproduce the tested MONA environment. The web package currently needs
the adjacent `src/` and `tools/` directories; deploy the source checkout, not an
untested standalone wheel.

## Main workflow

1. Load microscopy images or TDMS files.
2. Prepare particle crops and train a detector, or use your existing web-trained model.
3. Run single-model or composite detection.
4. Link detections and fill short gaps with linear interpolation.
5. Inspect trajectories, export results, and evaluate motion statistics.

The web detector catalog contains only the current user's web-trained models.
It does not import `trained_models_summary.yaml`. CLI models and research
checkpoints remain separate.

Two-sided LSTM gap refinement is optional and experimental. The historical gap
benchmark has target leakage; its reported advantage must not be treated as
validated superiority. Linear interpolation remains the default. ABP is a
baseline model whose assumptions need testing, not a guaranteed description of
the particles.

## Repository layout

- `src/detection/`: detector models, training, inference, and orientation.
- `src/tracking/`: linking, interpolation, visualization, and learned corrections.
- `src/analysis/`: MSD, ABP, interactions, and confinement diagnostics.
- `src/utils.py`: shared image and detection utilities.
- `web/`: API, browser interface, and JupyterHub launcher.
- `tools/`: image processing, logging, and ELab utilities.
- `test/`: unit, regression, and opt-in integration checks.
- `docs/`: user and maintainer documentation.
- `debug/`, `notebooks/`, and analysis output folders: research work and artifacts.

Generated models, data, and research results are not deployment instructions.
Do not delete or regenerate them as part of a code update.

## Specialist guides

- [Composite CLI model](COMPOSITE_MODEL_README.md)
- [Tools](tools/README.md)
- [ELab CLI](ELAB_CLI_SIMPLE_USAGE.md)

## Scientific reference and license

The detector is based on *Single-shot self-supervised object detection in
microscopy*, Midtvedt et al., Nature Communications 13, 7492 (2022),
DOI: 10.1038/s41467-022-35004-y.

This project is licensed under [GNU GPL-3.0](LICENSE).
