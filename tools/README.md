# Tools

Utilities for MONA LodeSTAR data processing and image manipulation.

## TDMS Explorer

TDMS inspection/export is provided by the installed `tdms_explorer` package in the MONA Python environment, not by a vendored copy under `tools/`.

- File/channel inspection
- Image export
- MP4 animation export
- Raw data access
- Statistics

```bash
tdms-explorer info input.tdms
tdms-explorer export input.tdms output_dir
tdms-explorer export input.tdms output_dir --start 0 --end 99 --prefix frame_
tdms-explorer export input.tdms output_dir --dtype float32 --format tiff
tdms-explorer export input.tdms output_dir --to-mp4 --fps 30
tdms-explorer animate input.tdms output.mp4 --fps 30
```

`convert` is deprecated; use `export` instead.

Jupyter:
```python
!tdms-explorer animate data/experiment.tdms output.mp4 --fps 30
```

The web app imports `TDMSFileExplorer` from the installed `tdms_explorer` package.

If JupyterLab shows two TDMS Explorer launcher tiles, check for stale package metadata:

```bash
/opt/mona_jupyterhub_env/bin/python -c "import importlib.metadata as md; print([f'{ep.name} {ep.dist.version}' for ep in md.entry_points(group='jupyter_serverproxy_servers') if ep.name == 'tdms-explorer'])"
```

In the current JupyterHub image, the duplicate is caused by old `~dms_explorer-1.0.0.dist-info` metadata next to the active `tdms_explorer-1.1.0.dist-info`. Removing it requires write access to `/opt/mona_jupyterhub_env`.

---

## Janus Orientation

`janus_crescent_ratio/` measures the bright or dark projected area of a Janus particle, reports the polar cosine angle and one-sided out-of-plane angle, and provides interactive crop, circle, and segmentation review tools.

```bash
python tools/janus_crescent_ratio/src/crescent_ratio.py --input-root data/ --output-dir tools/janus_crescent_ratio/outputs
python tools/janus_crescent_ratio/src/crescent_ratio_gui.py input.tdms
```

The MONA Track web interface imports this implementation directly and exposes it under Janus Orientation.

---

## `crop.py`

Interactive GUI for cropping images with square selection.

- Click-drag to draw square box
- Drag inside box to move
- Scroll wheel to resize
- Width display in pixels
- Center marker and diagonal guides

```bash
pip install matplotlib pillow numpy PyQt5
```

```bash
python tools/crop.py input.png output_cropped.png
```

---

## `mask.py`

Interactive GUI for circular ROI masking with noise background estimation.

- Two-phase workflow: ROI selection → noise region selection
- Click-drag to draw circle, drag to move, scroll to resize
- Calculates noise mean/std from background region
- Outputs masked image (black outside ROI)

```bash
pip install matplotlib pillow numpy PyQt5
```

```bash
python tools/mask.py input.png output_masked.png
```

---

## `merge_mp4.py`

Merge multiple MP4 files into a single video.

- Pattern matching for batch selection
- Configurable FPS

```bash
pip install imageio imageio-ffmpeg
```

```bash
python tools/merge_mp4.py video_dir/ -o merged.mp4
python tools/merge_mp4.py "video_{:03d}.mp4" -o merged.mp4 --start-index 1 --num-files 10
python tools/merge_mp4.py input.mp4 -o output.mp4 --fps 60
```

---

## `wandb_logging.py`

Abstracted WandB logging utility with optional wandb support.

- Works with or without wandb installed
- Provides `get_logger`, `get_run_id`, `set_summary`, `finish_run`
- `TrainingMetricsCallback` for Lightning training

```python
from tools.wandb_logging import get_logger, WANDB_AVAILABLE

logger = get_logger(config, particle_type)
```

---

## ELAB Integration (`elab/`)

ELAB integration tools for uploading training and test results to ELAB.

### Setup

```bash
export ELAB_HOST_URL="https://your-elab-instance.com"
export ELAB_API_KEY="your-api-key"
export ELAB_VERIFY_SSL="true"
```

### Usage

```bash
# Using root wrapper
python elab.py upload-training
python elab.py upload-test

# Using direct CLI (subcommands: upload-training, upload-test, link-resources)
python tools/elab_cli.py simple upload-training
python tools/elab_cli.py simple upload-test
```

### Configuration

- Reference config (structure/docs): `tools/elab/config/elab_config.yaml`. Current scripts do not load it; defaults are hardcoded.
- Reference: `elab_config.yaml` (root)

See [ELAB_CLI_SIMPLE_USAGE.md](../ELAB_CLI_SIMPLE_USAGE.md), [DUPLICATES_DOCUMENTATION.md](../DUPLICATES_DOCUMENTATION.md), and [TOOLS_VERIFICATION.md](../TOOLS_VERIFICATION.md) for entry points and verification.
