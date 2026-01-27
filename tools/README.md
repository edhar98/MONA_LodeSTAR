# Tools

Utilities for MONA LodeSTAR data processing and image manipulation.

## `tdms_to_png.py`

Converts TDMS files to PNG images with optional MP4 video output.

- Parallel processing
- Pattern matching for batch processing
- Automatic dtype conversion (uint8, uint16, float32)
- Normalization control (`--normed` / `--no-normed`)
- Skip existing files (resume capability)
- TDMS structure inspection

```bash
pip install nptdms pillow numpy
pip install imageio imageio-ffmpeg  # for MP4
```

```bash
python tools/tdms_to_png.py input.tdms -o output_dir
python tools/tdms_to_png.py "file_{:03d}.tdms" -o output --start-index 1 --num-files 10
python tools/tdms_to_png.py input.tdms --list-structure
python tools/tdms_to_png.py input.tdms -o output --to-mp4 --fps 30
python tools/tdms_to_png.py input.tdms -o output --no-normed  # preserve raw values
```

Jupyter:
```python
!python tools/tdms_to_png.py data/experiment.tdms -o output --to-mp4 --fps 30
```

Standalone executable (build with `./build_tdms_to_png.sh`):
```bash
./tdms_to_png input.tdms -o output --to-mp4 --fps 30
```

See [tdms_to_png_README.md](tdms_to_png_README.md) for full documentation.

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
