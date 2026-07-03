# Quick Reference Guide

**Last Updated:** 2026-05-16  
**Branch:** Documentation & Reporting

Quick reference for common commands, file locations, and import patterns in MONA_LodeSTAR.

## Table of Contents

1. [Common Commands](#common-commands)
2. [File Locations](#file-locations)
3. [Import Patterns](#import-patterns)
4. [Configuration Files](#configuration-files)
5. [Output Directories](#output-directories)

---

## Common Commands

Core CLI scripts expect **current working directory = repository root** when using default paths: `train_single_particle.py`, `test_single_particle.py`, `test_composite_model.py`, and `detect_particles.py` (with default `--config src/config.yaml`) use CWD for `data/`, `models/`, and `trained_models_summary.yaml`. `run_composite_pipeline.py` resolves config and summary from script location and works from any CWD.

### Training

```bash
# Train single particle type
python src/train_single_particle.py --particle Janus --config src/config.yaml

# Train all particle types
python src/train_single_particle.py --config src/config.yaml

# Train with checkpoint resume
python src/train_single_particle.py --particle Janus --checkpoint lightning_logs/<run_id>/checkpoints/epoch=10.ckpt

# Run complete training pipeline
python src/run_single_particle_pipeline.py
```

### Testing

```bash
# Test single model
python src/test_single_particle.py --particle Janus --model models/<run_id>/Janus_weights.pth

# Test composite model
python src/test_composite_model.py --config src/config.yaml

# Test with visualization
python src/test_single_particle.py --particle Janus --model models/<run_id>/Janus_weights.pth --visualize
```

### Data Generation

```bash
# Generate sample images
python src/generate_samples.py

# Generate datasets
python src/image_generator.py
```

### Detection

```bash
# Detect particles in image or image directory
python src/detect_particles.py --model models/<run_id>/Janus_weights.pth --input input.png --output results/

# Template-matching orientation detection
python src/detect_particles.py --model models/<run_id>/Janus_weights.pth --input input.png --output results/ \
  --detection-mode template --orientation-template crops/f000_d000_phi0245.9.png
```

### Detection Engine Benchmarks

```bash
# Trackpy linking baseline from an existing LodeSTAR detection CSV
python src/benchmark_trackpy.py \
  --input detection_results/.../csv/<name>_detections.csv \
  --output detection_results/.../tracks/trackpy_tracks.csv \
  --min-dist 20 --search-range 30 --memory 10 --min-track 5

# Compare trackpy.locate detections against a LodeSTAR detection CSV
python src/benchmark_trackpy_locate.py \
  --lodestar detection_results/.../csv/<name>_detections.csv \
  --images data/.../images \
  --output detection_results/.../benchmark/trackpy_locate \
  --diameter 41 --min-dist 20 --match-distance 20
```

Single-frame timing result on `JP_Fe_wf_2_40_slm075_574_001.png`: LodeSTAR `model.detect` took 180.1 ms on CUDA and 758.2 ms on CPU; `trackpy.locate(diameter=41)` took 265.1 ms on CPU. LodeSTAR wins with CUDA; trackpy wins on CPU-only detection.

### Web Interface

```bash
# Start web server
cd /home/edgarharutyunyan/MONA_LodeSTAR
uvicorn web.app:app --reload

# Access at http://localhost:8000
```

### ELAB Integration

```bash
# Set environment variables
export ELAB_HOST_URL="https://your-elab-instance.com"
export ELAB_API_KEY="your-api-key"
export ELAB_VERIFY_SSL="true"

# Upload training results (using root wrapper)
python elab.py upload-training

# Upload test results (using root wrapper)
python elab.py upload-test

# Using direct CLI (subcommands: upload-training, upload-test, link-resources)
python tools/elab_cli.py simple upload-training
python tools/elab_cli.py simple upload-test
```

### Data Processing Tools

```bash
# Export TDMS images
tdms-explorer export input.tdms output_dir

# Convert TDMS to MP4
tdms-explorer animate input.tdms output.mp4 --fps 30

# Crop image
python tools/crop.py input.png output_cropped.png

# Create mask
python tools/mask.py input.png output_masked.png

# Merge MP4 videos
python tools/merge_mp4.py video_dir/ -o merged.mp4
```

### Testing

```bash
# Run all tests
python test/run_tests.py

# Run specific test types
python test/run_tests.py --type unit
python test/run_tests.py --type regression
python test/run_tests.py --type integration

# Verbose output
python test/run_tests.py --verbose
```

### Maintenance

```bash
# Cleanup unused lightning logs
python cleanup_lightning_logs.py
```

---

## File Locations

### Core Source Files

| File | Location | Purpose |
|------|----------|---------|
| Training | `src/train_single_particle.py` | Main training script |
| Testing | `src/test_single_particle.py` | Single model testing |
| Composite Testing | `src/test_composite_model.py` | Composite model testing |
| Detection | `src/detect_particles.py` | LodeSTAR particle detection |
| Trackpy Linking Benchmark | `src/benchmark_trackpy.py` | Trackpy baseline from detection CSV |
| Trackpy Locate Benchmark | `src/benchmark_trackpy_locate.py` | Compare `trackpy.locate` against LodeSTAR detections |
| Composite Model | `src/composite_model.py` | Multi-class detection |
| Custom LodeSTAR | `src/custom_lodestar.py` | Paper-accurate implementation |
| Image Generator | `src/image_generator.py` | Synthetic image generation |
| Utilities | `src/utils.py` | Core utilities |

### Configuration Files

| File | Location | Purpose |
|------|----------|---------|
| Main Config | `src/config.yaml` | Training configuration |
| Samples | `src/samples.yaml` | Particle definitions (Janus, Ring, Spot, Ellipse, Rod) |
| Model summary | `trained_models_summary.yaml` (repo root) | Trained model tracking; used by composite pipeline and test scripts |
| ELAB Config (active) | `tools/elab/config/elab_config.yaml` | ELAB configuration |
| ELAB Config (reference) | `elab_config.yaml` | ELAB reference |

### Web Files

| File | Location | Purpose |
|------|----------|---------|
| Backend | `web/app.py` | FastAPI application |
| Frontend | `web/templates/index.html` | Web UI |
| User Data | `web/data/<username>/` | User-specific data (gitignored) |

### Tools

| File | Location | Purpose |
|------|----------|---------|
| TDMS Explorer | installed `tdms_explorer` package | TDMS export, animation, inspection package |
| Image Cropper | `tools/crop.py` | Interactive cropping |
| Mask Tool | `tools/mask.py` | Circular ROI masking |
| Video Merger | `tools/merge_mp4.py` | MP4 merging |
| WandB Logging | `tools/wandb_logging.py` | WandB abstraction |
| ELAB CLI | `tools/elab_cli.py` | ELAB CLI entry point |
| ELAB CLI (simple) | `tools/elab/cli/elab_cli_simple.py` | Simplified ELAB CLI |

### Documentation

| File | Location | Purpose |
|------|----------|---------|
| Main README | `README.md` | Project overview |
| Architecture | `docs/ARCHITECTURE.md` | Architecture overview |
| Branch Guides | `docs/BRANCH_GUIDES.md` | Branch-specific guides |
| Quick Reference | `docs/QUICK_REFERENCE.md` | This document |
| Inventory | `INVENTORY.md` | Codebase inventory |
| Composite Model | `COMPOSITE_MODEL_README.md` | Composite model docs |
| Detection Params | `MODEL_SPECIFIC_DETECTION_PARAMS.md` | Detection parameters |

### Output Directories

| Directory | Location | Purpose |
|-----------|----------|---------|
| Models | `models/<run_id>/` | Trained model weights |
| Checkpoints | `lightning_logs/<run_id>/checkpoints/` | Training checkpoints |
| Detection Results | `detection_results/` | Detection outputs |
| Logs | `logs/` | Training and execution logs |
| WandB Logs | `wandb_logs/` | WandB experiment logs |
| Generated Data | `data/` | Generated datasets |

---

## Import Patterns

### Web Imports (web/app.py)

```python
import sys
from pathlib import Path

# Add src to path (resolves to repo root via __file__)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import from committed code only
from tdms_explorer import TDMSFileExplorer
import utils
```

Web currently imports `TDMSFileExplorer` from the installed `tdms_explorer` package, plus `src/utils.py`. In the MONA JupyterHub environment, use `/opt/mona_jupyterhub_env` or the `mona_env` alias and verify with `tdms-explorer --help` before starting `web.app`.

### Core Imports (src/*.py)

```python
# Import utilities
import utils

# Import from tools
from tools.wandb_logging import get_logger, get_run_id, set_summary, finish_run

# Import DeepTrack/DeepPlay
import deeptrack as dt
import deeplay as dl

# Import PyTorch
import torch
import torch.nn as nn

# Import Lightning
import pytorch_lightning as pl
```

### Tools Imports (tools/*.py)

```python
# Tools are independent, no imports from Core
# External dependencies only
import numpy as np
from PIL import Image
import nptdms
import elabapi_python
```

### Research Imports (debug/*.py, notebooks)

```python
# Can import from anywhere for experimentation
import sys
sys.path.insert(0, '../src')
from custom_lodestar import CustomLodeSTAR
import utils
```

### Test Imports (test/*.py)

```python
import sys
import os
import unittest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from custom_lodestar import CustomLodeSTAR
import utils
```

---

## Configuration Files

### Training Configuration (`src/config.yaml`)

Key sections:
- `wandb`: WandB settings (project, entity, mode)
- `samples`: List of particle types to train
- `max_epochs`: Number of training epochs
- `batch_size`: Batch size
- `lr`: Learning rate
- `n_transforms`: Number of transforms
- `lodestar_version`: Model version (custom, default, skip_connections)
- `alpha`, `beta`, `cutoff`, `mode`: Detection parameters
- `mul_min`, `mul_max`: Multiplicative noise range
- `add_min`, `add_max`: Additive noise range

### Sample Configuration (`src/samples.yaml`)

Defines particle types and parameters:
- `Janus`: Janus particle parameters
- `Ring`: Ring particle parameters
- `Spot`: Spot particle parameters
- `Ellipse`: Ellipse particle parameters
- `Rod`: Rod particle parameters

### ELAB Configuration (`tools/elab/config/elab_config.yaml`)

ELAB integration settings:
- Default experiment settings
- Tags and metadata
- Directory mappings
- File patterns
- Archive settings

---

## Output Directories

### Model Storage

```
models/
└── <run_id>/
    ├── <particle_type>_weights.pth
    └── config.yaml
```

### Checkpoints

```
lightning_logs/
└── <run_id>/
    └── checkpoints/
        ├── epoch=<N>.ckpt
        └── <particle_type>_final_epoch.ckpt
```

### Detection Results

```
detection_results/
└── Testing_<snr>/
    ├── composite/
    │   ├── same_shape_same_size/
    │   ├── same_shape_different_size/
    │   ├── different_shape_same_size/
    │   └── different_shape_different_size/
    └── <particle_type>_<model_id>/
```

### Logs

```
logs/
├── train_single_particle_<timestamp>.log
├── test_single_particle_<timestamp>.log
└── run_single_particle_pipeline_<timestamp>.log
```

### Web User Data

```
web/data/
└── <username>/
    ├── uploads/
    ├── samples/
    ├── models/
    ├── results/
    └── masks/
```

---

## Environment Variables

### ELAB Integration

```bash
export ELAB_HOST_URL="https://your-elab-instance.com"
export ELAB_API_KEY="your-api-key"
export ELAB_VERIFY_SSL="true"
```

### WandB (optional)

```bash
export WANDB_API_KEY="your-wandb-key"
export WANDB_PROJECT="LodeSTAR"
export WANDB_ENTITY="your-entity"
```

---

## Git Operations

### Commit Message Format

```
- Short description of change
- One logical change per commit
- Start with dash (-)
```

### Branch Coordination

- Web Development: Only integrates committed Core code
- Research: Can use uncommitted Core code for prototyping
- Tools: Works with committed code from all branches
- Documentation: Documents only committed features

---

## Related Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - Architecture overview
- [BRANCH_GUIDES.md](BRANCH_GUIDES.md) - Branch-specific guides
- [INVENTORY.md](../INVENTORY.md) - Complete file catalog
- [README.md](../README.md) - Project overview
- [TOOLS_VERIFICATION.md](../TOOLS_VERIFICATION.md) - Tools checks, entry points, env vars
