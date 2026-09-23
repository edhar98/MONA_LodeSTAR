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

Core CLI scripts expect **current working directory = repository root** when using default paths. `src/` is split into subpackages: detection (`src/detection/`), tracking (`src/tracking/`), and physics analysis (`src/analysis/`), plus the shared `src/utils.py`. Detection scripts `src/detection/train_single_particle.py`, `src/detection/test_single_particle.py`, `src/detection/test_composite_model.py`, and `src/detection/detect_particles.py` (with default `--config src/config.yaml`) use CWD for `data/`, `models/`, and `trained_models_summary.yaml`. `src/detection/run_composite_pipeline.py` resolves config and summary from script location and works from any CWD.

### Training

```bash
# Train single particle type
python src/detection/train_single_particle.py --particle Janus --config src/config.yaml

# Train all particle types
python src/detection/train_single_particle.py --config src/config.yaml

# Train with checkpoint resume
python src/detection/train_single_particle.py --particle Janus --checkpoint lightning_logs/<run_id>/checkpoints/epoch=10.ckpt

# Run complete training pipeline
python src/detection/run_single_particle_pipeline.py
```

### Testing

```bash
# Test single model
python src/detection/test_single_particle.py --particle Janus --model models/<run_id>/Janus_weights.pth

# Test composite model
python src/detection/test_composite_model.py --config src/config.yaml

# Test with visualization
python src/detection/test_single_particle.py --particle Janus --model models/<run_id>/Janus_weights.pth --visualize
```

### Data Generation

```bash
# Generate sample images
python src/detection/generate_samples.py

# Generate datasets
python src/detection/image_generator.py
```

### Detection

```bash
# Detect particles in image or image directory
python src/detection/detect_particles.py --model models/<run_id>/Janus_weights.pth --input input.png --output results/

# Template-matching orientation detection
python src/detection/detect_particles.py --model models/<run_id>/Janus_weights.pth --input input.png --output results/ \
  --detection-mode template --orientation-template crops/f000_d000_phi0245.9.png
```

### Detection Engine Benchmarks

```bash
# Trackpy linking baseline from an existing LodeSTAR detection CSV
python src/detection/benchmark_trackpy.py \
  --input detection_results/.../csv/<name>_detections.csv \
  --output detection_results/.../tracks/trackpy_tracks.csv \
  --min-dist 20 --search-range 30 --memory 10 --min-track 5

# Compare trackpy.locate detections against a LodeSTAR detection CSV
python src/detection/benchmark_trackpy_locate.py \
  --lodestar detection_results/.../csv/<name>_detections.csv \
  --images data/.../images \
  --output detection_results/.../benchmark/trackpy_locate \
  --diameter 41 --min-dist 20 --match-distance 20
```

Single-frame timing result on `JP_Fe_wf_2_40_slm075_574_001.png`: LodeSTAR `model.detect` took 180.1 ms on CUDA and 758.2 ms on CPU; `trackpy.locate(diameter=41)` took 265.1 ms on CPU. LodeSTAR wins with CUDA; trackpy wins on CPU-only detection.

### Particle Tracking

```bash
# NMS -> Hungarian linking -> gap interpolation on a detection CSV
python src/tracking/track_particles.py \
  --input detection_results/.../csv/<name>_detections.csv \
  --output detection_results/.../tracks/<name>_tracks.csv \
  --min-dist 20 --max-link 30 --min-track 5 --max-gap 10

# Render track overview + video
python src/tracking/visualize_tracks.py \
  --tracks detection_results/.../tracks/<name>_tracks.csv \
  --images data/.../images --output detection_results/.../visualization/
```

### Gap Filling & Trajectory Correction (benchmark/probe)

```bash
# Causal LSTM next-state baseline
python src/tracking/lstm_track_predictor.py train --tracks <tracks.csv> --model-out lstm_outputs/<name>.pt
# Two-sided BiLSTM gap filler (+ Kalman probe)
python src/tracking/lstm_gap_filler.py train --tracks <tracks.csv> --model-out lstm_outputs/<name>.pt
# Reference-calibrated supervised corrector (pair -> train -> apply)
python src/tracking/build_supervised_correction_dataset.py --lodestar-tracks <tracks.csv> --reference-glob '<ref>/*_video.csv' --output-dir supervised_correction_outputs/<name>
python src/tracking/train_supervised_correction_lstm.py --dataset supervised_correction_outputs/<name>/<windows>.npz --model-out supervised_correction_outputs/<name>/model.pt
python src/tracking/apply_supervised_correction_lstm.py --tracks <tracks.csv> --model supervised_correction_outputs/<name>/model.pt --output supervised_correction_outputs/<name>/refined.csv
```

Linear gap interpolation remains the production baseline; LSTM/BiLSTM/Kalman are benchmark/probe stages. See AGENTS.md for the physics-first decision framing.

### Physics Analysis

```bash
# Fit ABP model (D_t, v0, D_r) and plot MSD
python src/analysis/analyze_tracks.py \
  --tracks detection_results/.../tracks/<name>_tracks.csv \
  --output detection_results/.../analysis/ --px-size 0.078

# Physics-first diagnostics
python src/analysis/analyze_motion_statistics.py --tracks <tracks.csv> --output analysis_outputs/motion_statistics/
python src/analysis/analyze_track_interactions.py --tracks <tracks.csv> --output analysis_outputs/interactions/
python src/analysis/analyze_confinement_drift.py --tracks <tracks.csv> --output analysis_outputs/confinement/

# Model comparison
python src/analysis/compare_filtered_abp.py --tracks <tracks.csv> --output analysis_outputs/model_comparison/filtered_abp/
python src/analysis/analyze_velocity_persistence.py --tracks <tracks.csv> --output analysis_outputs/model_comparison/velocity_persistence/
```

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
| Training | `src/detection/train_single_particle.py` | Main training script |
| Testing | `src/detection/test_single_particle.py` | Single model testing |
| Composite Testing | `src/detection/test_composite_model.py` | Composite model testing |
| Detection | `src/detection/detect_particles.py` | LodeSTAR particle detection |
| Trackpy Linking Benchmark | `src/detection/benchmark_trackpy.py` | Trackpy baseline from detection CSV |
| Trackpy Locate Benchmark | `src/detection/benchmark_trackpy_locate.py` | Compare `trackpy.locate` against LodeSTAR detections |
| Composite Model | `src/detection/composite_model.py` | Multi-class detection |
| Custom LodeSTAR | `src/detection/custom_lodestar.py` | Paper-accurate implementation |
| Image Generator | `src/detection/image_generator.py` | Synthetic image generation |
| Utilities | `src/utils.py` | Core utilities (shared hub at `src/` root) |
| Tracking | `src/tracking/track_particles.py` | NMS + Hungarian linking + gap interpolation |
| Track Visualization | `src/tracking/visualize_tracks.py` | Track overview/video rendering |
| LSTM Predictor | `src/tracking/lstm_track_predictor.py` | Causal next-state baseline |
| Gap Filler (BiLSTM) | `src/tracking/lstm_gap_filler.py` | Two-sided gap filling + Kalman probe |
| Supervised Correction | `src/tracking/build_supervised_correction_dataset.py`, `src/tracking/train_supervised_correction_lstm.py`, `src/tracking/apply_supervised_correction_lstm.py` | Reference-calibrated trajectory corrector |
| Physics Analysis | `src/analysis/analyze_tracks.py` | ABP/MSD fit (D_t, v0, D_r) |
| Motion / Interactions / Confinement | `src/analysis/analyze_motion_statistics.py`, `src/analysis/analyze_track_interactions.py`, `src/analysis/analyze_confinement_drift.py` | Physics-first diagnostics |
| Model Comparison | `src/analysis/compare_filtered_abp.py`, `src/analysis/analyze_velocity_persistence.py` | Filtered ABP / AOUP diagnostics |

### Configuration Files

| File | Location | Purpose |
|------|----------|---------|
| Main Config | `src/config.yaml` | Training configuration |
| Samples | `src/samples.yaml` | Particle definitions (Janus, Ring, Spot, Ellipse, Rod) |
| Model summary | `trained_models_summary.yaml` (repo root) | Trained model tracking; used by composite pipeline and test scripts |
| ELAB Config (reference) | `tools/elab/config/elab_config.yaml` | Reference only; scripts use CLI/env/hardcoded defaults |
| ELAB Config (reference) | `elab_config.yaml` | ELAB reference |

### Web Files

| File | Location | Purpose |
|------|----------|---------|
| Backend | `web/app.py` | FastAPI application |
| Routers | `web/routers/` | Route modules for files and TDMS workflows |
| Services | `web/services/` | TDMS operations, frame extraction, and cache helpers |
| Auth / Config / State | `web/auth.py`, `web/config.py`, `web/state.py` | Session helpers, paths/settings, and runtime state |
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
| January 2026 verification archive | `docs/archive/2026-01-verification/` | Superseded pre-restructure reports |
| Composite Model | `COMPOSITE_MODEL_README.md` | Composite model docs |
| Detection Params | `MODEL_SPECIFIC_DETECTION_PARAMS.md` | Detection parameters |

### Output Directories

| Directory | Location | Purpose |
|-----------|----------|---------|
| Models | `models/<run_id>/` | Trained model weights |
| Checkpoints | `lightning_logs/<run_id>/checkpoints/` | Training checkpoints |
| Detection Results | `detection_results/` | Detection outputs |
| Analysis Outputs | `analysis_outputs/` | Motion statistics, physics diagnostics, and model comparisons |
| LSTM Outputs | `lstm_outputs/` | Causal LSTM, BiLSTM gap filler, and Kalman benchmark outputs |
| Supervised Correction Outputs | `supervised_correction_outputs/` | Pairing datasets, trained correctors, and refined tracks |
| Orientation Test Outputs | `lodestar_orientation_test_out/` | Orientation experiment outputs |
| Logs | `logs/` | Training and execution logs |
| WandB Logs | `wandb_logs/` | WandB experiment logs |
| Generated Data | `data/` | Generated datasets |

---

## Import Patterns

### Web Imports (web/app.py)

```python
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR / "tracking"))
sys.path.insert(0, str(SRC_DIR / "analysis"))

from tdms_explorer import TDMSFileExplorer
import utils
# optional: track_particles, analyze_tracks (soft-fail if missing)
```

Web uses installed `tdms_explorer`, `src/utils`, and optionally Core tracking/analysis. Historical web verification notes are archived under `docs/archive/2026-01-verification/`. On JupyterHub use `/opt/mona_jupyterhub_env` / `mona_env`.

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
sys.path.insert(0, '../src/detection')
from custom_lodestar import customLodeSTAR
import utils
```

### Test Imports (test/*.py)

```python
import sys
import os
import unittest

# Add src and src/detection to path (per AGENTS.md import conventions)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'detection'))

from custom_lodestar import customLodeSTAR
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

Reference settings only; current scripts do not load this YAML. Use CLI arguments and environment variables for runtime settings.

Documented fields:
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
- [January 2026 verification archive](archive/2026-01-verification/README.md) - Superseded pre-restructure reports
- [README.md](../README.md) - Project overview
