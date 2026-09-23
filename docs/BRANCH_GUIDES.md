# Branch-Specific Guides

**Last Updated:** 2026-01-27  
**Branch:** Documentation & Reporting

This document provides guides for six ownership workstreams, historically called branches. Git branches remain `dev` and `main`. See [AGENT_WORKFLOW.md](AGENT_WORKFLOW.md) for coordinator, independent review, correction, and verification roles.

## Table of Contents

1. [Web Development](#web-development)
2. [Core Model Development](#core-model-development)
3. [Research & Experimentation](#research--experimentation)
4. [Tools & Automation](#tools--automation)
5. [Documentation & Reporting](#documentation--reporting)
6. [Maintenance & Operations](#maintenance--operations)

---

## Web Development

Historical status snapshot: `docs/archive/2026-01-verification/WEB_VERIFICATION.md`. Use `AGENTS.md` and current web files for live structure.

### Scope
- `web/app.py` — FastAPI application assembly
- `web/routers/` — route modules
- `web/services/` — TDMS, frame, and cache services
- `web/auth.py`, `web/config.py`, `web/state.py` — auth/session helpers, paths/settings, and runtime state
- `web/templates/index.html` — Sidebar SPA; see the current web integration inventory for controls and limitations
- `web/data/<username>/` — User data (runtime, gitignored)
- `web/users.json`, `web/training_jobs.json`, `web/background_jobs.json`

### Key Files
- **Backend:** `web/app.py`, `web/routers/`, `web/services/`, `web/auth.py`, `web/config.py`, `web/state.py`
- **Frontend:** `web/templates/index.html` (login overlay + app shell)
- **Data:** `web/data/<username>/{uploads,samples,models,results,masks}`

### API Endpoints (summary)

| Area | Endpoints |
|------|-----------|
| Auth | `POST /auth/register`, `POST /auth/login`, `GET /auth/check/{username}` |
| Health | `GET /health` |
| Files | chunked upload, `POST /files/load-path`, `POST /upload/csv`, list/delete/frame |
| Samples / masks | crop pool + circular mask |
| Training | `POST /train`, cancel, status, `WS /ws/train/{job_id}` |
| Models | list / delete / rename |
| Detection | `POST /detect/upload`, `GET /detect/frame/...`, `POST /detect` (legacy), `POST /detect/batch` |
| Tracking | `POST /track`, track visualize overview/video |
| Analysis | `POST /analyze/abp` |
| Jobs / results / TDMS / video / config | implemented across `web/app.py`, `web/routers/`, `web/services/`, `web/config.py`, and `web/state.py` |

### Data Flow

1. **Auth** — `web/users.json`; session `web/data/<username>/session.json`; login overlay hidden after success.
2. **Files** — Upload or server path; TDMS via installed `tdms_explorer.TDMSFileExplorer.extract_images()`.
3. **Training** — Multi-crop `samples/<name>/crop_*.jpg` (legacy `<name>.jpg`); inline DeepTrack + `dl.LodeSTAR` + `dl.Trainer`; **does not** call `src/detection/train_single_particle.py`; weights → `models/<name>_weights.pth`; progress via WebSocket.
4. **Detection** — UI: upload → `GET /detect/frame/...` with model_id/alpha/beta/cutoff. Single-model detection includes standard, area, watershed, and template modes. The model catalogue also discovers CLI summary entries; composite orchestration remains CLI-only. See [web inventory](WEB_INTEGRATION_AUDIT.md) for provenance and compatibility limits.
5. **Tracking / Analysis** — Calls Core `src/tracking/track_particles` and `src/analysis/analyze_tracks` when importable (`_tracking_available` / `_analysis_available`).

### Dependencies
- **sys.path:** `src`, `src/tracking`, `src/analysis` (not `tools/`).
- **Imports:** `utils`, `tdms_explorer.TDMSFileExplorer`, optional Core tracking/analysis.
- **External:** FastAPI, uvicorn, torch, deeptrack/deeplay, PIL, numpy, cv2, matplotlib, pandas.
- Integrates committed Core only; does not modify `src/` for Web features.

### Run assumptions
- Paths use `Path(__file__).parent.parent` (repo root).
- Recommended: from repo root, `uvicorn web.app:app --reload`.
- Jupyter proxy: frontend sets `API_BASE` from pathname `/proxy/<port>`.

### Example Usage

```bash
cd /path/to/MONA_LodeSTAR
uvicorn web.app:app --reload --host 0.0.0.0 --port 8000
# http://localhost:8000
```

---

## Core Model Development

Historical baseline snapshot: `docs/archive/2026-01-verification/BASELINE_REPORT.md`. Current path conventions are in `AGENTS.md` and `docs/QUICK_REFERENCE.md`.

### Scope
- `src/detection/`, `src/tracking/`, `src/analysis/`, and shared `src/utils.py`
- `src/config.yaml`, `src/samples.yaml`
- `src/requirements.txt`
- Model-related documentation

### Key Files

#### Training
- **`src/detection/train_single_particle.py`** - Main training script
  - Trains separate models for each particle type
  - Supports checkpointing and resuming
  - Integrates with WandB logging
  - Saves models to `models/<run_id>/`

- **`src/detection/train_enhanced.py`** - Enhanced training with multi-particle support
  - Supports single-particle and multi-particle modes
  - Alternative training approach

#### Testing
- **`src/detection/test_single_particle.py`** - Single particle model testing
  - Tests individual particle models
  - Generates test datasets (same/different shape/size)
  - Calculates precision, recall, F1-score

- **`src/detection/test_composite_model.py`** - Composite model testing
  - Tests multi-class detection
  - Uses model-specific detection parameters

#### Detection
- **`src/detection/detect_particles.py`** - Main particle detection script
  - Command-line detection interface
  - Supports batch processing

- **`src/detection/benchmark_trackpy.py`** - Trackpy linking baseline from existing detection CSVs

- **`src/detection/benchmark_trackpy_locate.py`** - `trackpy.locate` detector baseline against LodeSTAR detection CSVs

#### Models
- **`src/detection/custom_lodestar.py`** - Paper-accurate LodeSTAR implementation
  - Follows exact architecture from research paper
  - 3×Conv2D(3×3, 32) + ReLU → MaxPool2D(2×2) → 8×Conv2D(3×3, 32) + ReLU → Conv2D(1×1, 3)

- **`src/detection/composite_model.py`** - Composite model for multi-class detection
  - Combines multiple single-particle models
  - Weight-based classification
  - Detection merging with spatial clustering

#### Data Generation
- **`src/detection/image_generator.py`** - Image generation utilities
  - Synthetic microscopy image generation
  - Multiple dataset types
  - Trajectory generation

- **`src/detection/generate_samples.py`** - Sample image generation
  - Generates sample images for each particle type

#### Pipelines
- **`src/detection/run_single_particle_pipeline.py`** - Complete single-particle pipeline
  - Trains all particle types
  - Tests all trained models

- **`src/detection/run_composite_pipeline.py`** - Composite model pipeline
  - Tests composite model with all particle types

#### Tracking, Gap Filling, and Correction
- **`src/tracking/track_particles.py`** - NMS, Hungarian linking, and linear gap interpolation
- **`src/tracking/visualize_tracks.py`** - Track overview and video rendering
- **`src/tracking/lstm_track_predictor.py`** - Causal LSTM next-state baseline
- **`src/tracking/benchmark_lstm_gap_filling.py`** - Masked-gap benchmark for linear, persistence, velocity, and LSTM methods
- **`src/tracking/lstm_gap_filler.py`** - Two-sided LSTM/BiLSTM-style gap filler and Kalman probe
- **`src/tracking/build_supervised_correction_dataset.py`**, **`src/tracking/train_supervised_correction_lstm.py`**, **`src/tracking/apply_supervised_correction_lstm.py`** - Reference-calibrated LodeSTAR trajectory correction

#### Physics Analysis
- **`src/analysis/analyze_tracks.py`** - MSD and ABP model fitting
- **`src/analysis/analyze_motion_statistics.py`** - Raw motion statistics
- **`src/analysis/analyze_track_interactions.py`** - Nearest-neighbor and close-approach diagnostics
- **`src/analysis/analyze_confinement_drift.py`** - Radial occupancy and drift-field diagnostics
- **`src/analysis/compare_filtered_abp.py`** - ABP comparison under nearest-neighbor filters
- **`src/analysis/analyze_velocity_persistence.py`** - Velocity-persistence / AOUP-style diagnostic

#### Utilities
- **`src/utils.py`** - Core utilities
  - YAML loading/saving
  - XML parsing (Pascal VOC)
  - Logging setup
  - Visualization functions

### Configuration Files

#### `src/config.yaml`
Main training configuration:
- WandB settings (project, entity, mode)
- Training parameters (epochs, batch size, learning rate)
- Augmentation settings (multiplicative, additive noise)
- Detection settings (alpha, beta, cutoff, mode)
- Future detection-engine selection should expose `--detection-engine lodestar|trackpy`. On the JP benchmark frame, LodeSTAR `model.detect` took 180.1 ms on CUDA and 758.2 ms on CPU; `trackpy.locate(diameter=41)` took 265.1 ms on CPU.
- Model architecture (n_transforms, lodestar_version)
- Particle samples list

#### `src/samples.yaml`
Particle sample definitions:
- Particle types: Janus, Ring, Spot, Ellipse, Rod
- Parameters for each type (intensity, size, shape properties)

### Training Pipeline

1. **Data Preparation**
   ```python
   training_pipeline = create_single_particle_pipeline(config, particle_type)
   validation_pipeline = create_validation_pipeline(config, particle_type)
   ```

2. **Dataset Creation**
   ```python
   training_dataset = dt.pytorch.Dataset(training_pipeline, length=config['length'])
   validation_dataset = dt.pytorch.Dataset(validation_pipeline, length=config['length'] // 4)
   ```

3. **Model Initialization**
   ```python
   lodestar = dl.LodeSTAR(n_transforms=config['n_transforms']).build()
   ```

4. **Training**
   ```python
   trainer = dl.Trainer(max_epochs=config['max_epochs'], ...)
   trainer.fit(lodestar, train_dataloader, val_dataloader)
   ```

5. **Model Saving**
   - Checkpoints: `lightning_logs/<run_id>/checkpoints/`
   - Final weights: `models/<run_id>/<particle_type>_weights.pth`
   - Config: `models/<run_id>/config.yaml`

### Testing Pipeline

1. **Generate Test Datasets**
   - Same shape, same size
   - Same shape, different sizes
   - Different shapes, same size
   - Different shapes, different sizes

2. **Run Detection**
   - Load trained model
   - Process test images
   - Extract detections with coordinates and confidence

3. **Evaluate**
   - Compare with ground truth
   - Calculate metrics (precision, recall, F1)
   - Generate visualizations

### Dependencies
- **Imports:** `tools/wandb_logging.py`
- **External:** torch, lightning, deeptrack, deeplay, wandb, numpy, scipy, scikit-image, matplotlib, opencv-python, Pillow, PyYAML

### Coordination Rules
- Provides stable API for Web branch
- Commits changes before Web integration
- Maintains backward compatibility
- Model configs saved with each training run

### Example Usage

```bash
# Train single particle type
python src/detection/train_single_particle.py --particle Janus --config src/config.yaml

# Train all particle types
python src/detection/train_single_particle.py --config src/config.yaml

# Test single model
python src/detection/test_single_particle.py --particle Janus --model models/<run_id>/Janus_weights.pth

# Test composite model
python src/detection/test_composite_model.py --config src/config.yaml

# Run complete pipeline
python src/detection/run_single_particle_pipeline.py
```

---

## Research & Experimentation

Historical verification snapshot: `docs/archive/2026-01-verification/RESEARCH_VERIFICATION.md`.

### Scope
- `debug/` directory (all files)
- `src/*.ipynb` (all notebooks)
- Debug scripts: `src/debug_*.py`
- Experimental models: `src/detection/lodestar_*.py`

### Key Files

#### Diagnostics (`debug/diagnostics/`)
- **`diagnose_skip_connections.py`** - Skip connections analysis
  - Diagnoses skip connections implementation
  - Compares with standard LodeSTAR

#### Inspection (`debug/inspection/`)
- **`investigate_augmentations.py`** - Augmentation investigation
  - Analyzes data augmentation effects
  - Visualizes augmentation results

- **`architecture_diagram.py`** - Architecture visualization
  - Generates architecture diagrams
  - Visualizes model structure

- **`simple_architecture_diagram.py`** - Simplified architecture diagram
  - Simplified visualization

#### Notebooks (`src/`)
- **`Check_Augmentation.ipynb`** - Augmentation analysis
- **`Debug.ipynb`** - Debugging notebook
- **`detect_rings.ipynb`** - Ring detection experiments
- **`LodeStar.ipynb`** - Main LodeSTAR notebook

#### Experimental Models (`src/detection/`)
- **`src/detection/lodestar_with_skip_connections.py`** - Skip connections variant
  - Used by: `src/detection/train_single_particle.py` (conditional import)
  - Used by: `debug/diagnostics/diagnose_skip_connections.py`

- **`src/detection/lodestar_fixed_distributed.py`** - Fixed distributed training wrapper
  - Used by: `src/detection/lodestar_with_skip_connections.py` (inheritance)

- **`src/detection/lodestar_simple_skip.py`** - Simplified skip connections variant
  - Not directly imported (experimental)

#### Debug Scripts (`src/`)
- **`debug_area_detection.py`** - Debug area detection
- **`debug_disk_detection.py`** - Debug disk detection

### Usage Patterns

1. **Notebook-Based Analysis**
   - Use Jupyter notebooks for interactive analysis
   - Experiment with parameters and visualizations
   - Document findings in notebook markdown cells
   - Assume kernel CWD = `src/` or run from repo root per QUICK_REFERENCE and BASELINE_REPORT

2. **Diagnostic / Inspection Scripts**
   - Run from **repo root**. Scripts that import from `src/` need `PYTHONPATH=src`.
   - See `debug/README.md` for args, inputs, outputs per script.

3. **Experimental Models**
   - Test new architectures in separate files
   - Compare with standard implementations
   - Document findings before integration

### Coordination Rules
- Can use uncommitted Core code for prototyping
- Experimental findings inform Core development
- Temporary files should not be committed
- Notebooks are for exploration, not production

### Example Usage (from repo root)

```bash
PYTHONPATH=src:src/detection python debug/diagnostics/diagnose_skip_connections.py

PYTHONPATH=src python debug/inspection/investigate_augmentations.py [--particle Rod] [--config src/config_debug.yaml]

python debug/inspection/architecture_diagram.py

python debug/inspection/simple_architecture_diagram.py
```

---

## Tools & Automation

### Scope
- `tools/` directory (all files except notebooks)
- `elab.py` (root, convenience wrapper)
- ELAB-related scripts in root
- `tools/` documentation

### Key Files

#### Data Processing Tools
- **installed `tdms_explorer` package** - TDMS inspection/export package
  - Exports TDMS images
  - Creates MP4 animations
  - Provides file/channel inspection and statistics

- **`crop.py`** - Interactive image cropping
  - GUI for cropping images
  - Square selection with drag/resize

- **`mask.py`** - Circular ROI masking
  - Interactive circular mask creation
  - Noise background estimation

- **`merge_mp4.py`** - MP4 video merger
  - Merges multiple MP4 files
  - Pattern matching support

#### ELAB Integration (`tools/elab/`)
- **`elab/cli/elab_cli_simple.py`** - Simplified ELAB CLI (316 lines)
  - Simple interface for ELAB operations
  - Upload training/test results
  - Archive creation

- **`elab/cli/elab_cli.py`** - Full-featured ELAB CLI (1200 lines)
  - Complete ELAB API integration
  - Advanced features

- **`elab/scripts/upload_training.py`** - Upload training results
  - Uploads training results to ELAB
  - Creates experiments with metadata

- **`elab/scripts/upload_test.py`** - Upload test results
  - Uploads test results to ELAB
  - Creates experiments with test data

- **`elab/config/elab_config.yaml`** - ELAB configuration
  - Default experiment settings
  - Tags, directory mappings
  - File patterns

#### Logging
- **`wandb_logging.py`** - WandB logging abstraction
  - Optional wandb support
  - Provides `get_logger`, `get_run_id`, `set_summary`, `finish_run`
  - `TrainingMetricsCallback` for Lightning

#### Entry Points
- **`elab.py` (root)** - Convenience wrapper
  - Simple command mapping
  - Usage: `python elab.py upload-training`

- **`tools/elab_cli.py`** - Direct entry point
  - Usage: `python tools/elab_cli.py full` or `python tools/elab_cli.py simple`

### ELAB CLI Usage

#### Environment Setup
```bash
export ELAB_HOST_URL="https://your-elab-instance.com"
export ELAB_API_KEY="your-api-key"
export ELAB_VERIFY_SSL="true"
```

#### Upload Training Results
```bash
# Using root wrapper
python elab.py upload-training

# Using direct CLI (subcommand: upload-training)
python tools/elab_cli.py simple upload-training
```

#### Upload Test Results
```bash
# Using root wrapper
python elab.py upload-test

# Using direct CLI (subcommand: upload-test)
python tools/elab_cli.py simple upload-test
```

### Data Processing Usage

#### TDMS Conversion
```bash
# Single file
tdms-explorer export input.tdms output_dir

# To MP4
tdms-explorer animate input.tdms output.mp4 --fps 30
```

#### Image Cropping
```bash
python tools/crop.py input.png output_cropped.png
```

#### Masking
```bash
python tools/mask.py input.png output_masked.png
```

#### Video Merging
```bash
python tools/merge_mp4.py video_dir/ -o merged.mp4
```

### Dependencies
- **External:** elabapi-python, nptdms, imageio, imageio-ffmpeg, matplotlib, Pillow, numpy, PyQt5
- **Independent from core model**

### Coordination Rules
- Works with committed code from all branches
- Provides utilities for other branches
- Maintains backward compatibility
- ELAB config in `tools/elab/config/elab_config.yaml`

### Example Usage

```bash
# Convert TDMS to PNG
tdms-explorer export experiment.tdms output/

# Upload training results to ELAB
export ELAB_HOST_URL="https://elab.example.com"
export ELAB_API_KEY="your-key"
python elab.py upload-training

# Merge videos
python tools/merge_mp4.py video_dir/ -o merged.mp4
```

---

## Documentation & Reporting

### Scope
- All `.md` files in root
- `presentation/` directory
- Documentation in subdirectories
- PDF files in `docs/papers/`

### Key Files

#### Main Documentation
- **`README.md`** - Project overview and quick start
- **`docs/archive/2026-01-verification/`** - Historical January 2026 verification reports superseded by `AGENTS.md`

#### Feature Documentation
- **`COMPOSITE_MODEL_README.md`** - Composite model documentation
- **`MODEL_SPECIFIC_DETECTION_PARAMS.md`** - Detection parameters guide
- **`QUICK_START_COMPOSITE.md`** - Quick start for composite model

#### Implementation Documentation
- [Historical implementation summary](archive/2026-09-23-feature-notes/IMPLEMENTATION_SUMMARY.md)
- [Historical environment-specific distributed-training note](archive/2026-09-23-feature-notes/DEEPLAY_DISTRIBUTED_TRAINING_FIX.md)

#### Tools Documentation
- **`ELAB_CLI_SIMPLE_USAGE.md`** - ELAB CLI usage guide
- **`UPLOAD_TEST_RUNS.md`** - Test run upload documentation

#### Architecture Documentation (`docs/`)
- **`ARCHITECTURE.md`** - Architecture overview (this document's parent)
- **`BRANCH_GUIDES.md`** - This document
- **`QUICK_REFERENCE.md`** - Quick reference guide

#### Directory Documentation
- **`tools/README.md`** - Tools directory documentation
- **`test/README.md`** - Test directory documentation
- **`debug/README.md`** - Debug directory documentation

#### Presentation Materials (`presentation/`)
- LaTeX presentations
- Figures and diagrams
- Research paper references

### Documentation Standards

1. **Document Only Committed Features**
   - Verify features in git before documenting
   - Reference commit hashes for major changes

2. **Update When Branches Change**
   - Update docs when Core changes
   - Update docs when Web adds features
   - Keep architecture docs current

3. **Consistent Format**
   - Use markdown consistently
   - Include code examples
   - Reference related docs

4. **Clear Structure**
   - Table of contents for long docs
   - Clear section headings
   - Code examples with context

### Documentation Tasks

1. **Maintain README.md**
   - Keep current with project structure
   - Update file paths if changed
   - Reference new documentation

2. **Update Architecture Docs**
   - Reflect current branch structure
   - Document coordination rules
   - Map file ownership

3. **Branch-Specific Docs**
   - Document each branch's purpose
   - Provide usage examples
   - Explain coordination rules

4. **Quick Reference**
   - Common commands
   - File locations
   - Import patterns

### Coordination Rules
- Documents only committed features
- Updates documentation when branches change
- Maintains documentation standards
- References `AGENTS.md`, current docs, and the January 2026 archive only for historical context

### Example Usage

```bash
# Review documentation
cat README.md
cat docs/ARCHITECTURE.md
cat docs/BRANCH_GUIDES.md

# Update documentation after changes
# Edit relevant .md files
# Verify links work
# Check formatting
```

---

## Maintenance & Operations

### Scope
- `test/` directory (all files)
- `cleanup_lightning_logs.py`
- `.gitignore`
- Maintenance scripts
- Test documentation

### Key Files

#### Test Infrastructure (`test/`)
- **`run_tests.py`** - Test runner
  - Runs unit, regression, integration tests
  - Supports verbose output
  - Test type filtering

- **`unit/test_lodestar_models.py`** - LodeSTAR model unit tests
  - Tests model implementations
  - Validates architecture

- **`unit/test_utils.py`** - Utility function tests
  - Tests utility functions
  - Validates helper functions

- **`regression/test_backwards_compatibility.py`** - Backwards compatibility tests
  - Ensures no breaking changes
  - Validates API stability

- **`integration/`** - Integration tests (placeholder)
  - Full workflow tests
  - End-to-end validation

#### Maintenance Scripts
- **`cleanup_lightning_logs.py`** - Cleanup unused lightning logs
  - Removes unused log directories
  - Keeps logs for trained models

### Test Structure

#### Unit Tests (`test/unit/`)
Test individual functions, classes, and modules:
- Model implementations
- Utility functions
- Helper functions

#### Regression Tests (`test/regression/`)
Test that existing functionality continues to work:
- Backwards compatibility
- API stability
- Breaking change detection

#### Integration Tests (`test/integration/`)
Test complete workflows:
- Full training pipeline
- Detection pipeline
- End-to-end workflows

### Running Tests

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

### Maintenance Tasks

1. **Cleanup Operations**
   ```bash
   # Cleanup unused lightning logs
   python cleanup_lightning_logs.py
   ```

2. **Git Operations**
   - Maintain `.gitignore`
   - Coordinate commits across branches
   - Review file organization

3. **Dependency Management**
   - Update requirements files
   - Test dependency updates
   - Document breaking changes

### Coordination Rules
- Tests committed code from all branches
- Maintains test infrastructure
- Performs cleanup operations
- Coordinates git operations

### Example Usage

```bash
# Run all tests
python test/run_tests.py

# Cleanup logs
python cleanup_lightning_logs.py

# Check git status
git status
git ls-files
```

---

## Related Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - Architecture overview
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Quick reference guide
- [January 2026 verification archive](archive/2026-01-verification/README.md) - Superseded pre-restructure reports
