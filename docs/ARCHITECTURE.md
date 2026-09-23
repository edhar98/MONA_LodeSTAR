# MONA_LodeSTAR Architecture Overview

**Last Updated:** 2026-05-16  
**Branch:** Documentation & Reporting

## Six ownership workstreams

MONA_LodeSTAR groups ownership into six workstreams. These are responsibility domains, not six Git branches; Git uses `dev` and `main`. The [agent workflow](AGENT_WORKFLOW.md) describes coordination and independent review.

1. **Web Development** - User-facing web interface
2. **Core Model Development** - Training, testing, and detection pipelines
3. **Research & Experimentation** - Experimental code, notebooks, debugging
4. **Tools & Automation** - Data processing utilities and ELAB integration
5. **Documentation & Reporting** - Documentation, presentations, reports
6. **Maintenance & Operations** - Testing, cleanup, git operations

## Branch Ownership

### Web Development Branch
**Owns:**
- `web/` directory (all files)
- `setup.py` (web package setup)
- Web-specific documentation

**Dependencies:**
- Uses: `src/utils.py` (committed code only)
- Uses: installed `tdms_explorer.TDMSFileExplorer`
- External: FastAPI, uvicorn, jupyter-server-proxy

**Coordination Rules:**
- Only integrates committed Core code
- Does not modify `src/` files directly
- Calls existing functions via imports

### Core Model Development Branch
**Owns:**
- `src/detection/`, `src/tracking/`, `src/analysis/`, and shared `src/utils.py`
- `src/config.yaml`, `src/samples.yaml`
- `src/requirements.txt`
- Model-related documentation

**Dependencies:**
- Uses: `tools/wandb_logging.py`
- External: torch, lightning, deeptrack, deeplay, wandb

**Coordination Rules:**
- Provides stable API for Web branch
- Commits changes before Web integration
- Maintains backward compatibility

### Research & Experimentation Branch
**Owns:**
- `debug/` directory (all files)
- `src/*.ipynb` (all notebooks)
- Debug scripts: `src/debug_*.py`
- Experimental models: `src/detection/lodestar_*.py`

**Dependencies:**
- Uses: Core model files for experimentation
- May create temporary files

**Coordination Rules:**
- Can use uncommitted Core code for prototyping
- Experimental findings inform Core development
- Temporary files should not be committed

### Tools & Automation Branch
**Owns:**
- `tools/` directory (all files except notebooks)
- `elab.py` (root, convenience wrapper)
- ELAB-related scripts in root
- `tools/` documentation

**Dependencies:**
- External: elabapi-python, nptdms, imageio
- Independent from core model

**Coordination Rules:**
- Works with committed code from all branches
- Provides utilities for other branches
- Maintains backward compatibility

### Documentation & Reporting Branch
**Owns:**
- All `.md` files in root
- `presentation/` directory
- Documentation in subdirectories
- PDF files in `docs/papers/`

**Dependencies:**
- Documents all other branches

**Coordination Rules:**
- Documents only committed features
- Updates documentation when branches change
- Maintains documentation standards

### Maintenance & Operations Branch
**Owns:**
- `test/` directory (all files)
- `cleanup_lightning_logs.py`
- `.gitignore`
- Maintenance scripts
- Test documentation

**Dependencies:**
- Tests all other branches
- Uses: Core model files for testing

**Coordination Rules:**
- Tests committed code from all branches
- Maintains test infrastructure
- Performs cleanup operations

## File Organization

### Core Source (`src/`)
- **Detection:** `src/detection/train_single_particle.py`, `src/detection/test_single_particle.py`, `src/detection/detect_particles.py`
- **Detection Benchmarks:** `src/detection/benchmark_trackpy.py`, `src/detection/benchmark_trackpy_locate.py`
- **Detection Models:** `src/detection/custom_lodestar.py`, `src/detection/composite_model.py`
- **Data Generation:** `src/detection/image_generator.py`, `src/detection/generate_samples.py`
- **Tracking:** `src/tracking/track_particles.py`, `src/tracking/visualize_tracks.py`
- **Gap Filling:** `src/tracking/lstm_track_predictor.py`, `src/tracking/benchmark_lstm_gap_filling.py`, `src/tracking/lstm_gap_filler.py`
- **Supervised Correction:** `src/tracking/build_supervised_correction_dataset.py`, `src/tracking/train_supervised_correction_lstm.py`, `src/tracking/apply_supervised_correction_lstm.py`
- **Physics Analysis:** `src/analysis/analyze_tracks.py`, `src/analysis/analyze_motion_statistics.py`, `src/analysis/analyze_track_interactions.py`, `src/analysis/analyze_confinement_drift.py`, `src/analysis/compare_filtered_abp.py`, `src/analysis/analyze_velocity_persistence.py`
- **Utilities:** `utils.py`
- **Config:** `config.yaml`, `samples.yaml`

### Web Interface (`web/`)
- **App Assembly:** `app.py`
- **Routers:** `routers/`
- **Services:** `services/`
- **Auth/Config/State:** `auth.py`, `config.py`, `state.py`
- **Frontend:** `templates/index.html` (single-page app)
- **Data:** `data/<username>/` (runtime, gitignored)

### Tools (`tools/`)
- **Data Processing:** installed `tdms_explorer`, `crop.py`, `mask.py`, `merge_mp4.py`
- **ELAB Integration:** `elab/` directory
- **Logging:** `wandb_logging.py`

### Research (`debug/`)
- **Diagnostics:** `diagnostics/`
- **Inspection:** `inspection/`
- **Experiments:** `experiments/`

### Testing (`test/`)
- **Unit Tests:** `unit/`
- **Regression Tests:** `regression/`
- **Integration Tests:** `integration/`

## Dependency Map

### Core Dependencies
```
src/utils.py
  ├─ Used by: src/detection/detect_particles.py, src/detection/train_single_particle.py, src/detection/test_single_particle.py
  ├─ Used by: src/detection/composite_model.py, src/detection/run_composite_pipeline.py
  ├─ Used by: src/detection/image_generator.py, src/detection/generate_samples.py
  └─ Used by: web/app.py

src/detection/custom_lodestar.py
  ├─ Used by: src/detection/detect_particles.py, src/detection/train_single_particle.py
  ├─ Used by: debug_disk_detection.py, debug_area_detection.py
  └─ Used by: test/unit/test_lodestar_models.py

src/detection/composite_model.py
  ├─ Used by: src/detection/run_composite_pipeline.py
  └─ Used by: src/detection/test_composite_model.py
```

### Web Dependencies
```
web/app.py
  ├─ Imports: tdms_explorer.TDMSFileExplorer
  ├─ Imports: src/utils
  ├─ Includes: web/routers/
  └─ Delegates runtime work to: web/services/, web/auth.py, web/config.py, web/state.py
```

### Tools Dependencies
```
tools/elab/cli/elab_cli_simple.py
  └─ Used by: tools/elab/scripts/upload_test.py
  └─ Used by: tools/elab/scripts/upload_training.py
  └─ Used by: tools/elab_cli.py

tools/wandb_logging.py
  └─ Used by: src/detection/train_single_particle.py
```

## Coordination Rules

### Detection Engine Direction
- Current production detector is LodeSTAR via `src/detection/detect_particles.py`
- `trackpy.locate` is a strong classical baseline for position-only detection and should be exposed through a future `--detection-engine lodestar|trackpy` flag
- Single-frame benchmark on `JP_Fe_wf_2_40_slm075_574_001.png`: LodeSTAR `model.detect` took 180.1 ms on CUDA and 758.2 ms on CPU; `trackpy.locate(diameter=41)` took 265.1 ms on CPU
- Downstream orientation, tracking, gap interpolation, and ABP analysis should consume a shared detection CSV schema regardless of engine

### Git Strategy
- Two branches: `dev` (active development) and `main` (stable releases)
- All feature work happens on `dev`
- Merge to `main` only after testing and review
- Commit messages: start with dash (-), short and clear, one logical change per commit

### Branch Coordination
- **Web Development** integrates only committed Core changes (no uncommitted imports)
- **Research & Experimentation** can use uncommitted Core code for prototyping
- **Tools & Automation** works with committed code from all branches
- **Documentation** tracks committed features only

### Import Patterns
- Web imports from `src/` using `sys.path.insert`
- Core imports from `tools/` using relative imports where possible
- Tools are independent and don't import from Core
- Research can import from anywhere for experimentation

## Configuration Files

### Training Configuration
- **`src/config.yaml`** - Main training configuration
  - Used by: `src/detection/train_single_particle.py`, `src/detection/run_training.py`, `src/detection/train_enhanced.py`
  - Contains: WandB settings, training parameters, augmentation settings, particle samples

### Sample Configuration
- **`src/samples.yaml`** - Particle sample definitions
  - Used by: `src/detection/generate_samples.py`, `src/detection/image_generator.py`
  - Contains: Particle types (Janus, Ring, Spot, Ellipse, Rod) with parameters

### ELAB Configuration
- **`elab_config.yaml` (root)** - Reference/documentation for ELAB configuration
- **`tools/elab/config/elab_config.yaml`** - Reference configuration; current ELAB scripts do not load it
  - Actual defaults: CLI arguments, environment variables, and script constants
  - Contains: Default experiment settings, tags, directory mappings, file patterns

The January 2026 duplicate-file notes are archived under `docs/archive/2026-01-verification/`; use `AGENTS.md` for current source-of-truth guidance.

## Output Structure

### Generated Data
- **`data/`** - Generated datasets and sample images
- **`models/`** - Trained model weights and checkpoints
- **`detection_results/`** - Detection outputs and evaluation results
- **`analysis_outputs/`** - Motion statistics, ABP/model comparison, confinement, and interaction outputs
- **`lstm_outputs/`** - LSTM/BiLSTM/Kalman benchmark outputs
- **`supervised_correction_outputs/`** - Reference-calibrated correction datasets, models, and refined tracks
- **`lodestar_orientation_test_out/`** - Orientation experiment outputs

### Logs
- **`logs/`** - Training and execution logs
- **`lightning_logs/`** - PyTorch Lightning logs
- **`wandb_logs/`** - Weights & Biases logs

### Summary Files
- **`test_results_summary.yaml`** - Test results by particle type and dataset
- **`trained_models_summary.yaml`** - Model tracking information

## Related Documentation

- [BRANCH_GUIDES.md](BRANCH_GUIDES.md) - Detailed branch-specific guides
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Common commands and patterns
- [January 2026 verification archive](archive/2026-01-verification/README.md) - Superseded pre-restructure reports
