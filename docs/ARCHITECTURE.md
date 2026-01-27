# MONA_LodeSTAR Architecture Overview

**Last Updated:** 2026-01-27  
**Branch:** Documentation & Reporting

## 6-Branch Workflow Structure

MONA_LodeSTAR uses a 6-branch workflow to organize development across different domains:

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
- Uses: `tools/tdms_to_png.py` (committed code only)
- External: FastAPI, uvicorn, jupyter-server-proxy

**Coordination Rules:**
- Only integrates committed Core code
- Does not modify `src/` files directly
- Calls existing functions via imports

### Core Model Development Branch
**Owns:**
- `src/*.py` (all Python files except notebooks)
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
- Experimental models: `src/lodestar_*.py`

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
- **Training:** `train_single_particle.py`, `train_enhanced.py`
- **Testing:** `test_single_particle.py`, `test_composite_model.py`
- **Detection:** `detect_particles.py`
- **Models:** `custom_lodestar.py`, `composite_model.py`
- **Data Generation:** `image_generator.py`, `generate_samples.py`
- **Utilities:** `utils.py`
- **Config:** `config.yaml`, `samples.yaml`

### Web Interface (`web/`)
- **Backend:** `app.py` (FastAPI, 1235 lines)
- **Frontend:** `templates/index.html` (single-page app, 1688 lines)
- **Data:** `data/<username>/` (runtime, gitignored)

### Tools (`tools/`)
- **Data Processing:** `tdms_to_png.py`, `crop.py`, `mask.py`, `merge_mp4.py`
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
  ├─ Used by: detect_particles.py, train_single_particle.py, test_single_particle.py
  ├─ Used by: composite_model.py, run_composite_pipeline.py
  ├─ Used by: image_generator.py, generate_samples.py
  └─ Used by: web/app.py

src/custom_lodestar.py
  ├─ Used by: detect_particles.py, train_single_particle.py
  ├─ Used by: debug_disk_detection.py, debug_area_detection.py
  └─ Used by: test/unit/test_lodestar_models.py

src/composite_model.py
  ├─ Used by: run_composite_pipeline.py
  └─ Used by: test_composite_model.py
```

### Web Dependencies
```
web/app.py
  ├─ Imports: tools/tdms_to_png (extract_images_from_tdms)
  └─ Imports: src/utils
```

### Tools Dependencies
```
tools/elab/cli/elab_cli_simple.py
  └─ Used by: tools/elab/scripts/upload_test.py
  └─ Used by: tools/elab/scripts/upload_training.py
  └─ Used by: tools/elab_cli.py

tools/wandb_logging.py
  └─ Used by: src/train_single_particle.py
```

## Coordination Rules

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
  - Used by: `train_single_particle.py`, `run_training.py`, `train_enhanced.py`
  - Contains: WandB settings, training parameters, augmentation settings, particle samples

### Sample Configuration
- **`src/samples.yaml`** - Particle sample definitions
  - Used by: `generate_samples.py`, `image_generator.py`
  - Contains: Particle types (Janus, Ring, Spot, Ellipse, Rod) with parameters

### ELAB Configuration
- **`elab_config.yaml` (root)** - Reference/documentation for ELAB configuration
- **`tools/elab/config/elab_config.yaml`** - Active ELAB configuration with defaults
  - Used by: ELAB CLI tools
  - Contains: Default experiment settings, tags, directory mappings, file patterns

See [DUPLICATES_DOCUMENTATION.md](../DUPLICATES_DOCUMENTATION.md) for clarification on duplicate config files.

## Output Structure

### Generated Data
- **`data/`** - Generated datasets and sample images
- **`models/`** - Trained model weights and checkpoints
- **`detection_results/`** - Detection outputs and evaluation results

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
- [INVENTORY.md](../INVENTORY.md) - Complete file catalog
- [CLEANUP_REPORT.md](../CLEANUP_REPORT.md) - Cleanup actions executed
