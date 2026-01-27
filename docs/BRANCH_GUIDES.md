# Branch-Specific Guides

**Last Updated:** 2026-01-27  
**Branch:** Documentation & Reporting

This document provides detailed guides for each of the 6 branches in MONA_LodeSTAR.

## Table of Contents

1. [Web Development](#web-development)
2. [Core Model Development](#core-model-development)
3. [Research & Experimentation](#research--experimentation)
4. [Tools & Automation](#tools--automation)
5. [Documentation & Reporting](#documentation--reporting)
6. [Maintenance & Operations](#maintenance--operations)

---

## Web Development

### Scope
- `web/app.py` - FastAPI backend (1235 lines)
- `web/templates/index.html` - Single-page web UI (1688 lines)
- `web/data/<username>/` - User data management (runtime, gitignored)
- `setup.py` - Web package setup

### Key Files
- **Backend:** `web/app.py` - FastAPI application with REST API
- **Frontend:** `web/templates/index.html` - Vanilla JavaScript single-page app
- **Data:** User-specific directories under `web/data/<username>/`

### API Endpoints

#### Authentication
- `POST /auth/register` - Register new user
- `POST /auth/login` - User login
- `GET /auth/check/{username}` - Check if user exists

#### File Management
- `POST /upload` - Upload file (TDMS or image)
- `POST /upload/start` - Start chunked upload
- `POST /upload/chunk/{upload_id}` - Upload chunk
- `POST /upload/complete` - Complete chunked upload
- `GET /files/{username}` - List user files
- `DELETE /files/{username}/{file_id}` - Delete file
- `GET /frame/{username}/{file_id}/{index}` - Get frame from TDMS file

#### Sample Management
- `POST /sample` - Create sample from uploaded file
- `GET /samples/{username}` - List user samples
- `DELETE /sample/{username}/{particle_name}` - Delete sample

#### Masking
- `POST /mask` - Create mask from image
- `POST /mask/circular` - Create circular ROI mask

#### Training
- `POST /train` - Start training job
- `GET /train/active/{username}` - Get active training jobs
- `GET /train/{job_id}` - Get training job status

#### Model Management
- `GET /models/{username}` - List user models
- `DELETE /models/{username}/{model_id}` - Delete model
- `PUT /models/{username}/{model_id}/rename` - Rename model

#### Detection
- `POST /detect` - Run detection on uploaded image
- `POST /detect/upload` - Run detection on uploaded file
- `GET /detect/frame/{username}/{file_id}/{index}` - Detect on specific frame

#### TDMS Processing
- `GET /tdms/structure/{username}/{file_id}` - Get TDMS file structure
- `POST /tdms/export` - Export TDMS frames

#### Utilities
- `GET /config/defaults` - Get default configuration
- `POST /video/merge` - Merge MP4 videos
- `GET /results/{username}` - Get user results

### Data Flow

1. **User Registration/Login**
   - User data stored in `web/users.json`
   - Sessions stored in `web/data/<username>/session.json`
   - User directories created: `uploads/`, `samples/`, `models/`, `results/`, `masks/`

2. **File Upload**
   - TDMS files: Extracted using `tools/tdms_to_png.py`
   - Images: Direct storage
   - Settings: Image dimensions, channel index, normalization

3. **Training**
   - Background thread execution
   - Job tracking in `web/training_jobs.json`
   - Sample from `get_user_dir(username)/samples/<particle_name>/<particle_name>.jpg`
   - Does **not** call `src/train_single_particle.py`. Implements training inline: DeepTrack pipeline (LoadImage, Affine/Multiply/Add, etc.), `dl.LodeSTAR`, `dl.Trainer`
   - Models saved to `get_user_dir(username)/models/<particle_name>_weights.pth`

4. **Detection**
   - Loads `.pth` from user models dir via `load_model()`; uses `dl.LodeSTAR` and `lodestar.detect(alpha, beta, mode="constant", cutoff)`
   - Single-model detection only; no `src/composite_model` or `trained_models_summary.yaml`

### Dependencies
- **Imports:** `src/utils` (e.g. `load_yaml` for config defaults), `tools/tdms_to_png` (`extract_images_from_tdms`, `list_tdms_structure`, `save_images`, `save_video`). Committed code only; no branch-local paths.
- **External:** FastAPI, uvicorn, PIL, numpy, torch, cv2, matplotlib
- **No modifications to Core code**

### Coordination Rules
- Only integrates committed Core code
- Does not modify `src/` files directly
- Calls existing functions via imports
- User data is gitignored (runtime only)

### Run assumptions
- Paths use `Path(__file__).parent.parent` (repo root); resolution is CWD-independent when the app module is loaded from repo layout.
- Recommended: run from repo root, e.g. `uvicorn web.app:app --reload` or `python -m uvicorn web.app:app`. If run via Jupyter proxy or other CWD, ensure the process's `__file__` resolves to `web/app.py` under the repo so `parent.parent` is the repo root.

### Example Usage

```python
# Start web server
cd /path/to/MONA_LodeSTAR
uvicorn web.app:app --reload

# Access at http://localhost:8000
```

---

## Core Model Development

Baseline and compatibility notes: **BASELINE_REPORT.md** (path conventions, trained_models_summary format, deferred items).

### Scope
- `src/*.py` (all Python files except notebooks)
- `src/config.yaml`, `src/samples.yaml`
- `src/requirements.txt`
- Model-related documentation

### Key Files

#### Training
- **`train_single_particle.py`** - Main training script
  - Trains separate models for each particle type
  - Supports checkpointing and resuming
  - Integrates with WandB logging
  - Saves models to `models/<run_id>/`

- **`train_enhanced.py`** - Enhanced training with multi-particle support
  - Supports single-particle and multi-particle modes
  - Alternative training approach

#### Testing
- **`test_single_particle.py`** - Single particle model testing
  - Tests individual particle models
  - Generates test datasets (same/different shape/size)
  - Calculates precision, recall, F1-score

- **`test_composite_model.py`** - Composite model testing
  - Tests multi-class detection
  - Uses model-specific detection parameters

#### Detection
- **`detect_particles.py`** - Main particle detection script
  - Command-line detection interface
  - Supports batch processing

#### Models
- **`custom_lodestar.py`** - Paper-accurate LodeSTAR implementation
  - Follows exact architecture from research paper
  - 3×Conv2D(3×3, 32) + ReLU → MaxPool2D(2×2) → 8×Conv2D(3×3, 32) + ReLU → Conv2D(1×1, 3)

- **`composite_model.py`** - Composite model for multi-class detection
  - Combines multiple single-particle models
  - Weight-based classification
  - Detection merging with spatial clustering

#### Data Generation
- **`image_generator.py`** - Image generation utilities
  - Synthetic microscopy image generation
  - Multiple dataset types
  - Trajectory generation

- **`generate_samples.py`** - Sample image generation
  - Generates sample images for each particle type

#### Pipelines
- **`run_single_particle_pipeline.py`** - Complete single-particle pipeline
  - Trains all particle types
  - Tests all trained models

- **`run_composite_pipeline.py`** - Composite model pipeline
  - Tests composite model with all particle types

#### Utilities
- **`utils.py`** - Core utilities
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
python src/train_single_particle.py --particle Janus --config src/config.yaml

# Train all particle types
python src/train_single_particle.py --config src/config.yaml

# Test single model
python src/test_single_particle.py --particle Janus --model models/<run_id>/Janus_weights.pth

# Test composite model
python src/test_composite_model.py --config src/config.yaml

# Run complete pipeline
python src/run_single_particle_pipeline.py
```

---

## Research & Experimentation

Verification summary and CWD/output conventions: see **`RESEARCH_VERIFICATION.md`** at repo root.

### Scope
- `debug/` directory (all files)
- `src/*.ipynb` (all notebooks)
- Debug scripts: `src/debug_*.py`
- Experimental models: `src/lodestar_*.py`

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

#### Experimental Models (`src/`)
- **`lodestar_with_skip_connections.py`** - Skip connections variant
  - Used by: `train_single_particle.py` (conditional import)
  - Used by: `debug/diagnostics/diagnose_skip_connections.py`

- **`lodestar_fixed_distributed.py`** - Fixed distributed training wrapper
  - Used by: `lodestar_with_skip_connections.py` (inheritance)

- **`lodestar_simple_skip.py`** - Simplified skip connections variant
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
PYTHONPATH=src python debug/diagnostics/diagnose_skip_connections.py

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
- **`tdms_to_png.py`** - TDMS to PNG/MP4 converter
  - Converts TDMS files to PNG images or MP4 videos
  - Parallel processing support
  - Pattern matching for batch processing
  - Automatic dtype conversion

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
python tools/tdms_to_png.py input.tdms -o output_dir

# Batch processing
python tools/tdms_to_png.py "file_{:03d}.tdms" -o output --start-index 1 --num-files 10

# To MP4
python tools/tdms_to_png.py input.tdms -o output --to-mp4 --fps 30
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
python tools/tdms_to_png.py experiment.tdms -o output/

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
- **`INVENTORY.md`** - Complete codebase inventory
- **`CLEANUP_REPORT.md`** - Cleanup actions executed
- **`DUPLICATES_DOCUMENTATION.md`** - Clarification on duplicate files

#### Feature Documentation
- **`COMPOSITE_MODEL_README.md`** - Composite model documentation
- **`MODEL_SPECIFIC_DETECTION_PARAMS.md`** - Detection parameters guide
- **`QUICK_START_COMPOSITE.md`** - Quick start for composite model

#### Implementation Documentation
- **`IMPLEMENTATION_SUMMARY.md`** - Implementation summary
- **`DEEPLAY_DISTRIBUTED_TRAINING_FIX.md`** - Distributed training fix

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
- References INVENTORY.md and CLEANUP_REPORT.md

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
- [INVENTORY.md](../INVENTORY.md) - Complete file catalog
- [CLEANUP_REPORT.md](../CLEANUP_REPORT.md) - Cleanup actions executed
