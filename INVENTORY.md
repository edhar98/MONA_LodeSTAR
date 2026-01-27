# MONA_LodeSTAR Codebase Inventory

**Generated:** 2026-01-27  
**Last Updated:** 2026-01-27 (post-cleanup)  
**Purpose:** Comprehensive baseline inventory for all 6 branches  
**Branch:** Maintenance & Operations

**Cleanup Status:** ✅ Initial cleanup completed (see Section 8.1 and CLEANUP_REPORT.md)

---

## 1. FILE CATALOG BY BRANCH

### 1.1 Core Model Development (`src/`)

#### Python Files (Active)
- `composite_model.py` - Composite LodeSTAR model for multi-particle classification
- `custom_lodestar.py` - Custom LodeSTAR implementation
- `detect_particles.py` - Main particle detection script
- `generate_samples.py` - Sample image generation
- `image_generator.py` - Image generation utilities
- `run_composite_pipeline.py` - Composite model pipeline runner
- `run_single_particle_pipeline.py` - Single particle pipeline runner
- `run_training.py` - Training script wrapper
- `test_composite_model.py` - Composite model testing
- `test_model.py` - Model testing utilities
- `test_single_particle.py` - Single particle testing
- `train_enhanced.py` - Enhanced training with multi-particle support
- `train_single_particle.py` - Single particle training
- `utils.py` - Core utilities (YAML, XML, logging, visualization)

#### Python Files (Experimental/Alternative)
- `lodestar_fixed_distributed.py` - Fixed distributed training wrapper (USED: imported by train_single_particle.py)
- `lodestar_simple_skip.py` - Simplified skip connections variant (NOT DIRECTLY IMPORTED)
- `lodestar_with_skip_connections.py` - Skip connections variant (USED: imported by train_single_particle.py, debug/diagnostics)
- `compare_models.py` - Model comparison utilities
- `debug_area_detection.py` - Debug script for area detection
- `debug_disk_detection.py` - Debug script for disk detection
- `test_no_validation.py` - Test script for training without validation

#### Backup Files
- ~~`detect_particles_backup.py`~~ - **DELETED** (was backup of detect_particles.py)

#### Jupyter Notebooks (Research & Experimentation)
- `Check_Augmentation.ipynb` - Augmentation analysis
- `Debug.ipynb` - Debugging notebook
- `detect_rings.ipynb` - Ring detection experiments
- `LodeStar.ipynb` - Main LodeSTAR notebook

#### Configuration Files
- `config.yaml` - Main training configuration
- `config_debug.yaml` - Debug configuration
- `config_subset_example.yaml` - Example subset configuration
- `samples.yaml` - Particle sample definitions
- `requirements.txt` - Python dependencies

---

### 1.2 Web Development (`web/`)

#### Python Files
- `app.py` - FastAPI web application (1235 lines)
- `__init__.py` - Package initialization
- `jupyter_config.py` - Jupyter server proxy configuration

#### Templates & Static
- `templates/index.html` - Main web interface (1688 lines)
- `icon.svg` - Application icon

#### Data Files (Runtime)
- `samples/` - User sample data
- `uploads/` - User uploads
- `sessions.json` - User session data
- `training_jobs.json` - Training job tracking
- `users.json` - User authentication data

**Dependencies:**
- Imports: `tdms_to_png` (from tools), `utils` (from src)
- External: FastAPI, uvicorn, PIL, numpy, torch, cv2, matplotlib

---

### 1.3 Tools & Automation (`tools/`)

#### ELAB Integration
- `elab_cli.py` - Main ELAB CLI entry point (wrapper)
- `elab/__init__.py` - ELAB package
- `elab/cli/elab_cli.py` - Full-featured ELAB CLI (1200 lines)
- `elab/cli/elab_cli_simple.py` - Simplified ELAB CLI (316 lines)
- `elab/scripts/upload_test.py` - Upload test results script
- `elab/scripts/upload_training.py` - Upload training results script
- `elab/config/elab_config.yaml` - ELAB configuration

#### Data Processing Tools
- `tdms_to_png.py` - TDMS to PNG/MP4 converter
- `crop.py` - Interactive image cropping tool
- `mask.py` - Image masking tool with noise calculation
- `merge_mp4.py` - MP4 file merging utility
- `wandb_logging.py` - WandB logging callback

#### Build Scripts
- `build_tdms_to_png.sh` - Build standalone tdms_to_png executable
- `tdms_to_png` - Compiled binary (should be in .gitignore)

#### Documentation
- `README.md` - Tools documentation
- `tdms_to_png_README.md` - TDMS converter documentation

#### Notebooks
- `Untitled.ipynb` - Untitled notebook (cleanup candidate)

---

### 1.4 Research & Experimentation (`debug/`)

#### Diagnostics
- `diagnostics/diagnose_skip_connections.py` - Skip connections analysis

#### Inspection
- `inspection/architecture_diagram.py` - Architecture visualization
- `inspection/investigate_augmentations.py` - Augmentation investigation
- `inspection/simple_architecture_diagram.py` - Simplified architecture diagram

#### Documentation
- `README.md` - Debug directory documentation

---

### 1.5 Maintenance & Operations (`test/`)

#### Test Infrastructure
- `run_tests.py` - Test runner (unit, regression, integration)
- `unit/test_lodestar_models.py` - LodeSTAR model unit tests
- `unit/test_utils.py` - Utility function tests
- `regression/test_backwards_compatibility.py` - Backwards compatibility tests
- `integration/` - Integration tests (empty, placeholder)

#### Documentation
- `README.md` - Test directory documentation

---

### 1.6 Root-Level Scripts

#### ELAB Scripts
- `elab.py` - Convenience wrapper for ELAB tools (DUPLICATE: similar to tools/elab_cli.py)
- `check_existing_experiment.py` - ELAB experiment checker
- `test_template_24.py` - Template 24 compatibility test
- `test_template_24_metadata.py` - Template 24 metadata test
- `test_elab_connection.py` - ELAB connection test
- `test_minimal_upload.py` - Minimal upload test
- `debug_upload_test_run.py` - Debug upload test run

#### Maintenance Scripts
- `cleanup_lightning_logs.py` - Cleanup unused lightning logs

#### Shell Scripts
- `generate_and_train.sh` - Generate samples and train
- `run_rods.sh` - Run rod particle experiments
- `linked_resources_example.sh` - Linked resources example
- `upload_test_run_example.sh` - Upload test run example

#### Setup
- `setup.py` - Package setup for web interface

---

## 2. DEAD CODE & CLEANUP CANDIDATES

### 2.1 Backup Files (Remove)
- `src/detect_particles_backup.py` - Backup of detect_particles.py (never imported)

### 2.2 Temporary Files (Remove)
- ~~`count`~~ - **DELETED** (temporary file)
- ~~`ddd`~~ - **DELETED** (temporary file)
- ~~`diff.txt`~~ - **DELETED** (diff file, now in .gitignore)
- ~~`main.aux`~~ - **DELETED** (LaTeX auxiliary file)
- ~~`main.log`~~ - **DELETED** (LaTeX log file)
- ~~`main.out`~~ - **DELETED** (LaTeX output file)
- ~~`texput.log`~~ - **DELETED** (LaTeX log file)

### 2.3 HTML Files (Archive/Remove)
- ~~`body_from_155.html`~~ - **DELETED** (ELAB experiment body HTML)
- ~~`update_exp191_body.html`~~ - **DELETED** (ELAB experiment update HTML)

### 2.4 Untitled Notebooks (Remove)
- ~~`Untitled.ipynb` (root)~~ - **DELETED**
- ~~`tools/Untitled.ipynb`~~ - **DELETED**
- `draft.ipynb` - **REVIEW NEEDED** (very large file, 84MB - review manually)

### 2.5 Duplicate Files (Consolidate)
- `elab.py` (root) vs `tools/elab_cli.py` - Both provide ELAB CLI access
  - **Recommendation:** Keep `tools/elab_cli.py`, remove root `elab.py` or document which to use
- `elab_config.yaml` (root) vs `tools/elab/config/elab_config.yaml`
  - **Root version:** Reference/documentation style
  - **Tools version:** Actual configuration with defaults
  - **Recommendation:** Merge or clearly document purpose

### 2.6 Unused Experimental Files (Archive)
- `src/lodestar_simple_skip.py` - Not directly imported (only referenced in history)
- `src/test_no_validation.py` - One-time test script

### 2.7 Archive Files (Should be in .gitignore)
- `20250813-130006_detection_results.tar.gz`
- `20250813-130006_logs.tar.gz`
- `test_cli_simple_detection_results.tar.gz`
- `test_cli_simple_logs.tar.gz`
- `test_training_simple_logs.tar.gz`
- `test_training_simple_models.tar.gz`

### 2.8 Debug Notes (Archive)
- ~~`cursor_debugging_lodestar_detect_method.md`~~ - **ARCHIVED** to `.specstory/archive/`
- ~~`cursor_fix_logger_initialization_for_fi.md`~~ - **ARCHIVED** to `.specstory/archive/`
- ~~`cursor_generate_images_with_shape_and_s.md`~~ - **ARCHIVED** to `.specstory/archive/`
- ~~`cursor_merge_model_architectures_in_pyt.md`~~ - **ARCHIVED** to `.specstory/archive/`
- ~~`cursor_optimize_elab_cli_py_for_automat.md`~~ - **ARCHIVED** to `.specstory/archive/`

### 2.9 PDF Files (Documentation)
- ~~`2202.13546v1.pdf`~~ - **MOVED** to `docs/papers/`
- ~~`s41467-022-35004-y.pdf`~~ - **MOVED** to `docs/papers/`
- **Status:** Organized in `docs/papers/` directory

### 2.10 Test Results (Should be in .gitignore)
- `snr_test_results.txt`
- `test_results_summary.yaml`
- `test_composite_results_summary.yaml`
- `trained_models_summary.yaml` (may be needed for tracking)

---

## 3. DEPENDENCY MAP

### 3.1 Core Dependencies (src/)

**External Libraries:**
- `torch`, `torchvision` - PyTorch
- `lightning` - PyTorch Lightning
- `deeptrack`, `deeplay` - DeepTrack framework
- `numpy`, `scipy`, `scikit-image` - Scientific computing
- `matplotlib`, `pandas` - Visualization and data
- `opencv-python`, `Pillow` - Image processing
- `wandb` - Weights & Biases logging
- `PyYAML`, `Jinja2` - Configuration and templating

**Internal Dependencies:**
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

src/lodestar_with_skip_connections.py
  ├─ Used by: train_single_particle.py (conditional import)
  └─ Used by: debug/diagnostics/diagnose_skip_connections.py

src/lodestar_fixed_distributed.py
  └─ Used by: lodestar_with_skip_connections.py (inheritance)

src/image_generator.py
  ├─ Used by: generate_samples.py
  └─ Used by: test_model.py

src/composite_model.py
  ├─ Used by: run_composite_pipeline.py
  └─ Used by: test_composite_model.py
```

### 3.2 Web Dependencies (web/)

**External Libraries:**
- `fastapi`, `uvicorn` - Web framework
- `python-multipart` - File uploads
- `jupyter-server-proxy` - Jupyter integration

**Internal Dependencies:**
```
web/app.py
  ├─ Imports: tools/tdms_to_png (extract_images_from_tdms)
  └─ Imports: src/utils
```

### 3.3 Tools Dependencies (tools/)

**External Libraries:**
- `elabapi-python` - ELAB API client
- `nptdms` - TDMS file reading
- `imageio` - Video processing

**Internal Dependencies:**
```
tools/elab/cli/elab_cli_simple.py
  └─ Used by: tools/elab/scripts/upload_test.py
  └─ Used by: tools/elab/scripts/upload_training.py
  └─ Used by: tools/elab_cli.py

tools/elab/cli/elab_cli.py
  └─ Used by: tools/elab_cli.py

tools/wandb_logging.py
  └─ Used by: src/train_single_particle.py (imported as module)
```

### 3.4 Circular Dependencies

**None detected** - Clean dependency structure

### 3.5 Missing Dependencies

- `web/app.py` imports `tdms_to_png` - Path manipulation needed (sys.path.insert)
- `src/train_single_particle.py` imports `wandb_logging` - Should be `tools.wandb_logging`

---

## 4. GIT STATE ANALYSIS

### 4.1 Tracked Files (Modified)
- `src/samples.yaml` - Modified

### 4.2 Untracked Files (Should be Committed)

**Core Code:**
- `src/compare_models.py`
- `src/debug_area_detection.py`
- `src/debug_disk_detection.py`
- ~~`src/detect_particles_backup.py`~~ - **DELETED**
- `src/lodestar_fixed_distributed.py`
- `src/lodestar_simple_skip.py`
- `src/lodestar_with_skip_connections.py`
- `src/test_composite_model.py`
- `src/test_no_validation.py`
- `src/train_enhanced.py`

**Tools:**
- `tools/` directory (entire structure)
- `tools/elab/` (entire structure)

**Web:**
- `web/` directory (entire structure)

**Test:**
- `test/` directory (entire structure)

**Documentation:**
- All `.md` files in root (except README.md if tracked)
- `COMPOSITE_MODEL_README.md`
- `COMPOSITE_MODEL_UPDATE.md`
- `DEEPLAY_DISTRIBUTED_TRAINING_FIX.md`
- `ELAB_CLI_SIMPLE_USAGE.md`
- `IMPLEMENTATION_SUMMARY.md`
- `LINKED_RESOURCES.md`
- `MODEL_SPECIFIC_DETECTION_PARAMS.md`
- `QUICK_START_COMPOSITE.md`
- `UPLOAD_TEST_RUNS.md`
- `VISUALIZATION_UPDATE.md`

**Scripts:**
- `elab.py` (root)
- `check_existing_experiment.py`
- `cleanup_lightning_logs.py`
- `debug_upload_test_run.py`
- `test_elab_connection.py`
- `test_minimal_upload.py`
- `test_template_24.py`
- `test_template_24_metadata.py`

### 4.3 Files That Should Be in .gitignore

**Status:** ✅ **UPDATED** - All patterns added to `.gitignore`

**Temporary Files:**
- ✅ `count`, `ddd`, `diff.txt` - Added to .gitignore
- ✅ `*.aux`, `*.log`, `*.out` - Added to .gitignore
- ✅ `*.tar.gz` (archive files) - Added to .gitignore
- ✅ `snr_test_results.txt` - Added to .gitignore

**Build Artifacts:**
- ✅ `tools/tdms_to_png` (compiled binary) - Added to .gitignore
- `mona_track.egg-info/` - Should be added
- `__pycache__/` (already in .gitignore)
- `.ipynb_checkpoints/` (already in .gitignore)

**Runtime Data:**
- ✅ `web/sessions.json` - Added to .gitignore
- ✅ `web/training_jobs.json` - Added to .gitignore
- ✅ `web/users.json` - Added to .gitignore
- ✅ `web/uploads/` - Added to .gitignore
- ✅ `web/samples/` - Added to .gitignore

**Output Directories:**
- ✅ `detection_output/` - Added to .gitignore
- ✅ `debug_outputs/` - Added to .gitignore
- `logs/` (already in .gitignore)
- `lightning_logs/` (already in .gitignore)

### 4.4 Files That Should Be Committed

**Priority 1 (Core Functionality):**
- All `src/*.py` files (except backups)
- All `tools/` Python files
- All `web/` Python files
- All `test/` files
- Configuration files: `src/config.yaml`, `src/samples.yaml`, `tools/elab/config/elab_config.yaml`

**Priority 2 (Documentation):**
- All `.md` files in root
- `tools/README.md`, `tools/tdms_to_png_README.md`
- `test/README.md`, `debug/README.md`

**Priority 3 (Scripts):**
- `setup.py`
- `elab.py` (if keeping)
- Shell scripts (`.sh` files)

---

## 5. CONFIGURATION FILES INVENTORY

### 5.1 Training Configuration

**`src/config.yaml`**
- Purpose: Main training configuration
- Used by: `train_single_particle.py`, `run_training.py`, `train_enhanced.py`
- Contains: WandB settings, training parameters, augmentation settings, particle samples

**`src/config_debug.yaml`**
- Purpose: Debug/testing configuration
- Used by: `test_no_validation.py`
- Contains: Reduced parameters for quick testing

**`src/config_subset_example.yaml`**
- Purpose: Example configuration subset
- Status: Example/documentation file

### 5.2 Sample Configuration

**`src/samples.yaml`**
- Purpose: Particle sample definitions and parameters
- Used by: `generate_samples.py`, `image_generator.py`
- Contains: Particle types (Janus, Ring, Spot, Ellipse, Rod) with parameters

### 5.3 ELAB Configuration

**`elab_config.yaml` (root)**
- Purpose: Reference/documentation for ELAB configuration
- Contains: Template info, category/team IDs, example commands
- Status: Documentation style

**`tools/elab/config/elab_config.yaml`**
- Purpose: Actual ELAB configuration with defaults
- Used by: ELAB CLI tools
- Contains: Default experiment settings, tags, directory mappings, file patterns

**Status:** ✅ **DOCUMENTED** - See `DUPLICATES_DOCUMENTATION.md` for clarification
- Root version = Reference/documentation
- Tools version = Active configuration

### 5.4 Summary Files

**`trained_models_summary.yaml`**
- Purpose: Track trained models and checkpoints
- Used by: `cleanup_lightning_logs.py`
- Status: Runtime data, may need version control

**`test_results_summary.yaml`**
- Purpose: Test results tracking
- Status: Runtime data, should be in .gitignore

**`test_composite_results_summary.yaml`**
- Purpose: Composite model test results
- Status: Runtime data, should be in .gitignore

---

## 6. DOCUMENTATION INVENTORY

### 6.1 Root Documentation

**Main Documentation:**
- `README.md` - Project overview (should be tracked)

**Feature Documentation:**
- `COMPOSITE_MODEL_README.md` - Composite model documentation
- `COMPOSITE_MODEL_UPDATE.md` - Composite model updates
- `QUICK_START_COMPOSITE.md` - Quick start guide for composite model
- `MODEL_SPECIFIC_DETECTION_PARAMS.md` - Detection parameters documentation
- `VISUALIZATION_UPDATE.md` - Visualization updates

**Implementation Documentation:**
- `IMPLEMENTATION_SUMMARY.md` - Implementation summary
- `DEEPLAY_DISTRIBUTED_TRAINING_FIX.md` - Distributed training fix documentation

**Tools Documentation:**
- `ELAB_CLI_SIMPLE_USAGE.md` - ELAB CLI usage guide
- `UPLOAD_TEST_RUNS.md` - Test run upload documentation
- `LINKED_RESOURCES.md` - Linked resources documentation
- `DUPLICATES_DOCUMENTATION.md` - Documentation for duplicate files (ELAB CLI and config)

**Maintenance Documentation:**
- `CLEANUP_REPORT.md` - Cleanup actions report (2026-01-27)

**Debug Notes (Archived):**
- ~~`cursor_debugging_lodestar_detect_method.md`~~ - **ARCHIVED** to `.specstory/archive/`
- ~~`cursor_fix_logger_initialization_for_fi.md`~~ - **ARCHIVED** to `.specstory/archive/`
- ~~`cursor_generate_images_with_shape_and_s.md`~~ - **ARCHIVED** to `.specstory/archive/`
- ~~`cursor_merge_model_architectures_in_pyt.md`~~ - **ARCHIVED** to `.specstory/archive/`
- ~~`cursor_optimize_elab_cli_py_for_automat.md`~~ - **ARCHIVED** to `.specstory/archive/`

### 6.2 Directory Documentation

- `tools/README.md` - Tools directory documentation
- `tools/tdms_to_png_README.md` - TDMS converter documentation
- `test/README.md` - Test directory documentation
- `debug/README.md` - Debug directory documentation

### 6.3 Documentation Gaps

**Missing Documentation:**
1. **Web Interface:** No dedicated documentation for web interface usage
2. **Training Pipeline:** No comprehensive training pipeline guide
3. **Detection Pipeline:** No detection pipeline documentation
4. **Configuration:** No configuration reference guide
5. **Architecture:** No architecture overview document
6. **API Documentation:** No API documentation for web endpoints

**Outdated Documentation:**
- Check all `.md` files for accuracy against current codebase

---

## 7. BRANCH OWNERSHIP MATRIX

### 7.1 Web Development Branch

**Owns:**
- `web/` directory (all files)
- `setup.py` (web package setup)
- `web/` related documentation

**Dependencies:**
- Uses: `src/utils.py`
- Uses: `tools/tdms_to_png.py`
- External: FastAPI, uvicorn, jupyter-server-proxy

### 7.2 Core Model Development Branch

**Owns:**
- `src/*.py` (all Python files except notebooks)
- `src/config.yaml`, `src/samples.yaml`
- `src/requirements.txt`
- Model-related documentation

**Dependencies:**
- Uses: `tools/wandb_logging.py` (should be proper import)
- External: torch, lightning, deeptrack, deeplay, wandb

### 7.3 Research & Experimentation Branch

**Owns:**
- `debug/` directory (all files)
- `src/*.ipynb` (all notebooks)
- ~~`tools/Untitled.ipynb`~~ - **DELETED**
- `draft.ipynb` - **REVIEW NEEDED** (84MB, review manually)
- Debug scripts: `src/debug_*.py`
- Experimental models: `src/lodestar_*.py`

**Dependencies:**
- Uses: Core model files for experimentation
- May create temporary files

### 7.4 Tools & Automation Branch

**Owns:**
- `tools/` directory (all files except notebooks)
- `elab.py` (root, if keeping)
- ELAB-related scripts in root
- `tools/` documentation

**Dependencies:**
- External: elabapi-python, nptdms, imageio
- Independent from core model

### 7.5 Documentation & Reporting Branch

**Owns:**
- All `.md` files in root
- `presentation/` directory
- Documentation in subdirectories
- PDF files (research papers) - **ORGANIZED** in `docs/papers/`

**Dependencies:**
- Documents all other branches

### 7.6 Maintenance & Operations Branch

**Owns:**
- `test/` directory (all files)
- `cleanup_lightning_logs.py`
- `.gitignore`
- Maintenance scripts
- Test documentation

**Dependencies:**
- Tests all other branches
- Uses: Core model files for testing

---

## 8. RECOMMENDED ACTIONS

### 8.1 Immediate Cleanup (High Priority)

**Status:** ✅ **COMPLETED** (2026-01-27)

1. ✅ **Delete Backup Files:**
   - ~~`src/detect_particles_backup.py`~~ - **DELETED**

2. ✅ **Delete Temporary Files:**
   - ~~`main.aux`, `main.out`, `texput.log`~~ - **DELETED**
   - Note: `count`, `ddd`, `diff.txt`, `main.log` were already removed

3. ✅ **Delete Untitled Notebooks:**
   - ~~`Untitled.ipynb` (root), `tools/Untitled.ipynb`~~ - **DELETED**
   - ⚠️ `draft.ipynb` - **REVIEW NEEDED** (84MB file, review manually)

4. ✅ **Archive Debug Notes:**
   - All `cursor_*.md` files moved to `.specstory/archive/`

5. ✅ **Remove HTML Files:**
   - ~~`body_from_155.html`, `update_exp191_body.html`~~ - **DELETED**

6. ✅ **Organize PDFs:**
   - PDFs moved to `docs/papers/` directory

### 8.2 Update .gitignore

**Status:** ✅ **COMPLETED** - All patterns added

Added to `.gitignore`:
```
# Temporary files
count
ddd
diff.txt
*.aux
*.log
*.out

# Archive files
*.tar.gz

# Test results
snr_test_results.txt
test_results_summary.yaml
test_composite_results_summary.yaml

# Compiled binaries
tools/tdms_to_png

# Runtime data
web/sessions.json
web/training_jobs.json
web/users.json
web/uploads/
web/samples/

# Output directories
detection_output/
debug_outputs/
```

### 8.3 Consolidate Duplicates

**Status:** ✅ **DOCUMENTED** - See `DUPLICATES_DOCUMENTATION.md`

1. ✅ **ELAB CLI:**
   - Both files are maintained (intentional duplicates)
   - `elab.py` (root) = Convenience wrapper
   - `tools/elab_cli.py` = Direct entry point
   - Usage documented in `DUPLICATES_DOCUMENTATION.md`

2. ✅ **ELAB Config:**
   - Both files serve different purposes (intentional)
   - `elab_config.yaml` (root) = Reference/documentation
   - `tools/elab/config/elab_config.yaml` = Active configuration
   - Usage documented in `DUPLICATES_DOCUMENTATION.md`

### 8.4 Commit Strategy

**Phase 1: Core Functionality**
```bash
git add src/*.py src/*.yaml src/requirements.txt
git add tools/ tools/elab/
git add web/ setup.py
git add test/
```

**Phase 2: Documentation**
```bash
git add *.md
git add */README.md
```

**Phase 3: Scripts**
```bash
git add *.sh
git add *.py  # Root level scripts
```

### 8.5 Fix Import Issues

1. **wandb_logging import:**
   - `src/train_single_particle.py` imports `wandb_logging`
   - Should be: `from tools.wandb_logging import ...` or add tools to path

2. **tdms_to_png import:**
   - `web/app.py` uses sys.path manipulation
   - Consider proper package structure

### 8.6 Documentation Tasks

1. **Create missing documentation:**
   - Web interface user guide
   - Training pipeline guide
   - Detection pipeline guide
   - Configuration reference
   - Architecture overview

2. **Update existing documentation:**
   - Verify all `.md` files are current
   - Update README.md with current structure

### 8.7 Code Organization

1. ✅ **Move PDFs:**
   - Created `docs/papers/` directory
   - Moved `2202.13546v1.pdf` and `s41467-022-35004-y.pdf` to `docs/papers/`

2. **Organize root scripts:**
   - Consider `scripts/` directory for root-level scripts
   - Or clearly document which scripts belong to which branch

### 8.8 Testing Improvements

1. **Add integration tests:**
   - Currently `test/integration/` is empty
   - Add tests for full pipelines

2. **Test coverage:**
   - Ensure all core modules have unit tests
   - Add regression tests for critical paths

---

## 9. METRICS SUMMARY

### 9.1 File Counts

- **Python Files:** 65 total
  - Core (src/): 25
  - Web (web/): 3
  - Tools (tools/): 12
  - Debug (debug/): 4
  - Test (test/): 5
  - Root scripts: 16

- **Configuration Files:** 9 total
- **Documentation Files:** 52 total (.md files)
- **Notebooks:** 7 total
- **Shell Scripts:** 4 total

### 9.2 Code Statistics

- **Largest Files:**
  - `web/templates/index.html`: 1688 lines
  - `web/app.py`: 1235 lines
  - `tools/elab/cli/elab_cli.py`: ~1200 lines
  - `src/train_single_particle.py`: ~400 lines

### 9.3 Dependency Count

- **External Dependencies:** ~25 packages
- **Internal Modules:** ~15 core modules
- **Circular Dependencies:** 0

---

## 10. BRANCH-SPECIFIC RECOMMENDATIONS

### 10.1 Web Development
- Document web interface API
- Add web-specific tests
- Organize static assets

### 10.2 Core Model Development
- Consolidate model variants (lodestar_*.py)
- Document model architecture
- Create model comparison guide

### 10.3 Research & Experimentation
- Archive completed experiments
- Document experimental findings
- Clean up temporary notebooks

### 10.4 Tools & Automation
- Standardize ELAB CLI entry points
- Document tool usage
- Add tool-specific tests

### 10.5 Documentation & Reporting
- Create comprehensive user guide
- Update architecture diagrams
- Organize presentation materials

### 10.6 Maintenance & Operations
- Expand test coverage
- Create maintenance runbook
- Document cleanup procedures

---

## 11. CLEANUP STATUS SUMMARY

**Last Cleanup:** 2026-01-27  
**See:** `CLEANUP_REPORT.md` for detailed cleanup report

### Completed Actions

✅ **Files Deleted (3):**
- `main.aux`, `main.out`, `texput.log`

✅ **Files Archived (7):**
- 2 PDFs → `docs/papers/`
- 5 debug notes → `.specstory/archive/`

✅ **Configuration Updated:**
- `.gitignore` - Added comprehensive ignore patterns

✅ **Documentation Created:**
- `DUPLICATES_DOCUMENTATION.md` - Clarifies duplicate file usage
- `CLEANUP_REPORT.md` - Detailed cleanup report

✅ **Directories Created:**
- `docs/papers/` - For research papers
- `.specstory/archive/` - For archived debug notes

### Files Already Cleaned (Prior to 2026-01-27)
- `src/detect_particles_backup.py`
- `count`, `ddd`, `diff.txt`, `main.log`
- `Untitled.ipynb` files
- HTML files (`body_from_155.html`, `update_exp191_body.html`)

### Pending Review
- `draft.ipynb` - Very large file (84MB), requires manual review

---

**End of Inventory**
