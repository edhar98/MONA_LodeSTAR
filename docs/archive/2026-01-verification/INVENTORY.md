# MONA_LodeSTAR Codebase Inventory

**Generated:** 2026-01-27  
**Last Updated:** 2026-05-16 (detection-engine benchmark and docs sync)  
**Purpose:** Authoritative baseline inventory for branch coordination and maintenance  
**Primary baseline branch:** `dev` (ahead of `origin/dev` by 1 commit)

---

## 1. Current Repository Snapshot

- Baseline verification artifacts are present: `INVENTORY.md`, `BASELINE_REPORT.md`, `TOOLS_VERIFICATION.md`, `RESEARCH_VERIFICATION.md`, `WEB_VERIFICATION.md`, `CLEANUP_REPORT.md`, `DUPLICATES_DOCUMENTATION.md`
- Core project directories are present: `src/`, `tools/`, `web/`, `debug/`, `test/`, `docs/`
- Local and external-content directories/files are also present in workspace (`.specstory/`, `.vscode/`, `presentation/`, link placeholder dirs, model artifacts) and should remain excluded from baseline commits unless explicitly needed

---

## 2. File Catalog by Branch Ownership

### 2.1 Core Model Development (`src/`)

**Python files (25):**
- `benchmark_trackpy.py`, `benchmark_trackpy_locate.py`, `compare_models.py`, `composite_model.py`, `crop_detections.py`, `custom_lodestar.py`, `debug_area_detection.py`, `debug_disk_detection.py`, `detect_particles.py`, `generate_samples.py`, `image_generator.py`, `lodestar_fixed_distributed.py`, `lodestar_orientation.py`, `lodestar_simple_skip.py`, `lodestar_with_skip_connections.py`, `run_composite_pipeline.py`, `run_single_particle_pipeline.py`, `run_training.py`, `test_composite_model.py`, `test_model.py`, `test_no_validation.py`, `test_single_particle.py`, `train_enhanced.py`, `train_single_particle.py`, `utils.py`

**Notebooks:**
- `Check_Augmentation.ipynb`, `Debug.ipynb`, `LodeStar.ipynb`, `detect_rings.ipynb`

**Config and dependencies:**
- `config.yaml`, `config_debug.yaml`, `config_subset_example.yaml`, `samples.yaml`, `requirements.txt`

### 2.2 Web Development (`web/`)

**Python files (3):**
- `app.py`, `__init__.py`, `jupyter_config.py`

**Frontend assets:**
- `templates/index.html`, `icon.svg`

### 2.3 Tools & Automation (`tools/`)

**Core tools and ELAB:**
- `__init__.py`, `crop.py`, `mask.py`, `merge_mp4.py`, `wandb_logging.py`, `elab_cli.py`
- `elab/cli/elab_cli.py`, `elab/cli/elab_cli_simple.py`
- `elab/scripts/upload_test.py`, `elab/scripts/upload_training.py`
- `elab/config/elab_config.yaml`

**TDMSExplorer package:**
- TDMS support is provided by the installed `tdms_explorer` package in `/opt/mona_jupyterhub_env`, not by a vendored `tools/TDMSExplorer/` subtree.

**Note:** `tools/tdms_to_png.py`, `tools/tdms_to_png_README.md`, `tools/build_tdms_to_png.sh` are currently absent from tracked baseline content and should be treated as changed/removed until intentionally restored or deprecated.

### 2.4 Research & Experimentation (`debug/`)

**Python files (8):**
- `__init__.py`, `diagnostics/__init__.py`, `diagnostics/diagnose_skip_connections.py`, `experiments/__init__.py`, `inspection/__init__.py`, `inspection/architecture_diagram.py`, `inspection/investigate_augmentations.py`, `inspection/simple_architecture_diagram.py`

**Docs:**
- `debug/README.md`

### 2.5 Maintenance & Operations (`test/`)

**Python files (8):**
- `__init__.py`, `run_tests.py`, `integration/__init__.py`, `regression/__init__.py`, `regression/test_backwards_compatibility.py`, `unit/__init__.py`, `unit/test_lodestar_models.py`, `unit/test_utils.py`

**Docs:**
- `test/README.md`

### 2.6 Documentation & Reporting (`docs/` + root docs)

**Docs directory (current tracked set):**
- `docs/ARCHITECTURE.md`, `docs/BRANCH_GUIDES.md`, `docs/DOCUMENTATION_VERIFICATION.md`, `docs/QUICK_REFERENCE.md`

**Important root markdown artifacts:**
- `README.md`, `BASELINE_REPORT.md`, `TOOLS_VERIFICATION.md`, `RESEARCH_VERIFICATION.md`, `WEB_VERIFICATION.md`, `INVENTORY.md`, `CLEANUP_REPORT.md`, `DUPLICATES_DOCUMENTATION.md`

---

## 3. Dead Code, Drift, and Cleanup Candidates

### 3.1 Already-cleaned candidates

- `src/detect_particles_backup.py` removed
- temporary root files from prior cleanup removed
- old `cursor_*.md` notes archived under `.specstory/archive/`

### 3.2 Current candidate list

- `draft.ipynb` (large, local analysis candidate; keep only if active)
- `orientation_cnn_debug.ipynb` (new notebook; classify as research or archive)
- local model/test artifacts in root (`orientation_cnn_ckpt.pt`, generated test images, orientation output directory) should not enter baseline commits

### 3.3 Possible implementation drift requiring owner decision

- `tools/tdms_to_png.py` family appears absent; web currently imports installed `tdms_explorer.TDMSFileExplorer`
- `tdms_explorer` is the active TDMS direction; package installation/import contract should be verified before merge to main

---

## 4. Dependency Map (Current Practical View)

### 4.1 Cross-module imports

- `web/app.py` depends on `src/utils.py` and tool-side image conversion utilities
- core training/testing modules in `src/` share utility and model wrappers (`utils`, `custom_lodestar`, skip variants)
- test suite imports core `src` modules
- ELAB workflows exist via root `elab.py` and `tools/elab_cli.py` + `tools/elab/*`

### 4.2 External dependencies (observed from project files)

- ML/scientific: `torch`, `lightning`, `deeptrack/deeplay`, `numpy`, `scipy`, `opencv-python`, `matplotlib`, `pandas`
- web/API: `fastapi`, `uvicorn`, `python-multipart`, `jupyter-server-proxy`
- ELAB/data tools: `elabapi-python`, `urllib3`, `requests`, TDMS/image/video dependencies

### 4.3 Risk points

- Tooling drift risk if `tdms_explorer` is not installed/importable in environments that run `web/app.py`
- ELAB config ambiguity if readers assume `tools/elab/config/elab_config.yaml` is actively loaded by scripts
- multiple experimental model variants in `src/` increase maintenance burden

---

## 5. Git State (as of this update)

- Current branch: `dev`
- Tracking: `dev...origin/dev` (ahead by 1)
- Working tree includes additional modifications and untracked artifacts beyond baseline commit

**Modified tracked files currently visible:**
- `INVENTORY.md`
- `tools/crop.py`
- `web/app.py`
- `web/templates/index.html`
- deletions in tools TDMS files (`tools/build_tdms_to_png.sh`, `tools/tdms_to_png.py`, `tools/tdms_to_png_README.md`)

**Untracked notable additions currently visible:**
- `src/lodestar_orientation.py`, `orientation_cnn.py`, `test_lodestar_orientation.py`
- installed `tdms_explorer` package used by Web
- local notebook/artifact set (`orientation_cnn_debug.ipynb`, ckpt/images/output dir)
- local meta dirs/files (`.specstory/`, `.vscode/`, etc.)

---

## 6. Configuration Inventory

**YAML files currently present (9):**
- `src/config.yaml`
- `src/config_debug.yaml`
- `src/config_subset_example.yaml`
- `src/samples.yaml`
- `tools/elab/config/elab_config.yaml`
- `elab_config.yaml`
- `trained_models_summary.yaml`
- `test_results_summary.yaml`
- `test_composite_results_summary.yaml`

**Usage status:**
- Core runtime configs: `src/config.yaml`, `src/samples.yaml`
- Debug/experimental configs: `src/config_debug.yaml`, `src/config_subset_example.yaml`
- ELAB configs: documented in `DUPLICATES_DOCUMENTATION.md`; scripts currently rely on hardcoded defaults in places
- result summary YAMLs are runtime artifacts and are appropriately ignored where applicable

---

## 7. Documentation Inventory and Gaps

### 7.1 Present documentation strengths

- Branch/process docs: `docs/BRANCH_GUIDES.md`, `docs/QUICK_REFERENCE.md`
- Verification docs: `BASELINE_REPORT.md`, `TOOLS_VERIFICATION.md`, `RESEARCH_VERIFICATION.md`, `WEB_VERIFICATION.md`, `docs/DOCUMENTATION_VERIFICATION.md`
- cleanup and duplicates clarified: `CLEANUP_REPORT.md`, `DUPLICATES_DOCUMENTATION.md`

### 7.2 Remaining gaps

- Unified “current architecture” doc now reflects TDMSExplorer and the detection-engine benchmark direction
- Web endpoint/API usage guide remains incomplete
- Detection engine selection (`lodestar|trackpy`) should be implemented as code, not only documented
- Explicit “what is intentionally local/untracked” policy doc is still missing

---

## 8. Branch Ownership Matrix (Operational)

- **Web Development:** `web/`, web templates/assets, web-run workflow docs
- **Core Model Development:** `src/*.py`, model configs, training/testing core behavior
- **Research & Experimentation:** `debug/`, `src/*.ipynb`, exploratory notebooks and orientation experiments
- **Tools & Automation:** `tools/`, ELAB CLI stack, TDMS/automation tooling
- **Documentation & Reporting:** root/docs markdown corpus, baseline/verification reports
- **Maintenance & Operations:** `test/`, `.gitignore`, cleanup governance, baseline integrity checks

---

## 9. Recommended Actions (Current, Non-historical)

1. Verify the installed `tdms_explorer` import contract for web runtime and CI.
2. Resolve orientation pipeline ownership (`src/lodestar_orientation.py`, root orientation scripts/tests) and either integrate into `src/` workflow or isolate as research.
3. Keep baseline branch clean: split local artifacts from intentional code changes before next commit.
4. Add a real `--detection-engine lodestar|trackpy` option so benchmarks can become a supported runtime path.
5. Add integration tests under `test/integration/` for at least one end-to-end training+inference and one web-tool handshake path.

---

## 10. Ignore Policy (Enforced)

Current `.gitignore` correctly covers:
- cache/build/runtime dirs
- temporary and archive artifacts
- test summary YAML outputs
- compiled tool binary path
- runtime web state files
- output directories (`detection_output/`, `debug_outputs/`)

No change required at this time.

---

**End of Inventory**
