# Research & Experimentation – Verification Report

**Date:** 2026-01-27  
**Branch:** Research & Experimentation  
**Scope:** debug/, src/*.ipynb, src/debug_*.py, src/lodestar_*.py. No new experiments; runnability and documentation only.

## 1. Commands/Checks Run and Results

All commands below are intended to be run from **repo root** unless noted.

| Check | Command / action | Result | Notes |
|-------|------------------|--------|-------|
| diagnose_skip_connections | `python debug/diagnostics/diagnose_skip_connections.py` | **fail** | `ModuleNotFoundError: No module named 'lodestar_with_skip_connections'` — script adds `debug/diagnostics/` to path, not `src/`. |
| diagnose_skip_connections (with path) | `PYTHONPATH=src python debug/diagnostics/diagnose_skip_connections.py` | **ok** (by design) | Requires `lodestar_with_skip_connections` from src; uses committed or uncommitted src at runtime. |
| architecture_diagram | `python debug/inspection/architecture_diagram.py` | **ok** | No project imports; prints ASCII diagram. |
| simple_architecture_diagram | `python debug/inspection/simple_architecture_diagram.py` | **ok** | No project imports; prints flow diagram. |
| investigate_augmentations | `python debug/inspection/investigate_augmentations.py` | **fail** | `ModuleNotFoundError: No module named 'utils'` — needs `src` on path. |
| investigate_augmentations (with path) | `PYTHONPATH=src python debug/inspection/investigate_augmentations.py --particle Rod --config src/config_debug.yaml` | **ok** (by design) | Requires `config['data_dir']` and `config['samples']`; writes `debug_outputs/augmentation_investigation.png`. |
| debug/README vs scripts | Manual compare | **partial** | README lists scripts but omits `simple_architecture_diagram.py` in Usage; no args, CWD, or PYTHONPATH. |
| src/*.ipynb purpose | Inspect first cells / filenames | **ok** | See Section 2. |
| CWD/path assumptions (notebooks) | Inspect paths in notebooks | **ok** | See Section 2. |
| lodestar_* / debug_* importers | `grep` + INVENTORY | **ok** | See Section 3. |
| .gitignore research outputs | Inspect .gitignore | **ok** | `debug_outputs/`, `detection_output/` present. |
| Scripts writing under src/ or tools/ | grep savefig/save/output | **ok** | See Section 4. |
| BRANCH_GUIDES Research section | Read and compare to scripts/notebooks | **partial** | Lists all four debug scripts and notebooks; usage block does not show PYTHONPATH or `simple_architecture_diagram`. |

**Skipped (and why):** Actual execution of diagnose_skip_connections and investigate_augmentations was not run in this environment; “ok (by design)” is from code inspection. User runs those themselves from repo root with `PYTHONPATH=src` as above.

## 2. Notebooks: Purpose and CWD/Paths

| Notebook | Purpose (from first cell / filename) | CWD / path assumption | Writes under src/ or root? |
|----------|------------------------------------|------------------------|----------------------------|
| **Check_Augmentation.ipynb** | Augmentation visualization (training pipeline + grid) | Kernel CWD = `src/`; uses `config.yaml`, `data_dir = os.path.join('..', config['data_dir'])` | No; only `plt.show()`. |
| **Debug.ipynb** | Debugging (customLodeSTAR, OrientationAwareLodeSTAR, load_trained_model) | Kernel CWD = `src/`; `config.yaml`, checkpoint paths | No file writes in inspected cells. |
| **detect_rings.ipynb** | Ring detection experiments | Kernel CWD = `src/`; `CONFIG_PATH = "config.yaml"`, `MODEL_PATH = "../models/..."`, `IMAGE_PATH = "../data/..."` | Not inspected for saves; paths assume repo layout. |
| **LodeStar.ipynb** | Main LodeSTAR notebook | Uses absolute path in one cell (`image_dir = "/home/edgar/..."`); no single documented CWD | Not repo-root–specific; recommend documenting “run from repo root or set paths” in first cell. |

**Recommendation:** Per QUICK_REFERENCE and BASELINE_REPORT, document in each notebook or in this file: “Run from repo root” or “Kernel CWD = src/” as appropriate. For any new notebook that writes outputs, use a single research output dir (e.g. `debug_outputs/` or `.specstory/`) and avoid writing under `src/` or repo root by default.

## 3. Experimental Core Files: Who Imports Them

| File | Importers | Active vs experimental-only |
|------|------------|-----------------------------|
| **lodestar_with_skip_connections.py** | `train_single_particle.py` (conditional on `lodestar_version`), `debug/diagnostics/diagnose_skip_connections.py` | **Active** (train + debug). |
| **lodestar_fixed_distributed.py** | `lodestar_with_skip_connections.py` (inheritance) | **Active** (via skip-connections variant). |
| **lodestar_simple_skip.py** | None | **Experimental-only**; consider “archive or document” (e.g. “lodestar_simple_skip unused – archive?”). |
| **debug_area_detection.py** | Run as script only; imports `custom_lodestar` | **Experimental-only** (debug). |
| **debug_disk_detection.py** | Run as script only; imports `custom_lodestar` | **Experimental-only** (debug). |

**custom_lodestar** is used by train/test/detect and by debug_area_detection / debug_disk_detection; it is Core, not experimental-only.

## 4. Outputs and Cleanup

- **.gitignore:** `debug_outputs/` and `detection_output/` are present (confirmed); matches CLEANUP_REPORT / INVENTORY.
- **Scripts writing by default:**
  - `debug/inspection/investigate_augmentations.py` → `debug_outputs/augmentation_investigation.png` (good).
  - `src/debug_disk_detection.py`, `src/debug_area_detection.py` → `debug_disk_detection_output.png` in **CWD** (when run from `src/` or root, can clutter that directory).
- **Recommendation:** Have `src/debug_*.py` write under `debug_outputs/` (or a single research output dir) by default; or document “run from repo root and expect file in CWD” and add `debug_disk_detection_output.png` to .gitignore. Prefer one research output dir (e.g. `debug_outputs/`) for all such artifacts.

## 5. Documentation Updates

- **debug/README.md:** Updated to match actual scripts, include Usage for `simple_architecture_diagram.py`, document args for `investigate_augmentations.py`, and state that `diagnose_skip_connections` and `investigate_augmentations` require `PYTHONPATH=src` when run from repo root.
- **docs/BRANCH_GUIDES.md:** Research section updated to add run instructions with `PYTHONPATH=src` where needed and to include `simple_architecture_diagram.py` in the example usage block.

## 6. CWD / Path Conventions (Recommendations)

- **Debug scripts:** Run from repo root; use `PYTHONPATH=src` for scripts that import `utils` or `lodestar_*` from src.
- **Notebooks:** Prefer “Kernel CWD = `src/`” or “Run from repo root” and resolve paths from repo root (e.g. `_REPO_ROOT`) for consistency with Core and QUICK_REFERENCE.
- **New experiments:** Write outputs under `debug_outputs/` (or one chosen research dir); do not write under `src/` or `tools/` by default.

## 7. Open Questions for Maintenance / Handoff

- **lodestar_simple_skip.py** is unused; recommend “archive or document” only—do not delete without product owner decision.
- **src/debug_disk_detection.py** and **src/debug_area_detection.py** both write `debug_disk_detection_output.png` to CWD; consider switching default output to `debug_outputs/` and documenting in README or BRANCH_GUIDES.
- **LodeStar.ipynb:** Document required CWD or path assumptions in the first cell or in this file.
