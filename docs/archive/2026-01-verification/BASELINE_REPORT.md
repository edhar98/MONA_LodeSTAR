# Core Model Development – Baseline Report

**Date:** 2026-01-27  
**Branch:** Core Model Development  
**Scope:** `src/*.py` (except notebooks), `src/config.yaml`, `src/samples.yaml`, `trained_models_summary.yaml`

## 1. Summary

Baseline focused on consistency and correctness. No new features. Paths and imports were adjusted so Core scripts behave correctly when run from repo root or from `src/`; `run_composite_pipeline.py` was made CWD-independent for config and summary.

## 2. Files Changed and Reason

| File | Change | Reason |
|------|--------|--------|
| `src/run_composite_pipeline.py` | Resolve config and `trained_models_summary.yaml` from script/repo; resolve example data paths from repo root; add `_SCRIPT_DIR`, `_REPO_ROOT`, type hint for `run_composite_detection_example` | Config and summary used to be CWD-relative (`src/config.yaml`, `trained_models_summary.yaml`), so they broke when CWD was not repo root. Now they use `os.path.dirname(os.path.abspath(__file__))` and repo root. |
| `src/train_single_particle.py` | Use `os.path.abspath(...)` when adding `tools` to `sys.path` | Makes the tools path robust when the script is run from different working directories. |
| `docs/QUICK_REFERENCE.md` | Note that Core CLI expects CWD = repo root for default paths; document that `run_composite_pipeline.py` is CWD-independent; add `trained_models_summary.yaml` to Configuration Files; clarify samples in `samples.yaml` | Aligns docs with actual behaviour and INVENTORY. |
| `.gitignore` | Add `mona_track.egg-info/` | Per INVENTORY 4.3; build artifact should be ignored. |

## 3. What Was Verified

- **Config and paths:** `src/config.yaml` and `src/samples.yaml` match docs and INVENTORY. `data_dir`, `models/`, `lightning_logs/`, `logs/` are in `.gitignore`. No new config keys were added.
- **Imports:** `train_single_particle.py` imports `wandb_logging` from `tools/` via `sys.path` to the repo `tools` directory (now with `abspath`). `test_single_particle.py`, `composite_model.py`, `detect_particles.py` use `utils` and `custom_lodestar` from the same package (script directory = `src/` when run as `python src/<script>.py`). No imports from uncommitted locations.
- **Entry points and summary format:** `train_single_particle.py`, `test_single_particle.py`, `run_composite_pipeline.py`, `detect_particles.py` define CLIs and use `trained_models_summary.yaml` at repo root. Format: `{ particle_type: { model_path, models_dir, checkpoint_path [, additional_models] } }`, as expected by `composite_model.CompositeLodeSTAR` and docs.
- **Backward compatibility:** No changes to public function signatures or to config key names used by Web or Tools (see docs/BRANCH_GUIDES.md).

## 4. What Remains Deferred

- **Test execution:** `python test/run_tests.py` was not run in an environment with full deps (torch, numpy, lightning, deeptrack, deeplay, etc.). In a minimal env, unit/regression tests fail at import (e.g. `ModuleNotFoundError: No module named 'torch'`). **Owner:** Maintenance & Operations (environment and CI). Fix only what is needed for tests to pass once deps are installed; no major refactors.
- **Run from arbitrary CWD for train/test/detect:** `train_single_particle.py`, `test_single_particle.py`, `test_composite_model.py`, and `detect_particles.py` still treat `trained_models_summary.yaml` and default `--config src/config.yaml` as CWD-relative. Recommended usage: run from repo root. Making them CWD-independent would require passing or resolving paths from script/repo and is left for a later change.
- **Web/Tools integration:** Web and Tools were not exercised. Compatibility notes are below for the Documentation and Web branches.

## 5. Compatibility Notes for Other Branches

- **Web and Documentation:** Core did not change public function signatures or config key names. `run_composite_pipeline.run_composite_detection_example()` now takes no path args and loads config/summary from script and repo root; callers need not pass paths. If Web or Tools ever call this function, they can keep the same call.
- **trained_models_summary.yaml:** Location (repo root) and format are unchanged. Keys used by Core: `model_path`, `models_dir`, and optionally `checkpoint_path`, `additional_models` per particle type.
- **Config `samples`:** `config.samples` lists particle types to train/use. Entries must have sample data at `data_dir/Samples/<name>/<name>.jpg` (or `.png`). `samples.yaml` defines synthetic particle types (Janus, Ring, Spot, Ellipse, Rod) for data generation; `config.samples` can also include user-defined names (e.g. `JP_Fe_wf_2_40`) if that data exists.

## 6. Known Failing or Skipped Tests

| Test / area | Status | Owner |
|-------------|--------|--------|
| `test/unit/test_utils.py` | Fails at import without numpy (and thus full `src/requirements.txt`) | Maintenance & Operations |
| `test/unit/test_lodestar_models.py` | Fails at import without torch | Maintenance & Operations |
| `test/regression/test_backwards_compatibility.py` | Not run; same dep requirement | Maintenance & Operations |
| `test/integration/` | Empty/placeholder | Maintenance & Operations |

Running tests with deps installed: from repo root, `python test/run_tests.py` (or `--type unit` / `--type regression`). Fix only what is required for them to pass; document any remaining failures and owners as above.
