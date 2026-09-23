# MONA_LodeSTAR Codebase Review Report

**Date:** 2026-04-05  
**Documentation Sync:** 2026-05-16  
**Scope:** Full repository (src/, web/, tools/, debug/, test/, root scripts/configs, docs)  
**Inputs:** INVENTORY.md, CLEANUP_REPORT.md, DUPLICATES_DOCUMENTATION.md, live codebase (second pass)

---

## 1. EXECUTIVE SUMMARY

- **Web TDMS stack changed:** `web/app.py` uses `from tdms_explorer import TDMSFileExplorer` from the installed `/opt/mona_jupyterhub_env` package instead of the removed `tools/tdms_to_png.py`. TDMSExplorer is no longer vendored under `tools/`.
- **Security and robustness (unchanged):** No authorization binding requests to a logged-in user; `username` in paths/body is trusted. Path traversal via `username` remains possible. `general_exception_handler` still returns `str(exc)` to clients. Passwords remain unsalted SHA-256.
- **New surface area:** Root-level `orientation_cnn.py` and `test_lodestar_orientation.py` add orientation pipelines; the latter depends on `lodestar_link` (external/symlinked layout per repo state). Duplicated `refine_position_to_center` logic and unsafe `torch.load` usage on checkpoints.

---

## 2. FINDINGS BY DIMENSION

### 2.1 CODE QUALITY

**Finding:** Web still duplicates Core training and detection.  
**Where:** `web/app.py` — `run_training()`, `run_detection_on_image()`, `load_model()` use inlined DeepTrack/LodeSTAR; Core uses `train_single_particle.py`, `detect_particles.py`, `lodestar_version`, `customLodeSTAR`, area detection.  
**Issue:** Two implementations; Web will not track Core model variants or detection modes.  
**Suggestion:** Single Core API (train/detect) invoked from Web; document exception if Web must stay self-contained.  
**Owner:** Web Development + Core Model Development.

**Finding:** `tdms_explorer` import is fragile.  
**Where:** `web/app.py` lines 28–30: `sys.path.insert(0, … "src")` then `from tdms_explorer import TDMSFileExplorer`.  
**Issue:** Environments without installed `tdms_explorer` fail at startup.  
**Suggestion:** Keep `tdms_explorer` installed in `/opt/mona_jupyterhub_env`; document and test that install path.  
**Owner:** Web Development, Tools & Automation.

**Finding:** Redundant TDMS reads in export.  
**Where:** `web/app.py` `export_tdms` (~973–984): `explorer.extract_images()` is called to check `None`, then `frame_count` uses `explorer.extract_images().shape[0]` again.  
**Issue:** Duplicate extraction work on large files.  
**Suggestion:** Assign result of first `extract_images()` to a variable and reuse; or expose frame count on `TDMSFileExplorer` without full reload.  
**Owner:** Web Development.

**Finding:** Duplicated orientation / center-refinement logic.  
**Where:** `orientation_cnn.py` (`refine_position_to_center`, Hough + CoM) vs `test_lodestar_orientation.py` (similar function starting ~line 111).  
**Issue:** Bug fixes or parameter tuning must be done twice; behaviour can drift.  
**Suggestion:** One shared module (e.g. under `src/` or `tools/`) imported by both scripts.  
**Owner:** Research & Experimentation + Core Model Development.

**Finding:** CSV column selection in `orientation_cnn.load_csv_gt` is misleading.  
**Where:** `orientation_cnn.py` lines 161–164: `x_col = "x" if "x" in cols else "x"` (always `"x"`); same pattern for `y` and `phi`.  
**Issue:** If the first column is not `frame` but `x` is missing, code still assumes column name `"x"`; the fallback logic does not actually vary.  
**Suggestion:** Either enforce required column names or implement real fallbacks (e.g. positional indices) and document them.  
**Owner:** Research & Experimentation.

**Finding:** `torch.load` without safe loading.  
**Where:** `orientation_cnn.py` line ~281; `test_lodestar_orientation.py` lines ~440, ~718.  
**Issue:** Loading arbitrary checkpoints can execute pickle gadgets if a file is attacker-controlled.  
**Suggestion:** Use `weights_only=True` where supported (PyTorch 2.x) for state dicts only, or document “trusted checkpoints only”.  
**Owner:** Research & Experimentation, Core Model Development.

**Finding:** Comment/code mismatch in debug script (still present).  
**Where:** `debug/diagnostics/diagnose_skip_connections.py` — comment says add `src`; path append targets script directory.  
**Issue:** Import of `lodestar_with_skip_connections` fails when run from `debug/diagnostics/` without `PYTHONPATH`.  
**Suggestion:** Fix path to repo `src` and document run instructions.  
**Owner:** Research & Experimentation / Maintenance.

**Finding:** Inconsistent error exposure in Web.  
**Where:** `web/app.py` `general_exception_handler` (lines 166–171) returns `{"error": str(exc)}`.  
**Issue:** Internal details leak to clients.  
**Suggestion:** Generic 500 body + server-side logging only.  
**Owner:** Web Development.

---

### 2.2 STRUCTURE & OWNERSHIP

**Finding:** INVENTORY / ARCHITECTURE vs current Web TDMS dependency.  
**Where:** Previously, docs listed `tools/tdms_to_png`; `web/app.py` uses `tdms_explorer`. Git history shows `tools/tdms_to_png.py` removed.  
**Issue:** Stale docs would send branch owners to the wrong module; this was updated on 2026-05-16 in `INVENTORY.md`, `README.md`, `docs/ARCHITECTURE.md`, `docs/BRANCH_GUIDES.md`, `docs/QUICK_REFERENCE.md`, and `tools/README.md`.  
**Suggestion:** Keep `tdms_explorer` installation explicit in setup/CI; do not reintroduce `tdms_to_png` references unless the legacy tool is restored.  
**Owner:** Documentation & Reporting, Maintenance.

**Finding:** Root-level research scripts not in 6-branch catalog.  
**Where:** `orientation_cnn.py`, `test_lodestar_orientation.py`, `orientation_cnn_ckpt.pt`, `lodestar_orientation_test_out/` (if present).  
**Issue:** Unclear ownership (Research vs Core); not integrated with `src/` or Web.  
**Suggestion:** Move under `debug/` or `src/experiments/` with a one-line README pointer, or explicitly tag as “optional research” in INVENTORY.  
**Owner:** Research & Experimentation, Documentation.

**Finding:** `test_lodestar_orientation.py` depends on `lodestar_link`.  
**Where:** Imports `from lodestar_link.lodestar import LodeSTAR`.  
**Issue:** Repo may not clone with that dependency; CI and other developers need a documented install (submodule, pip, or local path).  
**Suggestion:** Document in README or `requirements-research.txt`; fail fast with a clear message if import missing.  
**Owner:** Research & Experimentation.

**Finding:** TDMSExplorer now lives outside the repo as an installed package.  
**Where:** `/opt/mona_jupyterhub_env/lib/python3.10/site-packages/tdms_explorer`.  
**Issue:** Runtime depends on the environment package and stale install metadata can create duplicate Jupyter launcher tiles.  
**Suggestion:** Keep a single installed `tdms_explorer` distribution and remove stale `~dms_explorer-*` metadata.  
**Owner:** Tools & Automation, Documentation.

**Finding:** Widespread `sys.path` manipulation (unchanged pattern).  
**Where:** Web (src only), `src/train_single_particle.py` (tools), tests, ELAB scripts, `orientation_cnn.py` (repo root).  
**Issue:** Fragile execution from arbitrary CWD.  
**Suggestion:** Editable install and package namespaces.  
**Owner:** Maintenance & Operations.

---

### 2.3 RELIABILITY & ROBUSTNESS

**Finding:** No request authentication tied to identity.  
**Where:** All endpoints using `username` in path or JSON/form body.  
**Issue:** Any client can read/write another user’s namespace by guessing `username`.  
**Suggestion:** Session/token; enforce match between credential and `username`.  
**Owner:** Web Development.

**Finding:** Path traversal via `username`.  
**Where:** `get_user_dir(username)` → `DATA_DIR / username` (line 71–72).  
**Issue:** `../` segments can escape `DATA_DIR`.  
**Suggestion:** Validate `username` (allowlist charset; reject `.` and `/`); resolve and assert prefix under `DATA_DIR`.  
**Owner:** Web Development.

**Finding:** Global in-memory `users`, `sessions`, `training_jobs`, `detect_files`; training in a thread.  
**Where:** `web/app.py` module globals and `threading.Thread` for `run_training`.  
**Issue:** Races; multi-worker deployments do not share state.  
**Suggestion:** Document single-process deployment or externalise state and jobs.  
**Owner:** Web Development.

**Finding:** TDMS settings simplified — possible UI/API drift.  
**Where:** `ChunkUploadStart` / `ChunkUploadComplete` / multipart upload only carry `normalize`; prior design had width/height/channel for TDMS.  
**Issue:** Frontends or docs expecting channel/geometry controls may be broken or behaviour changed silently.  
**Suggestion:** Confirm intentional removal; update `index.html` and API docs; if multi-channel TDMS is required, expose parameters on `TDMSFileExplorer` path.  
**Owner:** Web Development, Documentation.

---

### 2.4 PERFORMANCE & SCALABILITY

**Finding:** Repeated `extract_images()` in `export_tdms` (see 2.1).  
**Owner:** Web Development.

**Finding:** Video merge loads all frames into RAM.  
**Where:** `merge_videos` — `all_frames` list from all sources.  
**Issue:** Large MP4 sets can exhaust memory.  
**Suggestion:** Stream write or cap inputs; document limits.  
**Owner:** Web Development.

**Finding:** Base64 responses for frames, zip, video (unchanged).  
**Issue:** Memory spikes for large exports.  
**Owner:** Web Development.

---

### 2.5 SECURITY & SAFETY

**Finding:** Unauthenticated multi-tenant API (see 2.3).  
**Owner:** Web Development.

**Finding:** Exception text returned to clients (see 2.1).  
**Owner:** Web Development.

**Finding:** Weak password hashing (SHA-256, no salt).  
**Where:** `hash_password` in `web/app.py`.  
**Owner:** Web Development.

**Finding:** CORS `allow_origins=["*"]` with `allow_credentials=True`.  
**Where:** `web/app.py` middleware.  
**Owner:** Web Development.

**Finding:** Pickle-equivalent risk in `torch.load` for orientation scripts (see 2.1).  
**Owner:** Research & Experimentation.

---

### 2.6 MAINTAINABILITY & DEBT

**Finding:** `web/app.py` remains a large monolith (~1170 lines in this pass).  
**Finding:** `web/templates/index.html` still very large (not re-counted; assume similar to prior ~1688).  
**Suggestion:** Split by domain modules.  
**Owner:** Web Development.

**Finding:** `test/integration/` still only `__init__.py`.  
**Issue:** No automated end-to-end coverage (Web + Core + TDMS).  
**Owner:** Maintenance & Operations.

**Finding:** Installed TDMSExplorer CLI is outside this repo.  
**Issue:** CLI changes need to be reviewed in the TDMSExplorer package, not MONA_LodeSTAR.  
**Suggestion:** Add MONA smoke tests for `tdms_explorer` import and the `tdms-explorer` command.
**Owner:** Tools & Automation.

---

### 2.7 CONSISTENCY WITH WORKFLOW

**Finding:** “Web uses only committed Core code” still violated for training/detection (inlined LodeSTAR).  
**Owner:** Web Development + Core Model Development.

**Finding:** Docs previously said Web depends on `tdms_to_png`; code uses `tdms_explorer`.  
**Status:** Synced on 2026-05-16.  
**Owner:** Documentation & Reporting.

**Finding:** Root `elab_config.yaml` examples still reference `python src/elab_cli.py` (if unchanged) — wrong path vs `tools/elab_cli.py` / `elab.py`.  
**Owner:** Tools & Automation, Documentation (verify file; fix if still wrong).

---

## 3. OPEN QUESTIONS

1. Confirm the operational rule that **`tdms_explorer`** is the permanent replacement for **`tdms_to_png`** and is installed in `/opt/mona_jupyterhub_env`.
2. Should **`orientation_cnn.py`** / **`test_lodestar_orientation.py`** become part of Core (`src/`), stay as Research (`debug/`), or ship as optional extras with separate requirements?
3. How should **`lodestar_link`** be obtained in a reproducible way (submodule, version pin, internal package)?
4. Was **removing TDMS width/height/channel** from upload APIs intentional for all use cases, or should `TDMSFileExplorer` expose those controls again for multi-channel / non-standard layouts?
5. Single **`refine_position_to_center`** implementation: should it live next to **`utils`** in Core or in a small **`tools/image_geometry.py`** shared by Web/research?

---

## 4. OPTIMIZATION IDEAS

**Structural**
- One installable graph: `mona_track` + `tdms_explorer` (path dep) in one developer install command.
- Consolidate orientation/refinement helpers; unify checkpoint loading policy.
- Split `web/app.py`; keep TDMSExplorer CLI ownership in the external package.

**Process**
- After TDMS migration: add setup/CI coverage proving `tdms_explorer` imports in `web.app`.
- Add one integration test: install deps → start app import → open one TDMS fixture (if license permits a tiny fixture).

**Technical**
- Cache `extract_images()` result in `export_tdms` (or add `frame_count` on explorer).
- Restrict CORS; sanitize usernames; session-based auth.
- `torch.load(..., weights_only=True)` where applicable.

---

## 5. PRIORITY

**P0**
- Web **import/runtime**: ensure `tdms_explorer` is installable and documented; otherwise production and CI break.
- **No auth + path traversal + exception leakage** on Web (unchanged from prior review).

**P1**
- **Documentation drift** (INVENTORY/ARCHITECTURE vs `tdms_explorer`; ELAB example paths if still wrong).
- **Duplicate `extract_images()`** in `export_tdms`.
- **`torch.load` safety** for checkpoints in orientation scripts if files can be untrusted.
- **`lodestar_link`** dependency story for `test_lodestar_orientation.py`.

**P2**
- Web/Core **duplication** for train/detect; **CSV column logic** in `orientation_cnn`; **refine_position** duplication.
- **Monolithic** `web/app.py`, `index.html`.
- **Empty integration tests**.

**P3**
- Typing, API spec (OpenAPI already available from FastAPI), streaming large exports, job queue for training.

---

## 6. EXPERIMENT RESULTS — Orientation Detection (eLabFTW #290)

**Experiment:** LodeSTAR: JP orientation detection  
**eLabFTW ID:** 290 | **Category:** 4 | **Created:** 2026-03-02  
**Objective:** Determine particle orientation angle φ alongside (x, y) position from darkfield microscopy of Fe-capped Janus particles.  
**Test data:** `data/JP_FE/wf_2_40/04/` — 574-frame video, CSV ground truth (x, y, φ). Results on 100-frame subset (~60 particles/frame).  
**Training sample:** `data/Samples/JP_Fe_wf_2_40/JP_Fe_wf_2_40.png`

### Option A — 4-channel LodeSTAR (`test_lodestar_orientation.py`)

Extended LodeSTAR to 5 output channels (Δx, Δy, cos φ, sin φ, weight). Custom `Rotation4Ch` transform rotates position and orientation channels jointly via kornia. Magnitude regularisation loss enforces cos²+sin²→1. Trained 200 epochs on single sample with affine augmentations.

| Metric | Value |
|---|---|
| Architecture | 5-ch UNet (10 conv blocks), 1 MB checkpoint |
| Detections | ~60/frame, centre refined via Hough circle + CoM |
| Orientation | cos/sin channels did not converge — SSL consistency loss provides no gradient signal for orientation from single unlabelled sample |
| **Status** | **Unsuccessful for orientation; position detection works well** |

### Option B — Orientation CNN (`orientation_cnn.py`)

Small supervised CNN (4 conv layers 16→32→64→64, FC→2) trained on 32×32 crops centred at refined particle positions (Hough circle + CoM, mean shift 8.5 px from LodeSTAR peak to geometric centre). MSE loss on (cos φ, sin φ).

| Metric | Value |
|---|---|
| Architecture | OrientationCNN, patch_size=32, 257 KB checkpoint |
| Predictions | 5849 across 100 frames, 100% coverage |
| Key insight | Using refined positions (geometric centre) instead of raw LodeSTAR intensity peak significantly improved accuracy — properly centred crops resolve systematic bias |
| **Status** | **Good results with refined positions** |

### Option C — Velocity-vector orientation (`test_lodestar_orientation.py --test`)

After LodeSTAR detection + centre refinement, particles linked frame-to-frame (nearest-neighbour, max_link_dist=50 px). Displacement vector (dx, dy) → φ = atan2(dy, dx).

| Metric | Value |
|---|---|
| Coverage | 5112/5984 detections (85%) have velocity φ |
| Unlinked | 15% — last-frame or exceeding max_link_dist |
| Centre refinement | mean shift 8.46 px, max 43.3 px (5849 rows) |
| **Status** | **Usable baseline; accuracy depends on frame rate and detection precision** |

### Attached to eLabFTW #290

| File | Description |
|---|---|
| `detections.csv` | LodeSTAR detections + velocity orientation (5984 rows, 100 frames) |
| `detections_orientation_cnn.csv` | CNN-predicted orientation (5849 rows, 100 frames) |
| `centers.csv` | Hough/CoM refined centres (5849 rows) |
| `orientation_ckpt.pt` | 4-channel LodeSTAR checkpoint (Option A) |
| `orientation_cnn_ckpt.pt` | Orientation CNN checkpoint (Option B) |
| `vis_optionA_frame001.png` | Example: 4-ch LodeSTAR orientation (green=GT, red=predicted) |
| `vis_optionB_frame001.png` | Example: CNN orientation arrows |
| `vis_optionC_frame001.png` | Example: velocity-vector orientation |

---

**End of Review Report**
