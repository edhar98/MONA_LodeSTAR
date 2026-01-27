# Web App Verification Report

**Date:** 2026-01-27  
**Branch:** Web Development  
**Scope:** `web/app.py`, `web/templates/index.html`, `web/data/`, `web/*.json`. Verification only; no new features.

## 1. Check results

| Check | Result | Notes |
|-------|--------|-------|
| **Imports and paths** | ok | `web/app.py` lines 27–28: `sys.path.insert(0, str(Path(__file__).parent.parent / "src"))`, `sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))`. Resolve to repo root when app is loaded from repo (e.g. `uvicorn web.app:app` from repo root). |
| **Import tdms_to_png** | ok | `from tdms_to_png import extract_images_from_tdms` (line 29); `list_tdms_structure`, `save_images`, `save_video` imported locally in `/tdms/structure` and `/tdms/export`. From `tools` via sys.path; no uncommitted or branch-local paths. |
| **Import utils** | ok | `import utils` (line 31). From `src` via sys.path. Used in `/config/defaults` via `utils.load_yaml(str(config_path))`. |
| **run_training()** | ok | Uses one sample from `get_user_dir(username)/samples/<particle_name>/<particle_name>.jpg`, DeepTrack pipeline, `dl.LodeSTAR`, `dl.Trainer`; saves to `get_user_dir(username)/models/<particle_name>_weights.pth`. Does **not** call `src/train_single_particle.py`. Documented in BRANCH_GUIDES. |
| **load_model()** | ok | Loads `.pth` from user models dir; builds `dl.LodeSTAR` with config from model_info; `torch.load(..., map_location="cpu")`, `lodestar.eval()`. No `src/composite_model` or `trained_models_summary.yaml`. |
| **run_detection_on_image()** | ok | Uses `lodestar.detect(image_tensor, alpha=alpha, beta=beta, mode="constant", cutoff=cutoff)[0]`. Single-model detection only. |
| **Config defaults source** | ok | `/config/defaults` uses `config_path = Path(__file__).parent.parent / "src" / "config.yaml"`; reads via `utils.load_yaml(str(config_path))` when file exists. Matches BASELINE_REPORT and QUICK_REFERENCE (`src/config.yaml`). |
| **User data dirs** | ok | `web/data/<username>/{uploads,samples,models,results,masks}`. No writes under `src/` or `tools/`. |
| **Endpoints vs config/user/body** | ok | Auth, upload, samples, masks, train, models, detect, files, results: user dirs or request body. `/config/defaults`: reads `src/config.yaml` when present, else hardcoded defaults. No other config files. |
| **CWD / Path(__file__)** | ok | All repo-relative paths use `Path(__file__).parent` (WEB_DIR) or `Path(__file__).parent.parent` (repo root). CWD-independent when run from repo layout. Recommended run: from repo root, `uvicorn web.app:app`. Documented in BRANCH_GUIDES “Run assumptions”. |
| **TDMS tools usage** | ok | `extract_images_from_tdms`, `list_tdms_structure`, `save_images`, `save_video` from `tools/tdms_to_png.py`. Matches TOOLS_VERIFICATION.md and tools/README.md. Documented in QUICK_REFERENCE and BRANCH_GUIDES. |
| **Real training / ELAB** | skipped | No real training or ELAB runs; imports and path logic only. Manual run only for full flows. |

## 2. Endpoints and data sources

| Area | Endpoints | Config / user / body |
|------|-----------|----------------------|
| Auth | `POST /auth/register`, `POST /auth/login`, `GET /auth/check/{username}` | Request body; `web/users.json`; `web/data/<username>/session.json` |
| Upload | `POST /upload`, `POST /upload/start`, `POST /upload/chunk/{id}`, `POST /upload/complete` | Request body; user dir `uploads/` |
| Files | `GET /files/{username}`, `DELETE /files/{username}/{file_id}`, `GET /frame/...` | User session and uploads |
| Samples | `POST /sample`, `GET /samples/{username}`, `DELETE /sample/{username}/{particle_name}` | User session and `samples/` |
| Masks | `POST /mask`, `POST /mask/circular` | Request body; user dir `masks/` |
| Training | `POST /train`, `GET /train/active/{username}`, `GET /train/{job_id}` | Request body; `web/training_jobs.json`; user `models/` |
| Models | `GET /models/{username}`, `DELETE /models/...`, `PUT /models/.../rename` | User session and `models/` |
| Detection | `POST /detect`, `POST /detect/upload`, `GET /detect/frame/...` | Request body / detect_files; user models |
| TDMS | `GET /tdms/structure/...`, `POST /tdms/export` | User session; `tools/tdms_to_png` |
| Config | `GET /config/defaults` | `Path(__file__).parent.parent / "src" / "config.yaml"` + `utils.load_yaml`; fallback hardcoded defaults |
| Other | `GET /`, `GET /favicon.ico`, `GET /icon.svg`, `POST /test-upload`, `POST /video/merge`, `GET /results/{username}` | WEB_DIR templates/icon; user `results/` |

## 3. Single-model vs composite

- **Training:** Inline DeepTrack pipeline and `dl.LodeSTAR`/`dl.Trainer` only. No call to `src/train_single_particle.py`.
- **Detection:** Loads a single `.pth` per request via `load_model()`; uses `lodestar.detect(alpha, beta, mode="constant", cutoff)`. No `src/composite_model`, no `trained_models_summary.yaml`.
- **Documented** in BRANCH_GUIDES Data Flow and WEB_VERIFICATION (this file).

## 4. Consistency with REVIEW_REPORT and INVENTORY

- **REVIEW_REPORT:** Web reimplements training/detection (inline `dl.LodeSTAR`, pipelines, `run_detection_on_image`) instead of calling `src/train_single_particle.py` or `src/detect_particles.py` — **still accurate**. No code change in this verification; BRANCH_GUIDES and this report now state explicitly that Web does not call Core train/detect scripts.
- **REVIEW_REPORT:** `web/app.py` and `web/templates/index.html` are large single files — **per REVIEW_REPORT**; not re-implemented here.
- **REVIEW_REPORT:** Session/error handling, CORS, password hashing, threading — **per REVIEW_REPORT**; only verification and documentation in this pass.
- **INVENTORY:** Web uses `sys.path` to import from `src` and `tools`; user data under `web/data/<username>/` — **per INVENTORY**. INVENTORY also mentions `web/uploads/`, `web/samples/` in .gitignore; actual layout is `web/data/<username>/uploads`, `web/data/<username>/samples` (noted in REVIEW_REPORT; doc alignment is a separate task).

## 5. Doc updates made

- **docs/BRANCH_GUIDES.md (Web Development):** Data Flow “Training” and “Detection” updated to state that Web does not call `src/train_single_particle.py`; training is inline (DeepTrack, `dl.LodeSTAR`, `dl.Trainer`); detection uses `load_model` + `lodestar.detect(alpha, beta, mode="constant", cutoff)`; single-model only, no composite_model or trained_models_summary. Dependencies expanded to list tdms_to_png functions and utils. Added “Run assumptions” (Path(__file__), CWD-independent when run from repo layout; recommended run from repo root). Example usage path made generic (`/path/to/MONA_LodeSTAR`).
- **docs/QUICK_REFERENCE.md:** Web Imports section: clarified that path resolution uses `__file__`; added one-line summary of tdms_to_png functions used by Web and pointer to `tools/tdms_to_png.py` and TOOLS_VERIFICATION.md.

## 6. Run method and open questions

- **Recommended run:** From repo root, `uvicorn web.app:app --reload` (or `python -m uvicorn web.app:app`). Ensures `Path(__file__).parent.parent` is the repo root.
- **Skipped in this pass:** Browser/UI checks, real training runs, real ELAB, automated integration tests.
- **Open (unchanged):** Whether Web should ever call Core (`train_single_particle.py` / `detect_particles.py`); where “default” training/detection config should live for Web; INVENTORY/.gitignore alignment with `web/data/<username>/` layout (per REVIEW_REPORT).
