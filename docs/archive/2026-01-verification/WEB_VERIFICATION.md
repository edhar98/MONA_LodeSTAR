# Web App Verification Report

**Date:** 2026-07-15  
**Branch:** Web Development  
**Scope:** `web/app.py` (~1950 lines), `web/templates/index.html` (~1500 lines), `web/data/`, `web/*.json`.

This report supersedes the 2026-01-27 verification. The web app was substantially rewritten since then (sidebar UI, tracking/analysis panels, `tdms_explorer`, multi-crop training, WebSocket training progress, batch detection).

## 1. Architecture snapshot (current)

| Area | Current state |
|------|----------------|
| **UI** | Sidebar app: Files, Training, Detection, Tracking, Analysis. Login is a full-screen `#login-overlay` (z-index 100); app shown after auth. |
| **Backend** | FastAPI in `web/app.py`. Paths via `Path(__file__)`. |
| **sys.path** | `src`, `src/tracking`, `src/analysis` (not `tools/`). |
| **TDMS** | Installed package `tdms_explorer.TDMSFileExplorer` (not `tools/tdms_to_png`). |
| **Utils** | `import utils` from `src/`. Used for `load_yaml` in `/config/defaults`. |
| **Training** | Inline DeepTrack + `dl.LodeSTAR` + `dl.Trainer`. Multi-crop pool (`crop_*.jpg`) with legacy `<name>.jpg` fallback. Does **not** call `src/train_single_particle.py`. WebSocket `/ws/train/{job_id}` + poll. |
| **Detection** | Single-model `dl.LodeSTAR` only. No `composite_model` / `trained_models_summary.yaml`. GPU-aware `load_model()`. |
| **Tracking** | Optional import of Core `track_particles` (`apply_nms`, `link_tracks`, `interpolate_gaps`). Soft-fail if unavailable. |
| **Analysis** | Optional import of Core `analyze_tracks` (MSD / ABP). Soft-fail if unavailable. |
| **User data** | `web/data/<username>/{uploads,samples,models,results,masks}`. Jobs: `web/training_jobs.json`, `web/background_jobs.json`, `web/users.json`. |

## 2. Check results

| Check | Result | Notes |
|-------|--------|-------|
| **Imports / paths** | ok | `SRC_DIR = Path(__file__).parent.parent / "src"`; inserts `src`, `src/tracking`, `src/analysis`. |
| **Import utils** | ok | From `src` via sys.path. |
| **Import TDMS** | ok | `from tdms_explorer import TDMSFileExplorer` (env package). `extract_frame()` uses `explorer.extract_images()`. |
| **Import tracking/analysis** | ok / soft | Try/except; `_tracking_available` / `_analysis_available` flags. |
| **No tools/tdms_to_png** | ok | BRANCH_GUIDES previously said tdms_to_png; current code uses `tdms_explorer` only. |
| **run_training()** | ok | Multi-crop + legacy sample; saves `models/<particle_name>_weights.pth`; WS push; cancel endpoint. |
| **load_model() / detect** | ok | Builds `dl.LodeSTAR`, loads `.pth`, moves to cuda/cpu; `detect(..., mode="constant", cutoff)`. |
| **Config defaults** | ok | `SRC_DIR / "config.yaml"` + `utils.load_yaml`; includes tracking/ABP defaults. |
| **User dirs** | ok | No writes under `src/` or `tools/`. |
| **CWD** | ok | Path(__file__)-based; run from repo root recommended. |
| **Browser / real train** | skipped | Manual only. |

## 3. Endpoints (current)

| Area | Endpoints | Notes |
|------|-----------|-------|
| Auth | `POST /auth/register`, `POST /auth/login`, `GET /auth/check/{username}` | Session in `web/data/<user>/session.json` |
| Health | `GET /health` | GPU badge in UI |
| Upload | `POST /upload`, `/upload/start`, `/upload/chunk/{id}`, `/upload/complete`, `POST /files/load-path`, `POST /upload/csv` | Chunked + server-path load |
| Files | `GET/DELETE /files/...`, `GET /frame/...` | |
| Samples | `POST /sample`, multi-crop DELETE/preview | Crop pool for training |
| Masks | `POST /mask`, `/mask/circular` | |
| Training | `POST /train`, `POST /train/{id}/cancel`, `GET /train/...`, `WS /ws/train/{id}` | |
| Models | `GET/DELETE/PUT .../rename` | |
| Detection | `POST /detect/upload`, `GET /detect/frame/...`, `POST /detect`, `POST /detect/batch` | UI uses upload → `/detect/frame` (not bare `POST /detect`) |
| Tracking | `POST /track`, visualize overview/video | Background jobs |
| Analysis | `POST /analyze/abp` | |
| Jobs | `GET /jobs/{id}`, `GET /jobs/user/{username}` | |
| TDMS | `GET /tdms/structure/...`, `POST /tdms/export` | Via TDMSFileExplorer |
| Config | `GET /config/defaults` | |
| Results | `GET /results/...`, download | |
| Video | `POST /video/merge`, `/video/merge-from-files` | |

## 4. Detection flow (important for debugging)

**UI path (what the browser actually does):**
1. Upload image/TDMS via `POST /detect/upload` (or pick a file already in Files).
2. Select model + frame.
3. Run → `GET /detect/frame/{user}/{file_id}/{index}?model_id=&alpha=&beta=&cutoff=&return_weightmap=`.

**Legacy / unused by UI:** `POST /detect` (multipart file + form fields) still exists in the backend but the current UI does not call it for single-image detect.

**Likely debug targets:**
- `api()` does not check `r.ok`; FastAPI errors use `detail`, UI often checks `d.error` only → silent failures / broken canvas.
- Login overlay stays visible until Sign In succeeds; if `api()` hangs or returns oddly, overlay remains (perceived as “hung on login”). Overlay uses `position: fixed; inset: 0; z-index: 100`.
- Single-image requires a successful `/detect/upload` (or existing Files entry) so `detFileId` is set; selecting only a local file without completing upload leaves Detect disabled/alert.

## 5. Consistency with REVIEW_REPORT / INVENTORY

- Web still reimplements training/detection with inline `dl.LodeSTAR` (not Core CLI scripts) — **per REVIEW_REPORT**.
- Web now **does** call committed Core tracking/analysis modules when importable — partial alignment with “use committed Core”.
- Monolith `app.py` / `index.html` — **per REVIEW_REPORT / INVENTORY**.
- TDMS path changed: docs that still say `tools/tdms_to_png` for Web are outdated (this report + BRANCH_GUIDES updated).

## 6. Run method

```bash
cd /path/to/MONA_LodeSTAR
# Use env that has torch, deeptrack, tdms_explorer, fastapi, etc.
uvicorn web.app:app --reload --host 0.0.0.0 --port 8000
```

Jupyter proxy: UI derives `API_BASE` from pathname containing `proxy/<port>`.

## 7. Known issues / open debug list

### Files panel (2026-07-15 pass)
Fixed in UI:
- `api()` now checks HTTP status and maps FastAPI `detail` → `error`.
- Sign-in now loads session `files`/`models` (previously only auto-login via localStorage did).
- Preview selection highlight + frame_count/size meta refresh after first preview.
- Chunked upload checks chunk HTTP status; delete surfaces API errors.
- TDMS export requires a TDMS file selected via Preview; uses Normalize checkbox.

Still open (Files):
- Large server-path TDMS: frame_count stays 1 until Preview (by design; first preview can be slow).
- Export download embeds full zip/mp4 as base64 in JSON (memory-heavy for large files).
- `POST /upload` (non-chunked) unused by UI; chunked path is the only Files upload path.

### Other panels
1. **Login overlay stuck** — hung fetch or missing `.hidden` (partially helped by api() errors).
2. **Single-image detection** — upload → `/detect/frame`; verify after Files fixes.
3. **`POST /detect` vs UI** — dead path from UI.
4. **Core vs Web training** — no custom LodeSTAR / orientation / area mode.
5. **INVENTORY/.gitignore** vs `web/data/<user>/` layout still drifted.

## 8. Next work context

Intended next steps: add features and debug existing ones. Prefer:
- Fix `api()` error handling first (unblocks diagnosis of detect/login).
- Reproduce detect: upload PNG → select model → Detect → Network tab for `/detect/frame`.
- Keep Web integrating **committed** Core only; new physics/tracking features should call `src/tracking` / `src/analysis` rather than duplicating.
