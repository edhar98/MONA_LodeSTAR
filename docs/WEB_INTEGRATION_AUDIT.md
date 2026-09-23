# Web integration inventory

Date: 2026-09-23. Target: existing MONA JupyterHub deployment.

Follow-up: composite detection and optional learned trajectory inference were
subsequently integrated and tested in this checkout. See
[learned-model integration](WEB_LEARNED_MODELS.md) for current scope and evidence.
The phase-1 inventory below is retained as the pre-integration snapshot.

This phase-1 inventory describes the current working tree on `dev`, not a clean release or a deployed instance. The tree already contains modified documentation, web code, deletions, and untracked research/local files. No existing changes were reverted. This report adds no features and changes no application code.

The inventory agent inspected frontend controls and API calls, routes and their implementations, core entry points, and project guidance. It did not launch a service, run training/inference, invoke external integrations, inspect credentials, or validate the browser. The coordinating agent separately reports that `python test/run_tests.py --verbose` passed 10 tests in 0.470 seconds; that result is not end-to-end web validation.

## Classification

- **Integrated:** frontend invocation reaches substantive backend implementation. Runtime operation remains unverified unless explicitly stated.
- **Partial:** a narrower workflow is integrated, or backend functionality lacks a browser control.
- **CLI/research-only:** repository capability exists, but no corresponding browser/backend integration was found.
- **Broken/unverified:** a concrete static defect or unresolved compatibility risk affects the workflow. Confirmed defects are distinguished from runtime unknowns.

## Inventory

References are repository-relative paths and line numbers at inspection time.

| Capability | Classification | Evidence and limits |
|---|---|---|
| Image and TDMS ingestion | Integrated | Chunk start/upload/complete calls: `web/templates/index.html:2032`; handlers: `web/routers/files.py:47`. Accepted extensions: `web/config.py:60`. |
| Server file/path/glob ingestion | Integrated | Frontend: `web/templates/index.html:2005`; backend: `web/routers/files.py:160`. Files are registered without copying, under the Hub user's OS permissions. |
| Session file ordering and deletion | Integrated | Frontend reorder/bulk-delete: `web/templates/index.html:1664`, `:1697`; routes: `web/routers/files.py:277`. Registered server originals are retained by deletion logic at `:266`. |
| TIFF stacks and raw image depth | Partial | TIFF is accepted, but non-TDMS image loaders read one image, report one frame and convert to grayscale/uint8: `web/services/frames.py:51`, `:69`. Multipage TIFF stack ingestion is absent. |
| TDMS exploration | Integrated | Structure/channel/frame/histogram/filter/edges/profile/compare calls: `web/templates/index.html:1823`, `:1911`, `:1924`; routes: `web/routers/tdms_explorer.py:55`, `:97`, `:106`. Requires installed `tdms_explorer`: `web/services/tdms_cache.py:7`. |
| TDMS PNG/ZIP/MP4 export | Integrated | Browser call: `web/templates/index.html:2074`; export code: `web/routers/tdms_explorer.py:165`. Frame ranges and uint8/uint16 PNG supported. Runtime codecs unverified. |
| Multi-file video merge | Integrated | Browser call: `web/templates/index.html:2175`; job creation/implementation: `web/app.py:2213`, `:2243`. Merged video list/download/delete wired. |
| Crop pool and reference orientation | Integrated | Crop/reference direction payload: `web/templates/index.html:2634`; persistence: `web/app.py:455`; template resolution: `web/app.py:1073`. |
| General and circular training masks | Partial, backend-only | Implementations at `web/app.py:492`, `:558`; no corresponding frontend API calls found. Crescent-ratio mask preview is a separate feature. |
| Web LodeSTAR training | Partial | Browser training request: `web/templates/index.html:2736`; implementation: `web/app.py:651`, `:823`. Crop pools, augmentations, hyperparameters, progress and cancellation wired. Hardcodes default LodeSTAR; no architecture selection or checkpoint resume; disables checkpointing/logger. CLI architecture/resume support: `src/detection/train_single_particle.py:255`, `:367`. |
| Synthetic data and full training pipeline | CLI-only | `src/detection/generate_samples.py:22`, `:273` and `src/detection/run_single_particle_pipeline.py`; no browser orchestration found. |
| Model catalogue and deletion | Integrated, limited | Browser list/delete: `web/templates/index.html:2840`, `:2872`; catalogue combines user models and read-only CLI summary entries: `web/app.py:946`, `:1001`. |
| Model rename/import/export | Partial | Rename route exists at `web/app.py:1036`, but no frontend call found. No browser model upload/download workflow found. |
| CLI model architecture fidelity | Unverified compatibility risk | CLI entries inherit current `src/config.yaml` at `web/app.py:932`; loader selects architecture at `:988`. Historical weights do not carry independently verified architecture metadata here. Actual checkpoints were not loaded. |
| Single-model detection | Integrated | Browser call: `web/templates/index.html:3005`; route/core: `web/app.py:1373`, `:1181`. Standard, area, watershed and template modes; overlays and optional confidence image. |
| Template orientation | Integrated | UI parameters: `web/templates/index.html:685`, `:2925`; template bank/postprocessing: `web/app.py:1154`, `:1221`. Exports phi/NCC; no retraining needed. |
| Composite/multiclass detection | CLI-only | `src/detection/composite_model.py:19`; web requests one model ID at `web/app.py:227`. No composite invocation found. |
| Trackpy detection engine | Research/planned | Web inference exclusively uses LodeSTAR: `web/app.py:980`, `:1181`. AGENTS describes engine selection as future work. |
| Batch detection/global frames | Integrated, partial parity | Frontend selected-file call: `web/templates/index.html:3070`; cumulative global frame and local/source metadata: `web/app.py:1437`, `:1454`. Concatenates selected files in order; does not invoke CLI numeric stack-offset merging (`src/utils.py:97`) or expose first-stack/original-stack-offset controls. |
| Batch artifacts | Partial | Combined CSV at `web/app.py:1487`; no equivalent full CLI per-frame overlay/weightmap/per-stack artifact suite. Non-template phi/NCC are NaN; CSV includes pandas index at `:1493`. |
| Tracking | Integrated | CSV ingestion and tracking requests: `web/templates/index.html:3107`, `:3130`; shared NMS/Hungarian/linear-circular gap filling: `web/app.py:1563`. Distance, minimum length and maximum gap configurable. |
| Track overview/video | Integrated | Frontend: `web/templates/index.html:3238`, `:3258`; core calls: `web/app.py:1692`, `:1725`. Sources may be session files or server background directory (`:1683`). Frame mapping follows selected source order (`web/services/frames.py:98`); consistency requires runtime verification. |
| ABP/translational/angular MSD | Integrated, partial export parity | Browser interpolation toggle and fit request: `web/templates/index.html:3294`; core fitting/unit conversion: `web/app.py:1782`. Missing phi suppresses angular analysis. Returns fit values/PNG but does not persist MSD/AMSD CSVs like CLI (`src/analysis/analyze_tracks.py:320`). |
| Raw motion distributions/correlations | CLI-only | `src/analysis/analyze_motion_statistics.py:97`, `:177`, `:381`; no web control/route/import found. |
| Interactions and confinement | CLI-only | `src/analysis/analyze_track_interactions.py:111`, `:521`; `src/analysis/analyze_confinement_drift.py:141`, `:452`. |
| Filtered ABP and velocity persistence | CLI-only | `src/analysis/compare_filtered_abp.py:284`; `src/analysis/analyze_velocity_persistence.py:504`. No browser model-comparison orchestration. |
| Causal LSTM | CLI/research-only | Training/prediction: `src/tracking/lstm_track_predictor.py:327`, `:435`; benchmark: `src/tracking/benchmark_lstm_gap_filling.py:271`. |
| Two-sided LSTM and Kalman refinement | CLI/research-only | `src/tracking/lstm_gap_filler.py:48`, `:381`, `:555`; variants: `src/tracking/build_track_variants.py:148`. Separate past/future LSTMs, not a single bidirectional PyTorch LSTM. |
| Supervised reference-calibrated correction | CLI/research-only | Pairing: `src/tracking/build_supervised_correction_dataset.py:172`; training: `src/tracking/train_supervised_correction_lstm.py:175`; application: `src/tracking/apply_supervised_correction_lstm.py:106`. No web pairing/training/application/validation flow. |
| Janus crescent ratio | Integrated | Browser preview/save: `web/templates/index.html:3709`, `:3770`; measurement: `web/app.py:1954`, `:1986`; CSV/overlay export: `:2027`. |
| Result downloads | Integrated, partial | Browser results/downloads: `web/templates/index.html:3809`, `:3175`; handlers: `web/app.py:2107`, `:2122`. No complete reproducible analysis/model bundle. |
| ELab | CLI-only | Wrapper: `elab.py:12`, upload mapping at `:32`. No browser controls/routes found. |
| WandB | CLI-only | Core helper: `tools/wandb_logging.py:53`; web training sets `logger=False` in `web/app.py:651` implementation. No web setup/run-link/sync flow. |
| JupyterHub integration | Integrated, runtime unverified | Entry point: `setup.py:14`; user/home and loopback binding: `web/jupyter_launch.py:17`, `:32`; browser proxy prefixes: `web/templates/index.html:974`; automatic Hub identity: `:1230`. |
| Job persistence/restoration | Broken by inspection | Imported dictionary aliases (`web/app.py:83`) become stale after loaders rebind state dictionaries (`web/state.py:91`, `:106`). Frontend restoration (`web/templates/index.html:1475`) is therefore unreliable when persisted files are loaded. Execution is in daemon threads (`web/app.py:844`, `:1550`), not a durable queue. |

## Cross-cutting findings for independent review

1. **Confirmed dictionary alias defect:** endpoint/thread updates target old dictionaries after startup reload, while save functions serialize newly rebound dictionaries. Confirm this independently before fixing state ownership.
2. **Confirmed model overwrite behavior:** repeated training of one particle saves to the same `{particle_name}_weights.pth` and appends another model record (`web/app.py:777` vicinity). Multiple catalogue entries can reference overwritten weights.
3. **Confirmed unconstrained paths:** CSV upload uses the client filename directly (`web/routers/files.py:222`); output names also enter paths directly (`web/app.py:1533`, `:1623`). Under Hub the filesystem impact follows the user's OS permissions.
4. **Deployment boundary:** existing Hub server-proxy authentication and per-user OS isolation are the intended boundary. Standalone login issues no session token and username checks do not authenticate a request (`web/auth.py:67`, `web/state.py:49`); this remains a defect but is not equivalent to anonymous exposure of the chosen Hub deployment. Not every route checks supplied identity.
5. **Packaging limitation:** `setup.py:6` packages only `web`, while runtime imports depend on adjacent `src` and `tools` (`web/app.py:32`). Existing source/editable deployment can satisfy that layout; a clean installed wheel cannot be assumed complete.

## Verification limits and next phase

No runtime claim is made for feature routes, browser behavior, current checkpoints, codecs, external services, deployment configuration, or scientific accuracy. `/health` (`web/app.py:392`) reports optional imports and GPU availability, not full workflow readiness. `test/unit/test_web_frames.py` exists; no comprehensive HTTP/browser integration suite was found under `test/`.

The independent reviewer should verify classifications and evidence, especially backend-only controls, state reloading, model versioning, and batch frame conventions. Correction work is a separate phase. CLI-only research tools are not automatically approved production defaults: retain the project's linear interpolation baseline and reference-calibrated terminology, and assess scientific validation before proposing their integration.
