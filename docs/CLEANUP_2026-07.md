# Cleanup Report - July 2026

Date: 2026-07-27  
Branch context: Maintenance & Operations  
Source of truth used: `AGENTS.md`

## 1. Files Deleted

No root scratch image files were deleted in this turn because all four requested files were already absent from the repo root:

- `test_image.png`
- `test_image_phi0245.9.png`
- `test_image_phi0253.6.png`
- `scratch_512_overview.png`

Reference check performed before deletion:

```bash
rg -n "test_image\.png|test_image_phi0245\.9\.png|test_image_phi0253\.6\.png|scratch_512_overview\.png" -g '*.py' -g '*.md' -g '*.ipynb'
```

Result: no matches in Python, Markdown, or notebook files.

Root scratch file existence check:

```bash
find . -maxdepth 1 -type f \( -name 'test_image.png' -o -name 'test_image_phi0245.9.png' -o -name 'test_image_phi0253.6.png' -o -name 'scratch_512_overview.png' \) -print
```

Result: no files found.

## 2. Files Moved Or Archived

Moved historical January 2026 verification/report files to `docs/archive/2026-01-verification/`:

| Source | Destination |
|--------|-------------|
| `INVENTORY.md` | `docs/archive/2026-01-verification/INVENTORY.md` |
| `CLEANUP_REPORT.md` | `docs/archive/2026-01-verification/CLEANUP_REPORT.md` |
| `BASELINE_REPORT.md` | `docs/archive/2026-01-verification/BASELINE_REPORT.md` |
| `TOOLS_VERIFICATION.md` | `docs/archive/2026-01-verification/TOOLS_VERIFICATION.md` |
| `RESEARCH_VERIFICATION.md` | `docs/archive/2026-01-verification/RESEARCH_VERIFICATION.md` |
| `WEB_VERIFICATION.md` | `docs/archive/2026-01-verification/WEB_VERIFICATION.md` |
| `REVIEW_REPORT.md` | `docs/archive/2026-01-verification/REVIEW_REPORT.md` |
| `DUPLICATES_DOCUMENTATION.md` | `docs/archive/2026-01-verification/DUPLICATES_DOCUMENTATION.md` |
| `docs/DOCUMENTATION_VERIFICATION.md` | `docs/archive/2026-01-verification/DOCUMENTATION_VERIFICATION.md` |

Added `docs/archive/2026-01-verification/README.md` explaining that these are historical snapshots superseded by `AGENTS.md` because of the `src/` restructure and TDMSExplorer un-vendoring.

## 3. `.gitignore` Diff

```diff
diff --git a/.gitignore b/.gitignore
index a02c266..6ae402c 100644
--- a/.gitignore
+++ b/.gitignore
@@ -24,9 +24,6 @@ snr_test_results.txt
 test_results_summary.yaml
 test_composite_results_summary.yaml
 
-# Compiled binaries
-tools/tdms_to_png
-
 # Runtime data
 web/sessions.json
 web/training_jobs.json
@@ -40,3 +37,17 @@ web/feedback/
 # Output directories
 detection_output/
 debug_outputs/
+analysis_outputs/
+lstm_outputs/
+supervised_correction_outputs/
+lodestar_orientation_test_out/
+
+# Caches and editor state
+.matplotlib_cache/
+.cursorindexingignore
+.vscode/
+
+# Root scratch artifacts
+/test_image.png
+/test_image_phi*.png
+/scratch_512_overview.png
```

Reasons:

- Removed `tools/tdms_to_png`: stale vendored/binary ignore entry. TDMSExplorer is now the installed `tdms_explorer` package.
- Added `analysis_outputs/`: generated physics/model-comparison outputs referenced by current analysis work.
- Added `lstm_outputs/`: generated causal LSTM, BiLSTM gap filler, and Kalman benchmark outputs.
- Added `supervised_correction_outputs/`: generated supervised correction datasets, checkpoints, refined tracks, and validation outputs.
- Added `lodestar_orientation_test_out/`: generated orientation experiment outputs.
- Added `.matplotlib_cache/`: local plotting cache.
- Added `.cursorindexingignore`: editor/indexer state.
- Added `.vscode/`: editor state.
- Added `/test_image.png`, `/test_image_phi*.png`, `/scratch_512_overview.png`: root scratch artifacts verified as unreferenced so they do not reappear.

`elab_updates/` and `notebooks/` were not added to `.gitignore`; they contain real experimental/reporting artifacts referenced by `AGENTS.md` and ELab entries.

## 4. Documentation Corrections

### A. Active Docs Corrected In Place

`README.md`

- Replaced flat `src/` structure with `src/detection/`, `src/tracking/`, `src/analysis/`, and shared `src/utils.py`.
- Corrected commands to repo-root runnable paths such as `python src/detection/generate_samples.py`, `python src/detection/train_single_particle.py`, `python src/detection/test_single_particle.py`, `python src/detection/test_composite_model.py`, `python src/detection/compare_models.py`, and `python src/detection/run_single_particle_pipeline.py`.
- Added current tracking, LSTM/BiLSTM/Kalman benchmark, supervised correction, and physics-analysis capabilities.
- Updated web structure to mention `web/routers/`, `web/services/`, `web/auth.py`, `web/config.py`, and `web/state.py`.
- Replaced live references to old inventory/cleanup docs with the January 2026 archive.

`docs/ARCHITECTURE.md`

- Updated Core ownership from flat `src/*.py` to `src/detection/`, `src/tracking/`, `src/analysis/`, and `src/utils.py`.
- Corrected dependency-map paths for detection files.
- Added tracking, gap filling, supervised correction, and physics-analysis scripts.
- Updated web architecture from a monolithic `web/app.py` description to app assembly plus routers/services/auth/config/state.
- Replaced duplicate-doc reference with the archive note.

`docs/BRANCH_GUIDES.md`

- Corrected all stale command paths to `src/detection/...`, `src/tracking/...`, or `src/analysis/...`.
- Replaced live references to `WEB_VERIFICATION.md`, `BASELINE_REPORT.md`, and `RESEARCH_VERIFICATION.md` with archive references.
- Updated web scope and endpoint ownership to include routers/services/auth/config/state.
- Added tracking, gap filling, supervised correction, and physics-analysis key files.
- Corrected experimental model paths to `src/detection/lodestar_*.py`.

`docs/QUICK_REFERENCE.md`

- Confirmed repo-root runnable commands for detection, tracking, gap filling, supervised correction, and physics analysis.
- Added current web module locations.
- Added output directories for `analysis_outputs/`, `lstm_outputs/`, `supervised_correction_outputs/`, and `lodestar_orientation_test_out/`.
- Removed the live `tools/tdms_to_png` reference and pointed historical verification links to the archive.

### B. Source-Of-Truth Drift Corrected

`AGENTS.md` and `CLAUDE.md` were path-corrected only. Experimental results, decision framing, ELab entries, and conclusions were not rewritten.

Corrected stale paths include:

- `src/lstm_track_predictor.py` -> `src/tracking/lstm_track_predictor.py`
- `src/benchmark_lstm_gap_filling.py` -> `src/tracking/benchmark_lstm_gap_filling.py`
- `src/lstm_gap_filler.py` -> `src/tracking/lstm_gap_filler.py`
- `src/build_track_variants.py` -> `src/tracking/build_track_variants.py`
- `src/build_supervised_correction_dataset.py` -> `src/tracking/build_supervised_correction_dataset.py`
- `src/train_supervised_correction_lstm.py` -> `src/tracking/train_supervised_correction_lstm.py`
- `src/apply_supervised_correction_lstm.py` -> `src/tracking/apply_supervised_correction_lstm.py`
- `src/track_particles.py` -> `src/tracking/track_particles.py`
- `src/visualize_tracks.py` -> `src/tracking/visualize_tracks.py`
- `src/analyze_tracks.py` -> `src/analysis/analyze_tracks.py`
- `src/analyze_motion_statistics.py` -> `src/analysis/analyze_motion_statistics.py`
- `src/analyze_track_interactions.py` -> `src/analysis/analyze_track_interactions.py`
- `src/analyze_confinement_drift.py` -> `src/analysis/analyze_confinement_drift.py`
- `src/compare_filtered_abp.py` -> `src/analysis/compare_filtered_abp.py`
- `src/analyze_velocity_persistence.py` -> `src/analysis/analyze_velocity_persistence.py`
- `src/test_single_particle.py` -> `src/detection/test_single_particle.py`
- `src/lodestar_orientation.py` -> `src/detection/lodestar_orientation.py`

Verification:

```bash
rg -n "src/(train_single_particle|test_single_particle|detect_particles|crop_detections|generate_samples|image_generator|compare_models|test_composite_model|run_single_particle_pipeline|track_particles|visualize_tracks|lstm_track_predictor|benchmark_lstm_gap_filling|lstm_gap_filler|build_track_variants|analyze_tracks|compare_filtered_abp|analyze_velocity_persistence|analyze_motion_statistics|analyze_track_interactions|analyze_confinement_drift|build_supervised_correction_dataset|train_supervised_correction_lstm|apply_supervised_correction_lstm)\.py" README.md docs/ARCHITECTURE.md docs/BRANCH_GUIDES.md docs/QUICK_REFERENCE.md AGENTS.md CLAUDE.md
```

Result: no stale flat-path matches.

### C. Historical Artifacts Archived

The January 2026 reports were moved to `docs/archive/2026-01-verification/` and were not updated in place:

- `INVENTORY.md`
- `CLEANUP_REPORT.md`
- `BASELINE_REPORT.md`
- `TOOLS_VERIFICATION.md`
- `RESEARCH_VERIFICATION.md`
- `WEB_VERIFICATION.md`
- `REVIEW_REPORT.md`
- `docs/DOCUMENTATION_VERIFICATION.md`
- `DUPLICATES_DOCUMENTATION.md`

Reason: they describe the pre-restructure state and stale TDMS vendoring assumptions.

### D. Feature Docs Assessed; Paths Changed Only

Path fixes were applied where needed. No feature doc was archived or merged without user approval.

| File | Status | Recommendation |
|------|--------|----------------|
| `COMPOSITE_MODEL_README.md` | Active | Keep; it is the main detailed composite-model doc. |
| `COMPOSITE_MODEL_UPDATE.md` | Superseded by the current composite implementation | Archive or merge the still-useful config-loading note into `COMPOSITE_MODEL_README.md`. |
| `QUICK_START_COMPOSITE.md` | Active but partly overlapping | Keep short-term; merge command examples into `docs/QUICK_REFERENCE.md` later. |
| `MODEL_SPECIFIC_DETECTION_PARAMS.md` | Active | Keep or merge into composite docs if duplication becomes burdensome. |
| `IMPLEMENTATION_SUMMARY.md` | Superseded implementation-history document | Archive after extracting any still-useful rationale into the composite docs. |
| `UPLOAD_TEST_RUNS.md` | Active ELab/full-CLI workflow doc | Keep, but consider merging with ELab CLI docs. |
| `ELAB_CLI_SIMPLE_USAGE.md` | Active simple-CLI workflow doc | Keep. |
| `LINKED_RESOURCES.md` | Active full-CLI linked-resource workflow doc | Keep. |
| `VISUALIZATION_UPDATE.md` | Superseded by current composite docs | Archive or merge the label-color behavior into `COMPOSITE_MODEL_README.md`. |
| `DEEPLAY_DISTRIBUTED_TRAINING_FIX.md` | Superseded/diagnostic note | Archive unless this warning still recurs in current training. |

Feature-doc stale-path verification:

```bash
rg -n "src/[A-Za-z0-9_]+\.py|tools/tdms_to_png|tools/TDMSExplorer" COMPOSITE_MODEL_README.md COMPOSITE_MODEL_UPDATE.md QUICK_START_COMPOSITE.md MODEL_SPECIFIC_DETECTION_PARAMS.md IMPLEMENTATION_SUMMARY.md UPLOAD_TEST_RUNS.md ELAB_CLI_SIMPLE_USAGE.md LINKED_RESOURCES.md VISUALIZATION_UPDATE.md DEEPLAY_DISTRIBUTED_TRAINING_FIX.md
```

Result: no stale flat-path or vendored-TDMS matches.

## 5. Decisions Required From User

`draft.ipynb`

- Size: `85,722,238` bytes, about 81.8 MiB.
- Last modified: `2026-04-13 16:34:45 +0200`.
- Decision needed: keep tracked/visible, move under a notebook archive, or ignore locally. It was not deleted.

`orientation_cnn_ckpt.pt`

- Size: `262,877` bytes.
- Last modified: `2026-04-08 18:05:38 +0200`.
- Recommendation: move it under a results/output directory such as `lodestar_orientation_test_out/` or another explicit model-artifact directory, then ignore generated checkpoints there. It was not deleted.

`AGENTS.md` / `CLAUDE.md`

- Actual git status observed: `AGENTS.md` is untracked; `CLAUDE.md` is tracked and now modified with path-only corrections.
- Tradeoff: sharing these files preserves important agent operating context and hard-won experimental results; keeping them local avoids agent-specific noise in the repo. User decision needed for whether `AGENTS.md` should be tracked and whether `CLAUDE.md` should remain a shared tracked source of truth.

`.specstory/`

- Status: untracked.
- Contents include local agent/history logs under `.specstory/history/` plus `.specstory/debug/debug.log`.
- Tradeoff: useful context for future agents, but likely local/tooling noise and may contain sensitive or irrelevant conversation history. User decision needed: track selected summaries, ignore the directory, or archive externally.

Feature-doc decisions:

- Decide keep/archive/merge for each feature doc listed in section 4D.

## 6. Items Not Touched

- `lodestar_link`, `DEEPTRACK_Link`, and `YOLOTrack` were left in place.
  - `lodestar_link` -> `/opt/mona_jupyterhub_env/lib/python3.10/site-packages/deeplay/applications/detection/lodestar/`
  - `DEEPTRACK_Link` -> `/opt/mona_jupyterhub_env/lib/python3.10/site-packages/deeptrack/`
  - `YOLOTrack` -> `../LodeSTAR/Yolo/YOLOTrack-1.1/`
- `analysis_outputs/`, `lstm_outputs/`, `supervised_correction_outputs/`, `lodestar_orientation_test_out/`, `elab_updates/`, and `notebooks/` were not deleted. They contain real experimental outputs referenced by `AGENTS.md` and ELab entries.
- Root `.ipynb_checkpoints/` and `tools/.ipynb_checkpoints/` were not deleted because verification found notebook checkpoints with unique names/content and no clear live counterpart, including `elabFTW-checkpoint.ipynb`, `Untitled-checkpoint.ipynb`, and `tools/.ipynb_checkpoints/Untitled-checkpoint.ipynb`. This is ambiguous, so it needs a user decision.
- `tools/tdms_to_png` was not deleted. It is an untracked ELF binary and appears stale now that TDMSExplorer is installed as `tdms_explorer`, but the requested deletion list named `tools/tdms_to_png.py`, `tools/tdms_to_png_README.md`, and `tools/build_tdms_to_png.sh`, not this binary. Prefer user confirmation before removing it.
- Existing dirty web work in `web/app.py`, `web/routers/files.py`, `web/routers/tdms_explorer.py`, and `web/templates/index.html` was not reverted, staged, or committed.
