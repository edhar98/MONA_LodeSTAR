# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

All core scripts expect CWD = repository root when using default paths.

### Training
```bash
python src/detection/train_single_particle.py --particle Janus --config src/config.yaml
python src/detection/train_single_particle.py --config src/config.yaml          # train all
python src/detection/train_single_particle.py --particle Janus --checkpoint lightning_logs/<run_id>/checkpoints/epoch=10.ckpt
python src/detection/run_single_particle_pipeline.py                             # complete pipeline
```

### Testing & Evaluation
```bash
python src/detection/test_single_particle.py --particle Janus --model models/<run_id>/Janus_weights.pth
python src/detection/test_single_particle.py --particle Janus --model models/<run_id>/Janus_weights.pth --visualize
python src/detection/test_composite_model.py --config src/config.yaml
python src/detection/compare_models.py
```

### Data Generation
```bash
python src/detection/generate_samples.py    # generate sample images (required before training)
python src/detection/image_generator.py     # generate full datasets
```

### Detection
```bash
# Standard detection (local maxima)
python src/detection/detect_particles.py --model models/<run_id>/Janus_weights.pth --input input.png --output results/
# Area-based detection
python src/detection/detect_particles.py --model ... --input ... --detection-mode area
# Template-matching orientation detection (post-inference, no retraining needed)
python src/detection/detect_particles.py --model ... --input ... --detection-mode template \
  --orientation-template crops/f000_d000_phi0245.9.png \
  --template-angle-step 2 --template-refine-radius 25 --template-search-radius 5
python src/detection/test_single_particle.py --particle Janus --model models/<run_id>/Janus_weights.pth \
  --detection-mode template --orientation-template crops/f000_d000_phi0245.9.png
```

### Crop Detections (for template preparation)
```bash
python src/detection/crop_detections.py data_dir/ -o data_dir/crops -s 64
# data_dir must contain images/ and csv/ subdirectories
# output filenames encode phi: f{frame}_d{det}_phi{degrees}.png
```

### Web Interface
```bash
uvicorn web.app:app --reload      # starts at http://localhost:8000
```

### Tests
```bash
python test/run_tests.py
python test/run_tests.py --type unit        # unit | regression | integration
python test/run_tests.py --verbose
```

### Data Processing Tools
```bash
# TDMSExplorer is installed in /opt/mona_jupyterhub_env; use mona_env or the full env path
tdms-explorer export input.tdms output_dir
tdms-explorer animate input.tdms output.mp4 --fps 30
python tools/crop.py input.png output_cropped.png
python tools/crop.py input.png output_cropped.png --orientation  # append _phiNNN.N to crop filename
python tools/mask.py input.png output_masked.png
python tools/merge_mp4.py video_dir/ -o merged.mp4
```

### Particle Tracking
```bash
# Run full tracking pipeline on a detection CSV
python src/tracking/track_particles.py \
  --input detection_results/.../csv/<name>_detections.csv \
  --output detection_results/.../tracks/<name>_tracks.csv \
  --min-dist 20 --max-link 30 --min-track 5 --max-gap 10
```

### Physics Analysis (ABP Model)
```bash
# Fit MSD to ABP model, extract D_t, v0, D_r; output plots + CSV
python src/analysis/analyze_tracks.py \
  --tracks detection_results/.../tracks/<name>_tracks.csv \
  --output detection_results/.../analysis/ \
  --px-size 0.078
```

### ELAB Integration
```bash
# Required env vars: ELAB_HOST_URL, ELAB_API_KEY, ELAB_VERIFY_SSL
python elab.py upload-training
python elab.py upload-test
python tools/elab_cli.py simple upload-training
python tools/elab_cli.py simple upload-test
python tools/elab/cli/elab_cli_simple.py patch-item <id> --body "HTML body"
python tools/elab/cli/elab_cli_simple.py patch-item <id> --body-file update.html
```

Current progress-report ELab entries:

- Experiment `352`: `LodeSTAR: Reference-Calibrated LSTM Trajectory Correction`
  - URL: `https://139.18.53.128:3148/experiments.php?mode=view&id=352`
  - Contains supervised LSTM residual correction results, dataset 04 -> 01/02/03 transfer table, and raw-vs-refined MSD figures.
- Experiment `353`: `LodeSTAR: Physics Model Selection Beyond Plain ABP`
  - URL: `https://139.18.53.128:3148/experiments.php?mode=view&id=353`
  - Contains dataset 04 filtered ABP, velocity-persistence, and confinement diagnostics. Do not describe these outputs as dataset 02 results.
- Experiment `354`: `LodeSTAR: LSTM and BiLSTM Gap Filling`
  - URL: `https://139.18.53.128:3148/experiments.php?mode=view&id=354`
  - Contains single causal LSTM, two-sided LSTM/BiLSTM-style gap filler, masked-gap benchmark, production reach, and track 233 qualitative visualization.

For ELab edits, always download the current body first, patch from that live state, then verify title/body/uploads after submission. The local verified bodies are under `elab_updates/progress_2026-06-15/`.

### Maintenance
```bash
python cleanup_lightning_logs.py
```

## Architecture

### `src/` Structure

`src/` is split into three subpackages plus a shared root:

```
src/
├── utils.py            # shared hub — imported by all subpackages and web/app.py
├── config.yaml / samples.yaml / requirements.txt
├── detection/          # models, training, inference, orientation
├── tracking/           # track_particles, gap filling, LSTM, supervised correction
└── analysis/           # physics: MSD, ABP, confinement, interactions
```

Each subpackage has `__init__.py`. Scripts add `sys.path.insert(0, <src_dir>)` at the top so `import utils` resolves from `src/`. Same-package imports resolve automatically (Python adds the script's own directory to `sys.path` when run directly). `tools/` scripts are accessed with a two-level `../../tools` path from inside a subpackage.

### Core Model Pipeline (`src/detection/`)

The central algorithm is **LodeSTAR** — a self-supervised single-shot particle detector from the [Midtvedt et al. 2022 Nature Communications paper](https://doi.org/10.1038/s41467-022-35004-y). It outputs three channels per pixel: Δx, Δy (displacement vectors), and ρ (detection confidence).

Two implementations coexist:
- **`custom_lodestar.py`** — paper-accurate architecture: `3×Conv2D(3×3,32)+ReLU → MaxPool2D → 8×Conv2D(3×3,32)+ReLU → Conv2D(1×1,3)`
- **DeepTrack default** (`from deeplay import LodeSTAR`) — the upstream library's deeper variant; selected via `lodestar_version: default` in config

**`composite_model.py`** (`CompositeLodeSTAR`) runs all five particle-specific models in parallel, extracts confidence maps, clusters nearby detections (distance threshold = 20 px), and assigns class labels by highest per-model weight at each detection location.

Key dependency: **`src/utils.py`** is a shared utility hub imported by every other `src/` script as well as `web/app.py`.

### Particle Tracking Pipeline (`src/tracking/track_particles.py`)

Three-stage pipeline: within-frame NMS → Hungarian cross-frame linking → gap interpolation.

1. **NMS** (`min_det_distance=20px`): reuses `utils.nms_detections()`; keeps highest-NCC detection per cluster.
2. **Hungarian linking** (`max_link_distance=30px`, `max_gap_frames=10`): `scipy.optimize.linear_sum_assignment` for globally optimal assignment; unmatched tracks enter gap state, terminated after `max_gap_frames`.
3. **Gap interpolation**: x/y linear; φ circular via `np.unwrap`; filled rows flagged `is_interpolated=True`.

Output CSV columns: `track_id, frame, x, y, phi, ncc, is_interpolated`. Tracking config block lives under `tracking:` in `src/config.yaml`.

### LSTM Track Predictor (`src/lstm_track_predictor.py`)

This is the first learned baseline toward trajectory denoising and gap filling. It does not replace `track_particles.py` yet. It reads an existing tracked-particle CSV and trains a sequence model to predict the next particle state from the previous `seq_len` rows of the same track.

Current active baseline uses `--target-mode residual --feature-set motion`: the LSTM input is `[x, y, sin_phi, cos_phi, dx_prev, dy_prev, dt]`, and the target is `[dx, dy, sin_phi_next, cos_phi_next]`. Absolute predicted positions are reconstructed as `pred_x = last_x + dx`, `pred_y = last_y + dy`. The previous absolute-coordinate target is still available via `--target-mode absolute`, and the old 4-feature input is still available via `--feature-set basic`, but both are weaker baselines.

Current real-data test input:

```bash
detection_results/JP_FE/wf_2_40/JP_Fe_wf_2_40_5m4rtzfx/04/tracks/JP_Fe_wf_2_40_slm075_tracks.csv
```

This track file was produced from the detection CSV:

```bash
detection_results/JP_FE/wf_2_40/JP_Fe_wf_2_40_5m4rtzfx/04/csv/JP_Fe_wf_2_40_slm075_detections.csv
```

The LSTM code expects track CSV columns:

```text
track_id, frame, x, y, phi, ncc, is_interpolated
```

What the code does:

1. **Schema cleanup**: `load_tracks()` reads the CSV, normalizes missing optional columns, casts `track_id` and `frame` to integers, converts string booleans in `is_interpolated`, sorts by `track_id, frame`, and adds angle features.
2. **Angle encoding**: `_add_angle_features()` converts `phi` into `sin_phi` and `cos_phi`. The model does not directly regress raw angle because `phi=-pi` and `phi=+pi` are physically adjacent but numerically far apart.
3. **Feature vector**: basic mode represents each time step as `[x, y, sin_phi, cos_phi]`; motion mode adds `[dx_prev, dy_prev, dt]`.
4. **Window creation**: `make_windows()` groups rows by `track_id`. For each track it creates sliding windows: `seq_len` previous states become input `x`, and the following state becomes target `y`. By default interpolated rows are excluded during training so the first baseline learns from measured detections, not from linear gap-fill output.
5. **Target construction**: `make_targets()` supports `absolute` and `residual` modes. Residual mode predicts local displacement `[dx, dy]`, then reconstructs absolute position from the last input row.
6. **Normalization**: `fit_normalizer()` computes separate mean/std normalizers for inputs and targets. Training happens in normalized coordinate space, then predictions are transformed back to pixels and radians for reporting.
7. **Model architecture**: `LSTMTrackPredictor` is a small PyTorch model: multi-layer `nn.LSTM(batch_first=True)` followed by a two-layer MLP head. It uses only the final LSTM output (`out[:, -1]`) to predict four target values.
8. **Training**: `train()` splits windows into train/validation sets, optimizes with `AdamW`, uses `SmoothL1Loss`, tracks the best validation checkpoint, and saves model weights plus normalizer/config metadata to a `.pt` file.
9. **Metrics**: `compute_metrics()` reports x/y RMSE in pixels, mean/median Euclidean position error in pixels, and mean angular error in degrees using circular angle difference.
10. **Prediction CSV**: `predict()` reloads the checkpoint, rebuilds windows per track, writes actual next state and predicted next state as `x, y, phi, pred_x, pred_y, pred_phi`, and keeps `is_interpolated` for downstream filtering.

Residual baseline command used in this checkout:

```bash
/opt/mona_jupyterhub_env/bin/python src/lstm_track_predictor.py train \
  --tracks detection_results/JP_FE/wf_2_40/JP_Fe_wf_2_40_5m4rtzfx/04/tracks/JP_Fe_wf_2_40_slm075_tracks.csv \
  --model-out lstm_outputs/lstm_track_predictor_jp_fe_wf_2_40_slm075_motion.pt \
  --epochs 30 --batch-size 512 --seq-len 10 --min-track-length 30 \
  --target-mode residual --feature-set motion --device cpu \
  --wandb --wandb-project MONA_LodeSTAR_LSTM \
  --wandb-run-name JP_Fe_wf_2_40_slm075_motion_lstm
```

Use `--wandb-mode offline` when the JupyterHub node cannot reach WandB during training; sync later from `wandb_logs/`.

Prediction command:

```bash
/opt/mona_jupyterhub_env/bin/python src/lstm_track_predictor.py predict \
  --tracks detection_results/JP_FE/wf_2_40/JP_Fe_wf_2_40_5m4rtzfx/04/tracks/JP_Fe_wf_2_40_slm075_tracks.csv \
  --model lstm_outputs/lstm_track_predictor_jp_fe_wf_2_40_slm075_motion.pt \
  --output lstm_outputs/JP_Fe_wf_2_40_slm075_lstm_motion_predictions.csv \
  --device cpu
```

Residual run results:

| Item | Value |
|------|-------|
| Track rows | 120,597 |
| Tracks | 1,173 |
| Training windows | 94,373 |
| Epochs | 30 |
| Validation mean position error | 2.96 px |
| Validation median position error | 2.53 px |
| Full prediction mean position error | 2.70 px |
| Full prediction median position error | 2.30 px |
| Full prediction p95 position error | 5.86 px |
| Validation mean angular error | 23.4 deg |
| Prediction output rows | 109,132 |

Baseline comparison on the same prediction rows: persistence mean error ≈3.04 px, mean-velocity-over-10 mean error ≈3.05 px, motion-feature residual LSTM mean error ≈2.70 px.

Masked-gap benchmark lives in `src/benchmark_lstm_gap_filling.py`. It masks real consecutive detections and compares linear interpolation, persistence, constant velocity, and iterative LSTM rollout. Current JP Fe output files:

```text
lstm_outputs/JP_Fe_wf_2_40_slm075_gap_benchmark.csv
lstm_outputs/JP_Fe_wf_2_40_slm075_gap_benchmark_summary.csv
```

Masked-gap conclusion: causal LSTM rollout beats persistence and constant velocity, but linear interpolation wins for all tested gap lengths when the future endpoint is available. Example mean position errors: gap=1 linear ≈2.51 px vs LSTM ≈2.81 px; gap=10 linear ≈3.40 px vs LSTM ≈4.76 px. The next model should use both pre-gap and post-gap context rather than only causal one-step rollout.

Bidirectional gap filler is implemented in `src/lstm_gap_filler.py`. It trains on artificial masked gaps with both pre-gap and post-gap context, then predicts a correction to linear interpolation:

```bash
/opt/mona_jupyterhub_env/bin/python src/lstm_gap_filler.py train \
  --tracks detection_results/JP_FE/wf_2_40/JP_Fe_wf_2_40_5m4rtzfx/04/tracks/JP_Fe_wf_2_40_slm075_tracks.csv \
  --model-out lstm_outputs/lstm_gap_filler_jp_fe_wf_2_40_slm075.pt \
  --context-len 10 --gap-lengths 1,2,3,5,10 --max-samples 120000 \
  --epochs 30 --batch-size 512 --hidden-size 64 --layers 2 \
  --wandb --wandb-mode offline --wandb-run-name JP_Fe_wf_2_40_slm075_bilstm_gap_filler
```

Current benchmark files:

```text
lstm_outputs/JP_Fe_wf_2_40_slm075_bilstm_gap_benchmark.csv
lstm_outputs/JP_Fe_wf_2_40_slm075_bilstm_gap_benchmark_summary.csv
```

Bidirectional gap-filler conclusion: it beats linear interpolation across all tested gap lengths. Mean position error by gap length: gap=1 linear ≈2.42 px vs BiLSTM ≈1.30 px; gap=10 linear ≈3.30 px vs BiLSTM ≈2.89 px. WandB offline run: `wandb_logs/wandb/offline-run-20260531_010648-fa6o7fd6`.

Notebook split:

- `notebooks/JP_Fe_wf_2_40_slm075_LSTM_pipeline.ipynb` is the gradual development notebook from causal one-step LSTM to BiLSTM.
- `notebooks/JP_Fe_wf_2_40_slm075_BiLSTM_gap_filler.ipynb` is the focused latest-model notebook. It includes BiLSTM architecture, shared-sample benchmark, and qualitative application to actual `is_interpolated=True` gaps. The application section is fixed to `track_id = 233`, refines all four usable interpolated blocks (16 interpolated rows), and ends with a single whole-track real-frame overlap plot built from all 1011 available PNG frames for the track. The plot compares the full measured trajectory, original tracker interpolation, and BiLSTM-refined positions, but remains qualitative because true detections are missing for those frames.

Architecture-diagram correction for ELab/reporting:

- Do **not** reuse the old notebook-exported `single_lstm_architecture.png`: it incorrectly drew the two recurrent layers as parallel branches. The code uses one PyTorch `nn.LSTM(num_layers=2)`, i.e. stacked sequential recurrent layers, then `out[:, -1]` goes to `Linear(64,64) -> ReLU -> Linear(64,4)`.
- Do **not** reuse the old notebook-exported `bilstm_architecture_latest.png` without qualification: the gap filler is not a single `nn.LSTM(bidirectional=True)`. It is a two-sided context model with separate `past_lstm` and `future_lstm`; during prediction the future context is reversed before `future_lstm`.
- Corrected figures are in `elab_updates/progress_2026-06-15/attachments/gap_filling_notebook_visuals/`:
  - `single_lstm_architecture_v2_corrected.png`
  - `two_sided_lstm_gap_filler_architecture_v2_corrected.png`
- ELab experiment `354` was patched to use the corrected architecture figures and the old incorrect architecture uploads were deleted.

### Physics-First Track Quality Plan

The project goal is not to integrate LSTM/BiLSTM for its own sake. The goal is to turn LodeSTAR detections into reliable trajectories and extract defensible particle physics. Treat ABP as the first null model, not as a guaranteed final model. Plain ABP assumes isolated active particles with constant propulsion, no particle-particle interactions, no central potential/confinement, and no boundary effects; those assumptions must be tested on the real tracks.

Current decision framing:

1. Keep the current linear gap interpolation as the production baseline until a replacement proves value.
2. Use BiLSTM and Kalman smoothing as benchmark/probe models, not as automatic replacements.
3. A gap-filling method is useful only if it improves masked-gap error and does not introduce unwanted bias in downstream motion/physics statistics.
4. The decisive comparison is not just pixel error. Compare downstream raw statistics and fitted parameters for:
   - real detections only,
   - current linear-interpolated tracks,
   - BiLSTM-refined tracks,
   - Kalman-smoothed/refined tracks when available.
5. If the downstream physics is unchanged, linear interpolation is sufficient for this dataset and effort should move to orientation accuracy, collision handling, confidence filtering, or physical model selection.

Implementation steps for this phase:

1. **Raw motion statistics**: compute MSD, angular MSD, velocity autocorrelation, angular/orientation autocorrelation, displacement distributions, speed distributions, and turning-angle distributions for real-only, linear-interpolated, BiLSTM-refined, and Kalman variants.
2. **Interaction/collision analysis**: compute nearest-neighbor distances per frame, flag close-approach/collision windows, and compare motion statistics for isolated vs near-neighbor states. This tests whether isolated-particle ABP assumptions are valid.
3. **Central potential / confinement analysis**: estimate spatial drift fields, radial velocity, radial occupancy, speed-vs-radius, and possible inward/outward restoring drift. This tests whether ABP needs an external potential term.
4. **Model hierarchy decision**: compare Brownian diffusion, ABP, ABP + confinement, and interaction-filtered ABP. Use residuals and parameter stability rather than only fit curves.
5. **BiLSTM decision gate**: keep BiLSTM only if it improves masked-gap error and produces stable/improved raw motion statistics without hiding physical effects such as collisions or confinement.

The separate Kalman smoother baseline added to `src/lstm_gap_filler.py` produced shared-sample benchmark files:

```text
lstm_outputs/JP_Fe_wf_2_40_slm075_bilstm_gap_benchmark_with_kalman.csv
lstm_outputs/JP_Fe_wf_2_40_slm075_bilstm_gap_benchmark_with_kalman_summary.csv
```

Overall shared masked-frame mean position errors from that run:

| Method | Mean error [px] | Median error [px] |
|--------|------------------|-------------------|
| BiLSTM gap correction | 2.416 | 2.025 |
| Linear interpolation | 2.987 | 2.687 |
| Kalman smoother | 3.280 | 2.787 |

Interpretation: constant-velocity Kalman smoothing does not beat linear interpolation on this dataset; BiLSTM improves masked-gap pixel error, but the remaining question is whether the improvement matters for physical conclusions.

Implemented physics-first analysis tools:

- `src/analyze_motion_statistics.py`: raw motion statistics including translational MSD, angular MSD, displacement autocorrelation, orientation autocorrelation, displacement/speed/turning distributions, and metadata.
- `src/analyze_track_interactions.py`: nearest-neighbor and close-approach analysis, with isolated vs near-neighbor step/speed/turning comparisons.
- `src/analyze_confinement_drift.py`: radial occupancy, radial/tangential velocity, speed-vs-radius, spatial drift/quiver field, and radial drift residuals.
- `src/build_track_variants.py`: creates comparable track CSV variants under `analysis_outputs/track_variants/` for `linear`, `real_only`, `bilstm_refined`, and `kalman_refined`.
- `src/analyze_tracks.py` now supports `--include-interpolated` so ABP/MSD parameters can be computed on track variants that keep or refine filled rows. The default remains real-only.

Generated JP Fe wf 2 40 slm075 variant counts:

| Variant | Rows | Interpolated rows | Refined rows |
|---------|------|-------------------|--------------|
| real_only | 108,318 | 0 | 0 |
| linear | 120,597 | 12,279 | 0 |
| bilstm_refined | 120,597 | 12,279 | 1,241 |
| kalman_refined | 120,597 | 12,279 | 1,241 |

Only 1,241 of 12,279 interpolated rows have enough clean 10-frame pre/post context for the current BiLSTM/Kalman refinement. Therefore, even though BiLSTM improves masked-gap pixel error, the refined production-like track file is very close to the original linear-interpolated file.

ABP/MSD variant comparison using `src/analyze_tracks.py`:

| Variant | Included rows | D_t [um^2/s] | v0 [um/s] | D_r MSD [rad^2/s] | D_r angular [rad^2/s] |
|---------|---------------|--------------|-----------|-------------------|-----------------------|
| real_only | real only | 0.2551 | 3.4484 | 1.6915 | 3.3515 |
| linear | all rows | 0.1702 | 3.2393 | 1.4918 | 3.3606 |
| bilstm_refined | all rows | 0.1702 | 3.2401 | 1.4926 | 3.3631 |
| kalman_refined | all rows | 0.1730 | 3.2376 | 1.4914 | 3.3606 |

Interpretation: including interpolated rows shifts ABP translational parameters relative to real-only analysis; BiLSTM and Kalman refinement barely change the parameters relative to linear interpolation. For this dataset, the bigger issue is whether interpolated rows should enter physics analysis at all, not which smoother fills them.

Interaction analysis result for real detections:

- Close-neighbor instantaneous fractions: <=30 px: 1.60%, <=50 px: 41.81%, <=75 px: 79.02%.
- With +/-5 frame padding: <=30 px: 6.68%, <=50 px: 58.55%, <=75 px: 85.55%.
- At 50 px, median speed is 2.828 px/frame for isolated states vs 3.000 px/frame near neighbors; median absolute turn is 2.034 rad isolated vs 1.816 rad near neighbors.

Interpretation: the isolated-particle ABP assumption is questionable for the full dataset unless the analysis filters or stratifies by neighbor distance.

Confinement/drift analysis result for real detections:

- Positions: 108,318; steps: 107,145; tracks: 1,173.
- Estimated center: x=509.5 px, y=513.64 px.
- Mean speed: 96.43 px/s.
- Mean radial velocity: 0.95 px/s.
- Radial drift slope: 0.00806 1/s.

Interpretation: confinement/central drift is measurable enough to inspect in plots and include as a possible model extension, but interaction filtering should be considered before over-interpreting a global drift field.

Next physics-model direction:

Do not treat plain ABP as the final model. Use it as a null baseline and compare against models/diagnostics that relax its weakest assumptions:

1. **Filtered ABP**: fit ABP only on real detections after excluding states with close neighbors. Default thresholds to compare: no filter, nearest neighbor >30 px, >50 px, >75 px. If parameters stabilize after filtering, interactions are biasing the full-dataset ABP fit.
2. **Velocity-persistence / AOUP-style analysis**: estimate persistence directly from velocity autocorrelation instead of relying on NCC-derived `phi`. This is important because `D_r` from ABP MSD and angular MSD disagree by about a factor of two.
3. **Confinement-aware extension**: only fit/add an explicit central potential if drift-field/radial plots show a systematic, interpretable radial drift after interaction filtering.
4. **Interaction-aware reporting**: if nearest-neighbor filtering strongly changes motion statistics, report isolated-particle physics separately from close-approach/collision-window statistics rather than forcing one global ABP model.

Current approved implementation tasks:

- Add a filtered ABP comparison script under `src/`, writing outputs to `analysis_outputs/model_comparison/filtered_abp/`.
- Add a velocity-persistence/AOUP-style script under `src/`, writing outputs to `analysis_outputs/model_comparison/velocity_persistence/`.
- Integrate their outputs into a model-selection summary before changing presentation/report notebooks.

Implemented model-comparison scripts:

- `src/compare_filtered_abp.py`: filters real detections by nearest-neighbor distance and fits ABP/MSD per subset. It writes summaries and plots under `analysis_outputs/model_comparison/filtered_abp/`.
- `src/analyze_velocity_persistence.py`: fits velocity autocorrelation / persistent-random-walk style decay, with optional nearest-neighbor filtering. The 30 fps run is under `analysis_outputs/model_comparison/velocity_persistence_30fps/`.

Filtered ABP results on JP Fe wf 2 40 slm075:

| Filter | Rows | Tracks | Retained | D_t [um^2/s] | v0 [um/s] | D_r MSD [rad^2/s] | D_r AMSD [rad^2/s] |
|--------|------|--------|----------|--------------|-----------|-------------------|--------------------|
| no filter | 108,318 | 1,173 | 1.000 | 0.2339 | 3.1396 | 1.4335 | 3.2274 |
| nn >30 px | 106,583 | 1,173 | 0.984 | 0.2376 | 3.0538 | 1.3780 | 3.2617 |
| nn >50 px | 63,026 | 1,032 | 0.582 | ~0 | 2.5274 | 1.1664 | 3.8915 |
| nn >75 px | 22,720 | 592 | 0.210 | ~0 | 2.4650 | 2.1856 | 5.1049 |

The near-zero fitted `D_t` for stricter filters is a fit allocation issue: lag-1 MSD remains near 0.10 um^2, but the bounded ABP fit assigns short-lag motion mostly to the active term. Treat these fits as evidence that global ABP parameters are unstable under interaction filtering, not as final physical constants.

Velocity-persistence/AOUP-style results with frame rate 30 Hz:

| Filter | Velocity steps | tau_p [s] | Median speed [um/s] | VACF diffusion proxy |
|--------|----------------|-----------|---------------------|----------------------|
| no filter | 102,805 | 0.0258 | 6.62 | 0.627 |
| nn >30 px | 100,344 | 0.0249 | 6.62 | 0.617 |
| nn >50 px | 55,403 | 0.0172 | 6.62 | 0.547 |
| nn >75 px | 19,524 | 0.0141 | 5.23 | 0.530 |

Interpretation: velocity persistence is shorter than one frame at 30 fps and decreases under stricter neighbor filtering. This makes a clean long-persistence ABP interpretation questionable for this dataset; a velocity-based persistent random walk/AOUP diagnostic is useful, but it currently indicates very fast decorrelation rather than robust propulsion persistence.

### Supervised LodeSTAR Trajectory Correction

New active direction: use reference/YOLO-style CSV detections to train a supervised trajectory corrector from LodeSTAR tracks to reference positions. This is more defensible than self-supervised LSTM smoothing because the model learns residuals against an external reference source.

Important framing: the supervised model should be described as a **reference-calibrated trajectory corrector**, not as ground truth recovery. It learns to map LodeSTAR-derived trajectories toward the available reference/YOLO detections. Any bias or incompleteness in the reference data will be learned by the model.

Implemented scripts:

- `src/build_supervised_correction_dataset.py`: pairs LodeSTAR tracks with reference CSV detections frame-by-frame using Hungarian matching and writes:
  - matched pair CSV with residual targets `target_dx = ref_x - lode_x`, `target_dy = ref_y - lode_y`,
  - sequence-window NPZ for LSTM training.
- `src/train_supervised_correction_lstm.py`: trains a track-held-out LSTM residual corrector. It splits by `track_id`, not random windows.
- `src/apply_supervised_correction_lstm.py`: applies the trained corrector to tracks and exports raw plus refined coordinates.
- `utils.merge_detection_csvs()` and `tools/merge_detection_csvs.py`: merge per-stack LodeSTAR detection CSVs into one run-level global-frame detection CSV. The merge infers stack IDs from `_<stack>_detections.csv`, sorts stacks numerically, offsets local frames by `stack_index * frames_per_stack`, and adds `stack` plus `frame_local` columns.
- `src/test_single_particle.py --merge-detections --frames-per-stack 100`: future full-run detection can create the merged run-level detection CSV automatically after writing per-stack CSVs.

Current run status as of 2026-06-07:

- Run `01` is complete through detection, merge, tracking, raw visualization/analysis, supervised pairing, application of the `04`-trained corrector, refined visualization/analysis, and reference-error transfer validation.
- Run `01` merged detection CSV:
  `detection_results/JP_FE/wf_2_40/JP_Fe_wf_2_40_5m4rtzfx/01/csv/JP_Fe_wf_2_40_slm075_detections.csv`
- Run `01` merged tracks:
  `detection_results/JP_FE/wf_2_40/JP_Fe_wf_2_40_5m4rtzfx/01/tracks/JP_Fe_wf_2_40_slm075_tracks.csv`
- Run `01` refined-from-`04` tracks:
  `supervised_correction_outputs/JP_Fe_wf_2_40_01/JP_Fe_wf_2_40_slm075_tracks_supervised_refined_from04.csv`
- Run `02` and `03` full LodeSTAR detection are being launched by the user as a nohup job using `src/test_single_particle.py` with `dataset_types: ['02', '03']`, template mode, visualization enabled, and `--merge-detections`.
- Current `src/config.yaml` is intentionally set to `dataset_types: ['02', '03']`, `gt_from_csv: True`, `visualize: True` for this run. Reset it before re-running `01` or `04`.

Reference CSV format:

- Files such as `data/JP_FE/wf_2_40/04/csv/JP_Fe_wf_2_40_slm075_574_video.csv`.
- Columns: `x`, `y`, `phi`, `max_inensity`, `summed_inensity`, `frame`.
- In the current `04` dataset, frames after stack 574 are already global (`584` has frames `1000..1099`). The dataset builder supports `--reference-frame-mode auto|local|global`; use `auto` by default.

Current JP Fe wf 2 40 `04` supervised pairing command:

```bash
/opt/mona_jupyterhub_env/bin/python src/build_supervised_correction_dataset.py \
  --lodestar-tracks detection_results/JP_FE/wf_2_40/JP_Fe_wf_2_40_5m4rtzfx/04/tracks/JP_Fe_wf_2_40_slm075_tracks.csv \
  --reference-glob 'data/JP_FE/wf_2_40/04/csv/*_video.csv' \
  --output-dir supervised_correction_outputs/JP_Fe_wf_2_40_04 \
  --first-stack 574 --frames-per-stack 100 --reference-frame-mode auto \
  --max-match-distance 20 --seq-len 10
```

Pairing result:

| Item | Value |
|------|-------|
| LodeSTAR real rows used | 108,318 |
| Reference rows | 63,732 |
| Matched pairs | 31,644 |
| Matched frames | 1,100 |
| Matched LodeSTAR tracks | 353 |
| Sequence windows | 17,115 |
| Median LodeSTAR-to-reference residual | 5.47 px |
| Mean LodeSTAR-to-reference residual | 5.58 px |
| p95 residual | 8.88 px |

The sequence features are all deployable from LodeSTAR-side data: `lode_x`, `lode_y`, `sin_phi`, `cos_phi`, `lode_ncc`, `dx_prev`, `dy_prev`, `dt_prev`, `speed_prev`. The builder intentionally excludes `match_distance_px` from model features because it depends on reference data and would leak target information.

Training command:

```bash
/opt/mona_jupyterhub_env/bin/python src/train_supervised_correction_lstm.py \
  --dataset supervised_correction_outputs/JP_Fe_wf_2_40_04/JP_Fe_wf_2_40_slm075_tracks_correction_windows_seq10.npz \
  --model-out supervised_correction_outputs/JP_Fe_wf_2_40_04/supervised_lodestar_to_reference_lstm.pt \
  --epochs 40 --batch-size 512 --hidden-size 64 --layers 2 --device cpu
```

Held-out-by-track result:

| Split | Raw LodeSTAR mean error | Corrected mean error | Improvement |
|-------|--------------------------|----------------------|-------------|
| train | 5.41 px | 1.61 px | 3.80 px |
| validation | 5.75 px | 2.37 px | 3.39 px |
| test | 5.35 px | 1.91 px | 3.44 px |

This is the strongest evidence so far for an LSTM-based trajectory correction stage: against the reference detections, supervised residual correction substantially improves held-out track accuracy.

Application command:

```bash
/opt/mona_jupyterhub_env/bin/python src/apply_supervised_correction_lstm.py \
  --tracks detection_results/JP_FE/wf_2_40/JP_Fe_wf_2_40_5m4rtzfx/04/tracks/JP_Fe_wf_2_40_slm075_tracks.csv \
  --model supervised_correction_outputs/JP_Fe_wf_2_40_04/supervised_lodestar_to_reference_lstm.pt \
  --output supervised_correction_outputs/JP_Fe_wf_2_40_04/JP_Fe_wf_2_40_slm075_tracks_supervised_refined.csv \
  --device cpu --include-interpolated
```

The exported refined CSV contains raw columns (`x_raw`, `y_raw`, `phi_raw`), refined columns (`x_refined`, `y_refined`), residual columns (`supervised_dx`, `supervised_dy`, `supervised_shift_px`), and `is_model_refined`. In the first application pass, 110,206 of 120,597 rows were refined with mean correction magnitude 7.06 px. Because the model was trained on 353 matched tracks but applied to all 1,173 tracks, this export should be visually inspected before becoming the default trajectory export.

Visual inspection notebook:

- `notebooks/JP_Fe_wf_2_40_supervised_trajectory_correction_inspection.ipynb`

It loads the raw/refined/reference-pair CSVs, shows global correction distributions, lists worst corrected rows/tracks, lets the user set `TRACK_ID`, and plots:

- raw LodeSTAR vs supervised-refined path on a whole-track real-frame projection,
- reference/YOLO matched points when available,
- raw-to-refined correction vectors,
- x/y/shift time series,
- raw/refined error to reference on matched frames,
- a gallery of high-shift tracks.
- tracks with no YOLO/reference matches, including a dedicated unmatched-track overlay and high-correction unmatched-track gallery.

Default selected matched track is the reference-matched track with the most matched rows; current run picks track 57 with 673 rows, 513 matched reference rows, and 664 model-refined rows. The unmatched-track section reports 820 tracks with no reference matches; the default unmatched stress-test track is 810 with 80 rows and mean correction magnitude 11.40 px. Other high-correction unmatched examples include tracks 86, 768, 1212, and 960. Notebook syntax was validated.

### Supervised Correction Transfer Plan

Current proven scope:

- The supervised LSTM residual corrector improves LodeSTAR-to-reference error on held-out tracks within the same `JP_FE/wf_2_40/04` setup.
- It has not yet proven generalization to another setup, run, optical condition, particle size, LodeSTAR model, or reference-labeling convention.

Working assumption:

- The model can transfer to similar JP datasets if the learned correction is mainly a LodeSTAR localization/calibration bias.
- Transfer is less reliable if the learned correction depends strongly on motion dynamics, physical parameters, edge/periphery behavior, illumination, focus, particle size in pixels, or reference annotation conventions.

Reuse rule:

Do not describe a reused model as producing ground truth. Describe it as a **reference-calibrated LodeSTAR trajectory corrector**. A transferred model is accepted only after a small validation gate on the new dataset.

Validation gate for a new dataset:

1. Apply the trained model without overwriting raw values. Export:
   - `x_raw`, `y_raw`, `phi_raw`,
   - `x_refined`, `y_refined`,
   - `supervised_dx`, `supervised_dy`, `supervised_shift_px`,
   - `is_model_refined`,
   - quality/confidence flags.
2. Inspect correction statistics:
   - mean/median/p95 shift magnitude,
   - correction vector direction distribution,
   - shift vs frame position (`x`, `y`) to catch edge/periphery bias,
   - shift vs LodeSTAR confidence/NCC,
   - shift vs track length and interpolation state.
3. Visually inspect:
   - high-shift tracks,
   - long tracks,
   - peripheral tracks,
   - tracks with no reference matches,
   - tracks with interpolated rows.
4. If any reference/YOLO subset exists, match 50-200 frames and compare:
   - raw LodeSTAR -> reference error,
   - refined -> reference error,
   - failure cases where refined is worse.
5. Accept transfer only if refined error improves on the reference subset and the correction vectors look smooth/plausible on unmatched tracks.

Cross-run validation experiment:

1. Build supervised correction datasets for multiple runs/setups with the same pairing script:
   - `src/build_supervised_correction_dataset.py`
2. Train on some runs and test on a held-out run:
   - train: runs `01`, `02`, `03` if available,
   - validation: one held-out run,
   - test: a completely unseen run/setup.
3. Compare:
   - within-run held-out-track performance,
   - cross-run performance,
   - correction magnitude distributions,
   - edge/periphery behavior,
   - raw/refined/reference visual overlays.
4. If cross-run performance remains strong, promote the model from run-specific corrector to reusable JP corrector.

Initial stricter transfer check within `JP_FE/wf_2_40/04`:

`src/train_supervised_correction_lstm.py` now supports `--split-mode frame`, with inclusive validation/test frame ranges. This allows stack-held-out evaluation instead of random track-held-out evaluation.

Command used:

```bash
/opt/mona_jupyterhub_env/bin/python src/train_supervised_correction_lstm.py \
  --dataset supervised_correction_outputs/JP_Fe_wf_2_40_04/JP_Fe_wf_2_40_slm075_tracks_correction_windows_seq10.npz \
  --model-out supervised_correction_outputs/JP_Fe_wf_2_40_04/supervised_lodestar_to_reference_lstm_stack_split.pt \
  --epochs 40 --batch-size 512 --hidden-size 64 --layers 2 --device cpu \
  --split-mode frame --val-frame-ranges 900:999 --test-frame-ranges 1000:1099
```

This trains on frames 0-899, validates on stack/frame range 900-999, and tests on stack/frame range 1000-1099.

Stack-held-out result:

| Split | Windows | Raw LodeSTAR mean error | Corrected mean error | Improvement |
|-------|---------|--------------------------|----------------------|-------------|
| train frames 0-899 | 14,327 | 5.42 px | 1.61 px | 3.81 px |
| val frames 900-999 | 1,540 | 5.35 px | 2.26 px | 3.08 px |
| test frames 1000-1099 | 1,248 | 6.06 px | 2.67 px | 3.39 px |

Interpretation: the supervised residual corrector still improves substantially on a held-out future stack, so it is not only memorizing individual tracks. This is stronger than the original track-held-out result, but it is still within the same setup/run family. The next required transfer test is train on some runs (`01-03` if LodeSTAR tracks are generated) and test on a completely held-out run/setup.

Run `01` transfer result using the `04`-trained corrector:

- Pairing command used `--first-stack 420 --frames-per-stack 100 --reference-frame-mode auto --max-match-distance 20 --seq-len 10`.
- Pairing output directory: `supervised_correction_outputs/JP_Fe_wf_2_40_01/`
- Pairing summary: 245,152 LodeSTAR real rows, 1,912 LodeSTAR tracks, 62,497 reference rows, 48,740 matched pairs, 3,481 matched frames, 385 matched LodeSTAR tracks.
- Raw LodeSTAR-to-reference residuals on matched points: mean 5.49 px, median 5.39 px, p95 8.66 px.
- Applying `supervised_correction_outputs/JP_Fe_wf_2_40_04/supervised_lodestar_to_reference_lstm.pt` to `01` wrote:
  `supervised_correction_outputs/JP_Fe_wf_2_40_01/JP_Fe_wf_2_40_slm075_tracks_supervised_refined_from04.csv`
- Application summary: 267,339 rows, 250,451 model-refined rows, mean correction shift 7.04 px.
- Transfer validation against matched reference points:
  - all matched points: raw mean 5.49 px -> refined mean 2.58 px; raw median 5.39 px -> refined median 1.91 px; p95 8.66 px -> 7.49 px; 87.4% improved.
  - model-refined matched points only: raw mean 5.48 px -> refined mean 2.44 px; raw median 5.37 px -> refined median 1.84 px; p95 8.61 px -> 6.98 px; 91.3% improved.
- Error summary CSV:
  `supervised_correction_outputs/JP_Fe_wf_2_40_01/refined_from04_reference_error_summary.csv`
- Raw run `01` visualization:
  `detection_results/JP_FE/wf_2_40/JP_Fe_wf_2_40_5m4rtzfx/01/visualization/mona_slm075/JP_Fe_wf_2_40_slm075_tracks_overview.png`
  and `..._tracks_video.mp4`.
- Refined run `01` visualization:
  `supervised_correction_outputs/JP_Fe_wf_2_40_01/visualization/refined_from04/JP_Fe_wf_2_40_slm075_tracks_supervised_refined_from04_overview.png`
  and `..._video.mp4`.
- User visually inspected run `01` refined output and judged it sane.

Run `02`/`03` detection command currently intended/running via user terminal:

```bash
cd /home/edgarharutyunyan/MONA_LodeSTAR
nohup /opt/mona_jupyterhub_env/bin/python src/test_single_particle.py \
  --particle JP_Fe_wf_2_40 \
  --model models/5m4rtzfx/JP_Fe_wf_2_40_weights.pth \
  --config src/config.yaml \
  --detection-mode template \
  --orientation-template data/Samples/JP_Fe_wf_2_40/Samples/f000_d003_phi0234.0.png \
  --merge-detections \
  --frames-per-stack 100 \
  > logs/run02_03_lodestar_detection.log 2>&1 &
disown
```

Monitor with:

```bash
tail -f logs/run02_03_lodestar_detection.log
pgrep -af "test_single_particle.py.*JP_Fe_wf_2_40"
```

After `02`/`03` detection finishes, expected next steps:

1. Verify each run has per-stack CSVs, merged `JP_Fe_wf_2_40_slm075_detections.csv`, detection PNGs, and weightmaps.
2. Run `src/track_particles.py` once per run on the merged detection CSV to create `tracks/JP_Fe_wf_2_40_slm075_tracks.csv`.
3. Build `/tmp` global-frame image views for each run and run `src/visualize_tracks.py` plus `src/analyze_tracks.py` for raw tracks.
4. Build supervised correction datasets with `--first-stack 463` for `02` and `--first-stack 515` for `03`.
5. Apply the `04`-trained corrector to `02` and `03`, then compute raw-vs-refined reference errors as done for `01`.

Recommended model/export design for transfer:

- Continue predicting residuals, not absolute positions:
  - `x_refined = x_lode + dx_pred`,
  - `y_refined = y_lode + dy_pred`.
- Keep raw and refined values in every export.
- Normalize or include setup-scale metadata when possible:
  - pixel size,
  - frame size,
  - particle radius in pixels,
  - LodeSTAR model/run id,
  - detection mode,
  - confidence/NCC.
- Avoid non-deployable features such as reference-match distance during inference.
- Add an out-of-distribution/quality flag later, based on correction magnitude, feature ranges, and validation-set residuals.

Decision language:

- Safe claim now: “The supervised LSTM corrector improves LodeSTAR positions relative to reference detections on held-out tracks within this setup.”
- Claim after cross-run validation: “The supervised LSTM corrector generalizes across similar JP runs/setups.”
- Unsafe claim: “The LSTM recovers ground-truth trajectories.”

### Physics Analysis (`src/analyze_tracks.py`)

Fits tracks to the ABP (Active Brownian Particle) model. Requires tracks with ≥50 real (non-interpolated) frames.

- **Translational MSD**: ensemble-averaged, lag τ=1…100 frames, fit via `scipy.optimize.curve_fit` to `4D_t·τ + (2v₀²/D_r)[τ − (1−e^{−D_r·τ})/D_r]`
- **Angular MSD**: `np.unwrap(φ)`, linear fit on first 30% of lags, slope/2 = D_r
- Pixel→µm: D_t × px², v₀ × px, D_r unchanged (rad²/s); controlled via `--px-size` (default 0.078 µm/px)
- Outputs: `*_msd.csv`, `*_msd.png` (translational + angular MSD plots), printed parameter table

### Orientation Detection (post-inference, no retraining)

Janus particle orientation is determined after position detection via **template-matching NCC** — no changes to training are needed. The pipeline:

1. `utils.build_template_bank(sample_path, angle_step)` — loads a cropped particle PNG (phi encoded in filename as `phi{degrees}`), computes an annular mask from the radial intensity profile, and pre-rotates the template at `angle_step` increments across 0–360°.
2. `utils.orientation_postprocess(image, detections, template_bank)` — for each detection: refines position via HoughCircles (falls back to intensity-weighted center-of-mass), then finds the best-matching rotation angle by NCC over a `±search_r` pixel window. Returns `(N, 4)` array: `[x_refined, y_refined, phi_rad, ncc_score]`.
3. Results are written to CSV with `phi` and `orientation_ncc` columns; visualization draws arrows via `utils.draw_orientation_arrow()` and `cv2.arrowedLine()`.

Activated in both `detect_particles.py` and `test_single_particle.py` with `--detection-mode template --orientation-template <path>`. Config defaults live under the `orientation:` key in `src/config.yaml`.

Future detection work should add a selectable detection engine flag, e.g. `--detection-engine lodestar|trackpy`. `lodestar` remains the learned model path; `trackpy` is worth supporting as a classical microscopy blob-localization baseline for position detection, feeding the same downstream orientation, tracking, gap interpolation, and ABP analysis pipeline.

Current detection-only benchmark on one 1024x1024 JP frame (`JP_Fe_wf_2_40_slm075_574_001.png`, 30 timed reps after warmup):

| Engine | Hardware | Detections | Mean time |
|--------|----------|------------|-----------|
| LodeSTAR `model.detect` | CUDA | 120 | 180.1 ms |
| `trackpy.locate(diameter=41)` | CPU | 117 | 265.1 ms |
| LodeSTAR `model.detect` | CPU | 120 | 758.2 ms |
| `trackpy.locate(diameter=41)` | CPU | 117 | 265.7 ms |

Conclusion: LodeSTAR is faster when CUDA is available (~1.47x faster than trackpy on this frame), but CPU-only LodeSTAR is slower (~2.85x slower). Keep both engines benchmarkable and select based on deployment hardware and particle morphology.

### Training & Config

`train_single_particle.py` uses PyTorch Lightning. Training config lives in `src/config.yaml`; particle shape parameters live in `src/samples.yaml`. The `lodestar_version` key in config switches between custom and default implementations. WandB logging is abstracted in `tools/wandb_logging.py` and is optional (gracefully disabled when unavailable).

Trained weights are saved to `models/<run_id>/<ParticleType>_weights.pth`; Lightning checkpoints go to `lightning_logs/<run_id>/checkpoints/`. The file `trained_models_summary.yaml` (repo root) tracks all runs and is consumed by composite model and test scripts.

### Web Interface (`web/`)

FastAPI backend (`web/app.py`, ~1235 lines) with a single-page HTML frontend (`web/templates/index.html`, ~1688 lines). Supports user accounts (session-based), background training jobs persisted to `web/training_jobs.json`, and per-user data isolation under `web/data/<username>/`. The backend inserts repo `src/` into `sys.path` for `utils` and imports TDMS support from the installed `tdms_explorer` package.

TDMS support is provided by the installed `tdms_explorer` package, not by a repo-local `tools/TDMSExplorer/` copy. In the MONA JupyterHub deployment, Python lives under `/opt/mona_jupyterhub_env/` and the user shell has a `mona_env` alias for activation. Verify with:

```bash
/opt/mona_jupyterhub_env/bin/python -c "from tdms_explorer import TDMSFileExplorer; import tdms_explorer; print(tdms_explorer.__version__)"
tdms-explorer --help
```

The Jupyter launcher entry comes from the installed `tdms_explorer` package via `tdms_explorer.jupyter_config:setup_tdms_explorer`. If duplicate TDMS Explorer launcher tiles appear, check for stale `*dms_explorer*.dist-info` metadata in `/opt/mona_jupyterhub_env/lib/python3.10/site-packages`. The old stale `~dms_explorer-1.0.0.dist-info` metadata caused a duplicate launcher next to active `tdms_explorer-1.1.0.dist-info`.

Known TDMSExplorer launcher behavior: the Panel page may initially show only the blue header/blank body while Bokeh/Panel initializes. A delayed reload/render shows the widgets. If it stays blank, debug the external installed TDMSExplorer package rather than MONA_LodeSTAR.

### Import Conventions

| Context | Rule |
|---------|------|
| `web/app.py` | `sys.path.insert(0, .../src)`; imports installed `tdms_explorer.TDMSFileExplorer` and `src/utils.py` |
| `src/detection/*.py` | `sys.path.insert(0, <src_dir>)` then `import utils`; same-package imports work directly; tools via `../../tools` |
| `src/tracking/*.py` | Same pattern; same-package imports (e.g. `train_supervised_correction_lstm`) work directly |
| `src/analysis/*.py` | Same pattern; no cross-package imports |
| `tools/*.py` | No imports from `src/` or `web/` — fully independent |
| `test/*.py` | `sys.path.insert(0, .../../src)` + `sys.path.insert(0, .../../src/detection)` for detection symbols |
| `debug/` notebooks | Can import from anywhere for experimentation |

### Current Repository Status Notes

- TDMSExplorer is no longer vendored under `tools/TDMSExplorer/`; use the installed package.
- Legacy `tools/tdms_to_png.py`, `tools/tdms_to_png_README.md`, and `tools/build_tdms_to_png.sh` are deleted in the current working tree.
- The repo is on `dev` and has a dirty working tree with documentation, web, tool, and TDMS cleanup changes. Do not revert unrelated user changes.
- `CLAUDE.md` may be untracked in this checkout but is intended as local AI-agent operating context.
- Root-level orientation experiments (`orientation_cnn.py`, `test_lodestar_orientation.py`, `src/lodestar_orientation.py`, output/checkpoint artifacts) are present but ownership is still research/experimental unless explicitly integrated.

### Git Conventions

- Two long-lived branches: `dev` (active development) and `main` (stable)
- Commit messages start with a dash (`-`), short, one logical change per commit
- Web branch only integrates committed Core code (never uncommitted imports)
