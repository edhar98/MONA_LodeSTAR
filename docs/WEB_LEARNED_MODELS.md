# Composite detection and learned trajectory tools

This integration adds inference with existing trained models. It does not add
LSTM training, reference pairing, benchmark execution, or automatic model
promotion. Linear tracking remains the default. Source changes do not deploy
the separate JupyterHub checkout automatically.

## Composite detection

Select multiple particle-specific detector models in Detection. Composite
inference merges nearby detections and assigns the strongest model-confidence
class at each merged location. These confidence values are model weights, not
calibrated probabilities; comparing different model families or setups requires
validation on representative images.

The first web integration supports standard detection only. Per-class template
orientation and area/watershed composite detection are not included. Each model
must have a distinct particle label. Class-labelled CSVs are tracked separately
per class with globally unique track IDs, so detections from different classes
are not linked together. Classification flicker can still fragment trajectories.
The current ABP analysis pools all classes in the selected CSV and displays a
warning for multi-class inputs. Filter/export one class before interpreting
class-specific physics; no class-filter control is included in this integration.

## Optional learned trajectory tools

Use the Tracking panel with an existing tracks CSV and a compatible checkpoint:

| Method | Output | Meaning |
|---|---|---|
| Causal LSTM | Next-step prediction CSV | Diagnostic prediction, not replacement trajectories |
| Two-sided LSTM gap filler | Separate refined tracks CSV | Refines interpolated gaps with clean past/future context |
| Supervised LSTM corrector | Separate refined tracks CSV | Reference-calibrated position correction, not ground truth |

The gap filler has separate past and future recurrent encoders; it is not one
`nn.LSTM(bidirectional=True)`. Insufficient context leaves rows unchanged and
must be reflected in coverage counts. Refined files retain raw coordinates;
the input CSV is not overwritten. Choosing a refined file for visualization or
ABP analysis is an explicit decision, not a change to the default tracker.

Checkpoints are discovered from repository `lstm_outputs/` and
`supervised_correction_outputs/`. Arbitrary checkpoint uploads and user-supplied
checkpoint paths are not supported. Catalog availability depends on the actual
deployment checkout: Git code promotion does not transfer ignored model weights.

Model reuse across optical setups, particle types, frame rates, or localization
conventions requires validation. Gap error improvements do not establish better
physics estimates. Review correction magnitudes, coverage, raw/refined overlays,
and reference errors where available before using refined results scientifically.

These checkpoints require finite orientation values; supervised correction also
requires usable measured-row NCC. Use template-oriented tracks. Position-only
composite outputs currently cannot feed these angle-dependent checkpoints.
Composite detection and learned trajectory inference are separate additions,
not yet a class-specific orientation-to-LSTM pipeline.

Deployment preflight found neither checkpoint directory in the installed
`/home/mona/MONA_LodeSTAR` checkout. Approved weights must be provisioned there
separately before its learned-model catalog will be populated.

## Verification

Run from the repository root in the MONA environment:

```bash
/opt/mona_jupyterhub_env/bin/python -B test/run_tests.py
node test/integration/check_web_ui.cjs
/opt/mona_jupyterhub_env/bin/python -B test/integration/smoke_learned_web.py
```

The opt-in smoke uses existing detector/LSTM weights, one real image, and the
dataset-04 track-233 fixture. It writes only temporary isolated user state and
does not train, deploy, or update scientific outputs. Its 180-second hard timeout
can leave its temporary directory behind on failure. Tests of API schemas and
mocked UI behavior are not actual JupyterHub browser acceptance or scientific
accuracy validation.

The trajectory queue admits at most four pending/running jobs and runs one at a
time. There is no per-job runtime or row budget yet. Causal inference evaluates
individual windows; supervised inference builds sequence windows in memory.
Full-dataset latency and memory use have not been established by this small smoke.

## Verification result — 2026-09-23

The independent reviewer found no blocking integration defects and independently
passed all 13 new unit cases and the Node checks. The coordinator's final run
passed all 34 main tests, Node checks, syntax checks, and both real-model smokes.
The new smoke confirmed:

- Composite Janus `euk2wnni` plus Rod `uaqqndn3`: 95 detections on one real frame,
  both class labels retained; per-class tracking and pooled-ABP warning exercised.
- Real track 233: 1,011 input rows, 954 causal predictions, 4 eligible gap rows
  refined, and 957 measured rows reference-corrected. Counts describe this actual
  fixture/checkpoint execution, not the older notebook's historical counts.
- Source-file hash unchanged, raw coordinates preserved, finite model outputs,
  provenance artifacts, and cross-user catalog access rejected.
- Existing single-model smoke retained 367 detections across three frames,
  121 tracks/356 rows, and 120 template-oriented detections on one frame.

These changes reuse committed Core code without modifying it. No training,
research artifact replacement, installed-code changes, or service restarts were
performed. A representative real JupyterHub browser workflow and target model
provisioning remain deployment gates.
