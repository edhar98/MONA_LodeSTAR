# Tests

Run from the repository root using the MONA environment.

## Automated suite

```bash
/opt/mona_jupyterhub_env/bin/python -B test/run_tests.py
python test/run_tests.py --type unit
python test/run_tests.py --type regression
python test/run_tests.py --verbose
```

The runner discovers Python tests under `unit/`, `regression/`, and
`integration/` by their test filenames. The opt-in scripts below must be run
separately.

## Frontend checks

```bash
node test/integration/check_web_ui.cjs
node test/integration/check_select_refresh.cjs
node test/integration/check_tdms_layout.cjs
```

These check JavaScript syntax and UI contracts using simulated DOM elements,
including proxy paths, job recovery, selectors, normalization, and resizing.
They do not replace browser layout, pointer, or native-popup testing.

## Real model smoke tests

```bash
/opt/mona_jupyterhub_env/bin/python -B test/integration/smoke_hub_pipeline.py
/opt/mona_jupyterhub_env/bin/python -B test/integration/smoke_learned_web.py
```

Both scripts run on CPU with isolated temporary user storage. They copy the
required detector weights/configuration into temporary web-model fixtures;
they do not use web CLI discovery or change real user sessions. They perform
no training, deployment, or external ELab updates. Each has a 180-second POSIX
timeout; a forced timeout can leave its temporary directory behind.

The single-model smoke requires:

- `models/5m4rtzfx/JP_Fe_wf_2_40_weights.pth` and its saved `config.yaml`.
- Dataset-04 images `JP_Fe_wf_2_40_slm075_574_001.png` through `_003.png` under
  `data/JP_FE/wf_2_40/04/images/`.
- Orientation template
  `data/Samples/JP_Fe_wf_2_40/Samples/f000_d003_phi0234.0.png`.

It covers Hub-mode identity, upload/crop/server input, detection, orientation,
tracking, ABP plotting, export response bytes, and persisted state reload.

The composite/gap smoke additionally requires Janus run `euk2wnni`, Rod run
`uaqqndn3`, their saved configs, the JP dataset-04 tracks CSV, and the approved
historical gap checkpoint in `lstm_outputs/`. Exact fixture paths are defined in
the script. It verifies legacy rejection without changing that file, then uses
a temporary synthetic v2 checkpoint to test gap-only inference, raw/input
preservation, and provenance. This is a compatibility check, not an accuracy test.

## Gap training smoke test

```bash
/opt/mona_jupyterhub_env/bin/python -B test/integration/smoke_gap_training.py
```

This trains for two epochs on a temporary subset of the dataset-04 tracks and
benchmarks the saved test split. It checks split separation, provenance, equal
baseline samples, and overwrite refusal. It removes its temporary artifacts and
does not produce a deployment checkpoint or validate scientific superiority.

The corrected gap batch passed 83 Python tests, all three frontend checks, the
composite/gap compatibility smoke, and this training smoke on 2026-10-05.

## Last validated release

On 2026-10-04, the code released as `1fbc661` passed 74 Python tests and all three
frontend checks. The real-model checks returned:

- Single model: 367 detections across three frames; 121 tracks and 356 rows.
- Template orientation: 120 finite detections.
- Composite: 95 detections with Janus and Rod labels.
- Gap refinement: four refined rows among 16 interpolated rows in a 1,011-row
  fixture; measured rows and original input bytes unchanged.

The user also reported all nine manual web acceptance checks passed. This is
not a benchmark of scientific accuracy or proof of live Hub isolation.
Deployment still requires actual launcher HTTP/WebSocket and user-boundary
checks described in [deployment](../docs/DEPLOYMENT.md).

## Adding tests

Keep existing regression assertions. Test security failures before filesystem
mutation, preserve original inputs, and use temporary directories. Separate
intentional scientific-method changes from numerical-preserving optimizations.
See [known limitations](../docs/KNOWN_ISSUES.md) for the remaining regression work.
