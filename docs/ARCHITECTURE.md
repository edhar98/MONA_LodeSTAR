# Architecture

MONA LodeSTAR separates scientific computation from the web interface and
operational tools. The web interface reuses the same detector and tracking
implementations as the command-line workflows.

## Detection and tracking

`src/detection/` contains detector training, inference, synthetic data generation,
and template-based orientation. The custom model implements the paper-style
convolutional architecture; the default model uses Deeplay LodeSTAR. Both predict
displacement channels and a confidence map.

Composite detection runs multiple particle-specific models and merges nearby
detections. Confidence values are model weights, not calibrated probabilities.
The web wrapper in `web/services/composite_detection.py` retains particle labels
and model IDs. Web tracking processes classes separately to avoid linking
different particle classes.

`src/tracking/track_particles.py` applies within-frame suppression, Hungarian
linking, and linear gap interpolation. Tracks contain
`track_id, frame, x, y, phi, ncc, is_interpolated`.
Orientation is circular; missing frames must retain their actual frame indices.

Optional learned gap refinement uses separate past and future LSTM encoders.
It is not a single bidirectional PyTorch LSTM. Causal prediction and
reference-calibrated supervised correction remain research CLI workflows.

## Analysis

`src/analysis/` contains MSD and angular MSD calculations, ABP fitting,
interaction filtering, velocity persistence, and confinement diagnostics.
The default main ABP analysis excludes interpolated rows. Historical reports
predating frame-time corrections need explicit regeneration before scientific
reuse. See [known limitations](KNOWN_ISSUES.md).

## Web application

- `web/app.py`: application assembly, training, detection, tracking, and analysis routes.
- `web/routers/`: file handling, TDMS operations, and optional gap inference.
- `web/services/`: frame loading, cache freshness, composite inference, access checks,
  and output publication.
- `web/state.py`: user/session/job dictionaries and atomic JSON persistence.
- `web/config.py`: identity and storage paths.
- `web/auth.py`: standalone account handling and Hub identity responses.
- `web/templates/index.html`: browser UI, job polling, and WebSockets.
- `web/jupyter_config.py` and `web/jupyter_launch.py`: per-user Hub launcher.

Hub authentication belongs to JupyterHub. Its per-user proxy injects a private
header accepted by the backend; the browser does not receive this capability.
The backend rejects Hub-mode requests without it and rejects mismatched user
identities. OS permissions must still isolate users' files and processes.

Files linked from server paths remain editable outside the app. Frame metadata
and cache fingerprints are refreshed on reads. New results use separate names
and no-clobber publication; a code update must not replace user data.

## Imports and packaging

Core scripts expect the repository root as their working directory for default
paths. They add `src/` to the import path so `import utils` resolves consistently.
The web application also adds the relevant source subdirectories and the Janus
crescent tool package. TDMS support is an installed dependency, not vendored code.

`setup.py` packages the web modules and registers a Jupyter server-proxy launcher.
Runtime imports still need adjacent `src/` and `tools/`; a standalone wheel is
not currently supported.

## Development practices

Use `dev` for development and `main` for stable releases. Commit core changes
before integrating them into the web app. Keep scientific-method changes
separate from performance optimization, and preserve existing regression
assertions. A separate review should check security, numerical behavior, and
data preservation before release.

Research notebooks and experimental checkpoints are not automatically promoted
into production. Benchmark claims require held-out validation and reproducible
input/configuration provenance.
