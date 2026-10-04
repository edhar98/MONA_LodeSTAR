# Known limitations

These findings describe remaining work, not failures covered by the current
passing tests. Security and output-integrity corrections are included in release
`1fbc661`; the scientific and performance findings below remain open.

## Security and resource limits

Standalone usernames/passwords do not establish a secure multi-user API
session. Use the authenticated per-user JupyterHub launcher for deployment.
Hub proxy protection still depends on correct launcher configuration and OS
isolation; it does not defend against code running as the same OS user.

Chunk uploads have owner-bound capabilities, sequential writes, completion
revocation, and limits of 8 GiB per file and 4 MiB per chunk. There are at most
8 active chunk sessions per user and 32 per process, with one-hour idle expiry.
Multipart parsing can spool input before route-level size checks. CSV, sample,
and detection-specific upload paths still need consistent resource limits.
Restart invalidates unfinished upload capabilities; crash-orphan parts need
careful cleanup.

Most workloads do not share a bounded job scheduler. Heavy synchronous work can
block API responsiveness; training, detection, plots, and videos need common
admission limits, cancellation, and memory budgets. Gap inference alone has
a four-job admission limit and one worker, without a row/runtime budget.

Some CLI research loaders explicitly permit pickle-based artifacts
(`apply_supervised_correction_lstm.py` and `train_supervised_correction_lstm.py`).
Do not load untrusted checkpoints or datasets. Other loaders rely on the
installed Torch default; explicit restricted loading and pinned dependencies
remain desirable.

## Scientific validity

### Gap benchmark leakage

`src/tracking/lstm_gap_filler.py` computes motion features before masking the
target gap. The first future observation's displacement can depend on the last
hidden target. For a one-frame gap, that information can reconstruct the target.
Production uses interpolated context, creating a training/application mismatch.

The saved shared benchmark reports mean errors of 2.416 px for the two-sided
LSTM, 2.987 px for linear interpolation, and 3.280 px for Kalman. Those are
historical artifact values, not unbiased evidence of LSTM superiority. Do not
silently reinterpret or replace existing results.

Use a common context-only feature builder, version preprocessing, retrain, and
evaluate on held-out tracks/runs. Changing hidden target values must leave
model inputs unchanged. Linear interpolation remains the production default.

### Training and validation

Gap and causal training split overlapping windows/offsets and include validation
data in normalization. Some best-state snapshots use `detach().cpu()`, which can
alias live CPU parameters. Split independent tracks/time blocks before window
creation, normalize on training data only, and clone best-state tensors.

### Assignment and time handling

Tracking and supervised pairing apply the maximum-distance gate after Hungarian
assignment; this can reject links even when a different assignment could retain
more valid matches. Define the intended feasible-matching policy and test it
before changing trajectories.

The main tracking/MSD elapsed-frame handling was corrected, but historical
physics tables were not regenerated. Some other analysis paths still unwrap
angles across missing frames or generate invalid close-approach intervals.
The velocity-persistence tool can prefer a dataset-specific neighbor artifact
over the supplied dataset. Bind analysis inputs explicitly to their source.

### Input and evaluation consistency

Remaining findings include:

- Analysis loaders that convert nonempty boolean strings to true, truncate
  fractional IDs/frames, or accept duplicate observations.
- Non-normalized TDMS intensity conversion differs between preview, batch, and
  merge paths. A shared policy needs numerical regression tests.
- Greedy evaluation matching and unstable equal-score suppression can change
  metrics with input order.
- XML output needs escaping; some frame-to-image mappings use observed-frame
  rank and can misalign missing frames.
- Synthetic generation needs rectangular-image, empty-image, impossible-placement,
  and zero-normalization checks.
- Composite comparison has summary-schema inconsistencies.
- Some empty, all-filtered, or stationary datasets lack clear behavior.
- CLI single-particle evaluation can leave stale nonempty CSVs when a later run
  has no detections.

ABP is a null model, not a guaranteed physical law for these data. Check
interactions, orientation uncertainty, confinement, and parameter stability.
Interpolated rows can change fitted parameters more than the choice of smoother.

## Performance and maintainability

- Standard/template detection performs a forward pass for weights, then calls
  `detect()`, which performs another. Reuse the existing output only after
  coordinate, confidence, orientation, and call-count parity tests.
- Batch detection produces unused preview encodings.
- Several tools retain dense arrays or complete movies; stream results and
  bound memory instead.
- Composite merging is quadratic. Preserve its ordered, non-transitive grouping
  semantics if replacing it with spatial indexing.
- TDMS cache entry limits are not byte limits. Metadata fingerprints detect
  ordinary edits but are not cryptographic content checks.
- LSTM windows and repeated context encoding can be expensive.
- Some CLI wrappers have import/argument drift, and architecture choices differ
  between training and loading.
- WebSocket subscription/polling lifecycle and remaining direct selector
  rebuilds need cleanup.
- Shared helpers and thin route modules would reduce duplication; make these
  changes incrementally rather than rewriting the application wholesale.

## Data guarantees and limits

New web outputs use unique names and per-file no-clobber publication. A crescent
CSV and its overlay are not an all-or-nothing pair. TDMS exports publish a
completed directory. These safeguards do not replace backups or disk quotas.

External file changes are supported. Cached frame metadata is checked on reads;
edits that preserve the complete metadata fingerprint cannot be guaranteed
detectable. Existing registered duplicate IDs are retained to preserve saved
references.

## Required regression policy

Add a failing reproduction before correcting a finding. Preserve existing
assertions and numerical tolerances for performance changes. Scientific-method
changes require explicit versioning, held-out validation, and deliberate
regeneration of affected results. Passing UI/API smoke tests does not validate
physical conclusions.
