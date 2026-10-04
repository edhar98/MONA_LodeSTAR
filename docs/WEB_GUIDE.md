# Web guide

MONA Track provides image/TDMS loading, crop preparation, detector training,
detection, tracking, visualization, and export. Use the JupyterLab launcher for
the supported multi-user deployment. For development startup and remote access,
see [deployment](DEPLOYMENT.md).

## Load and inspect data

Upload a PNG, JPEG, TIFF, or TDMS file, or enter a server path, directory, or glob.
The shared Normalize TDMS checkbox applies to uploads and server-path loading;
its current value also controls preview and export. Changing it does not
retroactively rewrite stored detection settings for every loaded file: reload
a server input with the desired setting before using it for a new pipeline.

Server-linked files stay in their original locations. You may edit them from a
notebook, terminal, or file browser. Reloading the same resolved path refreshes
the existing entry and metadata rather than adding another entry. Old duplicate
registrations are not automatically removed because saved references may use them.

Select a file to inspect frames and TDMS metadata. If a file is deleted, moved,
or changes during reading, select the correct path and retry after editing
finishes. File ordering determines global frame order in multi-file operations.

## Prepare samples and train

Use the crop tools to create representative examples for each particle type,
then train a detector with appropriate augmentation and training settings.
Inspect training progress and results before detection. Training jobs do not
resume automatically after a server restart.

Only your web-trained detector models appear in the web model catalog. CLI
weights and `trained_models_summary.yaml` are not imported automatically.
Use the [CLI guide](QUICK_REFERENCE.md) for those workflows.

## Detect particles

Choose a model, source file or ordered set of files, and detection settings.
Preview a representative frame before starting a batch.

- Standard detection uses the model's local-maxima detection.
- Area and watershed modes operate on the model's weight map.
- Template mode adds orientation from a prepared template crop.
- Composite detection uses 2–8 models with distinct particle labels and supports
  standard mode only.

Composite confidence values are model weights, not calibrated class
probabilities. Tracking separates particle classes, but changing classifications
across frames can fragment trajectories.

## Track and analyze

Run tracking on a detection CSV. Set suppression distance, maximum link
distance, minimum track length, and maximum gap for the experiment's scale and
frame rate. Linear gap interpolation is the default and requires no checkpoint.
Inspect the trajectory overview and video before interpreting fitted parameters.

The main ABP analysis excludes interpolated rows by default. Use the correct
pixel size and frame interval. Multi-class CSV analysis pools classes; export
or filter a class before drawing class-specific conclusions. ABP assumptions
must be checked against interactions, boundaries, and confinement.

## Optional LSTM gap refinement

The tracker extra refines only eligible interpolated gaps and writes a separate
CSV with raw coordinates and provenance. Measured rows and the input CSV remain
unchanged. Choose the refined output explicitly if you want to analyze it.

The model has separate past/future context encoders. Current checkpoints require
finite orientation values, so position-only composite tracks are not compatible.
Insufficient clean context leaves the original linear-filled rows unchanged.

Compatible checkpoints must be provisioned in the deployment checkout's
`lstm_outputs/`. The web does not train or upload gap checkpoints. CLI detector
catalog removal does not remove this separate experimental checkpoint catalog.

Only checkpoints trained with `observed_context_v2` preprocessing can run.
Legacy files remain intact but are excluded from selection, with a retraining
message. Training instructions are in the [command reference](QUICK_REFERENCE.md).
Compatibility does not establish accuracy: the historical benchmark leaked
hidden target information. A newly trained model needs held-out evaluation and
downstream physics checks before use; see [known limitations](KNOWN_ISSUES.md).

## Export and layout

TDMS exports support PNG ZIPs and MP4. Every export uses a new output location,
and a ZIP contains only the frames requested in that operation. Detection,
tracking, analysis, and video outputs also use distinct names so repeated runs
do not replace earlier results.

Merge Selected creates a video from the chosen sources; unreadable inputs fail
the merge rather than silently publishing a partial movie.

Drag a card's bottom-right handle to resize it. Large cards extend the panel's
scrollable area. Double-click the handle to reset one card, or use Reset layout
to reset all saved sizes.

## Reporting a problem

Include the operation, selected input/model, error text, and relevant server
traceback. Do not include passwords, API keys, or private proxy headers.
For a connection problem, first distinguish an unreachable port from an HTTP
error returned by the app; the deployment guide covers both.
