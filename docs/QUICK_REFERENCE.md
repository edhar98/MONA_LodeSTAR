# Command reference

Run commands from the repository root. On MONA, use
`/opt/mona_jupyterhub_env/bin/python` or activate the corresponding environment.

## Detection and training

```bash
python src/detection/generate_samples.py
python src/detection/train_single_particle.py --particle Janus --config src/config.yaml
python src/detection/test_single_particle.py --particle Janus --model models/RUN_ID/Janus_weights.pth
python src/detection/detect_particles.py --model models/RUN_ID/Janus_weights.pth --input input.png --output results/
python src/detection/test_composite_model.py --config src/config.yaml
```

Replace `RUN_ID` and input/output paths with your own. Check `src/config.yaml`
before running: dataset selection, frame rate, model architecture, and paths are
experiment-specific. `src/samples.yaml` defines particle samples.
CLI training records models in `trained_models_summary.yaml`; the web interface
does not import that catalog.

For template orientation, add `--detection-mode template` and
`--orientation-template PATH_TO_CROP` to a supported detector command.
Template filenames can encode orientation as `phiNNN.N`.
See [composite models and detector parameters](../COMPOSITE_MODEL_README.md).

## Tracking and analysis

```bash
python src/tracking/track_particles.py \
  --input detections.csv --output tracks.csv \
  --min-dist 20 --max-link 30 --min-track 5 --max-gap 10

python src/analysis/analyze_tracks.py \
  --tracks tracks.csv --output analysis/ --px-size 0.078
```

Those numeric values are examples, not universal defaults for every experiment.
Use the correct pixel size and frame interval. Run a script with `--help` to
inspect its supported settings. Main ABP analysis excludes interpolated rows
unless `--include-interpolated` is requested.

Linear interpolation needs no checkpoint. Learned corrections are research
tools; read [known limitations](KNOWN_ISSUES.md) before training or interpreting
their benchmarks.

## Train and evaluate optional gap filling

Use tracked CSVs with finite `x`, `y`, and `phi`, integer `track_id` and `frame`,
and explicit `is_interpolated` flags. At least three tracks are needed, with
enough consecutive real observations in each split for the requested contexts.

```bash
python src/tracking/lstm_gap_filler.py train \
  --tracks tracks.csv --model-out lstm_outputs/gap_observed_v2.pt \
  --context-len 10 --gap-lengths 1,2,3,5,10 \
  --epochs 30 --val-fraction 0.2 --test-fraction 0.2 --device cpu

python src/tracking/lstm_gap_filler.py benchmark \
  --tracks tracks.csv --model lstm_outputs/gap_observed_v2.pt \
  --output lstm_outputs/gap_observed_v2_benchmark.csv \
  --summary-output lstm_outputs/gap_observed_v2_summary.csv --device cpu
```

Choose new output filenames for each experiment; existing files are refused.
Keep the input CSV unchanged between training and benchmarking. The benchmark
uses only the saved test tracks, compares all methods on the same masked gaps,
and records hashes and split details alongside the output. Do not tune the model
against this test set. A compatible checkpoint is not automatically a qualified
model; check accuracy and downstream physics before enabling it in the web app.

To create comparable full-track variants after qualification:

```bash
python src/tracking/build_track_variants.py \
  --tracks tracks.csv --model lstm_outputs/gap_observed_v2.pt \
  --output-dir analysis_outputs/track_variants --device cpu
```

This publishes a new uniquely named run directory after successful completion.
The manifest identifies its outputs; previous variants and source CSVs remain
unchanged. Legacy checkpoints require retraining, not a manual version-tag edit.

## Data tools

```bash
tdms-explorer export input.tdms output_dir
tdms-explorer animate input.tdms output.mp4 --fps 30
python src/detection/crop_detections.py data_dir/ -o data_dir/crops -s 64
python tools/crop.py input.png output_cropped.png
python tools/mask.py input.png output_masked.png
python tools/merge_mp4.py video_dir/ -o merged.mp4
```

`crop_detections.py` expects matching `images/` and `csv/` subdirectories.
See [tools](../tools/README.md) for details.

## Web and tests

```bash
python -m uvicorn web.app:app --host 127.0.0.1 --port 8002
python test/run_tests.py
```

Use the [web guide](WEB_GUIDE.md), [deployment guide](DEPLOYMENT.md), and
[test guide](../test/README.md) for remote access and additional checks.

## ELab

Set `ELAB_HOST_URL`, `ELAB_API_KEY`, and `ELAB_VERIFY_SSL` outside version
control. Never commit credentials.

```bash
python tools/elab_cli.py simple --help
python tools/elab_cli.py full --help
```

Uploading and patching change external records. Download the current body
before editing, apply changes to that live version, then verify the record and
attachments. See [ELab usage](../ELAB_CLI_SIMPLE_USAGE.md).

## Output locations

- CLI weights: `models/RUN_ID/`; Lightning checkpoints: `lightning_logs/`.
- Detection/analysis results: `detection_results/` and `analysis_outputs/`.
- Learned research artifacts: `lstm_outputs/` and `supervised_correction_outputs/`.
- Standalone web user data: `web/data/USERNAME/` by default.
- Hub user data: `~/mona_track/` by default.

Keep data backups separate from source-code backups.
