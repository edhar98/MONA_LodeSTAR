#!/usr/bin/env bash
# Print-only examples. Running this script never invokes Python or contacts ELab.
set -euo pipefail
printf '%s\n' \
  'Run the selected command from the repository root after checking its inputs.' \
  'Replace UPPERCASE placeholders with IDs from your own ELab installation.' \
  '' \
  '# Inspect CLI options (no upload)' \
  'python tools/elab_cli.py full upload-test-run --help' \
  '' \
  '# Read one experiment without modifying it' \
  'python tools/elab/diagnostics/inspect_experiment.py --experiment-id EXPERIMENT_ID' \
  '' \
  '# Create a new test-result experiment; archives existing result directories' \
  'python tools/elab_cli.py full upload-test-run --label RUN_LABEL --no-update' \
  '' \
  '# Link resources to an existing experiment (remote write)' \
  'python tools/elab_cli.py simple link-resources --experiment-id TARGET_ID --experiments SOURCE_ID --items ITEM_ID' \
  '' \
  '# Create a linked experiment (remote write)' \
  'python tools/elab_cli.py full create-with-links --title "Particle analysis" --body "Analysis summary" --template TEMPLATE_ID --experiments SOURCE_ID --items ITEM_ID' \
  '' \
  'Upload and linking commands modify ELab. Check reported failures and verify the resulting record.'
