#!/bin/bash

# Example script showing how to upload test run results to elab
# This script should be run after running test_single_particle.py

echo "Uploading test run results to elab..."

# Basic usage - uses timestamp as label
python src/elab_cli.py upload-test-run

# Or with custom label
# python src/elab_cli.py upload-test-run --label "janus_particle_test_2024"

# Or with custom title prefix
# python src/elab_cli.py upload-test-run --title-prefix "LodeSTAR Detection Test"

# With metadata matching the elab template
# python src/elab_cli.py upload-test-run \
#   --label "janus_particle_test_2024" \
#   --title-prefix "LodeSTAR Detection Test" \
#   --category 5 \
#   --team 1

# Minimal metadata (just category and team)
# python src/elab_cli.py upload-test-run \
#   --category 5 \
#   --team 1

echo "Done! Check the output above for the experiment ID."
