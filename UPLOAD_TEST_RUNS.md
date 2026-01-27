# Uploading Test Run Results to elab

This document explains how to upload test run results from `test_single_particle.py` to elab using the new CLI command.

## Overview

After running `test_single_particle.py`, the script generates:
- `logs/` - Evaluation metrics and test logs
- `detection_results/` - Visualized detection images
- `test_results_summary.yaml` - Summary of all test results

The new `upload-test-run` command in `elab_cli.py` allows you to upload these results to elab as a separate experiment with proper metadata matching the elab template.

## Prerequisites

1. Set environment variables:
   ```bash
   export ELAB_HOST_URL="your_elab_host_url"
   export ELAB_API_KEY="your_api_key"
   export ELAB_VERIFY_SSL="true"  # or "false" if needed
   ```

2. Ensure you have the required Python packages:
   ```bash
   pip install elabapi-python urllib3
   ```

## Usage

### Basic Usage
```bash
python src/elab_cli.py upload-test-run
```
This will:
- Use current timestamp as the run label
- Upload `logs/` and `detection_results/` directories
- Include `test_results_summary.yaml` if it exists
- Create an experiment with title "Test run results [timestamp]"
- Apply default tags "LodeSTAR|Test Run|ML"

### Custom Label
```bash
python src/elab_cli.py upload-test-run --label "janus_particle_test_2024"
```

### Custom Title Prefix
```bash
python src/elab_cli.py upload-test-run --title-prefix "LodeSTAR Detection Test"
```

### With Metadata (Matching elab Template)
```bash
python src/elab_cli.py upload-test-run \
  --label "janus_particle_test_2024" \
  --title-prefix "LodeSTAR Detection Test" \
  --category 5 \
  --team 1
```

### Minimal Metadata (just category and team)
```bash
python src/elab_cli.py upload-test-run \
  --category 5 \
  --team 1
```

## Metadata Parameters

The command supports metadata fields that align with the elab experiment template:

> **Tip**: See `elab_config.yaml` for common category IDs and team IDs.

> **Important**: Template ID 24 (LodeSTAR: single particle pipeline) supports category and team fields but does NOT support tags. Tags will be ignored when using this template.

### `--category`
- **Type**: Integer
- **Description**: Experiment category ID
- **Example**: `5` for "Full Run" category
- **Default**: `5` (Full Run) - automatically applied

### `--team`
- **Type**: Integer
- **Description**: Team ID for the experiment
- **Example**: `1` for "Molecular Nanophotonics Group"
- **Default**: `1` (Molecular Nanophotonics Group) - automatically applied

## Template Compatibility

The current implementation uses **Template ID 24** which has the following characteristics:

### **Template Details:**
- **ID**: 24
- **Title**: "LodeSTAR: single particle pipeline"
- **Purpose**: Specifically designed for LodeSTAR single particle pipeline experiments

### **Supported Fields:**
- `title` - Experiment title
- `body` - Experiment description  
- `template` - Template ID (fixed to 24)
- `category` - Category ID (default: 5 for "Full Run")
- `team` - Team ID (default: 1 for "Molecular Nanophotonics Group")

### **Not Supported:**
- `tags` - **Not supported** (causes 500 error)

### **Template Benefits:**
- Designed specifically for LodeSTAR experiments
- Supports proper categorization and team assignment
- Automatically applies appropriate defaults for LodeSTAR projects
- Stable and reliable for production use

## What Gets Uploaded

1. **Archived Directories:**
   - `logs/` → `[label]_logs.tar.gz`
   - `detection_results/` → `[label]_detection_results.tar.gz`

2. **Additional Files:**
   - `test_results_summary.yaml` (if exists)

3. **Experiment Metadata:**
   - Title: `[title_prefix] [label]`
   - Body: Description of uploaded content
   - Category: 5 (Full Run) - automatically applied
   - Team: 1 (Molecular Nanophotonics Group) - automatically applied
   - Template: 24 (LodeSTAR: single particle pipeline)
   - Type: Test run results

## Output

The command returns JSON output with:
- `experiment_id`: The created elab experiment ID
- `uploaded`: List of all uploaded files
- `test_run_type`: "single_particle_detection"
- `directories_archived`: List of archived directories
- `additional_files`: List of additional files uploaded
- `experiment_metadata`: Created experiment metadata including title, category, and team

## Example Workflow

1. Run your tests:
   ```bash
   python src/test_single_particle.py
   ```

2. Upload results to elab with metadata:
   ```bash
   python src/elab_cli.py upload-test-run \
     --label "janus_test_run" \
     --category 5 \
     --team 1
   ```

3. Check the output for the experiment ID and verify in elab

## Error Handling

The command will exit with appropriate error codes:
- `0`: Success
- `4`: No directories found to archive
- `5`: Failed to create experiment
- `6`: Unexpected experiment response
- `7`: Failed to upload files

## Notes

- The command is completely separate from the test execution
- It only uploads existing results; it doesn't run tests
- Archives are created with `.tar.gz` compression
- The experiment title and description clearly indicate it's a test run
- All uploaded files maintain their original names
- Metadata fields align with the elab experiment template structure
- Template ID 24 automatically applies appropriate category and team defaults
