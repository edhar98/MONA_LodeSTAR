# Simplified elab CLI Usage Guide

This simplified CLI focuses on the essential functionality for recording training and testing results in elab.

## Prerequisites

Set environment variables:
```bash
export ELAB_HOST_URL="your_elab_host_url"
export ELAB_API_KEY="your_api_key"
export ELAB_VERIFY_SSL="true"  # or "false" if needed
```

## Commands

### 1. Upload Training Results

Uploads training artifacts (logs, checkpoints, models) to elab:

```bash
python src/elab_cli_simple.py upload-training
```

**Options:**
- `--label`: Custom label (default: timestamp)
- `--title-prefix`: Title prefix (default: "Training Results")
- `--category`: Category ID (default: 5 for "Full Run")
- `--team`: Team ID (default: 1 for "Molecular Nanophotonics Group")
- `--experiments`: List of experiment IDs to link
- `--items`: List of item IDs to link

**Example:**
```bash
python src/elab_cli_simple.py upload-training \
  --label "janus_model_training" \
  --title-prefix "LodeSTAR Training" \
  --category 5 \
  --team 1
```

**What it uploads:**
- `logs/` directory → `janus_model_training_logs.tar.gz`
- `checkpoints/` directory → `janus_model_training_checkpoints.tar.gz`
- `models/` directory → `janus_model_training_models.tar.gz`

### 2. Upload Test Results

Uploads test results (logs, detection_results, test_results_summary.yaml) to elab:

```bash
python src/elab_cli_simple.py upload-test
```

**Options:**
- `--label`: Custom label (default: timestamp)
- `--title-prefix`: Title prefix (default: "Test Results")
- `--category`: Category ID (default: 5 for "Full Run")
- `--team`: Team ID (default: 1 for "Molecular Nanophotonics Group")
- `--experiments`: List of experiment IDs to link
- `--items`: List of item IDs to link

**Example:**
```bash
python src/elab_cli_simple.py upload-test \
  --label "janus_particle_detection" \
  --title-prefix "LodeSTAR Detection Test" \
  --category 5 \
  --team 1
```

**What it uploads:**
- `logs/` directory → `janus_particle_detection_logs.tar.gz`
- `detection_results/` directory → `janus_particle_detection_detection_results.tar.gz`
- `test_results_summary.yaml` (if exists)

### 3. Link Resources

Link existing experiments or items to an experiment:

```bash
python src/elab_cli_simple.py link-resources \
  --experiment-id <EXPERIMENT_ID> \
  --experiments <EXP_ID1> <EXP_ID2> \
  --items <ITEM_ID1> <ITEM_ID2>
```

**Example:**
```bash
python src/elab_cli_simple.py link-resources \
  --experiment-id 179 \
  --experiments 155 176 \
  --items 1270
```

## Typical Workflow

### After Training a Model:
```bash
# Upload training results
python src/elab_cli_simple.py upload-training \
  --label "janus_model_v1" \
  --title-prefix "LodeSTAR Janus Training"
```

### After Testing the Model:
```bash
# Upload test results
python src/elab_cli_simple.py upload-test \
  --label "janus_model_v1_test" \
  --title-prefix "LodeSTAR Janus Test"
```

### Link Related Experiments:
```bash
# Link training and test experiments
python src/elab_cli_simple.py link-resources \
  --experiment-id <TEST_EXPERIMENT_ID> \
  --experiments <TRAINING_EXPERIMENT_ID>
```

## Directory Structure Expected

### For Training:
```
logs/           # Training logs, metrics
checkpoints/    # Model checkpoints
models/         # Final model files
```

### For Testing:
```
logs/                    # Test logs, evaluation metrics
detection_results/       # Visualized detection results
test_results_summary.yaml # Summary file (optional)
```

## Benefits of Simplified Version

1. **Focused**: Only essential commands for training/testing workflow
2. **Simple**: No complex update logic or unnecessary features
3. **Reliable**: Straightforward upload without edge cases
4. **Maintainable**: Easy to understand and modify
5. **Consistent**: Same interface for both training and testing

## Migration from Original CLI

If you were using the original `elab_cli.py`:

- `upload-test-run` → `upload-test`
- `link-resources` → `link-resources` (same)
- Remove complex commands like cloning, comparing, syncing
- Simplified metadata handling (no tags, just category/team)
