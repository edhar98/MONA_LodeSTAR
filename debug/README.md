# Debug Directory

This directory contains inspection, diagnostic, and experimental scripts for debugging and analyzing the MONA LodeSTAR project.

## Structure

- **`inspection/`** - Scripts for inspecting model behavior and outputs
- **`diagnostics/`** - Diagnostic scripts for troubleshooting issues  
- **`experiments/`** - Experimental scripts for testing new features

## Contents

### Inspection Scripts (`inspection/`)
Scripts for visualizing model predictions, analyzing performance, and inspecting intermediate results:
- `investigate_augmentations.py` - Investigate data augmentation effects
- `architecture_diagram.py` - Generate architecture diagrams
- `simple_architecture_diagram.py` - Generate simplified architecture diagrams

### Diagnostic Scripts (`diagnostics/`)
Scripts for diagnosing problems, checking system health, and debugging specific issues:
- `diagnose_skip_connections.py` - Diagnose skip connections implementation

### Experimental Scripts (`experiments/`)
Scripts for experimenting with new architectures, parameters, and approaches before integrating into main codebase:
- (To be added as needed)

## Usage

Run from **repo root**. Scripts that import from `src/` need `PYTHONPATH=src`:

```bash
PYTHONPATH=src python debug/diagnostics/diagnose_skip_connections.py

PYTHONPATH=src python debug/inspection/investigate_augmentations.py [--particle Rod] [--config src/config_debug.yaml]

python debug/inspection/architecture_diagram.py

python debug/inspection/simple_architecture_diagram.py
```

### Script details

| Script | Args | Inputs | Outputs | Depends on |
|--------|------|--------|---------|------------|
| `diagnostics/diagnose_skip_connections.py` | none | none | stdout | `lodestar_with_skip_connections` (src/), deeptrack.deeplay |
| `inspection/investigate_augmentations.py` | `--particle Rod`, `--config src/config_debug.yaml` | config YAML, `config['data_dir']/Samples/<particle>/<particle>.jpg` | `debug_outputs/augmentation_investigation.png` | `utils` (src/) |
| `inspection/architecture_diagram.py` | none | none | stdout (ASCII diagram) | none |
| `inspection/simple_architecture_diagram.py` | none | none | stdout (flow diagram) | none |

## Notes

- These scripts are not part of the main codebase
- `diagnose_skip_connections` and `investigate_augmentations` use committed or uncommitted Core code from `src/`
- They are intended for development and debugging only
- Results should not be used in production
