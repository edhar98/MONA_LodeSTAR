# MONA LodeSTAR - Single Particle Detection

A comprehensive implementation of LodeSTAR (Localisation and detection from Symmetries, Translations And Rotations) for single particle detection and analysis in microscopy images.

## Overview

This repository implements the LodeSTAR algorithm as described in the research paper for detecting and localizing various particle types in microscopy images. The system can identify and track different particle shapes including Janus particles, rings, spots, ellipses, and rods.

## Features

- **Multi-Particle Support**: Detection of Janus, Ring, Spot, Ellipse, and Rod particles
- **Composite Model**: Multi-class detection and classification using ensemble of specialized models
- **Synthetic Data Generation**: Configurable image generation with realistic particle properties
- **Deep Learning Training**: PyTorch-based training pipeline with Lightning framework
- **Comprehensive Testing**: Multiple dataset types for robust model evaluation
- **Experiment Tracking**: Weights & Biases integration for training monitoring
- **CLI Workflows**: Training, detection, tracking, and analysis scripts; deployment readiness is assessed separately.

## Repository Structure

```
MONA_LodeSTAR/
├── src/                           # Core source code
│   ├── detection/                 # Training, inference, orientation, benchmarks
│   ├── tracking/                  # Tracking, gap filling, supervised correction
│   ├── analysis/                  # MSD, ABP, interactions, confinement
│   ├── config.yaml                # Configuration file
│   ├── samples.yaml               # Particle sample definitions
│   ├── utils.py                   # Utility functions
│   └── requirements.txt           # Dependencies
├── web/                           # Web interface
│   ├── app.py                     # FastAPI application assembly
│   ├── routers/                   # Files and TDMS routes
│   ├── services/                  # TDMS, frames, and cache operations
│   ├── auth.py                    # Session/user helpers
│   ├── config.py                  # Web paths/settings
│   ├── state.py                   # Runtime state persistence
│   ├── templates/index.html        # Web UI
│   └── data/                      # User data (runtime)
├── tools/                         # Data processing utilities
│   ├── crop.py                    # Interactive image cropping
│   ├── mask.py                    # Circular ROI masking
│   ├── merge_mp4.py               # MP4 video merger
│   ├── wandb_logging.py           # WandB logging abstraction
│   └── elab/                      # ELAB integration
├── debug/                         # Research & experimentation
│   ├── diagnostics/               # Diagnostic scripts
│   └── inspection/                 # Inspection tools
├── test/                          # Test infrastructure
│   ├── unit/                      # Unit tests
│   ├── regression/                # Regression tests
│   └── integration/               # Integration tests
├── docs/                          # Documentation
│   ├── papers/                    # Research papers
│   ├── ARCHITECTURE.md            # Architecture overview
│   ├── BRANCH_GUIDES.md           # Branch-specific guides
│   └── QUICK_REFERENCE.md         # Quick reference
├── presentation/                  # Presentation materials
└── COMPOSITE_MODEL_README.md      # Detailed composite model documentation
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the 6-branch workflow structure and [docs/BRANCH_GUIDES.md](docs/BRANCH_GUIDES.md) for branch-specific documentation.

## Deployment status

The existing JupyterHub installation is editable at `/home/mona/MONA_LodeSTAR`; it is distinct from this working checkout. Changes here are not automatically deployed there. Packaging currently depends on the adjacent source checkout (`src/` and `tools/`); a standalone wheel has not been validated. See the [working-tree web inventory](docs/WEB_INTEGRATION_AUDIT.md) and [agent workflow](docs/AGENT_WORKFLOW.md). The `--reload` examples below are development commands.

## Installation

### Prerequisites

- Python 3.10 is used by the current MONA environment; other versions have not been validated.
- CUDA-compatible GPU (recommended)
- PyTorch with CUDA support

### Dependencies

Install required packages:

```bash
pip install -r src/requirements.txt
```

**Note**: DeepTrack2 is included in the requirements and will be installed from the git repository automatically.

## Quick Start

### 1. Generate Sample Data

```bash
python src/detection/generate_samples.py
```

### 2. Train Models
```bash
python src/detection/train_single_particle.py
```

### 3. Generate datasets

```bash
python src/detection/image_generator.py
```

### 4. Test Models

Test individual models:
```bash
python src/detection/test_single_particle.py
```

Test composite model (multi-class detection):
```bash
python src/detection/test_composite_model.py
```

Compare single vs composite model performance:
```bash
python src/detection/compare_models.py
```

## Configuration

The main configuration file `src/config.yaml` contains:

- **Training Parameters**: Learning rate, batch size, epochs
- **Data Augmentation**: Intensity and multiplicative noise ranges
- **Detection Settings**: Alpha, beta, and cutoff thresholds
- **Model Architecture**: Number of transforms, device configuration

## Output Files and Directories

The system generates several output files and directories during execution:

### **Generated Data**
- **`data/`**: Contains generated datasets and sample images for each particle type
- **`models/`**: Stores trained model weights, checkpoints, and model configurations
- **`detection_results/`**: Contains model detection outputs, bounding boxes, and evaluation results

### **Logs and Tracking**
- **`logs/`**: Training and execution logs with timestamps and error information
- **`lightning_logs/`**: PyTorch Lightning framework logs with training metrics
- **`wandb_logs/`**: Weights & Biases experiment tracking logs and visualizations

### **Summary Files**
- **`test_results_summary.yaml`**: Test results organized by particle type and dataset category (same/different shape/size), containing precision, recall, F1-scores, and total true/false positive/negative counts for each test scenario
- **`trained_models_summary.yaml`**: Model tracking information organized by particle type, containing checkpoint paths, model weight paths, and model directories for each training run, including multiple model versions per particle type

**Note:** Research papers are located in `docs/papers/`. The old inventory is archived under `docs/archive/2026-01-verification/`; use `AGENTS.md` and current docs for live file organization.

## Data Generation

The `src/detection/image_generator.py` module creates synthetic microscopy images with:

- **Realistic Particle Properties**: Configurable intensity, size, and shape parameters
- **Multiple Dataset Types**:
  - Same shape, same size
  - Same shape, different sizes
  - Different shapes, same size
  - Different shapes, different sizes
- **Trajectory Generation**: Time-series data with particle movement
- **Annotation Export**: Pascal VOC format XML files

### Supported Particle Types

1. **Spot**: Gaussian intensity distribution
2. **Ring**: Annular intensity pattern
3. **Janus**: Asymmetric particle with orientation
4. **Ellipse**: Elliptical shape with rotation
5. **Rod**: Rectangular particle with length/width

## Training Pipeline

### Single Particle Training

The training pipeline (`src/detection/train_single_particle.py`) provides:

- **Model Architecture**: Paper-accurate LodeSTAR implementation
- **Data Augmentation**: Intensity and multiplicative noise
- **Validation**: Separate validation dataset with gentle augmentation
- **Metrics Tracking**: Comprehensive logging with Weights & Biases
- **Checkpointing**: Automatic model saving and restoration

### Training Process

1. **Data Preparation**: Load and augment training/validation data
2. **Model Initialization**: Create LodeSTAR model with specified transforms
3. **Training Loop**: PyTorch Lightning-based training with callbacks
4. **Validation**: Regular validation with metrics logging
5. **Checkpointing**: Save best models based on validation loss

## Testing and Evaluation

### Test Datasets

The system generates four types of test datasets:

1. **Same Shape, Same Size**: Tests detection consistency
2. **Same Shape, Different Sizes**: Tests scale invariance
3. **Different Shapes, Same Size**: Tests shape discrimination
4. **Different Shapes, Different Sizes**: Tests robustness

### Evaluation Metrics

- **Detection Accuracy**: Precision, recall, F1-score
- **Localization Error**: Mean squared error in position
- **Orientation Accuracy**: Angular error for oriented particles
- **Processing Speed**: Frames per second

### Detection Engine Baseline

LodeSTAR is the learned detector. `trackpy.locate` is now used as a classical microscopy baseline for position-only detection and should be available through a future engine flag such as `--detection-engine lodestar|trackpy`.

Benchmark on one 1024x1024 JP frame (`JP_Fe_wf_2_40_slm075_574_001.png`, detector call only, warmup excluded):

| Engine | Hardware | Detections | Mean time |
|--------|----------|------------|-----------|
| LodeSTAR `model.detect` | CUDA | 120 | 180.1 ms |
| `trackpy.locate(diameter=41)` | CPU | 117 | 265.1 ms |
| LodeSTAR `model.detect` | CPU | 120 | 758.2 ms |
| `trackpy.locate(diameter=41)` | CPU | 117 | 265.7 ms |

Use LodeSTAR when GPU inference or learned morphology is required. Use `trackpy.locate` as a strong CPU baseline for clean blob-like particles.

## Tracking and Physics Analysis

`src/tracking/track_particles.py` turns detection CSVs into trajectories using within-frame NMS, Hungarian cross-frame linking, and linear gap interpolation. Output tracks use `track_id, frame, x, y, phi, ncc, is_interpolated`.

The current learned trajectory tools are benchmark/probe stages, not automatic replacements for the linear production baseline:

- `src/tracking/lstm_track_predictor.py` trains a causal next-state LSTM baseline.
- `src/tracking/benchmark_lstm_gap_filling.py` benchmarks masked gaps against linear interpolation and simple motion baselines.
- `src/tracking/lstm_gap_filler.py` trains the two-sided LSTM/BiLSTM-style gap filler and includes a Kalman smoother probe.
- `src/tracking/build_supervised_correction_dataset.py`, `src/tracking/train_supervised_correction_lstm.py`, and `src/tracking/apply_supervised_correction_lstm.py` implement the reference-calibrated LodeSTAR trajectory corrector.

Physics analysis lives in `src/analysis/`. `src/analysis/analyze_tracks.py` fits MSD/ABP parameters. The physics-first diagnostics include motion statistics, nearest-neighbor interaction analysis, confinement drift, filtered ABP comparisons, and velocity-persistence/AOUP-style analysis.

## Composite Model Approach

The composite model enables **multi-class particle detection and classification** by combining multiple specialized single-particle models.

### Key Features

- **Ensemble Detection**: Runs all particle-specific models in parallel on the same image
- **Weight-Based Classification**: Assigns particle class based on highest confidence (weight) value
- **Detection Merging**: Combines detections from all models using spatial clustering
- **Interpretable Results**: Provides weight maps for each particle type

### How It Works

1. **Parallel Inference**: Each trained model (Janus, Ring, Spot, Ellipse, Rod) processes the input image
2. **Weight Map Extraction**: Extract confidence maps from each model's output
3. **Detection Merging**: Cluster nearby detections (distance threshold = 20 pixels)
4. **Classification**: For each detection, compare weight values across all models
5. **Label Assignment**: Assign the particle type with highest weight at detection location

### Usage Example

```python
import sys
sys.path[:0] = ['src', 'src/detection']  # Run from repository root
from composite_model import CompositeLodeSTAR
import utils

config = utils.load_yaml('src/config.yaml')
trained_models = utils.load_yaml('trained_models_summary.yaml')

composite = CompositeLodeSTAR(config, trained_models)
detections, labels, weight_maps, outputs = composite.detect_and_classify(image)
```

See `COMPOSITE_MODEL_README.md` for detailed documentation.

## Documentation

- **[Architecture Overview](docs/ARCHITECTURE.md)** - 6-branch workflow and system architecture
- **[Branch Guides](docs/BRANCH_GUIDES.md)** - Detailed guides for each branch
- **[Quick Reference](docs/QUICK_REFERENCE.md)** - Common commands and patterns
- **[Composite Model](COMPOSITE_MODEL_README.md)** - Multi-class detection documentation
- **[Model Detection Parameters](MODEL_SPECIFIC_DETECTION_PARAMS.md)** - Detection parameter guide
- **[January 2026 verification archive](docs/archive/2026-01-verification/README.md)** - Superseded pre-restructure reports

## Model Architecture

### Paper-Accurate Implementation (`custom_lodestar.py`)

This repository provides a **paper-accurate LodeSTAR implementation** that follows the exact architecture specified in the research paper:

```
Input → 3×Conv2D(3×3, 32) + ReLU → MaxPool2D(2×2) → 8×Conv2D(3×3, 32) + ReLU → Conv2D(1×1, 3)
```

### Default LodeSTAR Implementation

The **default LodeSTAR implementation** from the DeepTrack library differs from the paper specification:

```
Input → Conv2D(3×3, 32) → Conv2D(3×3, 32) → Conv2D(3×3, 64) → Pool → Conv2D(3×3, 64) → Conv2D(3×3, 64) → Conv2D(3×3, 64) → Conv2D(3×3, 64) → Conv2D(3×3, 64) → Conv2D(3×3, 64) → Conv2D(3×3, 64) → Conv2D(1×1, num_outputs + 1)
```

**Output Channels**:
- Channel 1: Δx (x-displacement)
- Channel 2: Δy (y-displacement)  
- Channel 3: ρ (detection confidence)

## CLI Tools

### Pipeline Runner

```bash
# Run complete training and testing pipeline
python src/detection/run_single_particle_pipeline.py

# Skip training and run testing only
python src/detection/run_single_particle_pipeline.py --test-only
```

### Individual Components

```bash
# Generate synthetic datasets
python src/detection/generate_samples.py

# Train specific particle type
python src/detection/train_single_particle.py --particle Janus

# Test specific model
python src/detection/test_single_particle.py --particle Janus --model models/janus.pth
```

## Experiment Tracking

The system integrates with Weights & Biases for:

- **Training Metrics**: Loss curves, accuracy plots
- **Model Parameters**: Architecture details, hyperparameters
- **Data Visualization**: Sample images, detection results
- **Experiment Comparison**: Multiple runs and configurations

## Performance

- **Training Time**: depends on particle type, crop pool, epochs, and GPU availability
- **Detection Speed**: on one 1024x1024 JP frame, LodeSTAR took 180.1 ms on CUDA and 758.2 ms on CPU; `trackpy.locate(diameter=41)` took 265.1 ms on CPU
- **Memory Usage**: depends on architecture, image size, batch size, and transforms
- **Model Size**: typically small single-particle `.pth` weights

## Troubleshooting

### Common Issues

1. **Missing Sample Images**: Run `python src/detection/generate_samples.py`
2. **CUDA Out of Memory**: Reduce batch size in config
3. **Training Divergence**: Check learning rate and data augmentation
4. **Poor Detection**: Verify alpha/beta/cutoff parameters

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Citation

If you use this implementation, please cite the original LodeSTAR paper:

```bibtex
@article{Midtvedt2022,
  author = {Midtvedt, Benjamin and Pineda, Jesús and Skärberg, Fredrik and Olsén, Erik and Bachimanchi, Harshith and Wesén, Emelie and Esbjörner, Elin K. and Selander, Erik and Höök, Fredrik and Midtvedt, Daniel and Volpe, Giovanni},
  title = {Single-shot self-supervised object detection in microscopy},
  journal = {Nature Communications},
  volume = {13},
  number = {1},
  pages = {7492},
  year = {2022},
  month = {12},
  day = {05},
  doi = {10.1038/s41467-022-35004-y},
  url = {https://doi.org/10.1038/s41467-022-35004-y},
  issn = {2041-1723}
}
```

## License

This project is licensed under the GNU GPL-3.0 License - see the LICENSE file for details.

## Contact

For questions and support:
- **Repository**: [MONA_LodeSTAR](https://github.com/edhar98/MONA_LodeSTAR)
- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for general questions
