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
- **Production Ready**: CLI tools and pipeline automation

## Repository Structure

```
MONA_LodeSTAR/
├── src/                           # Core source code
│   ├── image_generator.py         # Synthetic image generation
│   ├── train_single_particle.py   # Training pipeline
│   ├── test_single_particle.py    # Testing and evaluation
│   ├── composite_model.py         # Composite model for multi-class detection
│   ├── custom_lodestar.py         # Paper-accurate LodeSTAR implementation
│   ├── config.yaml                # Configuration file
│   ├── samples.yaml               # Particle sample definitions
│   ├── utils.py                   # Utility functions
│   └── requirements.txt           # Dependencies
├── web/                           # Web interface
│   ├── app.py                     # FastAPI backend
│   ├── templates/index.html        # Web UI
│   └── data/                      # User data (runtime)
├── tools/                         # Data processing utilities
│   ├── tdms_to_png.py             # TDMS to PNG/MP4 converter
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

## Installation

### Prerequisites

- Python 3.8+
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
python src/generate_samples.py
```

### 2. Train Models
```bash
python src/train_single_particle.py
```

### 3. Generate datasets

```bash
python src/image_generator.py
```

### 4. Test Models

Test individual models:
```bash
python src/test_single_particle.py
```

Test composite model (multi-class detection):
```bash
python src/test_composite_model.py
```

Compare single vs composite model performance:
```bash
python src/compare_models.py
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

**Note:** Research papers are located in `docs/papers/`. See [INVENTORY.md](INVENTORY.md) for complete file organization.

## Data Generation

The `image_generator.py` module creates synthetic microscopy images with:

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

The training pipeline (`train_single_particle.py`) provides:

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
- **[Duplicate Files](DUPLICATES_DOCUMENTATION.md)** - Clarification on duplicate files
- **[Inventory](INVENTORY.md)** - Complete codebase inventory
- **[Cleanup Report](CLEANUP_REPORT.md)** - Cleanup actions executed

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
python src/run_single_particle_pipeline.py

# Check prerequisites only
python src/run_single_particle_pipeline.py --check-only
```

### Individual Components

```bash
# Generate synthetic datasets
python src/generate_samples.py

# Train specific particle type
python src/train_single_particle.py --particle Janus

# Test specific model
python src/test_single_particle.py --particle Janus --model models/janus.pth
```

## Experiment Tracking

The system integrates with Weights & Biases for:

- **Training Metrics**: Loss curves, accuracy plots
- **Model Parameters**: Architecture details, hyperparameters
- **Data Visualization**: Sample images, detection results
- **Experiment Comparison**: Multiple runs and configurations

## Performance

- **Training Time**: ~2-4 hours per particle type (200 epochs)
- **Inference Speed**: 100+ FPS on GPU
- **Memory Usage**: 2-4 GB VRAM during training
- **Model Size**: ~1MB per trained model

## Troubleshooting

### Common Issues

1. **Missing Sample Images**: Run `python generate_samples.py`
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