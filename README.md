# MONA LodeSTAR - Single Particle Detection

A comprehensive implementation of LodeSTAR (Localization of Deformable Shapes using Transform-Augmented Regression) for single particle detection and analysis in microscopy images.

## Overview

This repository implements the LodeSTAR algorithm as described in the research paper for detecting and localizing various particle types in microscopy images. The system can identify and track different particle shapes including Janus particles, rings, spots, ellipses, and rods.

## Features

- **Multi-Particle Support**: Detection of Janus, Ring, Spot, Ellipse, and Rod particles
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
│   ├── custom_lodestar.py         # Paper-accurate LodeSTAR implementation
│   ├── run_single_particle_pipeline.py  # Complete pipeline runner
│   ├── config.yaml                # Configuration file
│   ├── utils.py                   # Utility functions
│   ├── generate_samples.py        # Sample generation script
|   └── requirements.txt           # Dependencies
```

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

```bash
python src/test_single_particle.py
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