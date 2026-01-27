# Composite Model for Multi-Class Particle Detection

## Overview

The composite model approach enables multi-class particle detection and classification by combining multiple single-particle LodeSTAR models. Each model is trained on a specific particle type, and during inference, all models analyze the same image. The classification is determined by comparing weight maps from all models at each detection location.

## Architecture

### Single Particle Models
Each particle type (Janus, Ring, Spot, Ellipse, Rod) has its own trained LodeSTAR model:
- **Input**: Grayscale image (H, W)
- **Output**: 3 channels (Δx, Δy, ρ) at half resolution (H/2, W/2)
  - Δx: X-displacement field
  - Δy: Y-displacement field  
  - ρ: Weight/confidence map

### Composite Model
The `CompositeLodeSTAR` class orchestrates multiple models:

1. **Load Models**: Loads trained particle models specified in `config['samples']` from `trained_models_summary.yaml`
2. **Parallel Inference**: Runs all loaded models on the same input image
3. **Detection Merging**: Combines detections from all models using spatial clustering
4. **Classification**: Assigns particle type based on highest weight value at detection location

**Note**: Only models for particle types listed in `config['samples']` will be loaded. This allows selective loading of specific particle models.

## Algorithm

### Detection and Classification Process

```
For each test image:
  1. Run all particle models in parallel
     - Extract weight maps (ρ) from each model
     - Get detections from each model using detect() method
  
  2. Merge all detections
     - Cluster nearby detections (distance_threshold=20 pixels)
     - Compute cluster centroids as unified detection positions
  
  3. Classify each detection
     - For each unified detection position (x, y):
       - Extract weight value from ALL model weight maps at (x, y)
       - Assign label of model with highest weight
       - confidence = max(weight values)
```

### Key Parameters

- `distance_threshold`: Maximum distance for clustering detections (default: 20 pixels)
- `alpha`: Object similarity metric - **loaded from each model's config**
- `beta`: 1 - alpha - **loaded from each model's config**
- `cutoff`: Detection threshold - **loaded from each model's config**
- `mode`: Detection mode - **loaded from each model's config**

**Note**: Detection parameters (alpha, beta, cutoff, mode) are now **model-specific**, loaded from each model's individual config file. This allows each particle type to use its optimal detection settings.

## Usage

### Training Individual Models

First, train individual models for each particle type:

```bash
python src/train_single_particle.py --particle Janus
python src/train_single_particle.py --particle Ring
python src/train_single_particle.py --particle Spot
python src/train_single_particle.py --particle Ellipse
python src/train_single_particle.py --particle Rod
```

Or train all at once:

```bash
python src/train_single_particle.py
```

### Testing with Composite Model

Test all particle types simultaneously with classification:

```bash
python src/test_composite_model.py --config src/config.yaml
```

Enable visualization of results:

```yaml
visualize: true
```

### Programmatic Usage

```python
from composite_model import CompositeLodeSTAR
import utils

config = utils.load_yaml('src/config.yaml')
trained_models = utils.load_yaml('trained_models_summary.yaml')

composite = CompositeLodeSTAR(config, trained_models)

# Uses model-specific parameters from each model's config (recommended)
detections, labels, weight_maps, outputs = composite.detect_and_classify(image)

# Override parameters for all models (if needed)
detections, labels, weight_maps, outputs = composite.detect_and_classify(
    image,
    alpha=0.2,
    beta=0.8,
    cutoff=0.2
)
```

## Output Structure

### Detection Results

```python
detections: np.ndarray  # Shape (N, 3) - [x, y, confidence]
labels: list           # Length N - ['Janus', 'Ring', ...]
weight_maps: dict      # {particle_type: weight_map (H, W)}
outputs: dict          # {particle_type: model_output (1, 3, H/2, W/2)}
```

### Evaluation Metrics

For each dataset type:
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 * (Precision * Recall) / (Precision + Recall)
- **True Positives (TP)**: Correct detections with correct labels
- **False Positives (FP)**: Incorrect detections or wrong labels
- **False Negatives (FN)**: Missed detections

## File Structure

```
src/
├── composite_model.py          # Composite model implementation
├── test_composite_model.py     # Testing script for composite model
├── train_single_particle.py    # Training script for individual models
├── test_single_particle.py     # Testing script for single models
├── custom_lodestar.py          # Custom LodeSTAR architecture
└── config.yaml                 # Configuration file

detection_results/
└── Testing_snr_10-10/
    ├── composite/              # Composite model results
    │   ├── same_shape_same_size/
    │   ├── same_shape_different_size/
    │   ├── different_shape_same_size/
    │   └── different_shape_different_size/
    └── {particle_type}_{model_id}/  # Single model results
```

## Visualization

The composite model generates visualizations with:

**Top Row:**
- Ground truth image with annotations
- Individual weight maps for each particle type (hot colormap)

**Bottom Row:**
- Combined detection results with color-coded markers and labels
- Ground truth markers (green circles) with colored text labels (format: "GT:ParticleType")
- Detection markers (colored circles) with colored text labels showing particle type
- Metrics overlay (Precision, Recall, F1, TP, FP, FN)

**Color Coding:**
Each particle type has a distinct color for easy identification:
- **Janus**: Red
- **Ring**: Blue
- **Spot**: Yellow
- **Ellipse**: Cyan
- **Rod**: Magenta

**Label Format:**
- Ground truth labels: White box with green border, colored text "GT:ParticleType"
- Detection labels: Black box with colored border matching particle type, colored text showing particle type

## Advantages

1. **Multi-Class Detection**: Simultaneously detects and classifies multiple particle types
2. **Leverages Specialization**: Each model specializes in its particle type
3. **Interpretable**: Weight maps show model confidence for each particle type
4. **Flexible**: Easy to add new particle types by training additional models

## Limitations

1. **Computational Cost**: Runs N models for N particle types
2. **Memory Usage**: Stores weight maps for all models
3. **Occlusion**: May struggle with overlapping particles of different types

## Configuration

Key settings in `src/config.yaml`:

```yaml
# Only these particle types will be loaded by the composite model
samples: [Janus, Ring, Spot, Ellipse, Rod]  # Can be subset: [Janus, Ring]

alpha: 0.2
beta: 0.8
cutoff: 0.2
mode: constant

visualize: false  # Set to true for visualization output

lodestar_version: custom  # or 'default', 'skip_connections'
n_transforms: 4
```

**Important**: The `samples` field controls which models are loaded. Only particle types listed in this field will be loaded from `trained_models_summary.yaml`. This allows you to:
- Test specific particle type combinations
- Reduce memory usage by loading fewer models
- Focus on specific detection tasks

## Results

Results are saved to:
- `test_composite_results_summary.yaml`: Overall metrics
- `detection_results/Testing_*/composite/*/`: Visualizations (if enabled)
- `logs/test_composite_model_*.log`: Detailed logs

