# Quick Start Guide: Composite Model

## Goal
Detect AND classify multiple particle types in the same image using an ensemble of specialized LodeSTAR models.

## Prerequisites

Ensure you have trained individual models for each particle type:

```bash
python src/train_single_particle.py
```

This creates `trained_models_summary.yaml` with paths to all trained models.

## Basic Usage

### 1. Test Composite Model on All Datasets

```bash
python src/test_composite_model.py --config src/config.yaml
```

This will:
- Load all trained particle models
- Test on 4 dataset types (same/different shape/size combinations)
- Calculate precision, recall, F1-score for multi-class detection
- Save results to `test_composite_results_summary.yaml`

### 2. Enable Visualization

Edit `src/config.yaml`:
```yaml
visualize: true
```

Then run:
```bash
python src/test_composite_model.py --config src/config.yaml
```

Results saved to: `detection_results/Testing_snr_10-10/composite/`

### 3. Quick Example

Run a single-image example:
```bash
python src/run_composite_pipeline.py
```

This demonstrates:
- Loading the composite model
- Processing one test image
- Displaying detections with labels
- Showing weight maps for all particle types

Output saved to: `detection_results/composite_example/`

### 4. Compare Performance

After running both single and composite model tests:

```bash
python src/test_single_particle.py  # If not already run
python src/test_composite_model.py
python src/compare_models.py
```

This generates:
- Comparison metrics in logs
- Comparison plot: `detection_results/comparison/model_comparison.png`

## Programmatic Usage

```python
from composite_model import CompositeLodeSTAR
import numpy as np
import utils

config = utils.load_yaml('src/config.yaml')
trained_models = utils.load_yaml('trained_models_summary.yaml')

composite = CompositeLodeSTAR(config, trained_models)

image = np.random.rand(416, 416)

# Uses model-specific detection parameters from each model's config
detections, labels, weight_maps, outputs = composite.detect_and_classify(image)

# Or override for all models
# detections, labels, _, _ = composite.detect_and_classify(image, alpha=0.5, cutoff=0.3)

for (x, y, conf), label in zip(detections, labels):
    print(f"{label}: ({x:.1f}, {y:.1f}) confidence={conf:.3f}")
```

## Understanding Outputs

### Detections
- **Format**: `np.ndarray` of shape `(N, 3)`
- **Contents**: `[x, y, confidence]` for each detection
- **N**: Number of detected particles

### Labels
- **Format**: `list` of length `N`
- **Contents**: Particle type strings: `['Janus', 'Ring', 'Spot', 'Ellipse', 'Rod']`

### Weight Maps
- **Format**: `dict` with particle types as keys
- **Contents**: `{particle_type: weight_map}` where weight_map is `(H, W)` array
- **Values**: Confidence values at each pixel location

### Model Outputs
- **Format**: `dict` with particle types as keys
- **Contents**: Raw model outputs `(1, 3, H/2, W/2)` with channels `[Δx, Δy, ρ]`

## Visualization

Composite model visualizations show:

**Top Row:**
- Ground truth with annotations
- Weight maps for each particle type (hot colormap)

**Bottom Row:**
- Combined detections with color-coded markers and labeled text:
  - **Janus**: Red marker and text with red border
  - **Ring**: Blue marker and text with blue border
  - **Spot**: Yellow marker and text with yellow border
  - **Ellipse**: Cyan marker and text with cyan border
  - **Rod**: Magenta marker and text with magenta border
- Ground truth markers (green circles) with "GT:ParticleType" labels in colored text
- Detection markers (colored circles) with particle type labels in matching colors
- Metrics overlay (Precision, Recall, F1, TP, FP, FN)

## Configuration

Key parameters in `src/config.yaml`:

```yaml
# Only particle types in this list will be loaded
samples: [Janus, Ring, Spot, Ellipse, Rod]  # Can be subset: [Janus, Spot]

alpha: 0.2
beta: 0.8
cutoff: 0.2
mode: constant

visualize: false

lodestar_version: custom
n_transforms: 4
```

**Tip**: To test only specific particle types, modify the `samples` list. For example:
```yaml
samples: [Janus, Ring]  # Only loads Janus and Ring models
```

## Troubleshooting

### Issue: "No valid models found for samples"
**Solution**: Ensure particle types in `config['samples']` match those in `trained_models_summary.yaml`.
```yaml
# config.yaml
samples: [Janus, Ring]  # Must match trained model names
```

### Issue: "No samples specified in config file"
**Solution**: Add `samples` field to your config file:
```yaml
samples: [Janus, Ring, Spot, Ellipse, Rod]
```

### Issue: "No valid models found"
**Solution**: Ensure `trained_models_summary.yaml` exists and contains valid model paths.
```bash
python src/train_single_particle.py
```

### Issue: Model paths not found
**Solution**: Check that model files exist at the paths in `trained_models_summary.yaml`.
```bash
ls models/
```

### Issue: Some models not loading
**Solution**: Check that particle types are in `config['samples']`. Only models listed in samples will be loaded.

### Issue: Out of memory
**Solution**: The composite model runs N models in parallel. Reduce image size or test fewer particle types.

### Issue: Slow inference
**Solution**: This is expected - composite model runs N forward passes. Consider:
- Using GPU acceleration
- Processing images in batches
- Reducing image resolution

## Expected Performance

The composite model should:
- **Detect** particles of all types in the same image
- **Classify** each detection to the correct particle type
- **Achieve** similar or better F1-score compared to single models on mixed datasets

## Next Steps

1. **Analyze Results**: Check `test_composite_results_summary.yaml`
2. **Visualize**: Enable visualization and inspect weight maps
3. **Compare**: Run comparison to see improvement over single models
4. **Tune**: Adjust detection parameters (alpha, beta, cutoff) if needed

## Documentation

- **Detailed Guide**: `COMPOSITE_MODEL_README.md`
- **Implementation Details**: `COMPOSITE_MODEL_IMPLEMENTATION.md`
- **Main README**: `README.md`

