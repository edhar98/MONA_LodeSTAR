# Model-Specific Detection Parameters

## Enhancement Summary

The composite model now uses **model-specific detection parameters** from each individual model's configuration file, rather than using global parameters for all models.

## Motivation

Each particle type may require different detection parameters for optimal performance:
- **Janus particles** might need different `alpha/beta` values than **Spot particles**
- **Rod particles** might require different `cutoff` thresholds than **Ring particles**
- Each model was trained with specific parameters that work best for its particle type

Previously, all models used the same global detection parameters, which could be suboptimal for some particle types.

## Implementation

### Changes Made

#### 1. **composite_model.py** - Modified `detect_and_classify()` method

**Before:**
```python
def detect_and_classify(self, image, alpha=0.2, beta=0.8, mode="constant", cutoff=0.2):
    # ... processing ...
    
    for particle_type, model in self.models.items():
        detections = model.detect(
            image_tensor, 
            alpha=alpha,      # Same for all models
            beta=beta,        # Same for all models
            mode=mode,        # Same for all models
            cutoff=cutoff     # Same for all models
        )[0]
```

**After:**
```python
def detect_and_classify(self, image, alpha=None, beta=None, mode=None, cutoff=None):
    # ... processing ...
    
    for particle_type, model in self.models.items():
        # Load model-specific parameters
        model_config = self.model_configs[particle_type]
        model_alpha = alpha if alpha is not None else model_config.get('alpha', 0.2)
        model_beta = beta if beta is not None else model_config.get('beta', 0.8)
        model_mode = mode if mode is not None else model_config.get('mode', 'constant')
        model_cutoff = cutoff if cutoff is not None else model_config.get('cutoff', 0.2)
        
        print(f"{particle_type} detection params: alpha={model_alpha}, beta={model_beta}, cutoff={model_cutoff}, mode={model_mode}")
        
        detections = model.detect(
            image_tensor, 
            alpha=model_alpha,     # Model-specific
            beta=model_beta,       # Model-specific
            mode=model_mode,       # Model-specific
            cutoff=model_cutoff    # Model-specific
        )[0]
```

**Key features:**
- Parameters default to `None` instead of hardcoded values
- If `None`, loads from model's config file: `model_configs[particle_type]`
- Can still override by passing explicit values
- Logs the parameters used for each model

#### 2. **test_composite_model.py** - Updated to use model-specific params

**Before:**
```python
detections, detection_labels, weight_maps, model_outputs = composite_model.detect_and_classify(
    image, 
    alpha=config.get('alpha', 0.2), 
    beta=config.get('beta', 0.8), 
    mode=config.get('mode', 'constant'), 
    cutoff=config.get('cutoff', 0.2)
)
```

**After:**
```python
# Uses model-specific parameters from each model's config
detections, detection_labels, weight_maps, model_outputs = composite_model.detect_and_classify(image)
```

#### 3. **run_composite_pipeline.py** - Updated example

**Before:**
```python
detections, labels, weight_maps, _ = composite.detect_and_classify(
    image,
    alpha=config.get('alpha', 0.2),
    beta=config.get('beta', 0.8),
    cutoff=config.get('cutoff', 0.2)
)
```

**After:**
```python
print("Using model-specific detection parameters from each model's config...")
detections, labels, weight_maps, _ = composite.detect_and_classify(image)
```

## How It Works

### 1. **Model Config Loading** (Already implemented in `__init__`)

```python
def __init__(self, config, trained_models_summary):
    for particle_type, model_info in trained_models_summary.items():
        model_config_path = model_info['models_dir'] + '/config.yaml'
        model_config = utils.load_yaml(model_config_path)
        
        # Store model config for later use
        self.model_configs[particle_type] = model_config
```

### 2. **Parameter Extraction** (New in `detect_and_classify`)

For each model, extract detection parameters from its config:
```python
model_config = self.model_configs[particle_type]
model_alpha = model_config.get('alpha', 0.2)    # Fallback to 0.2 if not in config
model_beta = model_config.get('beta', 0.8)      # Fallback to 0.8 if not in config
model_mode = model_config.get('mode', 'constant')
model_cutoff = model_config.get('cutoff', 0.2)
```

### 3. **Override Support**

Can still override for all models:
```python
# Use alpha=1.0 for all models
detections, labels, _, _ = composite.detect_and_classify(image, alpha=1.0)

# Use model-specific values
detections, labels, _, _ = composite.detect_and_classify(image)
```

## Example Model Configs

### Janus Model Config
```yaml
alpha: 1
beta: 0
cutoff: 0.8
mode: constant
```

### Spot Model Config
```yaml
alpha: 0.2
beta: 0.8
cutoff: 0.2
mode: constant
```

### Ring Model Config
```yaml
alpha: 0.5
beta: 0.5
cutoff: 0.5
mode: constant
```

## Output Example

When running detection, you'll see:
```
Composite model will load only samples from config: ['Janus', 'Ring', 'Spot']
Loaded Janus model from models/euk2wnni/Janus_weights.pth
Loaded Ring model from models/9ne6i5jr/Ring_weights.pth
Loaded Spot model from models/w4mrwg80/Spot_weights.pth
Composite model initialized with 3 models: ['Janus', 'Ring', 'Spot']

Using model-specific detection parameters from each model's config...
Janus detection params: alpha=1, beta=0, cutoff=0.8, mode=constant
Ring detection params: alpha=0.5, beta=0.5, cutoff=0.5, mode=constant
Spot detection params: alpha=0.2, beta=0.8, cutoff=0.2, mode=constant
```

## Benefits

1. **Optimal Performance**: Each model uses its best-tuned parameters
2. **Flexibility**: Different particle types can have different detection strategies
3. **Consistency**: Parameters match what was used during training
4. **Override Support**: Can still force specific parameters when needed
5. **Transparency**: Logs show exactly what parameters each model uses

## Detection Parameter Meanings

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `alpha` | Object similarity metric - weight given to position agreement | 0.0 - 1.0 |
| `beta` | Complement of alpha (usually 1 - alpha) | 0.0 - 1.0 |
| `cutoff` | Detection threshold - minimum confidence for detection | 0.0 - 1.0 |
| `mode` | Padding mode for coordinate transformations | 'constant', 'edge', 'reflect' |

### Parameter Effects

**High alpha (e.g., 1.0):**
- Emphasizes position consensus
- Good for particles with clear, consistent positions
- Example: Janus particles

**Low alpha (e.g., 0.2):**
- More weight-based detection
- Good for particles with varying appearances
- Example: Spot particles

**High cutoff (e.g., 0.8):**
- Fewer, more confident detections
- Reduces false positives
- May miss some true particles

**Low cutoff (e.g., 0.2):**
- More detections
- Catches more true particles
- May include false positives

## Usage

### Standard Usage (Model-Specific Params)
```python
from composite_model import CompositeLodeSTAR
import utils

config = utils.load_yaml('src/config.yaml')
trained_models = utils.load_yaml('trained_models_summary.yaml')

composite = CompositeLodeSTAR(config, trained_models)

# Uses each model's own detection parameters
detections, labels, weight_maps, outputs = composite.detect_and_classify(image)
```

### Override for All Models
```python
# Force all models to use alpha=0.5
detections, labels, weight_maps, outputs = composite.detect_and_classify(
    image, 
    alpha=0.5
)
```

### Partial Override
```python
# Override cutoff, but use model-specific alpha/beta
detections, labels, weight_maps, outputs = composite.detect_and_classify(
    image, 
    cutoff=0.3
)
```

## Testing

Run the composite model to see model-specific parameters in action:

```bash
# Test with model-specific parameters
python src/test_composite_model.py --config src/config.yaml

# Quick example
python src/run_composite_pipeline.py
```

Check the output logs to verify each model uses its own parameters.

## Future Enhancements

Potential improvements:
1. **Parameter Optimization**: Auto-tune parameters per model
2. **Adaptive Parameters**: Adjust based on image characteristics
3. **Parameter Visualization**: Show parameter effects in visualizations
4. **Parameter Analysis**: Compare detection quality with different parameters

## Backward Compatibility

Fully backward compatible:
- Old code with explicit parameters still works
- Default fallback values ensure robustness
- No breaking changes to API

## Related Enhancement

This enhancement sets the foundation for future improvements using the Δx and Δy displacement channels (saved in memory ID: 9771523), which will further improve classification accuracy by combining displacement field consensus with weight-based scoring.

## Summary

Model-specific detection parameters ensure each particle type uses optimal settings for detection, improving overall accuracy and leveraging the specialized training of each model. The implementation is clean, transparent, and maintains full backward compatibility.

