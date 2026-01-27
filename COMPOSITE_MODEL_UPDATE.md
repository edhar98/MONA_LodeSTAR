# Composite Model Update: Config-Based Model Loading

## Change Summary

The `CompositeLodeSTAR` class now respects the `samples` field in the configuration file when loading models. Only models for particle types specified in `config['samples']` will be loaded from `trained_models_summary.yaml`.

## Motivation

Previously, the composite model would load **all** trained models from `trained_models_summary.yaml` regardless of configuration. This had several limitations:

1. **No selective loading**: Couldn't choose which models to use
2. **Excessive memory usage**: All models loaded even if not needed
3. **No flexibility**: Testing specific particle combinations required code changes

## Implementation

### Before
```python
for particle_type, model_info in trained_models_summary.items():
    model_path = model_info['model_path']
    if os.path.exists(model_path):
        model = self._load_model(model_path, config)
        if model is not None:
            self.models[particle_type] = model
```

### After
```python
config_samples = config.get('samples', [])
if not config_samples:
    raise ValueError("No samples specified in config file")

for particle_type, model_info in trained_models_summary.items():
    if particle_type not in config_samples:
        print(f"Skipping {particle_type} (not in config samples)")
        continue
    
    model_path = model_info['model_path']
    if os.path.exists(model_path):
        model = self._load_model(model_path, config)
        if model is not None:
            self.models[particle_type] = model
```

## Usage

### Load All Models
```yaml
# config.yaml
samples: [Janus, Ring, Spot, Ellipse, Rod]
```

```bash
python src/test_composite_model.py --config src/config.yaml
```

Output:
```
Composite model will load only samples from config: ['Janus', 'Ring', 'Spot', 'Ellipse', 'Rod']
Loaded Janus model from models/euk2wnni/Janus_weights.pth
Loaded Ring model from models/9ne6i5jr/Ring_weights.pth
Loaded Spot model from models/w4mrwg80/Spot_weights.pth
Loaded Ellipse model from models/e767fhbg/Ellipse_weights.pth
Loaded Rod model from models/5z4h6luk/Rod_weights.pth
Composite model initialized with 5 models: ['Janus', 'Ring', 'Spot', 'Ellipse', 'Rod']
```

### Load Subset of Models
```yaml
# config_subset_example.yaml
samples: [Janus, Spot]
```

```bash
python src/test_composite_model.py --config src/config_subset_example.yaml
```

Output:
```
Composite model will load only samples from config: ['Janus', 'Spot']
Loaded Janus model from models/euk2wnni/Janus_weights.pth
Skipping Ring (not in config samples)
Loaded Spot model from models/w4mrwg80/Spot_weights.pth
Skipping Ellipse (not in config samples)
Skipping Rod (not in config samples)
Composite model initialized with 2 models: ['Janus', 'Spot']
```

### Programmatic Usage
```python
from composite_model import CompositeLodeSTAR
import utils

# Load config with specific samples
config = utils.load_yaml('src/config_subset_example.yaml')
trained_models = utils.load_yaml('trained_models_summary.yaml')

# Only models in config['samples'] will be loaded
composite = CompositeLodeSTAR(config, trained_models)

print(f"Loaded models: {composite.particle_types}")
# Output: Loaded models: ['Janus', 'Spot']
```

## Benefits

1. **Selective Loading**: Choose which particle types to detect
2. **Memory Efficiency**: Load only needed models
3. **Faster Initialization**: Fewer models to load
4. **Flexible Testing**: Easy to test specific combinations
5. **Configuration-Driven**: No code changes needed

## Use Cases

### 1. Binary Classification
Test detection/classification of only two particle types:
```yaml
samples: [Janus, Ring]
```

### 2. Specific Research Focus
Focus on particular particle types of interest:
```yaml
samples: [Ellipse, Rod]
```

### 3. Memory-Constrained Environments
Load fewer models when memory is limited:
```yaml
samples: [Spot]  # Single model
```

### 4. Incremental Testing
Test models as they become available:
```yaml
samples: [Janus, Ring, Spot]  # First three trained models
```

## Error Handling

### No samples in config
```python
# Raises: ValueError("No samples specified in config file")
```

### No valid models found
```python
# Raises: ValueError(f"No valid models found for samples: {config_samples}")
```

### Sample not in trained models
```
# Prints warning: "Warning: Model path not found for {particle_type}: {model_path}"
```

## Backward Compatibility

The change is fully backward compatible. Existing configs with `samples` field will work exactly as before, loading all specified models. The only difference is that models **not** in the samples list will be skipped.

## Example Scenarios

### Scenario 1: Development Testing
During development, test with a single model:
```yaml
samples: [Spot]
```

### Scenario 2: Comparing Particle Pairs
Compare detection of similar particles:
```yaml
samples: [Janus, Ring]  # Both circular particles
```
or
```yaml
samples: [Ellipse, Rod]  # Both elongated particles
```

### Scenario 3: Full Production
Use all available models:
```yaml
samples: [Janus, Ring, Spot, Ellipse, Rod]
```

## Documentation Updates

Updated documentation files:
- `src/composite_model.py`: Added config check and logging
- `COMPOSITE_MODEL_README.md`: Added note about selective loading
- `QUICK_START_COMPOSITE.md`: Added configuration tips and troubleshooting
- `src/config_subset_example.yaml`: Example config with subset

## Testing

Test the change with different configurations:

```bash
# All models
python src/test_composite_model.py --config src/config.yaml

# Subset of models
python src/test_composite_model.py --config src/config_subset_example.yaml

# Single model
python src/run_composite_pipeline.py  # Edit config.yaml: samples: [Janus]
```

## Summary

This update provides fine-grained control over which particle models are loaded by the composite model, enabling:
- More efficient resource usage
- Flexible testing scenarios
- Configuration-driven model selection
- Better error messages and logging

The implementation is clean, maintains backward compatibility, and aligns with the project's minimalist design philosophy.

