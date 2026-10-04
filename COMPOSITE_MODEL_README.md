# Composite particle detection

Composite detection combines particle-specific LodeSTAR models to detect and classify different particle types in one image. Each selected model processes the image; nearby detections are clustered and assigned the class with the highest weight at the merged position. That weight is not a calibrated probability. Models are evaluated sequentially, so inference cost grows with their number.

## Web interface

In **Detection → Setup**, enable **Composite classification**, select 2–8 web-trained models with distinct particle names, and choose the clustering distance. Run a preview before batch detection.

- Only standard detection is supported. Composite output has no orientation; it cannot feed orientation-dependent gap refinement.
- The selected detection settings apply to every member model.
- Detection CSVs preserve particle labels; web tracking links particles within their class.
- CLI models in `trained_models_summary.yaml` are not discovered by the web interface.
- A pooled physics fit is not a class-specific estimate. Export or filter one class before interpreting its parameters.

## Command-line workflow

Run from the repository root. Train the required particle types first, then select them in `src/config.yaml`:

```yaml
samples: [Janus, Ring]
visualize: true
```

```bash
python src/detection/train_single_particle.py --particle Janus --config src/config.yaml
python src/detection/train_single_particle.py --particle Ring --config src/config.yaml
python src/detection/test_composite_model.py --config src/config.yaml
python src/detection/run_composite_pipeline.py
python src/detection/compare_models.py
```

The CLI reads model paths and model directories from `trained_models_summary.yaml`, then reads each model directory's `config.yaml`. Comparison requires existing single-model and composite test results. The composite test writes `test_composite_results_summary.yaml`; visualization destinations depend on dataset configuration under `detection_results/`.

## Python interface and parameters

```python
import sys
sys.path[:0] = ['src', 'src/detection']
from composite_model import CompositeLodeSTAR
import utils

model = CompositeLodeSTAR(
    utils.load_yaml('src/config.yaml'),
    utils.load_yaml('trained_models_summary.yaml'),
)
# image: a two-dimensional NumPy grayscale array
detections, labels, weight_maps, outputs = model.detect_and_classify(image)
# An explicit argument overrides that parameter for every member model:
detections, labels, weight_maps, outputs = model.detect_and_classify(image, cutoff=0.3)
```

Without overrides, the CLI uses each model's saved `alpha`, `beta`, `cutoff`, and `mode`; missing values fall back to `0.2`, `0.8`, `0.2`, and `constant`. Alpha and beta control detection weighting; cutoff controls the acceptance threshold. Increasing cutoff generally reduces detections. Tune on representative validation images rather than assuming one setting suits all particle types. The CLI currently fixes the clustering distance at 20 pixels; the web adapter exposes it as a setting.

For nonempty results, `detections` has shape `(N, 3)` with `[x, y, winning_weight]`; `labels` contains the corresponding particle names. `weight_maps` maps each class to an image-sized confidence map. `outputs` contains the raw model tensors, whose spatial size depends on architecture. Check `detections.size` before indexing an empty CLI result.

## Validation and troubleshooting

Check that configured sample names match summary entries and that every referenced weight file and saved configuration exists. Review warnings: the CLI can skip failed detections, whereas the web adapter rejects missing/nonfinite composite outputs. Do not assume a partially loaded ensemble is equivalent to the intended one.

Compare class labels, localization and precision/recall against representative annotated data. Similar particle appearances, overlapping objects and incomparable confidence scales can impair classification; an ensemble does not guarantee better accuracy. Reduce selected models or image size if memory is insufficient.

See the [project overview](README.md) for installation and the [command reference](docs/QUICK_REFERENCE.md) for single-model detection, orientation and tracking.
