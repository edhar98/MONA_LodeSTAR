# Visualization Update: Colored Particle Labels

## Change Summary

Updated the composite model visualization to display colored text labels for each detected and ground truth particle, making it easier to identify particle types at a glance.

## What Changed

### Before
- Detection markers were color-coded
- No text labels on detections
- Difficult to identify particle type without matching to color legend

### After
- Detection markers are color-coded (unchanged)
- **Text labels added** next to each detection showing particle type
- **Labels have matching colored borders** (red for Janus, blue for Ring, etc.)
- Ground truth markers now also show "GT:ParticleType" labels
- Labels are bold with semi-transparent black background for readability

## Visual Style

### Detection Labels
Each detected particle now displays:
- **Marker**: Colored circle at particle position
- **Text Label**: Particle type name (e.g., "Janus", "Ring")
- **Label Style**:
  - Bold font (fontsize=8, fontweight='bold')
  - Colored text matching the particle type
  - Black semi-transparent background (alpha=0.6)
  - Colored border matching the particle type (linewidth=1.5)
  - Rounded box with padding (pad=0.3)
  - Positioned at (x+5, y-5) relative to detection point

### Ground Truth Labels
Ground truth particles now display:
- **Marker**: Green circle at ground truth position
- **Text Label**: "GT:ParticleType" format
- **Label Style**:
  - Bold font (fontsize=7, fontweight='bold')
  - Colored text matching the particle type
  - White semi-transparent background (alpha=0.7)
  - Green border (linewidth=1.5)
  - Positioned at (x-15, y-5) relative to ground truth point

## Color Scheme

| Particle Type | Marker Color | Label Border Color |
|---------------|--------------|-------------------|
| Janus         | Red          | Red               |
| Ring          | Blue         | Blue              |
| Spot          | Yellow       | Yellow            |
| Ellipse       | Cyan         | Cyan              |
| Rod           | Magenta      | Magenta           |
| Ground Truth  | Green        | Green             |

## Updated Files

1. **`src/test_composite_model.py`**
   - Function: `visualize_composite_results()`
   - Added text labels for both detections and ground truth
   - Lines 210-211: Ground truth labels
   - Lines 219-220: Detection labels

2. **`src/run_composite_pipeline.py`**
   - Updated example visualization to match
   - Lines 54-55: Detection labels with colored borders

3. **Documentation**
   - `COMPOSITE_MODEL_README.md`: Updated visualization section
   - `QUICK_START_COMPOSITE.md`: Updated visualization description

## Code Implementation

### Detection Labels
```python
if len(detections) > 0:
    colors = {'Janus': 'red', 'Ring': 'blue', 'Spot': 'yellow', 
              'Ellipse': 'cyan', 'Rod': 'magenta'}
    for i, (x, y, conf) in enumerate(detections):
        label = detection_labels[i] if detection_labels else 'Unknown'
        color = colors.get(label, 'white')
        
        # Plot marker
        axes[1, 0].plot(x, y, 'o', color=color, markersize=5, 
                       markeredgecolor='white', markeredgewidth=1)
        
        # Add colored text label
        axes[1, 0].text(x + 5, y - 5, label, color=color, 
                       fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='black', alpha=0.6, 
                                edgecolor=color, linewidth=1.5))
```

### Ground Truth Labels
```python
if len(gt_bboxes) > 0:
    colors = {'Janus': 'red', 'Ring': 'blue', 'Spot': 'yellow', 
              'Ellipse': 'cyan', 'Rod': 'magenta'}
    for i, (x, y) in enumerate(gt_bboxes):
        label = gt_labels[i] if gt_labels else 'Unknown'
        
        # Plot green marker
        axes[1, 0].plot(x, y, 'go', markersize=5, 
                       markeredgecolor='white', markeredgewidth=1)
        
        # Add colored text label with green border
        color = colors.get(label, 'green')
        axes[1, 0].text(x - 15, y - 5, f'GT:{label}', color=color,
                       fontsize=7, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='white', alpha=0.7,
                                edgecolor='green', linewidth=1.5))
```

## Benefits

1. **Instant Identification**: Particle types are immediately visible without checking legend
2. **Color Consistency**: Labels use same colors as markers for intuitive understanding
3. **GT vs Detection**: Clear distinction between ground truth (GT: prefix) and detections
4. **Readability**: Black/white backgrounds ensure text is readable on any image
5. **Professional Look**: Colored borders provide polished, publication-ready visualizations

## Example Output

When visualizing a test image:
- Janus particles show **red circles** with **"Janus"** in red text with red border
- Ring particles show **blue circles** with **"Ring"** in blue text with blue border
- Ground truth shows **green circles** with **"GT:ParticleType"** in colored text with green border
- All labels have semi-transparent backgrounds for readability

## Usage

No code changes needed - the update is automatic:

```bash
# Enable visualization in config
visualize: true

# Run composite model testing
python src/test_composite_model.py --config src/config.yaml

# Or run example
python src/run_composite_pipeline.py
```

Results will show colored labels automatically in:
- `detection_results/Testing_snr_10-10/composite/`
- `detection_results/composite_example/`

## Backward Compatibility

This change is fully backward compatible:
- No API changes
- No configuration changes needed
- Existing visualization code still works
- Only visual output improved

## Testing

Verify the update works correctly:

```bash
# Test with full dataset
python src/test_composite_model.py --config src/config.yaml

# Quick example
python src/run_composite_pipeline.py
```

Check output images for:
- ✓ Colored labels next to each detection
- ✓ Labels match marker colors
- ✓ Ground truth shows "GT:" prefix
- ✓ Text is readable on dark and bright areas
- ✓ Borders match particle type colors

## Summary

The visualization enhancement adds colored, labeled text to each detection and ground truth marker, making particle identification immediate and intuitive. The professional styling with colored borders and semi-transparent backgrounds ensures readability while maintaining visual appeal.
