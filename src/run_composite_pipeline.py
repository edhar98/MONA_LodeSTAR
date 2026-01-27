import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import deeptrack as dt
import utils
from composite_model import CompositeLodeSTAR

_SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT: str = os.path.dirname(_SCRIPT_DIR)
_DEFAULT_CONFIG_PATH: str = os.path.join(_SCRIPT_DIR, 'config.yaml')
_DEFAULT_SUMMARY_PATH: str = os.path.join(_REPO_ROOT, 'trained_models_summary.yaml')


def run_composite_detection_example(visualize: bool = False) -> None:
    config = utils.load_yaml(_DEFAULT_CONFIG_PATH)
    trained_models = utils.load_yaml(_DEFAULT_SUMMARY_PATH)
    
    print("=== Composite LodeSTAR Detection Example ===")
    print(f"Loading models for particle types: {list(trained_models.keys())}")
    
    composite = CompositeLodeSTAR(config, trained_models)
    print(f"Composite model loaded with {len(composite.models)} models")
    
    file_name = 'image_0012.jpg'
    test_image_path = os.path.join(_REPO_ROOT, 'data', 'Testing_snr_10-10', 'different_shape_same_size', 'images', file_name)
    annotation_path = os.path.join(_REPO_ROOT, 'data', 'Testing_snr_10-10', 'different_shape_same_size', 'annotations', file_name.replace('.jpg', '.xml'))
    
    gt_bboxes, gt_labels, _ = utils.parse_xml_annotations(annotation_path)
    
    image = np.array(dt.LoadImage(test_image_path).resolve())
    print("Running composite detection and classification...")
    print("Using model-specific detection parameters from each model's config...")
    detections, labels, weight_maps, _ = composite.detect_and_classify(image)

    for i, (detection, label) in enumerate(zip(detections, labels)):
        print(f"Detection {i+1}: {label} - x: {detection[0]}, y: {detection[1]}, confidence: {detection[2]}")

    metrics = utils.calculate_detection_metrics(gt_bboxes, detections, gt_labels=gt_labels, detection_labels=labels)
    print(f"\nDetection Metrics: \n\tPrecision: {metrics['precision']:.3f}\n\tRecall: {metrics['recall']:.3f}\n\tF1-Score: {metrics['f1_score']:.3f}\n\tTP: {metrics['tp']}\n\tFP: {metrics['fp']}\n\tFN: {metrics['fn']}")
    print(f"\nTotal detections: {len(detections)}")
    
    n_models = len(weight_maps)
    if visualize:
        # Add an extra column for the colorbar, so its axis has same size as all other axes
        fig, axes = plt.subplots(1, n_models + 2, figsize=(5 * (n_models + 1), 5), gridspec_kw={'width_ratios': [1] * (n_models + 1) + [0.07]})

        # axes layout: [detections, weight_map_1, ..., weight_map_n, colorbar_axis]
        img_axis = axes[0]
        cbar_axis = axes[-1]
        weight_map_axes = axes[1:n_models + 1]

        img_axis.imshow(image, cmap='gray')
        
        if len(gt_bboxes) > 0:
            colors = {'Janus': 'red', 'Ring': 'blue', 'Spot': 'yellow', 'Ellipse': 'cyan', 'Rod': 'magenta'}
            for i, (x, y) in enumerate(gt_bboxes):
                label = gt_labels[i] if gt_labels and i < len(gt_labels) else 'Unknown'
                img_axis.plot(x, y, 'go', markersize=5, markeredgecolor='white', markeredgewidth=1)
                color = colors.get(label, 'green')
                img_axis.text(x - 15, y - 5, f'GT:{label}', color=color, fontsize=7, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='green', linewidth=0))
    
        if len(detections) > 0:
            colors = {'Janus': 'red', 'Ring': 'blue', 'Spot': 'yellow', 'Ellipse': 'cyan', 'Rod': 'magenta'}
            for (x, y, _), label in zip(detections, labels):
                color = colors.get(label, 'white')
                img_axis.plot(x, y, 'o', color=color, markersize=10, markeredgecolor='white', markeredgewidth=2)
                img_axis.text(x + 5, y - 5, label, color=color, fontsize=7, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5, edgecolor=color, linewidth=1.5))
        img_axis.set_title(f"Detections ({len(detections)} particles)")
        img_axis.axis('off')

        vmin = 0.0  # for normalized weight maps
        vmax = 1.0

        im_list = []
        for ax, (particle_type, weight_map) in zip(weight_map_axes, weight_maps.items()):
            im = ax.imshow(weight_map, cmap='hot', vmin=vmin, vmax=vmax)
            ax.set_title(f"{particle_type} Weight Map")
            ax.axis('off')
            im_list.append(im)

        # Add colorbar in its own axis (with same height and width as other axes)
        plt.tight_layout()
        plt.colorbar(im_list[-1], cax=cbar_axis, label='Weight', orientation='vertical')

        output_dir = 'detection_results/composite_example'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'example_detection.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_path}")
        plt.close()
    
    print("\nWeight Map Statistics:")
    for particle_type, weight_map in weight_maps.items():
        print(f"  {particle_type}: min={weight_map.min():.3f}, max={weight_map.max():.3f}, mean={weight_map.mean():.3f}")


if __name__ == '__main__':
    run_composite_detection_example(True)

