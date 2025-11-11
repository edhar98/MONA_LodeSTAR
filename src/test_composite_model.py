import os
import torch
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import deeptrack as dt
import utils
from composite_model import CompositeLodeSTAR

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f'test_composite_model_{timestamp}.log')
logger = utils.setup_logger('test_composite_model', log_file=log_file)


def evaluate_composite_model_on_dataset(composite_model, dataset_dir, config):
    logger.info(f"\n=== Evaluating composite model on dataset ===")
    
    images_dir = os.path.join(dataset_dir, 'images')
    annotations_dir = os.path.join(dataset_dir, 'annotations')
    
    if not os.path.exists(images_dir) or not os.path.exists(annotations_dir):
        logger.warning(f"Dataset directories not found: {dataset_dir}")
        return None
    
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    image_files.sort()
    
    all_metrics = []
    detection_results = []
    
    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        image = np.array(dt.LoadImage(image_path).resolve()).astype(np.float32)
        
        annotation_file = image_file.replace('.jpg', '.xml')
        annotation_path = os.path.join(annotations_dir, annotation_file)
        
        if not os.path.exists(annotation_path):
            logger.warning(f"Annotation not found: {annotation_path}")
            continue
        
        gt_bboxes, gt_labels, snr = utils.parse_xml_annotations(annotation_path)
        
        detections, detection_labels, weight_maps, model_outputs = composite_model.detect_and_classify(image)
        
        metrics = utils.calculate_detection_metrics(gt_bboxes, detections, gt_labels=gt_labels, detection_labels=detection_labels)
        
        all_metrics.append(metrics)
        
        detection_results.append({
            'image_file': image_file,
            'gt_bboxes': gt_bboxes,
            'gt_labels': gt_labels,
            'detections': detections,
            'detection_labels': detection_labels,
            'weight_maps': weight_maps,
            'metrics': metrics,
            'snr': snr,
            'model_outputs': model_outputs
        })
        
        logger.info(f"  {image_file}: TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}, "
              f"Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, SNR={snr:.2f}")
    
    if all_metrics:
        avg_metrics = {
            'precision': np.mean([m['precision'] for m in all_metrics]),
            'recall': np.mean([m['recall'] for m in all_metrics]),
            'f1_score': np.mean([m['f1_score'] for m in all_metrics]),
            'total_tp': sum([m['tp'] for m in all_metrics]),
            'total_fp': sum([m['fp'] for m in all_metrics]),
            'total_fn': sum([m['fn'] for m in all_metrics])
        }
        
        logger.info(f"\nAverage Metrics:")
        logger.info(f"  Precision: {avg_metrics['precision']:.3f}")
        logger.info(f"  Recall: {avg_metrics['recall']:.3f}")
        logger.info(f"  F1-Score: {avg_metrics['f1_score']:.3f}")
        logger.info(f"  Total TP: {avg_metrics['total_tp']}")
        logger.info(f"  Total FP: {avg_metrics['total_fp']}")
        logger.info(f"  Total FN: {avg_metrics['total_fn']}")
        
        return avg_metrics, detection_results
    
    return None, []


def visualize_composite_results(image, gt_bboxes, detections, weight_maps, title="Composite Detection", save_dir="detection_results", snr=None, gt_labels=None, detection_labels=None, metrics=None, particle_types=None):
    n_models = len(weight_maps)
    fig, axes = plt.subplots(2, n_models + 1, figsize=(5 * (n_models + 1), 10))
    
    if n_models == 1:
        axes = axes.reshape(2, -1)
    
    axes[0, 0].imshow(image, cmap='gray')
    if len(gt_bboxes) > 0:
        for i, (x, y) in enumerate(gt_bboxes):
            label = gt_labels[i] if gt_labels and i < len(gt_labels) else 'Unknown'
            axes[0, 0].plot(x, y, 'go', markersize=5, markeredgecolor='white', markeredgewidth=1)
    axes[0, 0].set_title(f"Ground Truth: {len(gt_bboxes)} particles")
    axes[0, 0].axis('off')
    
    for idx, (particle_type, weight_map) in enumerate(weight_maps.items()):
        weight_normalized = (weight_map - weight_map.min()) / (weight_map.max() - weight_map.min() + 1e-8)
        axes[0, idx + 1].imshow(weight_normalized, cmap='hot')
        axes[0, idx + 1].set_title(f"{particle_type} Weights")
        axes[0, idx + 1].axis('off')
    
    axes[1, 0].imshow(image, cmap='gray')
    
    if len(gt_bboxes) > 0:
        colors = {'Janus': 'red', 'Ring': 'blue', 'Spot': 'yellow', 'Ellipse': 'cyan', 'Rod': 'magenta'}
        for i, (x, y) in enumerate(gt_bboxes):
            label = gt_labels[i] if gt_labels and i < len(gt_labels) else 'Unknown'
            axes[1, 0].plot(x, y, 'go', markersize=5, markeredgecolor='white', markeredgewidth=1)
            color = colors.get(label, 'green')
            axes[1, 0].text(x - 15, y - 5, f'GT:{label}', color=color, fontsize=7, fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='green', linewidth=1.5))
    
    if len(detections) > 0:
        colors = {'Janus': 'red', 'Ring': 'blue', 'Spot': 'yellow', 'Ellipse': 'cyan', 'Rod': 'magenta'}
        for i, (x, y, conf) in enumerate(detections):
            label = detection_labels[i] if detection_labels and i < len(detection_labels) else 'Unknown'
            color = colors.get(label, 'white')
            axes[1, 0].plot(x, y, 'o', color=color, markersize=5, markeredgecolor='white', markeredgewidth=1)
            axes[1, 0].text(x + 5, y - 5, label, color=color, fontsize=8, fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6, edgecolor=color, linewidth=1.5))
    
    if metrics:
        metrics_text = f"F1: {metrics['f1_score']:.3f}\nP: {metrics['precision']:.3f}\nR: {metrics['recall']:.3f}\nTP: {metrics['tp']}\nFP: {metrics['fp']}\nFN: {metrics['fn']}"
        axes[1, 0].text(0.02, 0.98, metrics_text, transform=axes[1, 0].transAxes, 
                    fontsize=10, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='black'))
    
    axes[1, 0].set_title("Combined Results")
    axes[1, 0].axis('off')
    
    for idx in range(1, n_models + 1):
        axes[1, idx].axis('off')
    
    fig_title_parts = [title]
    if snr is not None:
        fig_title_parts.append(f"SNR: {snr:.2f}")
    fig.suptitle(' - '.join(fig_title_parts), fontsize=14, y=0.98)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{title.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.close()


def test_composite_model(config):
    logger.info(f"\n=== Testing Composite LodeSTAR Model ===")
    
    summary_path = 'trained_models_summary.yaml'
    if not os.path.exists(summary_path):
        logger.error(f"Training summary not found: {summary_path}")
        return None
    
    trained_models = utils.load_yaml(summary_path)
    
    composite_model = CompositeLodeSTAR(config, trained_models)
    logger.info(f"Loaded composite model with {len(composite_model.models)} particle types: {composite_model.particle_types}")
    
    testing_type = 'Testing_snr_10-10'
    testing_dir = os.path.join(config['data_dir'], testing_type)
    dataset_types = ['same_shape_same_size', 'same_shape_different_size', 
                     'different_shape_same_size', 'different_shape_different_size']
    
    results_dir = f'detection_results/{testing_type}/composite'
    os.makedirs(results_dir, exist_ok=True)
    all_results = {}
    
    for dataset_type in dataset_types:
        dataset_dir = os.path.join(testing_dir, dataset_type)
        save_dir = os.path.join(results_dir, f'{dataset_type}')
        os.makedirs(save_dir, exist_ok=True)
        
        if os.path.exists(dataset_dir):
            logger.info(f"\n--- Testing on {dataset_type} dataset ---")
            metrics, results = evaluate_composite_model_on_dataset(composite_model, dataset_dir, config)
            
            if metrics:
                all_results[dataset_type] = {'metrics': metrics}
                
                if config.get('visualize', False):
                    for result in results:
                        image_path = os.path.join(dataset_dir, 'images', result['image_file'])
                        image = np.array(dt.LoadImage(image_path).resolve())
                        
                        base_filename = os.path.splitext(result['image_file'])[0]
                        metrics_title = f"composite_{dataset_type}_{base_filename}"
                        
                        visualize_composite_results(
                            image, 
                            result['gt_bboxes'], 
                            result['detections'], 
                            result['weight_maps'],
                            metrics_title,
                            save_dir,
                            result['snr'],
                            result['gt_labels'],
                            result['detection_labels'],
                            result['metrics'],
                            composite_model.particle_types
                        )
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Test Composite LodeSTAR model for multi-class particle detection')
    parser.add_argument('--config', type=str, default='src/config.yaml', help='Path to configuration file')
    args = parser.parse_args()
    
    config = utils.load_yaml(args.config)
    
    logger.info(f"=== Composite LodeSTAR Testing Started ===")
    logger.info(f"Using model-specific detection parameters from each model's config")
    
    test_results = test_composite_model(config)
    
    if test_results:
        results_path = 'test_composite_results_summary.yaml'
        try:
            utils.save_yaml(test_results, results_path)
            logger.info(f"\nTest results saved to {results_path}")
        except Exception as e:
            logger.error(f"Error saving test results: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        logger.info(f"\n=== Composite Test Results Summary ===")
        for dataset_type, result in test_results.items():
            if result and 'metrics' in result:
                metrics = result['metrics']
                logger.info(f"\n{dataset_type}:")
                logger.info(f"  Precision: {metrics['precision']:.3f}")
                logger.info(f"  Recall: {metrics['recall']:.3f}")
                logger.info(f"  F1-Score: {metrics['f1_score']:.3f}")
                logger.info(f"  Total TP: {metrics['total_tp']}")
                logger.info(f"  Total FP: {metrics['total_fp']}")
                logger.info(f"  Total FN: {metrics['total_fn']}")
    else:
        logger.warning("No models were successfully tested")


if __name__ == '__main__':
    main()

