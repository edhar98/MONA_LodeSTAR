import os
import torch
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from deeplay import LodeSTAR
import deeptrack.deeplay as dl
import deeptrack as dt
import utils
from image_generator import generateImage, Object
import xml.etree.ElementTree as ET
from sklearn.metrics import precision_recall_curve, average_precision_score
import cv2
from scipy.spatial.distance import cdist
from scipy.ndimage import maximum_filter, generate_binary_structure
from scipy import ndimage

# Import customLodeSTAR from separate file
from custom_lodestar import customLodeSTAR

# Setup logger with file output
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f'test_single_particle_{timestamp}.log')
logger = utils.setup_logger('test_single_particle', log_file=log_file)


def load_trained_model(model_path, config):
    """Load trained LodeSTAR model"""
    
    # Create LodeSTAR model
    if config['lodestar_version'] == 'default':
        lodestar = dl.LodeSTAR(
            n_transforms=config['n_transforms'], 
            optimizer=dl.Adam(lr=config['lr'])
        ).build()
    else:
        lodestar = customLodeSTAR(
            n_transforms=config['n_transforms'], 
            optimizer=dl.Adam(lr=config['lr'])
        ).build()
    
    # Load trained weights
    if os.path.exists(model_path):
        lodestar.load_state_dict(torch.load(model_path))
        logger.info(f"Loaded model from {model_path}")
    else:
        logger.warning(f"Warning: Model not found at {model_path}")
        return None
    
    lodestar.eval()
    return lodestar


def parse_xml_annotations(xml_path):
    """Parse XML annotations to get ground truth bounding boxes and SNR"""
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    bboxes = []
    labels = []
    snr = None
    
    # Extract SNR if available
    snr_elem = root.find('snr')
    if snr_elem is not None:
        snr = float(snr_elem.text)
    
    for obj in root.findall('object'):
        label = obj.find('name').text
        bbox = obj.find('bndbox')
        
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        # Convert to center coordinates
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        
        bboxes.append([center_x, center_y])
        labels.append(label)
    
    return np.array(bboxes), labels, snr


def detect_particles(model, image, config, particle_type=None):
    """Detect particles using trained LodeSTAR model"""
    
    # Convert image to numpy array if it's not already
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    # Ensure image is 2D grayscale
    if len(image.shape) == 3:
        if image.shape[-1] == 3:  # RGB image (H, W, 3)
            image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
        elif image.shape[0] == 3:  # RGB image (3, H, W)
            image = np.dot(image[:3].transpose(1, 2, 0), [0.299, 0.587, 0.114])
        elif image.shape[-1] == 1:  # Single channel (H, W, 1)
            image = image[..., 0]
        elif image.shape[0] == 1:  # Single channel (1, H, W)
            image = image[0]
        else:
            image = image[0] if image.shape[0] < image.shape[-1] else image[..., 0]
    elif len(image.shape) > 3:
        if image.shape[1] == 3:  # (B, 3, H, W)
            image = np.dot(image[0].transpose(1, 2, 0), [0.299, 0.587, 0.114])
        else:
            image = image[0, 0] if len(image.shape) == 4 else image[0]
    
    if len(image.shape) != 2:
        raise ValueError(f"Image must be 2D after processing, got shape {image.shape}")
    
    # Prepare image for model: (H, W) -> (1, 1, H, W)
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
    
    # Get prediction from model
    with torch.no_grad():
        model_output = model(image_tensor)
        
        # Extract weights tensor (last channel)
        if len(model_output.shape) == 4 and model_output.shape[1] >= 3:
            weights = model_output[0, -1].detach().numpy()
            prediction = {'weights': weights}
        else:
            prediction = model_output[0].detach().numpy() if len(model_output.shape) == 4 else model_output.detach().numpy()
    
    # Scale weights tensor if needed
    h, w = image.shape
    if isinstance(prediction, dict) and 'weights' in prediction:
        weights_tensor = prediction['weights']
        if weights_tensor.shape != (h, w):
            prediction['weights'] = cv2.resize(weights_tensor, (w, h), interpolation=cv2.INTER_LINEAR)
    elif isinstance(prediction, np.ndarray) and prediction.shape[:2] != (h, w):
        if len(prediction.shape) == 3:
            scaled_prediction = np.zeros((prediction.shape[0], h, w))
            for c in range(prediction.shape[0]):
                scaled_prediction[c] = cv2.resize(prediction[c], (w, h), interpolation=cv2.INTER_LINEAR)
            prediction = scaled_prediction
        else:
            prediction = cv2.resize(prediction, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Get detections using model's detect method
    try:
        detections = model.detect(image_tensor, alpha=config.get('alpha', 0.2), beta=config.get('beta', 0.8), mode=config.get('mode', "constant"), cutoff=config.get('cutoff', 0.2))[0]
        
        if len(detections) > 0:
            # Swap coordinates: [y, x] -> [x, y] and cluster
            detections_xy = detections[:, [1, 0]]
            clustered_detections = cluster_nearby_detections(detections_xy, distance_threshold=20)
            detections_with_confidence = np.column_stack([clustered_detections, np.ones(len(clustered_detections))])
            detection_labels = [particle_type] * len(clustered_detections) if particle_type else None
        else:
            detections_with_confidence = np.empty((0, 3))
            detection_labels = []
    except AttributeError:
        logger.error("AttributeError: detect method doesn't exist")
        detections_with_confidence = np.empty((0, 3))
        detection_labels = []
    
    return detections_with_confidence, prediction, detection_labels, model_output


def cluster_nearby_detections(detections, distance_threshold=20):
    """
    Cluster nearby detections that likely belong to the same object.
    
    Parameters:
    -----------
    detections : np.ndarray
        Array of detections with shape (N, 2) where each row is [x, y]
    distance_threshold : float
        Maximum distance between detections to be considered the same object
    
    Returns:
    --------
    np.ndarray
        Clustered detections with shape (M, 2) where M <= N
    """
    if len(detections) <= 1:
        return detections
    
    # Calculate pairwise distances
    distances = cdist(detections, detections)
    
    # Create clusters using a simple greedy approach
    clusters = []
    used = set()
    
    for i in range(len(detections)):
        if i in used:
            continue
        
        # Start a new cluster
        cluster = [i]
        used.add(i)
        
        # Find all nearby detections
        for j in range(i + 1, len(detections)):
            if j not in used and distances[i, j] <= distance_threshold:
                cluster.append(j)
                used.add(j)
        
        clusters.append(cluster)
    
    # Compute cluster centroids
    clustered_detections = []
    for cluster in clusters:
        if len(cluster) == 1:
            # Single detection, keep as is
            clustered_detections.append(detections[cluster[0]])
        else:
            # Multiple detections, compute centroid
            centroid = np.mean(detections[cluster], axis=0)
            clustered_detections.append(centroid)
    
    return np.array(clustered_detections)


def calculate_detection_metrics(gt_bboxes, detections, gt_labels=None, detection_labels=None, iou_threshold=0.5, distance_threshold=20):
    """Calculate detection metrics using IoU and distance-based matching, considering object types"""
    
    if len(gt_bboxes) == 0 and len(detections) == 0:
        return {'precision': 1.0, 'recall': 1.0, 'f1_score': 1.0, 'tp': 0, 'fp': 0, 'fn': 0}
    
    if len(gt_bboxes) == 0:
        return {'precision': 0.0, 'recall': 1.0, 'f1_score': 0.0, 'tp': 0, 'fp': len(detections), 'fn': 0}
    
    if len(detections) == 0:
        return {'precision': 1.0, 'recall': 0.0, 'f1_score': 0.0, 'tp': 0, 'fp': 0, 'fn': len(gt_bboxes)}
    
    # Calculate distances between ground truth and detections
    gt_positions = gt_bboxes[:, :2]  # Center coordinates
    det_positions = detections[:, :2]  # Center coordinates
    
    # Calculate distance matrix
    distances = cdist(gt_positions, det_positions)
    
    # Match detections to ground truth
    matched_gt = set()
    matched_det = set()
    true_positives = 0
    
    # Find matches within distance threshold and same object type
    for gt_idx in range(len(gt_positions)):
        for det_idx in range(len(det_positions)):
            # Check distance threshold
            if distances[gt_idx, det_idx] <= distance_threshold:
                # Check object type if labels are provided
                type_match = True
                if gt_labels is not None and detection_labels is not None:
                    if gt_idx < len(gt_labels) and det_idx < len(detection_labels):
                        type_match = gt_labels[gt_idx] == detection_labels[det_idx]
                
                if type_match and gt_idx not in matched_gt and det_idx not in matched_det:
                    matched_gt.add(gt_idx)
                    matched_det.add(det_idx)
                    true_positives += 1
    
    # Calculate metrics
    false_positives = len(detections) - true_positives
    false_negatives = len(gt_bboxes) - true_positives
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'tp': true_positives,
        'fp': false_positives,
        'fn': false_negatives
    }


def evaluate_model_on_dataset(model, dataset_dir, particle_type, config):
    """Evaluate model on generated dataset with annotations"""
    
    logger.info(f"\n=== Evaluating {particle_type} model on generated dataset ===")
    
    # Find all images and annotations
    images_dir = os.path.join(dataset_dir, 'images')
    annotations_dir = os.path.join(dataset_dir, 'annotations')
    
    if not os.path.exists(images_dir) or not os.path.exists(annotations_dir):
        logger.warning(f"Dataset directories not found: {dataset_dir}")
        return None
    
    # Get all image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    
    all_metrics = []
    detection_results = []  # Store results for all images (for visualization)
    
    for i, image_file in enumerate(image_files):
        # Load image
        image_path = os.path.join(images_dir, image_file)
        image = np.array(dt.LoadImage(image_path).resolve()).astype(np.float32)
        
        # Load annotations
        annotation_file = image_file.replace('.jpg', '.xml')
        annotation_path = os.path.join(annotations_dir, annotation_file)
        
        if not os.path.exists(annotation_path):
            logger.warning(f"Annotation not found: {annotation_path}")
            continue
        
        # Parse annotations
        gt_bboxes, gt_labels, snr = parse_xml_annotations(annotation_path)
        
        # Filter ground truth to only include objects of the same type as the model
        if len(gt_bboxes) > 0 and gt_labels:
            # Find indices of objects that match the particle type
            matching_indices = [i for i, label in enumerate(gt_labels) if label == particle_type]
            
            if matching_indices:
                # Keep only the matching objects
                gt_bboxes = gt_bboxes[matching_indices]
                gt_labels = [gt_labels[i] for i in matching_indices]
            else:
                # No objects of this type in the image
                gt_bboxes = np.empty((0, 2))
                gt_labels = []
        
        # Detect particles
        detections, prediction, detection_labels, model_output = detect_particles(model, image, config, particle_type=particle_type)
        
        # Calculate metrics
        metrics = calculate_detection_metrics(gt_bboxes, detections, gt_labels=gt_labels, detection_labels=detection_labels)
        
        all_metrics.append(metrics)
        
        # Store detection results for all images (for visualization)
        detection_results.append({
            'image_file': image_file,
            'gt_bboxes': gt_bboxes,
            'gt_labels': gt_labels,
            'detections': detections,
            'detection_labels': detection_labels,
            'prediction': prediction,  
            'metrics': metrics,
            'snr': snr,
            'model_output': model_output  # Store model output for weighted prediction
        })
        
        logger.info(f"  {image_file}: TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}, "
              f"Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, SNR={snr:.2f}, "
              f"GT_{particle_type}={len(gt_bboxes)}")
    
    # Calculate average metrics
    if all_metrics:
        avg_metrics = {
            'precision': np.mean([m['precision'] for m in all_metrics]),
            'recall': np.mean([m['recall'] for m in all_metrics]),
            'f1_score': np.mean([m['f1_score'] for m in all_metrics]),
            'total_tp': sum([m['tp'] for m in all_metrics]),
            'total_fp': sum([m['fp'] for m in all_metrics]),
            'total_fn': sum([m['fn'] for m in all_metrics])
        }
        
        logger.info(f"\nAverage Metrics for {particle_type}:")
        logger.info(f"  Precision: {avg_metrics['precision']:.3f}")
        logger.info(f"  Recall: {avg_metrics['recall']:.3f}")
        logger.info(f"  F1-Score: {avg_metrics['f1_score']:.3f}")
        logger.info(f"  Total TP: {avg_metrics['total_tp']}")
        logger.info(f"  Total FP: {avg_metrics['total_fp']}")
        logger.info(f"  Total FN: {avg_metrics['total_fn']}")
        
        return avg_metrics, detection_results
    
    return None, []


def visualize_detection_results(image, gt_bboxes, detections, prediction, title="Detection Results", save_dir="detection_results", snr=None, gt_labels=None, detection_labels=None, metrics=None, model_output=None, model=None):
    """Visualize detection results with ground truth and predictions - similar to LodeStar.ipynb"""
    
    # Create figure with equal-sized subplots using a simpler approach
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'aspect': 'equal'})
    
   
    # Original image with ground truth
    axes[0].imshow(image, cmap='gray')
    if len(gt_bboxes) > 0:
        for i, (x, y) in enumerate(gt_bboxes):
            label = gt_labels[i] if gt_labels and i < len(gt_labels) else 'Unknown'
            axes[0].plot(x, y, 'go', markersize=5, markeredgecolor='white', markeredgewidth=1, label=f'GT: {label}' if i == 0 else "")
    
    # Add sample training image if available
    # particle_type = title.split('_')[0]  # Use the first label as particle type
    # sample_path = os.path.join('data', 'Samples', particle_type, f'{particle_type}.jpg')
    # if os.path.exists(sample_path):
    #     try:
    #         # Load sample training image using matplotlib instead of deeptrack
    #         sample_image = mpimg.imread(sample_path)
            
    #         # Convert to grayscale if needed
    #         if len(sample_image.shape) == 3 and sample_image.shape[-1] == 3:
    #             sample_image = np.dot(sample_image[..., :3], [0.299, 0.587, 0.114])
    #         elif len(sample_image.shape) == 3 and sample_image.shape[-1] == 4:
    #             # RGBA image
    #             sample_image = np.dot(sample_image[..., :3], [0.299, 0.587, 0.114])
            
    #         # Ensure the image is in the correct range [0, 1]
    #         if sample_image.max() > 1.0:
    #             sample_image = sample_image / 255.0
            
    #         # Create inset axes for sample image
    #         axins = inset_axes(axes[0], width=f"{sample_image.shape[1]/image.shape[1] * 100}%", height=f"{sample_image.shape[0]/image.shape[0] * 100}%", loc='lower left', borderpad=0.2)
    #         axins.imshow(sample_image, cmap='gray', aspect='equal')
    #         #axins.set_title(f'Trained \n{particle_type}', fontsize=8)
    #         axins.axis('off')
            
    #         # Add a border to the inset by setting the spine color
    #         for spine in axins.spines.values():
    #             spine.set_color('red')
    #             spine.set_linewidth(1)
            
    #     except Exception as e:
    #         logger.warning(f"Error loading sample training image: {e}")
    #         import traceback
    #         logger.warning(f"Traceback: {traceback.format_exc()}")

    axes[0].set_title(f"Ground Truth: {len(gt_bboxes)} particles")
    axes[0].axis('off')
    
    # Weighted prediction visualization
    if model_output is not None and isinstance(prediction, dict) and 'weights' in prediction:
        try:
            # Extract X, Y, and rho from model output (similar to notebook)
            if len(model_output.shape) == 4 and model_output.shape[1] >= 3:
                # model_output shape is (B, C, H, W) where C >= 3
                X = model_output[0, 0].detach().cpu().numpy()  # (H, W) - X coordinates
                Y = model_output[0, 1].detach().cpu().numpy()  # (H, W) - Y coordinates
                rho = model_output[0, -1].detach().cpu().numpy()  # (H, W) - weights/rho
                
                # Normalize rho for better visualization
                rho_normalized = (rho - rho.min()) / (rho.max() - rho.min() + 1e-8)
                
                # Show the weight map directly
                axes[1].imshow(rho_normalized, cmap='hot', aspect='equal')
                axes[1].set_title("Prediction (Weight Map)")
                
        except Exception as e:
            logger.warning(f"Error plotting weighted predictions: {e}")
            import traceback
            logger.warning(f"Traceback: {traceback.format_exc()}")
            axes[1].imshow(image, cmap='gray')
            axes[1].set_title("Prediction: Error")
    else:
        axes[1].imshow(image, cmap='gray')
        axes[1].set_title("Prediction: N/A")
    axes[1].axis('off')
    
    # Combined results - only detections and ground truth
    axes[2].imshow(image, cmap='gray')
    
    # Plot ground truth
    if len(gt_bboxes) > 0:
        for i, (x, y) in enumerate(gt_bboxes):
            label = gt_labels[i] if gt_labels and i < len(gt_labels) else 'Unknown'
            axes[2].plot(x, y, 'go', markersize=5, markeredgecolor='white', markeredgewidth=1, label=f'GT: {label}' if i == 0 else "")
    
    # Plot detections
    if len(detections) > 0:
        for i, (x, y, conf) in enumerate(detections):
            label = detection_labels[i] if detection_labels and i < len(detection_labels) else 'Unknown'
            axes[2].plot(x, y, 'ro', markersize=5, markeredgecolor='white', markeredgewidth=1, label=f'Det: {label}' if i == 0 else "")
            #axes[2].text(x + 5, y + 5, f'{conf:.2f}', color='red', fontsize=8)
    
    if metrics:
        metrics_text = f"F1: {metrics['f1_score']:.3f}\nP: {metrics['precision']:.3f}\nR: {metrics['recall']:.3f}\nTP: {metrics['tp']}\nFP: {metrics['fp']}\nFN: {metrics['fn']}"
        axes[2].text(0.02, 0.98, metrics_text, transform=axes[2].transAxes, 
                    fontsize=10, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='black'))
        
          
    axes[2].set_title("Combined Results")
    axes[2].axis('off')
    axes[2].legend(loc='upper right', fontsize=8, framealpha=0.3)
    
    # Add figure title with SNR
    fig_title_parts = [title]
    if snr is not None:
        fig_title_parts.append(f"SNR: {snr:.2f}")
    fig.suptitle(' - '.join(fig_title_parts), fontsize=14, y=0.98)
    
    # Adjust layout to ensure equal sizes
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{title.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.close()


def test_single_particle_model(particle_type, model_path, config, visualize=False):
    """Test a single particle model on generated images"""
    
    logger.info(f"\n=== Testing {particle_type} model ===")
    
    # Load model
    model = load_trained_model(model_path, config)
    if model is None:
        return None
    
    # Test on all available testing datasets
    testing_type = 'Testing' # 'SmallTesting'
    testing_dir = os.path.join(config['data_dir'], testing_type)
    #dataset_types = ['same_shape_same_size', 'different_shape_same_size']
    dataset_types = ['same_shape_same_size', 'same_shape_different_size', 
                    'different_shape_same_size', 'different_shape_different_size']
    #dataset_types = ['same_shape_same_size']
    model_id = model_path.split("/")[-2]
    results_dir = f'detection_results/{testing_type}/{particle_type}_{model_id}'
    os.makedirs(results_dir,exist_ok=True)
    all_results = {}
    
    for dataset_type in dataset_types:
        dataset_dir = os.path.join(testing_dir, dataset_type)
        save_dir = os.path.join(results_dir, f'{dataset_type}')
        os.makedirs(save_dir,exist_ok=True)
        if os.path.exists(dataset_dir):
            logger.info(f"\n--- Testing on {dataset_type} dataset ---")
            metrics, results = evaluate_model_on_dataset(model, dataset_dir, particle_type, config)
            
            if metrics:
                all_results[dataset_type] = {
                    'metrics': metrics
                }
                
                if visualize:
                    # Visualize all results from this dataset
                    for i, result in enumerate(results):
                        image_path = os.path.join(dataset_dir, 'images', result['image_file'])
                        image = np.array(dt.LoadImage(image_path).resolve())
                        
                        # Create title with metrics
                        metrics_title = f"{particle_type}_{dataset_type}_sample_{i+1}"
                        
                        visualize_detection_results(
                            image, 
                            result['gt_bboxes'], 
                            result['detections'], 
                            result['prediction'],
                            metrics_title,
                            save_dir,
                            result['snr'],
                            result['gt_labels'],
                            result['detection_labels'],
                            result['metrics'],  # Pass individual metrics for display
                            result['model_output'], # Pass model_output for weighted prediction
                            model # Pass the model instance
                        )
    
    return all_results


def main():
    """Test trained models on their respective datasets"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test LodeSTAR model for particle detection')
    parser.add_argument('--particle', type=str, help='Specific particle type to test (e.g., Janus, Ring, Spot, Ellipse, Rod)')
    parser.add_argument('--model', type=str, help='Path to specific model file to test')
    parser.add_argument('--config', type=str, default='src/config.yaml', help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = utils.load_yaml(args.config)
    
    # Log detection configuration parameters
    detection_config = {
        'alpha': config.get('alpha', 0.2),
        'beta': config.get('beta', 0.8),
        'cutoff': config.get('cutoff', 0.2),
        'mode': config.get('mode', 'constant')
    }
    logger.info(f"=== LodeSTAR Testing Started ===")
    logger.info(f"Detection configuration: alpha={detection_config['alpha']}, beta={detection_config['beta']}, cutoff={detection_config['cutoff']}, mode={detection_config['mode']}")
    
    # Load training summary
    summary_path = 'trained_models_summary.yaml'
    if not os.path.exists(summary_path):
        logger.warning(f"Training summary not found: {summary_path}")
        logger.warning("Please run training first: python src/train_single_particle.py")
        return
    
    trained_models = utils.load_yaml(summary_path)
    
    # Determine which models to test
    if args.particle and args.model:
        # Test specific particle type with specific model
        if args.particle not in trained_models:
            logger.error(f"Particle type '{args.particle}' not found in training summary")
            return
        
        if not os.path.exists(args.model):
            logger.error(f"Model file not found: {args.model}")
            return
        
        logger.info(f"Testing {args.particle} with specific model: {args.model}")
        test_results = {}
        metrics = test_single_particle_model(args.particle, args.model, config)
        if metrics:
            test_results[args.particle] = metrics
            logger.info(f"Successfully tested {args.particle} model")
        else:
            logger.warning(f"Failed to test {args.particle} model")
            return
        
    elif args.particle:
        # Test specific particle type using model from training summary
        if args.particle not in trained_models:
            logger.error(f"Particle type '{args.particle}' not found in training summary")
            return
        
        logger.info(f"Testing {args.particle} model from training summary")
        test_results = {}
        model_info = trained_models[args.particle]
        model_path = model_info['model_path']
        
        if os.path.exists(model_path):
            metrics = test_single_particle_model(args.particle, model_path, config, visualize=config.get('visualize', False))
            if metrics:
                test_results[args.particle] = metrics
                logger.info(f"Successfully tested {args.particle} model")
            else:
                logger.warning(f"Failed to test {args.particle} model")
                return
        else:
            logger.error(f"Model not found: {model_path}")
            return
    
    elif args.model:
        # Test specific model file (need to determine particle type)
        if not os.path.exists(args.model):
            logger.error(f"Model file not found: {args.model}")
            return
        
        # Try to determine particle type from model path or ask user
        particle_type = None
        for pt in trained_models.keys():
            if pt.lower() in args.model.lower():
                particle_type = pt
                break
        
        if not particle_type:
            logger.error("Could not determine particle type from model path. Please specify --particle")
            return
        
        logger.info(f"Testing {particle_type} with model: {args.model}")
        test_results = {}
        metrics = test_single_particle_model(particle_type, args.model, config, visualize=config.get('visualize', False))
        if metrics:
            test_results[particle_type] = metrics
            logger.info(f"Successfully tested {particle_type} model")
        else:
            logger.warning(f"Failed to test {particle_type} model")
            return
    
    else:
        # Test all models (default behavior)
        logger.info("Testing all trained models")
        test_results = {}
        
        for particle_type, model_info in trained_models.items():
            model_path = model_info['model_path']
            
            if os.path.exists(model_path):
                metrics = test_single_particle_model(particle_type, model_path, config, visualize=config.get('visualize', False))
                if metrics:
                    test_results[particle_type] = metrics
                    logger.info(f"Successfully tested {particle_type} model")
                else:
                    logger.warning(f"Failed to test {particle_type} model")
            else:
                logger.warning(f"Model not found: {model_path}")
    
    # Save test results
    if test_results:
        results_path = 'test_results_summary.yaml'
        try:
            utils.save_yaml(test_results, results_path)
            logger.info(f"\nTest results saved to {results_path}")
        except Exception as e:
            logger.error(f"Error saving test results: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Print summary
        logger.info(f"\n=== Test Results Summary ===")
        for particle_type, dataset_results in test_results.items():
            logger.info(f"\n{particle_type}:")
            
            # Calculate overall metrics across all datasets
            all_metrics = []
            for dataset_type, result in dataset_results.items():
                if result and 'metrics' in result:
                    all_metrics.append(result['metrics'])
            
            if all_metrics:
                # Calculate average metrics across all datasets
                avg_precision = np.mean([m['precision'] for m in all_metrics])
                avg_recall = np.mean([m['recall'] for m in all_metrics])
                avg_f1 = np.mean([m['f1_score'] for m in all_metrics])
                total_tp = sum([m['total_tp'] for m in all_metrics])
                total_fp = sum([m['total_fp'] for m in all_metrics])
                total_fn = sum([m['total_fn'] for m in all_metrics])
                
                logger.info(f"  Overall Precision: {avg_precision:.3f}")
                logger.info(f"  Overall Recall: {avg_recall:.3f}")
                logger.info(f"  Overall F1-Score: {avg_f1:.3f}")
                logger.info(f"  Total TP: {total_tp}")
                logger.info(f"  Total FP: {total_fp}")
                logger.info(f"  Total FN: {total_fn}")
                
                # Print per-dataset results
                for dataset_type, result in dataset_results.items():
                    if result and 'metrics' in result:
                        metrics = result['metrics']
                        logger.info(f"    {dataset_type}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}")
            else:
                logger.warning(f"  No valid metrics found for {particle_type}")
    
    else:
        logger.warning("No models were successfully tested")


if __name__ == '__main__':
    main() 