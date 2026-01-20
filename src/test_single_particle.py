import os
import torch
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import deeptrack.deeplay as dl
import deeptrack as dt
import utils
import cv2

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

def detect_particles(model, image, config, particle_type=None, detection_mode='standard'):
    image = utils.preprocess_image(image)
    h, w = image.shape
    
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
    
    with torch.no_grad():
        model_output = model(image_tensor)
        
        if len(model_output.shape) == 4 and model_output.shape[1] >= 3:
            weights = model_output[0, -1].detach().numpy()
            if weights.shape != (h, w):
                weights = cv2.resize(weights, (w, h), interpolation=cv2.INTER_LINEAR)
            prediction = {'weights': weights}
        else:
            weights = None
            prediction = model_output[0].detach().numpy() if len(model_output.shape) == 4 else model_output.detach().numpy()
        
        if detection_mode == 'area':
            area_config = config.get('area_detection', {})
            clustered_detections = utils.detect_by_area(
                weights,
                cutoff=config.get('cutoff', 0.9),
                min_area=area_config.get('min_area', 100),
                max_area=area_config.get('max_area', 2500)
            )
        else:
            try:
                detections = model.detect(
                    image_tensor, 
                    alpha=config.get('alpha', 0.2), 
                    beta=config.get('beta', 0.8), 
                    mode=config.get('mode', 'constant'), 
                    cutoff=config.get('cutoff', 0.2)
                )[0]
                
                if len(detections) > 0:
                    detections_xy = detections[:, [1, 0]]
                    clustered_detections = utils.cluster_nearby_detections(detections_xy, distance_threshold=20)
                else:
                    clustered_detections = np.empty((0, 2))
            except AttributeError:
                logger.error("Model detect method not available")
                clustered_detections = np.empty((0, 2))
    
    if len(clustered_detections) > 0:
        detections_with_confidence = np.column_stack([clustered_detections, np.ones(len(clustered_detections))])
        detection_labels = [particle_type] * len(clustered_detections) if particle_type else None
    else:
        detections_with_confidence = np.empty((0, 3))
        detection_labels = []
    
    return detections_with_confidence, prediction, detection_labels, model_output


def parse_image_filename(image_file):
    base_name = os.path.splitext(image_file)[0]
    parts = base_name.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        video_id = parts[0]
        frame_idx = int(parts[1]) - 1
        return video_id, frame_idx
    return None, None


def save_detections_to_csv(detection_results, save_dir):
    detections_by_video = {}
    
    for result in detection_results:
        video_id, frame_idx = parse_image_filename(result['image_file'])
        if video_id is None:
            continue
        
        if video_id not in detections_by_video:
            detections_by_video[video_id] = []
        
        if len(result['detections']) > 0:
            for det in result['detections']:
                detections_by_video[video_id].append({
                    'x': det[0],
                    'y': det[1],
                    'confidence': det[2] if len(det) > 2 else 1.0,
                    'frame': frame_idx
                })
    
    csv_dir = os.path.join(save_dir, 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    
    for video_id, detections in detections_by_video.items():
        if detections:
            df = pd.DataFrame(detections)
            df = df.sort_values('frame').reset_index(drop=True)
            csv_path = os.path.join(csv_dir, f'{video_id}_detections.csv')
            df.to_csv(csv_path)
            logger.info(f"Saved detections CSV: {csv_path} ({len(df)} detections)")


def evaluate_model_on_dataset(model, dataset_dir, particle_type, config, detection_mode='standard'):
    logger.info(f"\n=== Evaluating {particle_type} model on generated dataset ===")
    
    images_dir = os.path.join(dataset_dir, 'images')
    annotations_dir = os.path.join(dataset_dir, 'annotations')
    csv_dir = os.path.join(dataset_dir, 'csv')
    
    gt_from_csv = False
    csv_gt_by_video = {}
    
    if os.path.exists(csv_dir):
        csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
        if csv_files:
            gt_from_csv = True
            for csv_file in csv_files:
                video_id = csv_file.replace('_video.csv', '')
                csv_path = os.path.join(csv_dir, csv_file)
                csv_gt_by_video[video_id] = utils.load_csv_ground_truth(csv_path)
            logger.info(f"Loaded CSV ground truth for {len(csv_gt_by_video)} videos")
    
    if not gt_from_csv and (not os.path.exists(images_dir) or not os.path.exists(annotations_dir)):
        logger.warning(f"Dataset directories not found: {dataset_dir}")
        return None, []
    
    if gt_from_csv:
        image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.tif', '.tiff')) and not f.endswith('.mp4')])
    else:
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
    
    all_metrics = []
    detection_results = []
    
    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        image = np.array(dt.LoadImage(image_path).resolve()).astype(np.float32)
        
        if gt_from_csv:
            video_id, frame_idx = parse_image_filename(image_file)
            if video_id and video_id in csv_gt_by_video and frame_idx in csv_gt_by_video[video_id]:
                gt_bboxes = csv_gt_by_video[video_id][frame_idx]['positions']
                gt_labels = [particle_type] * len(gt_bboxes)
                snr = None
            else:
                gt_bboxes = np.empty((0, 2))
                gt_labels = []
                snr = None
        else:
            annotation_file = image_file.replace('.jpg', '.xml')
            annotation_path = os.path.join(annotations_dir, annotation_file)
            
            if not os.path.exists(annotation_path):
                logger.warning(f"Annotation not found: {annotation_path}")
                continue
            
            gt_bboxes, gt_labels, snr = utils.parse_xml_annotations(annotation_path)
            
            if len(gt_bboxes) > 0 and gt_labels:
                matching_indices = [idx for idx, label in enumerate(gt_labels) if label == particle_type]
                if matching_indices:
                    gt_bboxes = gt_bboxes[matching_indices]
                    gt_labels = [gt_labels[idx] for idx in matching_indices]
                else:
                    gt_bboxes = np.empty((0, 2))
                    gt_labels = []
        
        detections, prediction, detection_labels, model_output = detect_particles(
            model, image, config, particle_type=particle_type, detection_mode=detection_mode
        )
        
        metrics = utils.calculate_detection_metrics(gt_bboxes, detections, gt_labels=gt_labels, detection_labels=detection_labels)
        all_metrics.append(metrics)
        
        detection_results.append({
            'image_file': image_file,
            'gt_bboxes': gt_bboxes,
            'gt_labels': gt_labels,
            'detections': detections,
            'detection_labels': detection_labels,
            'prediction': prediction,  
            'metrics': metrics,
            'snr': snr,
            'model_output': model_output
        })
        
        snr_str = f"SNR={snr:.2f}" if snr is not None else "SNR=N/A"
        logger.info(f"  {image_file}: TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}, "
              f"Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, {snr_str}, "
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


def visualize_detection_results(image, gt_bboxes, detections, prediction, title="Detection Results", 
                                save_dir="detection_results", snr=None, gt_labels=None, 
                                detection_labels=None, metrics=None, model_output=None, model=None):
    image = utils.preprocess_image(image)
    
    weightmaps_dir = os.path.join(save_dir, 'weightmaps')
    detections_dir = os.path.join(save_dir, 'detections')
    os.makedirs(weightmaps_dir, exist_ok=True)
    os.makedirs(detections_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'aspect': 'equal'})
    
    axes[0].imshow(image, cmap='gray')
    if len(gt_bboxes) > 0:
        axes[0].plot(gt_bboxes[:, 0], gt_bboxes[:, 1], 'go', markersize=5, 
                     markeredgecolor='white', markeredgewidth=1, label='GT')
    axes[0].set_title(f"Ground Truth: {len(gt_bboxes)} particles")
    axes[0].axis('off')
    
    if isinstance(prediction, dict) and 'weights' in prediction:
        weights = prediction['weights']
        weights_norm = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
        axes[1].imshow(weights_norm, cmap='hot', aspect='equal')
        axes[1].set_title("Weight Map")
    else:
        axes[1].imshow(image, cmap='gray')
        axes[1].set_title("Weight Map: N/A")
    axes[1].axis('off')
    
    axes[2].imshow(image, cmap='gray')
    if len(gt_bboxes) > 0:
        axes[2].plot(gt_bboxes[:, 0], gt_bboxes[:, 1], 'go', markersize=5, 
                     markeredgecolor='white', markeredgewidth=1, label='GT')
    if len(detections) > 0:
        axes[2].plot(detections[:, 0], detections[:, 1], 'ro', markersize=5, 
                     markeredgecolor='white', markeredgewidth=1, label='Det')
    
    if metrics:
        metrics_text = f"F1: {metrics['f1_score']:.3f}\nP: {metrics['precision']:.3f}\nR: {metrics['recall']:.3f}\nTP: {metrics['tp']} FP: {metrics['fp']} FN: {metrics['fn']}"
        axes[2].text(0.02, 0.98, metrics_text, transform=axes[2].transAxes, 
                    fontsize=9, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    axes[2].set_title("Combined Results")
    axes[2].axis('off')
    axes[2].legend(loc='upper right', fontsize=8, framealpha=0.3)
    
    fig_title = title if snr is None else f"{title} - SNR: {snr:.2f}"
    fig.suptitle(fig_title, fontsize=14, y=0.98)
    
    plt.tight_layout()
    plt.savefig(os.path.join(weightmaps_dir, f'{title.replace(" ", "_")}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    det_only_path = os.path.join(detections_dir, f'{title.replace(" ", "_")}.png')
    utils.save_image_with_detections(image, detections, det_only_path, gt_bboxes=gt_bboxes)


def test_single_particle_model(particle_type, model_path, config, visualize=False, 
                               detection_mode='standard', generate_video=False):
    logger.info(f"\n=== Testing {particle_type} model (mode: {detection_mode}) ===")
    
    model = load_trained_model(model_path, config)
    if model is None:
        return None
    
    testing_type = config.get('testing_type', 'JP_FE/wf_2_40')
    testing_dir = os.path.join(config['data_dir'], testing_type)
    dataset_types = config.get('dataset_types', ['01'])
    
    model_id = model_path.split("/")[-2]
    results_dir = f'detection_results/{testing_type}/{particle_type}_{model_id}'
    os.makedirs(results_dir, exist_ok=True)
    all_results = {}
    
    for dataset_type in dataset_types:
        dataset_dir = os.path.join(testing_dir, dataset_type)
        save_dir = os.path.join(results_dir, f'{dataset_type}')
        os.makedirs(save_dir, exist_ok=True)
        
        if os.path.exists(dataset_dir):
            logger.info(f"\n--- Testing on {dataset_type} dataset ---")
            metrics, results = evaluate_model_on_dataset(
                model, dataset_dir, particle_type, config, detection_mode=detection_mode
            )
            
            if metrics:
                all_results[dataset_type] = {'metrics': metrics}
                
                save_detections_to_csv(results, save_dir)
                
                if visualize:
                    for result in results:
                        image_path = os.path.join(dataset_dir, 'images', result['image_file'])
                        image = np.array(dt.LoadImage(image_path).resolve())
                        
                        base_filename = os.path.splitext(result['image_file'])[0]
                        metrics_title = f"{particle_type}_{dataset_type}_{base_filename}"
                        
                        visualize_detection_results(
                            image, result['gt_bboxes'], result['detections'], 
                            result['prediction'], metrics_title, save_dir,
                            result['snr'], result['gt_labels'], result['detection_labels'],
                            result['metrics'], result['model_output'], model
                        )
                
                if generate_video and len(results) > 1:
                    det_dir = os.path.join(save_dir, 'detections')
                    video_path = os.path.join(results_dir, f'{particle_type}_{dataset_type}_detections.mp4')
                    try:
                        utils.create_video_from_detections(det_dir, video_path, fps=config.get('fps', 30))
                        logger.info(f"Video saved: {video_path}")
                    except Exception as e:
                        logger.warning(f"Failed to create video: {e}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Test LodeSTAR model for particle detection')
    parser.add_argument('--particle', type=str, help='Specific particle type to test')
    parser.add_argument('--model', type=str, help='Path to specific model file to test')
    parser.add_argument('--config', type=str, default='src/config.yaml', help='Path to configuration file')
    parser.add_argument('--detection-mode', type=str, default='standard', choices=['standard', 'area'],
                        help='Detection mode: standard (local maxima) or area (area-based filtering)')
    parser.add_argument('--video', action='store_true', help='Generate video from detection results')
    args = parser.parse_args()
    
    config = utils.load_yaml(args.config)
    detection_mode = args.detection_mode
    generate_video = args.video
    
    logger.info(f"=== LodeSTAR Testing Started ===")
    logger.info(f"Detection mode: {detection_mode}")
    if detection_mode == 'area':
        area_cfg = config.get('area_detection', {})
        logger.info(f"Area params: min_area={area_cfg.get('min_area', 100)}, max_area={area_cfg.get('max_area', 2500)}, cutoff={config.get('cutoff', 0.9)}")
    else:
        logger.info(f"Detection params: alpha={config.get('alpha', 0.2)}, beta={config.get('beta', 0.8)}, cutoff={config.get('cutoff', 0.2)}")
    
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
        metrics = test_single_particle_model(args.particle, args.model, config, 
                                             visualize=config.get('visualize', False),
                                             detection_mode=detection_mode, generate_video=generate_video)
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
            metrics = test_single_particle_model(args.particle, model_path, config, 
                                                 visualize=config.get('visualize', False),
                                                 detection_mode=detection_mode, generate_video=generate_video)
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
        metrics = test_single_particle_model(particle_type, args.model, config, 
                                             visualize=config.get('visualize', False),
                                             detection_mode=detection_mode, generate_video=generate_video)
        if metrics:
            test_results[particle_type] = metrics
            logger.info(f"Successfully tested {particle_type} model")
        else:
            logger.warning(f"Failed to test {particle_type} model")
            return
    
    else:
        logger.info("Testing all trained models")
        test_results = {}
        
        for particle_type, model_info in trained_models.items():
            model_path = model_info['model_path']
            
            if os.path.exists(model_path):
                metrics = test_single_particle_model(particle_type, model_path, config, 
                                                     visualize=config.get('visualize', False),
                                                     detection_mode=detection_mode, generate_video=generate_video)
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