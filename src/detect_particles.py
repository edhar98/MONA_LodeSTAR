import os
import torch
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import deeptrack.deeplay as dl
import deeptrack as dt
import utils
import cv2
from custom_lodestar import customLodeSTAR

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f'detect_particles_{timestamp}.log')
logger = utils.setup_logger('detect_particles', log_file=log_file)


def load_trained_model(model_path: str, config: dict):
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
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    lodestar.load_state_dict(torch.load(model_path))
    logger.info(f"Loaded model from {model_path}")
    lodestar.eval()
    return lodestar




def detect_particles(model, image: np.ndarray, config: dict, 
                     detection_mode: str = 'standard') -> tuple:
    image = utils.preprocess_image(image)
    h, w = image.shape
    
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
    
    with torch.no_grad():
        model_output = model(image_tensor)
        
        if len(model_output.shape) == 4 and model_output.shape[1] >= 3:
            weights = model_output[0, -1].detach().numpy()
            if weights.shape != (h, w):
                weights = cv2.resize(weights, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            weights = None
        
        if detection_mode == 'area':
            area_config = config.get('area_detection', {})
            clustered_detections = utils.detect_by_area(
                weights,
                cutoff=config.get('cutoff', 0.9),
                min_area=area_config.get('min_area', 100),
                max_area=area_config.get('max_area', 2500)
            )
            logger.info(f"Area detection: found {len(clustered_detections)} particles")
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
    
    return clustered_detections, weights, model_output


def save_image_with_detections(image: np.ndarray, detections: np.ndarray, save_path: str,
                               marker_color: tuple = (255, 0, 0), marker_radius: int = 3, 
                               marker_thickness: int = 1):
    utils.save_image_with_detections(image, detections, save_path, 
                                      det_color=marker_color, marker_radius=marker_radius,
                                      marker_thickness=marker_thickness)
    logger.info(f"Saved image with detections: {save_path}")


def visualize_detections(image: np.ndarray, detections: np.ndarray, weights: np.ndarray, 
                         title: str, save_path: str, cutoff: float = 0.9):
    image = utils.preprocess_image(image)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(image, cmap='gray')
    if len(detections) > 0:
        for x, y in detections:
            axes[0].plot(x, y, 'ro', markersize=2, markeredgecolor='white', markeredgewidth=0.5)
    axes[0].set_title(f"Detections: {len(detections)} particles")
    axes[0].axis('off')

    if weights is not None:
        axes[1].imshow(weights, cmap='hot', vmin=0, vmax=1)
        axes[1].set_title(f"Weight Map [{weights.min():.2f}, {weights.max():.2f}]")
        
        binary_mask = (weights > cutoff).astype(np.uint8)
        axes[2].imshow(binary_mask, cmap='gray')
        axes[2].set_title(f"Binary Mask (cutoff={cutoff})")
    else:
        axes[1].imshow(image, cmap='gray')
        axes[1].set_title("Weight Map (N/A)")
        axes[2].imshow(image, cmap='gray')
        axes[2].set_title("Binary Mask (N/A)")
    axes[1].axis('off')
    axes[2].axis('off')
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


def process_directory(model, input_dir: str, output_dir: str, config: dict, 
                      particle_type: str = 'Unknown',
                      extensions: tuple = ('.jpg', '.png', '.tif', '.tiff'),
                      detection_mode: str = 'standard'):
    base_output = os.path.join(output_dir, particle_type)
    detections_dir = os.path.join(base_output, 'detections')
    weight_maps_dir = os.path.join(base_output, 'detections_with_weight_maps')
    
    os.makedirs(detections_dir, exist_ok=True)
    os.makedirs(weight_maps_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(extensions)]
    image_files.sort()
    
    if not image_files:
        logger.warning(f"No images found in {input_dir}")
        return [], base_output
    
    logger.info(f"Processing {len(image_files)} images from {input_dir} (mode: {detection_mode})")
    
    results = []
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        image = np.array(dt.LoadImage(image_path).resolve()).astype(np.float32)
        
        detections, weights, _ = detect_particles(model, image, config, detection_mode)
        
        base_name = os.path.splitext(image_file)[0]
        
        weight_map_path = os.path.join(weight_maps_dir, f"{base_name}.png")
        visualize_detections(image, detections, weights, f"Detection: {image_file}", weight_map_path, 
                             cutoff=config.get('cutoff', 0.9))
        
        detection_path = os.path.join(detections_dir, f"{base_name}.png")
        save_image_with_detections(image, detections, detection_path)
        
        results.append({
            'image': image_file,
            'num_detections': len(detections),
            'detections': detections.tolist() if len(detections) > 0 else []
        })
        
        logger.info(f"  {image_file}: {len(detections)} detections")
    
    return results, base_output


def process_single_image(model, image_path: str, output_dir: str, config: dict, 
                         particle_type: str = 'Unknown',
                         detection_mode: str = 'standard'):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    base_output = os.path.join(output_dir, particle_type)
    detections_dir = os.path.join(base_output, 'detections')
    weight_maps_dir = os.path.join(base_output, 'detections_with_weight_maps')
    
    os.makedirs(detections_dir, exist_ok=True)
    os.makedirs(weight_maps_dir, exist_ok=True)
    
    image = np.array(dt.LoadImage(image_path).resolve()).astype(np.float32)
    detections, weights, _ = detect_particles(model, image, config, detection_mode)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    weight_map_path = os.path.join(weight_maps_dir, f"{base_name}.png")
    visualize_detections(image, detections, weights, f"Detection: {os.path.basename(image_path)}", weight_map_path,
                         cutoff=config.get('cutoff', 0.9))
    
    detection_path = os.path.join(detections_dir, f"{base_name}.png")
    save_image_with_detections(image, detections, detection_path)
    
    logger.info(f"Detected {len(detections)} particles in {image_path} (mode: {detection_mode})")
    return detections, base_output


def main():
    parser = argparse.ArgumentParser(description='Detect particles in images without ground truth')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model file')
    parser.add_argument('--input', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--output', type=str, default='detection_results', help='Output directory for results')
    parser.add_argument('--particle', type=str, default='Unknown', help='Particle type name for output folder')
    parser.add_argument('--config', type=str, default='src/config.yaml', help='Path to configuration file')
    parser.add_argument('--fps', type=float, default=30.0, help='Frames per second for output video')
    parser.add_argument('--no-video', action='store_true', help='Disable video generation')
    parser.add_argument('--detection-mode', type=str, default='standard', choices=['standard', 'area'],
                        help='Detection mode: standard (local maxima) or area (area-based filtering)')
    args = parser.parse_args()
    
    config = utils.load_yaml(args.config)
    
    logger.info(f"=== Particle Detection Started ===")
    logger.info(f"Model: {args.model}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Particle type: {args.particle}")
    logger.info(f"Detection mode: {args.detection_mode}")
    if args.detection_mode == 'area':
        area_cfg = config.get('area_detection', {})
        logger.info(f"Area params: min_area={area_cfg.get('min_area', 100)}, max_area={area_cfg.get('max_area', 2500)}")
    else:
        logger.info(f"Detection params: alpha={config.get('alpha', 0.2)}, beta={config.get('beta', 0.8)}, cutoff={config.get('cutoff', 0.2)}")
    
    model = load_trained_model(args.model, config)
    
    if os.path.isdir(args.input):
        results, base_output = process_directory(model, args.input, args.output, config, 
                                                  particle_type=args.particle,
                                                  detection_mode=args.detection_mode)
        total = sum(r['num_detections'] for r in results)
        logger.info(f"\n=== Summary ===")
        logger.info(f"Processed {len(results)} images")
        logger.info(f"Total detections: {total}")
        
        summary_path = os.path.join(base_output, 'detection_summary.yaml')
        utils.save_yaml({'images': results, 'total_detections': total}, summary_path)
        logger.info(f"Summary saved to {summary_path}")
        
        if not args.no_video and len(results) > 1:
            detections_dir = os.path.join(base_output, 'detections')
            video_path = os.path.join(base_output, f'{args.particle}_detections.mp4')
            utils.create_video_from_detections(detections_dir, video_path, fps=args.fps)
            logger.info(f"Video saved to {video_path}")
    else:
        detections, base_output = process_single_image(model, args.input, args.output, config, 
                                                        particle_type=args.particle,
                                                        detection_mode=args.detection_mode)
        logger.info(f"\n=== Result ===")
        logger.info(f"Detected {len(detections)} particles")
        
        summary_path = os.path.join(base_output, 'detection_summary.yaml')
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        utils.save_yaml({
            'images': [{'image': base_name, 'num_detections': len(detections), 
                       'detections': detections.tolist() if len(detections) > 0 else []}],
            'total_detections': len(detections)
        }, summary_path)
        logger.info(f"Summary saved to {summary_path}")


if __name__ == '__main__':
    main()
