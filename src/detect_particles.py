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
                     detection_mode: str = 'standard',
                     template_bank: dict = None,
                     template_refine_radius: int = 25,
                     template_search_radius: int = 5) -> tuple:
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
        
        orientations = None
        orientation_ncc = None
        if detection_mode == 'area':
            area_config = config.get('area_detection', {})
            clustered_detections = utils.detect_by_area(
                weights,
                cutoff=config.get('cutoff', 0.9),
                min_area=area_config.get('min_area', 100),
                max_area=area_config.get('max_area', 2500)
            )
            logger.info(f"Area detection: found {len(clustered_detections)} particles")
        elif detection_mode == 'watershed':
            ws_config = config.get('watershed_detection', {})
            clustered_detections = utils.detect_by_watershed(
                weights,
                cutoff=config.get('cutoff', 0.3),
                min_distance=ws_config.get('min_distance', 10),
                min_area=ws_config.get('min_area', 20),
            )
            logger.info(f"Watershed detection: found {len(clustered_detections)} particles")
        elif detection_mode == 'template':
            if template_bank is None:
                raise ValueError("template_bank is required for detection_mode='template'")
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
                oriented = utils.orientation_postprocess(
                    image=image,
                    detections=clustered_detections,
                    template_bank=template_bank,
                    refine_radius=template_refine_radius,
                    search_r=template_search_radius,
                )
                clustered_detections = oriented[:, :2]
                orientations = oriented[:, 2]
                orientation_ncc = oriented[:, 3]
            else:
                clustered_detections = np.empty((0, 2))
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

    return clustered_detections, weights, model_output, orientations, orientation_ncc


def save_image_with_detections(image: np.ndarray, detections: np.ndarray, save_path: str,
                               marker_color: tuple = (255, 0, 0), marker_radius: int = 3,
                               marker_thickness: int = 1,
                               orientations: np.ndarray = None):
    utils.save_image_with_detections(image, detections, save_path, 
                                      det_color=marker_color, marker_radius=marker_radius,
                                      marker_thickness=marker_thickness,
                                      orientations=orientations)
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
                      detection_mode: str = 'standard',
                      template_bank: dict = None,
                      template_refine_radius: int = 25,
                      template_search_radius: int = 5):
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
        
        detections, weights, _, orientations, orientation_ncc = detect_particles(
            model, image, config, detection_mode,
            template_bank=template_bank,
            template_refine_radius=template_refine_radius,
            template_search_radius=template_search_radius,
        )
        
        base_name = os.path.splitext(image_file)[0]
        
        weight_map_path = os.path.join(weight_maps_dir, f"{base_name}.png")
        visualize_detections(image, detections, weights, f"Detection: {image_file}", weight_map_path, 
                             cutoff=config.get('cutoff', 0.9))
        
        detection_path = os.path.join(detections_dir, f"{base_name}.png")
        save_image_with_detections(image, detections, detection_path, orientations=orientations)
        
        row = {
            'image': image_file,
            'num_detections': len(detections),
            'detections': detections.tolist() if len(detections) > 0 else []
        }
        if orientations is not None:
            row['orientations'] = orientations.tolist()
        if orientation_ncc is not None:
            row['orientation_ncc'] = orientation_ncc.tolist()
        results.append(row)
        
        logger.info(f"  {image_file}: {len(detections)} detections")
    
    return results, base_output


def process_single_image(model, image_path: str, output_dir: str, config: dict, 
                         particle_type: str = 'Unknown',
                         detection_mode: str = 'standard',
                         template_bank: dict = None,
                         template_refine_radius: int = 25,
                         template_search_radius: int = 5):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    base_output = os.path.join(output_dir, particle_type)
    detections_dir = os.path.join(base_output, 'detections')
    weight_maps_dir = os.path.join(base_output, 'detections_with_weight_maps')
    
    os.makedirs(detections_dir, exist_ok=True)
    os.makedirs(weight_maps_dir, exist_ok=True)
    
    image = np.array(dt.LoadImage(image_path).resolve()).astype(np.float32)
    detections, weights, _, orientations, _ = detect_particles(
        model, image, config, detection_mode,
        template_bank=template_bank,
        template_refine_radius=template_refine_radius,
        template_search_radius=template_search_radius,
    )
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    weight_map_path = os.path.join(weight_maps_dir, f"{base_name}.png")
    visualize_detections(image, detections, weights, f"Detection: {os.path.basename(image_path)}", weight_map_path,
                         cutoff=config.get('cutoff', 0.9))
    
    detection_path = os.path.join(detections_dir, f"{base_name}.png")
    save_image_with_detections(image, detections, detection_path, orientations=orientations)
    
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
    parser.add_argument('--detection-mode', type=str, default='standard', choices=['standard', 'area', 'watershed', 'template'],
                        help='Detection mode: standard (local maxima), area, watershed (touching particles), or template orientation')
    parser.add_argument('--orientation-template', type=str, default=None, help='Template image path for template mode')
    parser.add_argument('--template-angle-step', type=int, default=2, help='Template rotation step in degrees')
    parser.add_argument('--template-phi-deg', type=float, default=None, help='Template reference phi in degrees')
    parser.add_argument('--template-refine-radius', type=int, default=25, help='Center refine radius in pixels')
    parser.add_argument('--template-search-radius', type=int, default=5, help='Template local search radius in pixels')
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
        template_bank = None
    elif args.detection_mode == 'watershed':
        ws_cfg = config.get('watershed_detection', {})
        logger.info(f"Watershed params: min_distance={ws_cfg.get('min_distance', 10)}, min_area={ws_cfg.get('min_area', 20)}, cutoff={config.get('cutoff', 0.3)}")
        template_bank = None
    elif args.detection_mode == 'template':
        if not args.orientation_template:
            raise ValueError("--orientation-template is required for --detection-mode template")
        template_bank = utils.build_template_bank(
            sample_path=args.orientation_template,
            angle_step=args.template_angle_step,
            template_phi_deg=args.template_phi_deg,
        )
        logger.info(f"Template mode: template={args.orientation_template}, angle_step={args.template_angle_step}, refine_radius={args.template_refine_radius}, search_radius={args.template_search_radius}")
    else:
        logger.info(f"Detection params: alpha={config.get('alpha', 0.2)}, beta={config.get('beta', 0.8)}, cutoff={config.get('cutoff', 0.2)}")
        template_bank = None
    
    model = load_trained_model(args.model, config)
    
    if os.path.isdir(args.input):
        results, base_output = process_directory(model, args.input, args.output, config, 
                                                  particle_type=args.particle,
                                                  detection_mode=args.detection_mode,
                                                  template_bank=template_bank,
                                                  template_refine_radius=args.template_refine_radius,
                                                  template_search_radius=args.template_search_radius)
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
                                                        detection_mode=args.detection_mode,
                                                        template_bank=template_bank,
                                                        template_refine_radius=args.template_refine_radius,
                                                        template_search_radius=args.template_search_radius)
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
