import os
import torch
import numpy as np
import cv2
import deeptrack.deeplay as dl
from custom_lodestar import customLodeSTAR
import utils


class CompositeLodeSTAR:
    
    def __init__(self, config, trained_models_summary):
        self.config = config
        self.models = {}
        self.model_configs = {}
        self.particle_types = []
        
        config_samples = config.get('samples', [])
        if not config_samples:
            raise ValueError("No samples specified in config file")
        
        print(f"Composite model will load only samples from config: {config_samples}")
        
        for particle_type, model_info in trained_models_summary.items():
            if particle_type not in config_samples:
                print(f"Skipping {particle_type} (not in config samples)")
                continue
            
            model_path = model_info['model_path']
            model_config_path = model_info['models_dir'] + '/config.yaml'
            model_config = utils.load_yaml(model_config_path)
            if os.path.exists(model_path):
                model = self._load_model(model_path, model_config)
                if model is not None:
                    self.models[particle_type] = model
                    self.particle_types.append(particle_type)
                    self.model_configs[particle_type] = model_config
                    print(f"Loaded {particle_type} model from {model_path}")
            else:
                print(f"Warning: Model path not found for {particle_type}: {model_path}")
        
        if not self.models:
            raise ValueError(f"No valid models found for samples: {config_samples}")
        
        print(f"Composite model initialized with {len(self.models)} models: {self.particle_types}")
    
    def _load_model(self, model_path, config):
        if config['lodestar_version'] == 'default':
            lodestar = dl.LodeSTAR(
                n_transforms=config['n_transforms'], 
                optimizer=dl.Adam(lr=config['lr'])
            ).build()
            print('Adding default LodeSTAR model')
        else:
            lodestar = customLodeSTAR(
                n_transforms=config['n_transforms'], 
                optimizer=dl.Adam(lr=config['lr'])
            ).build()
            print('Adding custom LodeSTAR model')
        if os.path.exists(model_path):
            lodestar.load_state_dict(torch.load(model_path))
            lodestar.eval()
            return lodestar
        return None
    
    def detect_and_classify(self, image, alpha=None, beta=None, mode=None, cutoff=None):
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        if len(image.shape) == 3:
            if image.shape[-1] == 3:
                image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
            elif image.shape[0] == 3:
                image = np.dot(image[:3].transpose(1, 2, 0), [0.299, 0.587, 0.114])
            elif image.shape[-1] == 1:
                image = image[..., 0]
            elif image.shape[0] == 1:
                image = image[0]
        
        if len(image.shape) != 2:
            raise ValueError(f"Image must be 2D after processing, got shape {image.shape}")
        
        h, w = image.shape
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
        
        all_outputs = {}
        all_weight_maps = {}
        all_detections = {}
        
        with torch.no_grad():
            for particle_type, model in self.models.items():
                model_output = model(image_tensor)
                all_outputs[particle_type] = model_output
                
                if len(model_output.shape) == 4 and model_output.shape[1] >= 3:
                    weights = model_output[0, -1].detach().cpu().numpy()
                    
                    if weights.shape != (h, w):
                        weights = cv2.resize(weights, (w, h), interpolation=cv2.INTER_LINEAR)
                    
                    all_weight_maps[particle_type] = weights
                
                model_config = self.model_configs[particle_type]
                model_alpha = alpha if alpha is not None else model_config.get('alpha', 0.2)
                model_beta = beta if beta is not None else model_config.get('beta', 0.8)
                model_mode = mode if mode is not None else model_config.get('mode', 'constant')
                model_cutoff = cutoff if cutoff is not None else model_config.get('cutoff', 0.2)
                
                print(f"{particle_type} detection params: alpha={model_alpha}, beta={model_beta}, cutoff={model_cutoff}, mode={model_mode}")
                
                try:
                    detections = model.detect(
                        image_tensor, 
                        alpha=model_alpha, 
                        beta=model_beta, 
                        mode=model_mode, 
                        cutoff=model_cutoff
                    )[0]
                    
                    if len(detections) > 0:
                        detections_xy = detections[:, [1, 0]]
                        all_detections[particle_type] = detections_xy
                    else:
                        all_detections[particle_type] = np.empty((0, 2))
                except Exception as e:
                    print(f"Error detecting with {particle_type} model: {e}")
                    all_detections[particle_type] = np.empty((0, 2))
        
        unified_detections = self._merge_detections(all_detections, distance_threshold=20)
        
        classified_detections = []
        classified_labels = []
        
        for det_pos in unified_detections:
            x, y = det_pos
            x_int, y_int = int(round(x)), int(round(y))
            x_int = np.clip(x_int, 0, w - 1)
            y_int = np.clip(y_int, 0, h - 1)
            
            weights_at_detection = {}
            for particle_type, weight_map in all_weight_maps.items():
                weights_at_detection[particle_type] = weight_map[y_int, x_int]
            
            best_particle_type = max(weights_at_detection, key=weights_at_detection.get)
            best_weight = weights_at_detection[best_particle_type]
            
            classified_detections.append([x, y, best_weight])
            classified_labels.append(best_particle_type)
        
        return np.array(classified_detections), classified_labels, all_weight_maps, all_outputs
    
    def _merge_detections(self, all_detections, distance_threshold=20):
        all_points = []
        for detections in all_detections.values():
            if len(detections) > 0:
                all_points.extend(detections)
        
        if not all_points:
            return np.empty((0, 2))
        
        all_points = np.array(all_points)
        
        merged = []
        used = set()
        
        for i in range(len(all_points)):
            if i in used:
                continue
            
            cluster = [i]
            used.add(i)
            
            for j in range(i + 1, len(all_points)):
                if j not in used:
                    dist = np.linalg.norm(all_points[i] - all_points[j])
                    if dist <= distance_threshold:
                        cluster.append(j)
                        used.add(j)
            
            centroid = np.mean(all_points[cluster], axis=0)
            merged.append(centroid)
        
        return np.array(merged)

