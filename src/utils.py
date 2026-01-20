import os
import yaml
import logging
from jinja2 import Environment, BaseLoader
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import cv2
import pandas as pd

def calculate_detection_metrics(gt_bboxes, detections, gt_labels=None, detection_labels=None, distance_threshold=20):
    if len(gt_bboxes) == 0 and len(detections) == 0:
        return {'precision': 1.0, 'recall': 1.0, 'f1_score': 1.0, 'tp': 0, 'fp': 0, 'fn': 0}
    
    if len(gt_bboxes) == 0:
        return {'precision': 0.0, 'recall': 1.0, 'f1_score': 0.0, 'tp': 0, 'fp': len(detections), 'fn': 0}
    
    if len(detections) == 0:
        return {'precision': 1.0, 'recall': 0.0, 'f1_score': 0.0, 'tp': 0, 'fp': 0, 'fn': len(gt_bboxes)}
    
    gt_positions = gt_bboxes[:, :2]
    det_positions = detections[:, :2]
    
    distances = cdist(gt_positions, det_positions)
    
    matched_gt = set()
    matched_det = set()
    true_positives = 0
    
    for gt_idx in range(len(gt_positions)):
        for det_idx in range(len(det_positions)):
            if distances[gt_idx, det_idx] <= distance_threshold:
                type_match = True
                if gt_labels is not None and detection_labels is not None:
                    if gt_idx < len(gt_labels) and det_idx < len(detection_labels):
                        type_match = gt_labels[gt_idx] == detection_labels[det_idx]
                
                if type_match and gt_idx not in matched_gt and det_idx not in matched_det:
                    matched_gt.add(gt_idx)
                    matched_det.add(det_idx)
                    true_positives += 1
    
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

def _load_config(file_name='samples.yaml'):
    config_path = os.path.join(os.path.dirname(__file__), file_name)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_yaml(path_file):
    """
    Load a YAML file.
    """
    with open(path_file, encoding="utf-8") as f:
        dictionary = yaml.safe_load(f)
    return dictionary


def save_yaml(dictionary, path_file):
    """
    Save a dictionary as a YAML file.
    """
    # Only create directory if path_file contains a directory
    dir_path = os.path.dirname(path_file)
    if dir_path:  # Only create directory if it's not empty
        os.makedirs(dir_path, exist_ok=True)
    with open(path_file, 'w', encoding="utf-8") as f:
        yaml.dump(dictionary, f, default_flow_style=False)


def setup_logger(name, log_file=None, level=logging.INFO):
    """Set up logger with console and optional file handler"""
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

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

class XMLWriter:
    """XML annotation writer for Pascal VOC format"""
    
    def __init__(self, path, width, height, depth=3, database='Unknown', segmented=0):
        # XML template for Pascal VOC format
        self.xml_template = """<annotation>
    <folder>{{ folder }}</folder>
    <filename>{{ filename }}</filename>
    <path>{{ path }}</path>
    <source>
        <database>{{ database }}</database>
    </source>
    <size>
        <width>{{ width }}</width>
        <height>{{ height }}</height>
        <depth>{{ depth }}</depth>
    </size>
    <segmented>{{ segmented }}</segmented>
{% if snr is not none %}    <snr>{{ snr }}</snr>{% endif %}
{% for object in objects %}    <object>
        <name>{{ object.name }}</name>
        <orientation>{{ object.orientation }}</orientation>
        <bndbox>
            <xmin>{{ object.xmin }}</xmin>
            <ymin>{{ object.ymin }}</ymin>
            <xmax>{{ object.xmax }}</xmax>
            <ymax>{{ object.ymax }}</ymax>
        </bndbox>
    </object>{% endfor %}
</annotation>"""
        
        # Create Jinja2 environment
        self.environment = Environment(loader=BaseLoader())
        self.template = self.environment.from_string(self.xml_template)
        
        abspath = os.path.abspath(path)
        
        self.template_parameters = {
            'path': abspath,
            'filename': os.path.basename(abspath),
            'folder': os.path.basename(os.path.dirname(abspath)),
            'width': width,
            'height': height,
            'depth': depth,
            'database': database,
            'segmented': segmented,
            'objects': [],
            'snr': None
        }
    
    def addObject(self, name, xmin, ymin, xmax, ymax, orientation=0):
        """Add an object annotation to the XML"""
        self.template_parameters['objects'].append({
            'name': name,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'orientation': orientation,
        })
    
    def setSNR(self, snr):
        """Set the SNR value for the annotation"""
        self.template_parameters['snr'] = snr
    
    def save(self, annotation_path):
        """Save the XML annotation to file"""
        with open(annotation_path, 'w') as file:
            content = self.template.render(**self.template_parameters)
            file.write(content)

# Alias for backward compatibility
Writer = XMLWriter


def preprocess_image(image: np.ndarray) -> np.ndarray:
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
        else:
            image = image[0] if image.shape[0] < image.shape[-1] else image[..., 0]
    elif len(image.shape) > 3:
        if image.shape[1] == 3:
            image = np.dot(image[0].transpose(1, 2, 0), [0.299, 0.587, 0.114])
        else:
            image = image[0, 0] if len(image.shape) == 4 else image[0]
    
    if len(image.shape) != 2:
        raise ValueError(f"Image must be 2D after processing, got shape {image.shape}")
    
    return image


def cluster_nearby_detections(detections: np.ndarray, distance_threshold: float = 20) -> np.ndarray:
    if len(detections) <= 1:
        return detections
    
    tree = cKDTree(detections)
    pairs = tree.query_pairs(r=distance_threshold)
    
    n = len(detections)
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    for i, j in pairs:
        union(i, j)
    
    clusters = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(i)
    
    return np.array([np.mean(detections[indices], axis=0) for indices in clusters.values()])


def load_csv_ground_truth(csv_path: str) -> dict:
    df = pd.read_csv(csv_path, index_col=0)
    
    sorted_frames = sorted(df['frame'].unique())
    
    frames = {}
    for image_idx, frame_val in enumerate(sorted_frames):
        frame_data = df[df['frame'] == frame_val]
        frames[image_idx] = {
            'positions': frame_data[['x', 'y']].values,
            'phi': frame_data['phi'].values,
            'max_intensity': frame_data['max_inensity'].values,
            'summed_intensity': frame_data['summed_inensity'].values,
            'frame': int(frame_val)
        }
    return frames


def detect_by_area(weights: np.ndarray, cutoff: float = 0.9,
                   min_area: int = 100, max_area: int = 2500) -> np.ndarray:
    if weights is None:
        return np.empty((0, 2))
    
    binary_mask = (weights > cutoff).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        detections.append([cx, cy])
    
    return np.array(detections) if detections else np.empty((0, 2))


def save_image_with_detections(image: np.ndarray, detections: np.ndarray, save_path: str,
                               gt_bboxes: np.ndarray = None,
                               det_color: tuple = (255, 0, 0), gt_color: tuple = (0, 255, 0),
                               marker_radius: int = 3, marker_thickness: int = 1):
    image = preprocess_image(image)
    
    if image.max() <= 1.0:
        image_uint8 = (image * 255).astype(np.uint8)
    else:
        image_uint8 = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
    
    image_rgb = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)
    
    if gt_bboxes is not None and len(gt_bboxes) > 0:
        for x, y in gt_bboxes[:, :2]:
            cv2.circle(image_rgb, (int(x), int(y)), marker_radius, gt_color, marker_thickness)
    
    if len(detections) > 0:
        for det in detections:
            x, y = det[0], det[1]
            cv2.circle(image_rgb, (int(x), int(y)), marker_radius, det_color, marker_thickness)
    
    cv2.imwrite(save_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def create_video_from_detections(images_dir: str, output_path: str, fps: int = 10, 
                                  extensions: tuple = ('.jpg', '.png', '.tif', '.tiff')) -> str:
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(extensions)])
    
    if not image_files:
        raise ValueError(f"No images found in {images_dir}")
    
    first_image = cv2.imread(os.path.join(images_dir, image_files[0]))
    if first_image is None:
        raise ValueError(f"Could not read first image: {image_files[0]}")
    
    height, width = first_image.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        frame = cv2.imread(image_path)
        if frame is not None:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            video_writer.write(frame)
    
    video_writer.release()
    return output_path