import os
import yaml
import logging
from jinja2 import Environment, BaseLoader
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.distance import cdist

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