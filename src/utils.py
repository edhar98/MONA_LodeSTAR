import os
import yaml
import logging
from datetime import datetime
from jinja2 import Environment, BaseLoader


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