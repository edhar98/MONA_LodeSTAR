import os
import yaml
import logging
from datetime import datetime


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


def setup_logger(name='default', log_file=None):
    """
    Setup logger configuration.
    
    Args:
        name (str): Logger name
        log_file (str): Optional log filename. If None, creates timestamped filename.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create timestamped log filename if not provided
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'{name}_{timestamp}.log'
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(f'logs/{log_file}', mode='w')
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger