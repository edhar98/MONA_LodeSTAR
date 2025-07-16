import os
import yaml
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
    os.makedirs(os.path.dirname(path_file), exist_ok=True)
    with open(path_file, 'w', encoding="utf-8") as f:
        yaml.dump(dictionary, f, default_flow_style=False)