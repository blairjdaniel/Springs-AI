# config_loader.py
import yaml

def load_baseline_responses(config_path):
    """
    Load baseline responses from a YAML file.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)