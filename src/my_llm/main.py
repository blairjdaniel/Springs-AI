# Load configuration in a Python file (e.g., main.py or a notebook cell)
import yaml

with open("config/model_config.yaml", "r") as file:
    config = yaml.safe_load(file)

print(config)