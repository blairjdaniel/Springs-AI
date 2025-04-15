import logging.config
import yaml

def setup_logging(config_path="config/logging_config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        logging.config.dictConfig(config)

# Initialize logging configuration at the start of your application
setup_logging()

logger = logging.getLogger("myapp")
logger.info("Logging configuration loaded successfully!")