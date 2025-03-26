import yaml
from pathlib import Path

CONFIG_PATH = Path("config.yaml")

def load_config():
    with CONFIG_PATH.open("r") as f:
        return yaml.safe_load(f)

# Always reload the config dynamically
def get_config():
    return load_config()

config = get_config()