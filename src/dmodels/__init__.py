import yaml
from pathlib import Path

data_root = Path(__file__).parent / "data"

def get_cfg(name):
    path = data_root / name
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)
    