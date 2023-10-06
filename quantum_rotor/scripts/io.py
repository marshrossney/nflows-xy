from datetime import datetime
import importlib.metadata
from pathlib import Path
import logging
import subprocess

import torch
import yaml

from quantum_rotor.model import FlowBasedModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHECKPOINT_FILE = "checkpoint.pt"
CONFIG_FILE = "config.yaml"
META_FILE = "metadata.yaml"


def get_version():
    return importlib.metadata.version("quantum_rotor")


def get_commit():
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(Path(__file__).resolve().parent),
                "rev-parse",
                "--short",
                "HEAD",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        logger.info(f"Unable to obtain git commit hash: {e}")
        return "unknown"
    else:
        return result.stdout.strip()


def get_meta() -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    version = get_version()
    commit = get_commit()
    ret = (
        f"# Run on {timestamp} using quantum_rotor v{version}, commit {commit}"
    )
    return ret


def save_model(
    path: str | Path,
    model: FlowBasedModel,
    config: str | dict,
) -> None:
    path.mkdir(parents=True, exist_ok=False)

    if isinstance(config, dict):
        config = yaml.safe_dump(config, indent=4)

    config = get_meta() + "\n" + config

    config_file = path / CONFIG_FILE
    logger.info(f"Saving config to {config_file}")
    with config_file.open("w") as file:
        file.write(config)

    checkpoint_file = path / CHECKPOINT_FILE
    logger.info(f"Saving model to {checkpoint_file}")
    torch.save(model.state_dict(), checkpoint_file)


def load_model(path: str | Path) -> FlowBasedModel:
    path = path if isinstance(path, Path) else Path(path)

    from quantum_rotor.scripts.train import parser

    config_file = path / CONFIG_FILE
    config = parser.parse_path(config_file)
    config = parser.instantiate_classes(config)
    model = config.model

    checkpoint_file = path / CHECKPOINT_FILE
    checkpoint = torch.load(checkpoint_file)

    logger.info(f"Loading model from {checkpoint_file}")
    model.load_state_dict(checkpoint)

    return model
