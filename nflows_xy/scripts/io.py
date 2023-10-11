from datetime import datetime
import importlib.metadata
from pathlib import Path
import logging
import subprocess

import pandas as pd
import torch
import yaml

from nflows_xy.core import FlowBasedSampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHECKPOINT_FILE = "checkpoint.pt"
CONFIG_FILE = "config.yaml"
META_FILE = "metadata.yaml"
TRAINING_METRICS_FILE = "training_metrics.csv"


def get_version():
    return importlib.metadata.version("nflows_xy")


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
    ret = f"# Run on {timestamp} using nflows_xy v{version}, commit {commit}"
    return ret


def save_model(
    path: str | Path,
    model: FlowBasedSampler,
    config: str | dict,
    training_metrics: pd.DataFrame | None = None,
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

    if training_metrics is not None:
        metrics_file = path / TRAINING_METRICS_FILE
        logger.info(f"Saving training metrics to {metrics_file}")
        training_metrics.to_csv(metrics_file, index=False)



def load_model(path: str | Path) -> FlowBasedSampler:
    path = path if isinstance(path, Path) else Path(path)

    from nflows_xy.scripts.train import parser

    config_file = path / CONFIG_FILE
    config = parser.parse_path(config_file)
    config = parser.instantiate_classes(config)

    model = FlowBasedSampler(config.flow, config.target)

    checkpoint_file = path / CHECKPOINT_FILE
    checkpoint = torch.load(checkpoint_file)
    logger.info(f"Loading model from {checkpoint_file}")
    model.load_state_dict(checkpoint)

    return model
