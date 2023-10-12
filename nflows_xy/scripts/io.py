from datetime import datetime
import importlib.metadata
from pathlib import Path
import logging
import shutil
import subprocess

from jsonargparse import Namespace
import pandas as pd
import torch

from nflows_xy.flows import Flow

Tensor = torch.Tensor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class _ExistingDirectory:
    def __init__(self, path: str | Path):
        path = path if isinstance(path, Path) else Path(path)
        path = path.resolve()
        if not path.is_dir():
            raise NotADirectoryError(f"No directory found at {path}")

        self._path = path

    @property
    def path(self) -> Path:
        return self._path


class TrainingDirectory(_ExistingDirectory):
    _config_file = "config.yaml"
    _checkpoint_file = "checkpoint.pt"
    _metrics_file = "training_metrics.csv"

    @classmethod
    def new(
        cls,
        path: str | Path,
        *,
        model: Flow,
        config: Namespace,
        metrics: pd.DataFrame,
    ) -> "TrainingDirectory":
        path = path if isinstance(path, Path) else Path(path)
        path = path.resolve()

        logger.info(f"Creating directory at {path}")
        path.mkdir(parents=True, exist_ok=False)

        from nflows_xy.scripts.train import parser

        config = parser.dump(config, skip_none=False)
        config = get_meta() + "\n" + config

        config_file = path / cls._config_file
        logger.info(f"Saving config to {config_file}")
        with config_file.open("w") as file:
            file.write(config)

        checkpoint_file = path / cls._checkpoint_file
        logger.info(f"Saving model to {checkpoint_file}")
        torch.save(model.state_dict(), checkpoint_file)

        metrics_file = path / cls._metrics_file
        logger.info(f"Saving training metrics to {metrics_file}")
        metrics.to_csv(metrics_file, index=False)

        return cls(path)

    @property
    def config_file(self) -> Path:
        return self.path / self._config_file

    @property
    def checkpoint_file(self) -> Path:
        return self.path / self._checkpoint_file

    @property
    def metrics_file(self) -> Path:
        return self.path / self._metrics_file

    def load_config(self) -> Namespace:
        from nflows_xy.scripts.train import parser

        config = parser.parse_path(self.config_file)
        config = parser.instantiate_classes(config)

        logger.info(f"Loading parameters from {self.checkpoint_file}")
        checkpoint = torch.load(self.checkpoint_file)
        config.model.load_state_dict(checkpoint)

        return config


class SamplingDirectory(_ExistingDirectory):
    _config_file = "config.yaml"
    _training_config_file = "training_config.yaml"
    _sample_file = "sample.pt"
    _metrics_file = "metrics.csv"

    @classmethod
    def new(
        cls,
        path: str | Path,
        *,
        sample: Tensor,
        config: Namespace,
        metrics: pd.Series,
    ) -> "SamplingDirectory":
        path = path if isinstance(path, Path) else Path(path)
        path = path.resolve()

        logger.info(f"Creating directory at {path}")
        path.mkdir(parents=True, exist_ok=False)

        if "model" in config:
            model_path = config.model
            logger.info(f"Copying training config from {model_path}")
            shutil.copy(
                TrainingDirectory(model_path).config_file,
                path / cls._training_config_file,
            )

            from nflows_xy.scripts.fhmc import parser

        else:
            from nflows_xy.scripts.hmc import parser

        config = parser.dump(config, skip_none=False)
        config = get_meta() + "\n" + config

        config_file = path / cls._config_file
        logger.info(f"Saving config to {config_file}")
        with config_file.open("w") as file:
            file.write(config)

        sample_file = path / cls._sample_file
        logger.info(f"Saving sample to {sample_file}")
        torch.save(sample, sample_file)

        metrics_file = path / cls._metrics_file
        logger.info(f"Saving metrics to {metrics_file}")
        metrics.to_csv(metrics_file)

        return cls(path)

    @property
    def config_file(self) -> Path:
        return self.path / self._config_file

    @property
    def training_config_file(self) -> Path:
        return self.path / self._training_config_file

    @property
    def sample_file(self) -> Path:
        return self.path / self._sample_file

    @property
    def metrics_file(self) -> Path:
        return self.path / self._metrics_file

    def load_config(self) -> Namespace:
        from nflows_xy.scripts.hmc import parser

        return parser.parse_path(self.config_file)

    def load_sample(self) -> Tensor:
        return torch.load(self.sample_file)
