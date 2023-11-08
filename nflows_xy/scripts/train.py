from copy import deepcopy
from pathlib import Path
import logging

from jsonargparse import (
    ArgumentParser,
    ActionConfigFile,
    ActionYesNo,
    class_from_function,
    Namespace,
)
from jsonargparse.typing import Path_dc
import torch

from nflows_xy.core import Flow, FlowBasedSampler
from nflows_xy.plot import plot_training_metrics, plot_test_metrics, plot_spins
from nflows_xy.scripts.io import TrainingDirectory
from nflows_xy.train import train, test
from nflows_xy.utils import make_banner
from nflows_xy.xy import action

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = ArgumentParser(
    prog="train", description="Train a Normalising Flow model from scratch."
)
parser.add_argument(
    "--model",
    type=Flow,
    required=True,
    help="Specifies the Normalising Flow model.",
)
parser.add_class_arguments(class_from_function(action), "target")
parser.add_function_arguments(train, "train", skip=["model"])
parser.add_function_arguments(test, "test", skip=["model"])
parser.add_argument("--cuda", action=ActionYesNo, help="Train using CUDA.")
parser.add_argument(
    "--double", action=ActionYesNo, help="Use double precision."
)
parser.add_argument(
    "-o", "--output", type=Path_dc, help="Location to save trained model."
)
parser.add_argument("-c", "--config", action=ActionConfigFile)


def main(config: Namespace) -> None:
    config_copy = deepcopy(config)

    config = parser.instantiate_classes(config)

    if config.output is None:
        logger.warning(
            "No output directory specified: trained model will not be saved!"
        )
        output_path = None

    else:
        output_path = Path(config.output)
        if output_path.exists():
            raise FileExistsError(f"{output_path} already exists!")

    model = FlowBasedSampler(config.model, config.target)

    device = "cuda" if config.cuda else "cpu"
    dtype = torch.float64 if config.double else torch.float32
    model = model.to(device=device, dtype=dtype)

    training_metrics = train(model, **config.train)

    fields, actions = model(500000)
    fig = plot_spins(fields.outputs)
    print(fig)

    logger.info("Plotting training metrics...")
    figs = plot_training_metrics(training_metrics)
    print(make_banner("Training metrics"))
    print("\n".join(list(figs.values())))

    test_metrics = test(model, **config.test)

    logger.info("Plotting test metrics...")
    figs = plot_test_metrics(test_metrics)
    print(make_banner("Test metrics"))
    print("\n".join(list(figs.values())))

    print(test_metrics.describe())

    if output_path is not None:
        _ = TrainingDirectory.new(
            output_path,
            model=model.flow,
            config=config_copy,
            metrics=training_metrics,
        )
