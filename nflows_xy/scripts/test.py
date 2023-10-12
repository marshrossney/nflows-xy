import logging

from jsonargparse import ArgumentParser, Namespace
from jsonargparse.typing import Path_dw

from nflows_xy.core import FlowBasedSampler
from nflows_xy.train import test
from nflows_xy.scripts.io import TrainingDirectory
from nflows_xy.plot import plot_test_metrics

from nflows_xy.transforms.module import (
    UnivariateTransformModule,
    dilute_module,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = ArgumentParser(
    prog="test", description="Test a trained Normalising Flow model."
)
parser.add_argument("model", type=Path_dw, help="Path to a trained model.")
parser.add_function_arguments(test, "test", skip=["model"])
modifications = parser.add_argument_group(
    "modify",
    "Modifications to be applied to the existing flow model or target action.",
)
modifications.add_argument(
    "--modify.beta",
    type=float,
    help="Modify the target beta.",
)
modifications.add_argument(
    "--modify.lattice_size",
    type=int,
    help="Modify the target lattice size.",
)
modifications.add_argument(
    "--modify.dilution",
    type=float,
    help="Dilute the flow by mixing with identity.",
)


def main(config: Namespace) -> None:
    train_config = TrainingDirectory(config.model).load_config()

    model = train_config.model
    target = train_config.target

    if config.modify.beta is not None:
        target.beta = config.modify.beta
    if config.modify.lattice_size is not None:
        target.lattice_size = config.modify.lattice_size
    if config.modify.dilution is not None:
        for module in model.modules():
            if isinstance(module, UnivariateTransformModule):
                dilute_module(module, config.modify.dilution)

    model = FlowBasedSampler(model, target)

    metrics = test(model, **config.test)
    logger.info("Plotting test metrics")
    figs = plot_test_metrics(metrics)
    print("\n".join(list(figs.values())))

    print(metrics.describe())
