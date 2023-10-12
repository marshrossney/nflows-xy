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

parser = ArgumentParser(prog="test")
parser.add_argument("model", type=Path_dw, help="path to a trained model")
parser.add_function_arguments(test, "test", skip=["model"])
parser.add_argument("--dilution", type=float)


def main(config: Namespace) -> None:
    train_config = TrainingDirectory(config.model).load_config()

    model = train_config.model

    if config.dilution is not None:
        for module in model.modules():
            if isinstance(module, UnivariateTransformModule):
                dilute_module(module, config.dilution)

    model = FlowBasedSampler(flow=model, target=train_config.target)

    metrics = test(model, **config.test)
    logger.info("Plotting test metrics")
    figs = plot_test_metrics(metrics)
    print("\n".join(list(figs.values())))

    print(metrics.describe())
