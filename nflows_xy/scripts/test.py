from jsonargparse import ArgumentParser, Namespace
from jsonargparse.typing import Path_dw

from nflows_xy.train import test
from nflows_xy.scripts.io import load_model

from nflows_xy.transforms.module import (
    UnivariateTransformModule,
    dilute_module,
)

parser = ArgumentParser(prog="test")

parser.add_argument("model", type=Path_dw, help="path to a trained model")
parser.add_function_arguments(test, "test", skip=["model"])
parser.add_argument("--dilution", type=float, default=0.0)


def main(config: Namespace) -> None:
    model = load_model(config.model)

    if config.dilution is not None:
        for module in model.flow.modules():
            if isinstance(module, UnivariateTransformModule):
                dilute_module(module, config.dilution)

    metrics = test(model, **config.test)
    print(metrics.describe(include="all"))

    from nflows_xy.plot import plot_spins_1d

    fields, actions = model(100000)
    φ = fields.outputs

    fig = plot_spins_1d(φ)
    fig.savefig("spins.png")
