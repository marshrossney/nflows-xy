from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import logging

from jsonargparse import ArgumentParser, ActionConfigFile, Namespace
from jsonargparse.typing import Path_dw, Path_dc
import pandas as pd

from nflows_xy.autocorr import autocorrelations
from nflows_xy.core import FlowBasedSampler, PullbackAction
from nflows_xy.hmc import hmc
from nflows_xy.plot import plot_observable, plot_spin_correlation
from nflows_xy.scripts.io import TrainingDirectory, SamplingDirectory
from nflows_xy.train import test
from nflows_xy.transforms.module import (
    UnivariateTransformModule,
    dilute_module,
)
from nflows_xy.utils import make_banner
from nflows_xy.xy import top_charge, spin_correlation, fit_spin_correlation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = ArgumentParser(
    prog="hmc",
    description="Run a 'flowed' Hybrid Monte Carlo simulation.",
)
parser.add_argument(
    "-m",
    "--model",
    type=Path_dw,
    required=True,
    help="path to a trained Normalising Flow model",
)
parser.add_function_arguments(test, "test", skip=["model"])
parser.add_function_arguments(hmc, "hmc", skip=["action"])
parser.add_argument("-c", "--config", action=ActionConfigFile)
parser.add_argument(
    "-o", "--output", type=Path_dc, help="location to save outputs"
)

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
    config_copy = deepcopy(config)

    if config.output is None:
        logger.warning(
            "No output directory specified: Sample outputs will not be saved!"
        )
        output_path = None

    else:
        output_path = Path(config.output)
        if output_path.exists():
            raise FileExistsError(f"{output_path} already exists!")

    logger.info(f"Loading trained model from {config.model}")
    train_config = TrainingDirectory(config.model).load_config()

    model = train_config.model
    target = train_config.target
    print(model)

    if config.modify.beta is not None:
        target.beta = config.modify.beta
    if config.modify.lattice_size is not None:
        target.lattice_size = config.modify.lattice_size
    if config.modify.dilution is not None:
        for module in model.modules():
            if isinstance(module, UnivariateTransformModule):
                dilute_module(module, config.modify.dilution)

    test_metrics = test(FlowBasedSampler(model, target), **config.test)

    print(make_banner("Test metrics"))
    print(test_metrics.describe())

    φ, hmc_metrics = hmc(PullbackAction(model, target), **config.hmc)

    S = target(φ)
    Γ_S = autocorrelations(S)
    logger.info("Plotting the action...")
    figs = plot_observable(S, Γ_S, "S")

    print(make_banner("Action plots"))
    print("\n".join(list(figs.values())))

    Q = top_charge(φ)
    Γ_Q = autocorrelations(Q)
    logger.info("Plotting the topological charge...")
    figs = plot_observable(Q, Γ_Q, "Q")

    print(make_banner("Topological charge plots"))
    print("\n".join(list(figs.values())))

    G = spin_correlation(φ)
    (ξ, c), cov = fit_spin_correlation(G)
    logger.info("Plotting the spin correlation function...")
    fig = plot_spin_correlation(G.mean(0).log(), ξ, c)

    print(make_banner("Spin correlation"))
    print(fig)

    metrics = pd.Series(
        test_metrics.median().to_dict()
        | asdict(hmc_metrics)
        | {
            "corr_len": ξ.item(),
            "tau_int_S": Γ_S.integrated,
            "tau_int_Q": Γ_Q.integrated,
        }
    )
    print(make_banner("Summary"))
    print(metrics)

    if output_path is not None:
        _ = SamplingDirectory.new(
            output_path,
            sample=φ,
            config=config_copy,
            metrics=metrics,
        )
